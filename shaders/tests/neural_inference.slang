// 8 -> 16 -> 4 MLP. Weights/biases are FP16; matrices use the device's
// inference-optimal layout, supplied through VRI's host conversion API.
[[vk::binding(0, 0)]] RWByteAddressBuffer uWeights;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> uOut;

struct Offsets
{
    uint firstMatrix;
    uint firstBias;
    uint secondMatrix;
    uint secondBias;
};
[[vk::push_constant]] ConstantBuffer<Offsets> uOffsets;

[shader("compute")]
[numthreads(32, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= 65)
        return;
    CoopVec<half, 8> input;
    for (int i = 0; i < 8; ++i)
        input[i] = half((int(tid.x % 9) - 4) * (i + 1) / 32.0);
    CoopVec<half, 16> hidden = coopVecMatMulAdd<half, 16, 8>(
        input, CoopVecComponentType.Float16, uWeights, int(uOffsets.firstMatrix),
        CoopVecComponentType.Float16, uWeights, int(uOffsets.firstBias),
        CoopVecComponentType.Float16, CoopVecMatrixLayout.InferencingOptimal, false, 0);
    for (int i = 0; i < 16; ++i)
        hidden[i] = max(hidden[i], half(0));
    CoopVec<half, 4> output = coopVecMatMulAdd<half, 4, 16>(
        hidden, CoopVecComponentType.Float16, uWeights, int(uOffsets.secondMatrix),
        CoopVecComponentType.Float16, uWeights, int(uOffsets.secondBias),
        CoopVecComponentType.Float16, CoopVecMatrixLayout.InferencingOptimal, false, 0);
    uOut[tid.x] = float4(output[0], output[1], output[2], output[3]);
}
