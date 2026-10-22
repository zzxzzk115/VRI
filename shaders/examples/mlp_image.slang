// Trained 2 -> 32 -> 32 -> 3 RGB regressor. All network evaluation runs here.
// Pixel columns: analytic reference, FP16 MLP, absolute error amplified for inspection.
[[vk::binding(0, 0)]] RWByteAddressBuffer weights;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> predictions;
[[vk::binding(2, 0)]] [format("rgba8")] RWTexture2D<float4> image;

struct Params
{
    uint firstMatrix;
    uint firstBias;
    uint secondMatrix;
    uint secondBias;
    uint thirdMatrix;
    uint thirdBias;
    uint size;
    float errorScale;
};
[[vk::push_constant]] ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.size || id.y >= params.size)
    {
        return;
    }
    float2 point = (float2(id.xy) + 0.5) / float(params.size) * 2.0 - 1.0;
    CoopVec<half, 2> input;
    input[0] = half(point.x);
    input[1] = half(point.y);
    CoopVec<half, 32> hidden1 = coopVecMatMulAdd<half, 32, 2>(
        input, CoopVecComponentType.Float16, weights, int(params.firstMatrix),
        CoopVecComponentType.Float16, weights, int(params.firstBias), CoopVecComponentType.Float16,
        CoopVecMatrixLayout.InferencingOptimal, false, 0);
    for (int i = 0; i < 32; ++i)
    {
        hidden1[i] = max(hidden1[i], half(0));
    }
    CoopVec<half, 32> hidden2 = coopVecMatMulAdd<half, 32, 32>(
        hidden1, CoopVecComponentType.Float16, weights, int(params.secondMatrix),
        CoopVecComponentType.Float16, weights, int(params.secondBias), CoopVecComponentType.Float16,
        CoopVecMatrixLayout.InferencingOptimal, false, 0);
    for (int i = 0; i < 32; ++i)
    {
        hidden2[i] = max(hidden2[i], half(0));
    }
    CoopVec<half, 3> result = coopVecMatMulAdd<half, 3, 32>(
        hidden2, CoopVecComponentType.Float16, weights, int(params.thirdMatrix),
        CoopVecComponentType.Float16, weights, int(params.thirdBias), CoopVecComponentType.Float16,
        CoopVecMatrixLayout.InferencingOptimal, false, 0);
    float3 inferred = saturate(float3(result[0], result[1], result[2]));
    float3 reference = float3(0.5 + 0.4 * sin(2.4 * point.x + 1.1 * point.y),
                              0.5 + 0.4 * sin(2.0 * point.y - 1.3 * point.x),
                              0.5 + 0.4 * cos(2.7 * length(point)));
    predictions[id.y * params.size + id.x] = float4(inferred, 1.0);
    image[id.xy] = float4(reference, 1.0);
    image[id.xy + uint2(params.size, 0)] = float4(inferred, 1.0);
    image[id.xy + uint2(2 * params.size, 0)] = float4(saturate(abs(inferred - reference) * params.errorScale), 1.0);
}
