// Byte lengths must describe the bound view, including its offset and range.
[[vk::binding(0, 0)]] StructuredBuffer<float4> inputBuffer;
[[vk::binding(0, 1)]] RWStructuredBuffer<uint> outputBuffer;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain()
{
    uint count, stride;
    inputBuffer.GetDimensions(count, stride);
    outputBuffer[0] = count;
    outputBuffer.GetDimensions(count, stride);
    outputBuffer[1] = count;
    outputBuffer[2] = uint(inputBuffer[0].x);
}
