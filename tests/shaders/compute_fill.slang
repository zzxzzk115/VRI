// Compute shader that fills a storage buffer: out[i] = i + 100. One dispatch of a
// single 64-wide workgroup writes 64 elements. Validates compute pipelines +
// dispatch + storage-buffer (UAV) descriptors.
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> uOut;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uOut[tid.x] = tid.x + 100u;
}
