// Subgroup / wave operations: each lane contributes 1 to WaveActiveSum, so the result
// equals the number of active lanes in the wave (= the subgroup/wave size for a full
// wave). The test dispatches one full wave and checks the result matches the device's
// reported subgroupSize. Maps to Vulkan subgroups (SPIR-V) and SM6.0 wave intrinsics
// (DXIL cs_6_0).
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> outBuf;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    outBuf[tid.x] = WaveActiveSum(1u); // active lane count of this thread's wave
}
