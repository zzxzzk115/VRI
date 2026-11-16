// 64-bit atomics on a storage buffer (VriFeature_ShaderAtomicInt64). Each of 256 invocations packs a key in
// the high 32 bits and its own index in the low 32 bits, then takes the maximum and adds 1 to a counter.
// The key peaks at invocation 200, so the maximum names it; values above 2^32 prove the high half survives.
[[vk::binding(0, 0)]] RWStructuredBuffer<uint64_t> uResult;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    const uint     key    = tid.x == 200u ? 0xFFFFFFF0u : tid.x;
    const uint64_t packed = (uint64_t(key) << 32) | uint64_t(tid.x);
    InterlockedMax(uResult[0], packed);
    InterlockedAdd(uResult[1], 0x100000001ull);
}
