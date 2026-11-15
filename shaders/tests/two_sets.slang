// Adds texel (0, 0) of a texture from set 0 and one from set 1 into every texel of a storage texture.
// D3D12 registers follow VRI's mapping: each set is its own space, table resources count from 0.
[[vk::binding(0, 0)]] RWTexture2D<float4> uTarget : register(u0);
[[vk::binding(1, 0)]] Texture2D<float4> uFirst : register(t0);
[[vk::binding(0, 1)]] Texture2D<float4> uSecond : register(t0, space1);

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint w, h;
    uTarget.GetDimensions(w, h);
    if (tid.x < w && tid.y < h)
        uTarget[tid.xy] = uFirst.Load(int3(0, 0, 0)) + uSecond.Load(int3(0, 0, 0));
}
