// Samples a texture at LOD 0 between texel centers and writes the result to every texel of a storage
// texture, so the sampler's magnification filter decides the value.
[[vk::binding(0, 0)]] RWTexture2D<float4> uTarget : register(u0);
[[vk::binding(1, 0)]] Texture2D<float4> uSource : register(t0);
[[vk::binding(2, 0)]] SamplerState uSampler : register(s0);

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint w, h;
    uTarget.GetDimensions(w, h);
    if (tid.x < w && tid.y < h)
        uTarget[tid.xy] = uSource.SampleLevel(uSampler, float2(0.4, 0.4), 0);
}
