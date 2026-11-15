// One step of a mip reduction: reads the level above through a single-mip sampled view and
// writes the next level through a single-mip storage view, adding 1 to z so each level is
// distinguishable from the one it was read from.
[[vk::binding(0, 0)]] Texture2D<float4> uSource;
[[vk::binding(1, 0)]] RWTexture2D<float4> uTarget;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint w, h;
    uTarget.GetDimensions(w, h);
    if (tid.x < w && tid.y < h)
        uTarget[tid.xy] = uSource.Load(int3(tid.xy * 2, 0)) + float4(0, 0, 1, 0);
}
