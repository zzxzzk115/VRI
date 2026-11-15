// Writes each texel's coordinate into a 3D storage texture.
[[vk::binding(0, 0)]] RWTexture3D<float4> uTarget;

[shader("compute")]
[numthreads(4, 4, 4)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint w, h, d;
    uTarget.GetDimensions(w, h, d);
    if (all(tid < uint3(w, h, d)))
        uTarget[tid] = float4(tid, 1);
}
