// Writes (width, height, layers, 1) of the bound array storage view into every texel of every
// layer it covers.
[[vk::binding(0, 0)]] RWTexture2DArray<float4> uTarget;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint w, h, layers;
    uTarget.GetDimensions(w, h, layers);
    if (tid.x < w && tid.y < h && tid.z < layers)
        uTarget[tid] = float4(w, h, layers, 1);
}
