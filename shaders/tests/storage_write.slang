// Writes (width, height, 7, 1) of the bound storage view into every texel. The D3D12 view tests
// read back which subresource received the write and which size the view reported.
[[vk::binding(0, 0)]] RWTexture2D<float4> uTarget;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint w, h;
    uTarget.GetDimensions(w, h);
    if (tid.x < w && tid.y < h)
        uTarget[tid.xy] = float4(w, h, 7, 1);
}
