// Copies element 1 of a read-only structured buffer into every texel of a storage texture.
// D3D12 registers follow VRI's mapping: both are table resources, counted from 0 per class.
[[vk::binding(0, 0)]] StructuredBuffer<float4> uInput : register(t0);
[[vk::binding(1, 0)]] RWTexture2D<float4> uTarget : register(u0);

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint w, h;
    uTarget.GetDimensions(w, h);
    if (tid.x < w && tid.y < h)
        uTarget[tid.xy] = uInput[1];
}
