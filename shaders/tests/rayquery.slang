// Ray query (inline ray tracing) in a COMPUTE shader: trace orthographic rays against
// the TLAS and write hit (red) / miss (black) to a storage image. No RT pipeline, no
// shader binding table - just RayQuery against a bound acceleration structure. Maps to
// VK_KHR_ray_query (SPIR-V) and DXR 1.1 inline RT (DXIL cs_6_5).
[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outImage;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint w, h;
    outImage.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h)
        return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(w, h) * 2.0 - 1.0;

    RayDesc ray;
    ray.Origin = float3(uv.x, uv.y, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.001;
    ray.TMax = 10.0;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    float3 color = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? float3(1.0, 0.0, 0.0) : float3(0.0, 0.0, 0.0);
    outImage[tid.xy] = float4(color, 1.0);
}
