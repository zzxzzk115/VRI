// Hardware ray tracer (compute, inline RayQuery) - the Vulkan + D3D12 path for examples/raytracing.
// Each thread shoots a primary ray, RayQuery-traces it against the scene's TLAS, then a shadow ray
// (accept-first-hit) toward the directional light. Shading reads the hit triangle's normal & albedo
// from the SAME constant buffer the SOFTWARE path uses (rt_software.slang), indexed by the committed
// primitive index - the BLAS is one geometry of triangles in scene order under an identity instance,
// so primitive index == triangle index == world space. Both paths therefore produce the same image.
static const int kMaxTris = 64;

struct Tri
{
    float4 v0;     // xyz: vertex 0 (unused here; geometry lives in the TLAS)
    float4 v1;
    float4 v2;
    float4 nrm;    // xyz: face normal (world space, identity instance)
    float4 albedo; // rgb
};

struct Scene
{
    float4 eye;
    float4 camRight;
    float4 camUp;
    float4 camFwd;
    float4 lightDir;
    float4 params;  // x: triangle count (unused here)  y: ambient
    Tri    tris[kMaxTris];
};

// The constant buffer sits at binding 0 / register(b0): D3D12 binds it as a root CBV whose register
// number is the range's baseRegister, so b0 must pair with baseRegister 0. The TLAS (SRV) and the
// storage image (UAV) go into D3D12 descriptor tables that restart at t0 / u0 regardless of their
// binding numbers, so their Vulkan bindings (1, 2) and DXIL auto-registers (t0, u0) both line up.
[[vk::binding(0, 0)]] ConstantBuffer<Scene> g : register(b0);
[[vk::binding(1, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(2, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outImage;

bool occluded(float3 ro, float3 rd)
{
    RayDesc ray;
    ray.Origin = ro; ray.Direction = rd; ray.TMin = 1e-3; ray.TMax = 1e30;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    q.Proceed();
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint w, h;
    outImage.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;

    float2 uv = (float2(tid.xy) + 0.5) / float2(w, h);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0); // top-left origin

    float3 ro = g.eye.xyz;
    float3 rd = normalize(g.camFwd.xyz + ndc.x * g.camRight.xyz + ndc.y * g.camUp.xyz);

    RayDesc ray;
    ray.Origin = ro; ray.Direction = rd; ray.TMin = 1e-3; ray.TMax = 1e30;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed()) {} // all geometry opaque -> the closest hit is auto-committed

    float3 color;
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
    {
        float s = saturate(rd.y * 0.5 + 0.5);
        color = lerp(float3(0.10, 0.12, 0.16), float3(0.45, 0.62, 0.85), s);
    }
    else
    {
        int prim = q.CommittedPrimitiveIndex();
        float3 P = ro + rd * q.CommittedRayT();
        float3 N = normalize(g.tris[prim].nrm.xyz);
        float3 L = normalize(g.lightDir.xyz);
        float shadow = occluded(P + N * 2e-3, L) ? 0.0 : 1.0;
        float ndl = max(dot(N, L), 0.0);
        color = g.tris[prim].albedo.xyz * (g.params.y + (1.0 - g.params.y) * ndl * shadow);
    }
    outImage[tid.xy] = float4(color, 1.0);
}
