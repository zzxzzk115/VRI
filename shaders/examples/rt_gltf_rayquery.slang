// Ray-traced glTF model via INLINE ray query in a compute shader - the Metal path for
// examples/raytracing (and any backend that has ray query but not the DXR/VK ray-tracing pipeline).
// This follows Metal's native ray-tracing model (Apple WWDC 2020 "Discover ray tracing with Metal"):
// ONE compute kernel does ray generation + traversal (raytracing::intersector) + shading, instead of
// separate raygen/miss/closesthit stages dispatched through a shader binding table (Metal has no SBT).
//
// It traces the SAME model TLAS and reads the SAME vertex/index/GeometryNode SSBOs + material texture
// array as the RT-pipeline shader (rt_gltf.slang), so both paths render an identical image. The hit's
// geometry is identified inline via Committed{GeometryIndex,PrimitiveIndex,TriangleBarycentrics}().
static const uint kMaxTextures = 64; // must match the example / rt_gltf.slang

// Packed as two float4 (NOT float3 members): a structured buffer lays float3 out 16-byte aligned,
// which would make the shader-side stride 48B and mismatch the host's tightly-packed 32B GltfVertex.
// Two float4 are unambiguously 32B (a@0, b@16). a.xyz=pos, a.w=normal.x, b.xy=normal.yz, b.zw=uv.
struct Vertex { float4 a; float4 b; };
float3 vNormal(Vertex v) { return float3(v.a.w, v.b.x, v.b.y); }
float2 vUv(Vertex v) { return v.b.zw; }
struct GeometryNode { uint firstIndex; int baseColorTexture; int occlusionTexture; uint _pad; }; // 16

struct Camera
{
    float4 eye; float4 camRight; float4 camUp; float4 camFwd; float4 lightDir; float4 params; // params.x = ambient
};

// Binding order MUST match rt_gltf.slang and the example's descriptor layout (AS@0, output@1,
// camera@2, vertices@3, indices@4, geometryNodes@5, sampler@6, textures@7) so one set drives both.
[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outImage;
[[vk::binding(2, 0)]] ConstantBuffer<Camera> cam;
[[vk::binding(3, 0)]] RWStructuredBuffer<Vertex> vertices;       // read-only use
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> indices;          // read-only use
[[vk::binding(5, 0)]] RWStructuredBuffer<GeometryNode> geometryNodes;
[[vk::binding(6, 0)]] SamplerState samp;
[[vk::binding(7, 0)]] Texture2D<float4> textures[kMaxTextures];

// Trace one primary ray (ndc in [-1,1]) and shade the hit, or return the background on a miss.
float3 traceShade(float2 ndc)
{
    RayDesc ray;
    ray.Origin = cam.eye.xyz;
    ray.Direction = normalize(cam.camFwd.xyz + ndc.x * cam.camRight.xyz + ndc.y * cam.camUp.xyz);
    ray.TMin = 1e-3;
    ray.TMax = 1e30;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed()) {} // all geometry opaque -> closest hit auto-committed

    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return float3(0.13, 0.16, 0.22); // background

    GeometryNode gn = geometryNodes[q.CommittedGeometryIndex()];
    uint base = gn.firstIndex + q.CommittedPrimitiveIndex() * 3;
    Vertex v0 = vertices[indices[base + 0]];
    Vertex v1 = vertices[indices[base + 1]];
    Vertex v2 = vertices[indices[base + 2]];

    float2 b2 = q.CommittedTriangleBarycentrics();
    float3 bary = float3(1.0 - b2.x - b2.y, b2.x, b2.y);
    float3 N = normalize(vNormal(v0) * bary.x + vNormal(v1) * bary.y + vNormal(v2) * bary.z);
    float2 texUv = vUv(v0) * bary.x + vUv(v1) * bary.y + vUv(v2) * bary.z;

    float3 albedo = float3(0.8, 0.8, 0.8);
    if (gn.baseColorTexture >= 0)
        albedo = textures[NonUniformResourceIndex(gn.baseColorTexture)].SampleLevel(samp, texUv, 0).rgb;
    if (gn.occlusionTexture >= 0)
        albedo *= textures[NonUniformResourceIndex(gn.occlusionTexture)].SampleLevel(samp, texUv, 0).r;

    // hemispheric sky fill (brighter from above) + one directional sun, so unlit sides keep
    // their texture instead of going black. params.x scales the ambient fill.
    float3 L = normalize(cam.lightDir.xyz);
    float ndl = max(dot(N, L), 0.0);
    float hemi = 0.5 + 0.5 * N.y;
    float3 fill = lerp(float3(0.22, 0.24, 0.30), float3(0.68, 0.72, 0.80), hemi);
    return albedo * (fill * (cam.params.x * 2.5) + ndl);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint w, h;
    outImage.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;

    // 4x rotated-grid supersampling for anti-aliasing: average 4 sub-pixel-jittered primary rays.
    // A fixed pattern (not per-frame random) keeps edges stable while the camera orbits.
    const float2 kAA[4] = { float2(-0.125, -0.375), float2(0.375, -0.125), float2(0.125, 0.375), float2(-0.375, 0.125) };
    float3 color = float3(0, 0, 0);
    for (int s = 0; s < 4; ++s)
    {
        // ndc.y rises with the row so show_tex displays upright (see rt_rayquery.slang).
        float2 uv = (float2(tid.xy) + 0.5 + kAA[s]) / float2(w, h);
        color += traceShade(float2(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0));
    }
    outImage[tid.xy] = float4(color * 0.25, 1.0);
}
