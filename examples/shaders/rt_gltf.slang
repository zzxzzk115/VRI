// Ray-traced glTF model (RT *pipeline*: raygen / miss / closesthit + SBT) - the "real" ray tracing
// path for examples/raytracing, as opposed to inline ray query (examples/rayquery). A primary ray
// per pixel is traced against the model's TLAS; the closest hit fetches the hit triangle's vertices
// from the model's global vertex/index SSBOs (indexed by GeometryIndex()+PrimitiveIndex()),
// interpolates normal & uv with the barycentrics, samples the material's baseColor (+occlusion)
// texture from a descriptor array, and does simple directional lighting. Miss = sky gradient.
//
// SPIR-V (Vulkan) + DXIL (D3D12, lib_6_3). The descriptor register mapping is chosen so ONE source
// satisfies both backends (see examples/rayquery's rt_rayquery.slang for the underlying rules):
//   - ConstantBuffer -> D3D12 root CBV, register(b#) == the range's baseRegister.
//   - StorageBuffer  -> D3D12 root UAV, register(u#) == the range's baseRegister.
//   - Sampler        -> sampler table, s# restarts at 0 in declaration order.
//   - TLAS / Texture -> SRV table, t# restarts at 0 (TLAS=t0, the texture array follows at t1).
//   - StorageTexture -> UAV table, u# restarts at 0 (the output image is u0).
// The output image (table u0) and the SSBOs (root u3/u4/u5) share register space 0 but never collide.
static const uint kMaxTextures = 64; // upper bound on the model's texture count (array fully bound)

// Packed as two float4, NOT float3 members: a structured buffer lays float3 out 16-byte aligned
// (shader stride 48B), which would mismatch the host's tightly-packed 32B vriex::GltfVertex. Two
// float4 are unambiguously 32B (a@0, b@16). a.xyz=pos, a.w=normal.x, b.xy=normal.yz, b.zw=uv.
struct Vertex { float4 a; float4 b; };
float3 vNormal(Vertex v) { return float3(v.a.w, v.b.x, v.b.y); }
float2 vUv(Vertex v) { return v.b.zw; }

struct GeometryNode // matches the host GeometryNode upload (16 bytes); one per BLAS geometry / primitive
{
    uint firstIndex;        // start of this primitive's indices in the global index buffer
    int  baseColorTexture;  // index into textures[]  (-1 == none)
    int  occlusionTexture;  // index into textures[]  (-1 == none)
    uint _pad;
};

struct Camera
{
    float4 eye;       // xyz: ray origin (camera position)
    float4 camRight;  // xyz: right basis, pre-scaled by tan(fovX/2)*aspect
    float4 camUp;     // xyz: up basis,    pre-scaled by tan(fovY/2)
    float4 camFwd;    // xyz: forward (view direction)
    float4 lightDir;  // xyz: direction TO the light (normalized)
    float4 params;    // x: ambient
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas : register(t0);
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outImage : register(u0);
[[vk::binding(2, 0)]] ConstantBuffer<Camera> cam : register(b2);
[[vk::binding(3, 0)]] RWStructuredBuffer<Vertex> vertices : register(u3);      // read-only use
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> indices : register(u4);         // read-only use
[[vk::binding(5, 0)]] RWStructuredBuffer<GeometryNode> geometryNodes : register(u5);
[[vk::binding(6, 0)]] SamplerState samp : register(s0);
[[vk::binding(7, 0)]] Texture2D<float4> textures[kMaxTextures] : register(t1);

struct Payload { float3 color; };

[shader("raygeneration")]
void rayGenMain()
{
    uint2 idx = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;

    // 4x rotated-grid supersampling AA: average 4 sub-pixel-jittered primary rays. Fixed pattern
    // (not per-frame random) so edges stay stable while the camera orbits.
    const float2 kAA[4] = { float2(-0.125, -0.375), float2(0.375, -0.125), float2(0.125, 0.375), float2(-0.375, 0.125) };
    float3 color = float3(0, 0, 0);
    for (int s = 0; s < 4; ++s)
    {
        // show_tex samples image row 0 at the screen bottom, so let ndc.y rise with the row -> upright.
        float2 uv = (float2(idx) + 0.5 + kAA[s]) / float2(dim);
        float2 ndc = float2(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0);

        RayDesc ray;
        ray.Origin = cam.eye.xyz;
        ray.Direction = normalize(cam.camFwd.xyz + ndc.x * cam.camRight.xyz + ndc.y * cam.camUp.xyz);
        ray.TMin = 1e-3;
        ray.TMax = 1e30;

        Payload payload;
        payload.color = float3(0, 0, 0);
        TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);
        color += payload.color;
    }
    outImage[idx] = float4(color * 0.25, 1.0);
}

[shader("miss")]
void missMain(inout Payload payload)
{
    // sky gradient based on ray direction (reconstructed from the dispatch is unavailable in miss;
    // use a fixed pleasant background instead - the model fills most of the frame).
    payload.color = float3(0.13, 0.16, 0.22);
}

[shader("closesthit")]
void closestHitMain(inout Payload payload, BuiltInTriangleIntersectionAttributes attr)
{
    GeometryNode gn = geometryNodes[GeometryIndex()];
    uint base = gn.firstIndex + PrimitiveIndex() * 3;
    Vertex v0 = vertices[indices[base + 0]];
    Vertex v1 = vertices[indices[base + 1]];
    Vertex v2 = vertices[indices[base + 2]];

    float3 bary = float3(1.0 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
    float3 N = normalize(vNormal(v0) * bary.x + vNormal(v1) * bary.y + vNormal(v2) * bary.z);
    float2 texUv = vUv(v0) * bary.x + vUv(v1) * bary.y + vUv(v2) * bary.z;

    float3 albedo = float3(0.8, 0.8, 0.8);
    if (gn.baseColorTexture >= 0)
        albedo = textures[NonUniformResourceIndex(gn.baseColorTexture)].SampleLevel(samp, texUv, 0).rgb;
    if (gn.occlusionTexture >= 0)
        albedo *= textures[NonUniformResourceIndex(gn.occlusionTexture)].SampleLevel(samp, texUv, 0).r;

    // hemispheric sky fill (brighter from above) + one directional sun, so unlit sides keep their
    // texture instead of going black. params.x scales the ambient fill.
    float3 L = normalize(cam.lightDir.xyz);
    float ndl = max(dot(N, L), 0.0);
    float hemi = 0.5 + 0.5 * N.y;
    float3 fill = lerp(float3(0.22, 0.24, 0.30), float3(0.68, 0.72, 0.80), hemi);
    payload.color = albedo * (fill * (cam.params.x * 2.5) + ndl);
}
