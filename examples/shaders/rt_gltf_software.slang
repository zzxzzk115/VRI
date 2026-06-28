#include "rt_gltf_software_common.slangh"

struct Camera
{
    float4 eye; float4 camRight; float4 camUp; float4 camFwd; float4 lightDir; float4 params;
};

[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outImage;
[[vk::binding(2, 0)]] ConstantBuffer<Camera> cam;
[[vk::binding(3, 0)]] RWStructuredBuffer<Vertex> vertices;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> indices;
[[vk::binding(5, 0)]] RWStructuredBuffer<GeometryNode> geometryNodes;
[[vk::binding(6, 0)]] SamplerState samp;
[[vk::binding(7, 0)]] Texture2DArray<float4> textures; // one layer per glTF texture (GL/WebGPU non-bindless)
[[vk::binding(8, 0)]] RWStructuredBuffer<BvhNode> bvhNodes;
[[vk::binding(9, 0)]] RWStructuredBuffer<TriRef> triRefs;

float3 traceShade(float2 ndc)
{
    float3 ro = cam.eye.xyz;
    float3 rd = normalize(cam.camFwd.xyz + ndc.x * cam.camRight.xyz + ndc.y * cam.camUp.xyz);
    Hit h = traceBvh(ro, rd, vertices, indices, bvhNodes, triRefs);
    if (!h.hit) return float3(0.13, 0.16, 0.22);

    GeometryNode gn = geometryNodes[h.geomNodeIndex];
    Vertex v0 = vertices[indices[h.indexBase + 0]];
    Vertex v1 = vertices[indices[h.indexBase + 1]];
    Vertex v2 = vertices[indices[h.indexBase + 2]];
    float3 bary = float3(1.0 - h.u - h.v, h.u, h.v);
    float3 N = normalize(vNormal(v0) * bary.x + vNormal(v1) * bary.y + vNormal(v2) * bary.z);
    float2 texUv = vUv(v0) * bary.x + vUv(v1) * bary.y + vUv(v2) * bary.z;

    float3 albedo = float3(0.8, 0.8, 0.8);
    if (gn.baseColorTexture >= 0)
        albedo = textures.SampleLevel(samp, float3(texUv, float(gn.baseColorTexture)), 0).rgb;
    if (gn.occlusionTexture >= 0)
        albedo *= textures.SampleLevel(samp, float3(texUv, float(gn.occlusionTexture)), 0).r;

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
    const float2 kAA[4] = { float2(-0.125, -0.375), float2(0.375, -0.125), float2(0.125, 0.375), float2(-0.375, 0.125) };
    float3 color = float3(0, 0, 0);
    for (int s = 0; s < 4; ++s)
    {
        float2 uv = (float2(tid.xy) + 0.5 + kAA[s]) / float2(w, h);
        color += traceShade(float2(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0));
    }
    outImage[tid.xy] = float4(color * 0.25, 1.0);
}
