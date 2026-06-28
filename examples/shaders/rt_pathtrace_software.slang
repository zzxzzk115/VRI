#include "rt_gltf_software_common.slangh"

static const float PI = 3.14159265358979323846;

struct PtCamera
{
    float4 eye; float4 camRight; float4 camUp; float4 camFwd;
    float4 sky;
    float4 control;
};

[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outImage;
[[vk::binding(2, 0)]] ConstantBuffer<PtCamera> cam;
[[vk::binding(3, 0)]] RWStructuredBuffer<Vertex> vertices;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> indices;
[[vk::binding(5, 0)]] RWStructuredBuffer<GeometryNode> geometryNodes;
[[vk::binding(6, 0)]] SamplerState samp;
[[vk::binding(7, 0)]] Texture2DArray<float4> textures; // one layer per glTF texture (GL/WebGPU non-bindless)
[[vk::binding(8, 0)]] RWStructuredBuffer<float4> accumBuffer;
[[vk::binding(9, 0)]] RWStructuredBuffer<BvhNode> bvhNodes;
[[vk::binding(10, 0)]] RWStructuredBuffer<TriRef> triRefs;

uint pcg(inout uint s) { s = s * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28) + 4u)) ^ s) * 277803737u; return (w >> 22) ^ w; }
float rnd(inout uint s) { return float(pcg(s) & 0x00FFFFFFu) / float(0x01000000); }

float3 skyColor(float3 dir)
{
    float t = saturate(dir.y * 0.5 + 0.5);
    float3 horizon = float3(0.55, 0.6, 0.66), zenith = cam.sky.xyz;
    return lerp(horizon, zenith, t) * cam.sky.w;
}

float3 cosineSample(float3 n, inout uint s)
{
    float u1 = rnd(s), u2 = rnd(s);
    float r = sqrt(u1), phi = 2.0 * PI * u2;
    float3 up = abs(n.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 t = normalize(cross(up, n));
    float3 b = cross(n, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint w, h;
    outImage.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;
    uint pix = tid.y * w + tid.x; // NOT 'flat': that is a reserved interpolation keyword in GLSL (WebGPU/GL)

    float sampleCount = cam.control.x;
    uint maxBounces = (uint)cam.control.y;
    bool reset = cam.control.z > 0.5;
    uint seed = (tid.x * 1973u) ^ (tid.y * 9277u) ^ ((uint)sampleCount * 26699u) ^ 0x9e3779b9u;

    float2 jitter = float2(rnd(seed), rnd(seed));
    float2 uv = (float2(tid.xy) + jitter) / float2(w, h);
    float2 ndc = float2(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0);

    float3 ro = cam.eye.xyz;
    float3 rd = normalize(cam.camFwd.xyz + ndc.x * cam.camRight.xyz + ndc.y * cam.camUp.xyz);
    float3 L = float3(0, 0, 0);
    float3 beta = float3(1, 1, 1);

    for (uint bounce = 0; bounce <= maxBounces; ++bounce)
    {
        Hit ht = traceBvh(ro, rd, vertices, indices, bvhNodes, triRefs);
        if (!ht.hit)
        {
            L += beta * skyColor(rd);
            break;
        }

        GeometryNode gn = geometryNodes[ht.geomNodeIndex];
        Vertex v0 = vertices[indices[ht.indexBase + 0]];
        Vertex v1 = vertices[indices[ht.indexBase + 1]];
        Vertex v2 = vertices[indices[ht.indexBase + 2]];
        float3 bary = float3(1.0 - ht.u - ht.v, ht.u, ht.v);
        float3 N = normalize(vNormal(v0) * bary.x + vNormal(v1) * bary.y + vNormal(v2) * bary.z);
        float2 texUv = vUv(v0) * bary.x + vUv(v1) * bary.y + vUv(v2) * bary.z;
        float3 P = ro + rd * ht.t;

        float3 albedo = float3(0.8, 0.8, 0.8);
        if (gn.baseColorTexture >= 0)
            albedo = textures.SampleLevel(samp, float3(texUv, float(gn.baseColorTexture)), 0).rgb;

        if (dot(N, rd) > 0.0) N = -N;
        beta *= albedo;
        if (bounce >= 3)
        {
            float p = max(beta.x, max(beta.y, beta.z));
            if (rnd(seed) > p) break;
            beta /= max(p, 1e-4);
        }
        ro = P + N * 1e-3;
        rd = cosineSample(N, seed);
    }

    float3 prev = reset ? float3(0, 0, 0) : accumBuffer[pix].xyz;
    float3 acc = prev + L;
    accumBuffer[pix] = float4(acc, 1.0);
    float3 col = acc / (sampleCount + 1.0);
    col = col / (col + 1.0);
    col = pow(max(col, 0.0), 1.0 / 2.2);
    outImage[tid.xy] = float4(col, 1.0);
}
