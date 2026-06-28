// Progressive multi-bounce path tracer with diffuse global illumination, as a SINGLE compute
// megakernel using inline ray query - Metal's native ray-tracing model (Apple WWDC 2020), and the
// portable shape for a path tracer (one kernel does ray gen + traversal + shading + the bounce loop).
// Runs on any VRI backend with ray query (Metal, Vulkan/D3D12 inline).
//
// Each frame casts ONE jittered sample per pixel, bouncing up to `maxBounces` times. At each hit it
// fetches the triangle from the model's global vertex/index SSBOs, samples the baseColor as the
// Lambertian albedo, and continues along a cosine-weighted hemisphere direction (importance sampled).
// The only light is the sky gradient (image-based), so indirect bounces produce real GI - color
// bleeding, contact darkening, soft sky shadows. Radiance accumulates in accumBuffer across frames
// (reset when the camera/lighting changes); the displayed image is accum / sampleCount, tonemapped.
static const uint kMaxTextures = 64;
static const float PI = 3.14159265358979323846;

// Two float4, NOT float3 members (a structured-buffer float3 is 16B-aligned -> 48B stride, which
// would mismatch the host's tight 32B GltfVertex). a.xyz=pos, a.w=normal.x, b.xy=normal.yz, b.zw=uv.
struct Vertex { float4 a; float4 b; };
float3 vNormal(Vertex v) { return float3(v.a.w, v.b.x, v.b.y); }
float2 vUv(Vertex v) { return v.b.zw; }
struct GeometryNode { uint firstIndex; int baseColorTexture; int occlusionTexture; uint _pad; };

struct PtCamera
{
    float4 eye; float4 camRight; float4 camUp; float4 camFwd;
    float4 sky;     // xyz: sky/zenith tint, w: sky intensity
    float4 control; // x: sampleCount-so-far (this frame is the (x+1)-th), y: maxBounces, z: reset(>0.5)
};

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas : register(t0);
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outImage : register(u0);   // tonemapped display
[[vk::binding(2, 0)]] ConstantBuffer<PtCamera> cam : register(b2);
[[vk::binding(3, 0)]] RWStructuredBuffer<Vertex> vertices : register(u3);        // read-only use
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> indices : register(u4);           // read-only use
[[vk::binding(5, 0)]] RWStructuredBuffer<GeometryNode> geometryNodes : register(u5);
[[vk::binding(6, 0)]] SamplerState samp : register(s0);
[[vk::binding(7, 0)]] Texture2D<float4> textures[kMaxTextures] : register(t1);
[[vk::binding(8, 0)]] RWStructuredBuffer<float4> accumBuffer : register(u8);     // persistent radiance accumulator

// PCG hash RNG (one state word per thread, advanced per draw).
uint pcg(inout uint s) { s = s * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28) + 4u)) ^ s) * 277803737u; return (w >> 22) ^ w; }
float rnd(inout uint s) { return float(pcg(s) & 0x00FFFFFFu) / float(0x01000000); }

float3 skyColor(float3 dir)
{
    float t = saturate(dir.y * 0.5 + 0.5);
    float3 horizon = float3(0.55, 0.6, 0.66), zenith = cam.sky.xyz;
    return lerp(horizon, zenith, t) * cam.sky.w;
}

// Cosine-weighted hemisphere sample around n.
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
    uint flat = tid.y * w + tid.x;

    float sampleCount = cam.control.x;          // samples already accumulated
    uint maxBounces = (uint)cam.control.y;
    bool reset = cam.control.z > 0.5;

    uint seed = (tid.x * 1973u) ^ (tid.y * 9277u) ^ ((uint)sampleCount * 26699u) ^ 0x9e3779b9u;

    // jittered primary ray (ndc.y rises with the row so show_tex displays upright)
    float2 jitter = float2(rnd(seed), rnd(seed));
    float2 uv = (float2(tid.xy) + jitter) / float2(w, h);
    float2 ndc = float2(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0);

    float3 ro = cam.eye.xyz;
    float3 rd = normalize(cam.camFwd.xyz + ndc.x * cam.camRight.xyz + ndc.y * cam.camUp.xyz);

    float3 L = float3(0, 0, 0);
    float3 beta = float3(1, 1, 1);
    for (uint bounce = 0; bounce <= maxBounces; ++bounce)
    {
        RayDesc ray; ray.Origin = ro; ray.Direction = rd; ray.TMin = 1e-3; ray.TMax = 1e30;
        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
        while (q.Proceed()) {}

        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        {
            L += beta * skyColor(rd); // ray escaped to the sky (the only emitter)
            break;
        }

        GeometryNode gn = geometryNodes[q.CommittedGeometryIndex()];
        uint base = gn.firstIndex + q.CommittedPrimitiveIndex() * 3;
        Vertex v0 = vertices[indices[base + 0]];
        Vertex v1 = vertices[indices[base + 1]];
        Vertex v2 = vertices[indices[base + 2]];
        float2 b2 = q.CommittedTriangleBarycentrics();
        float3 bary = float3(1.0 - b2.x - b2.y, b2.x, b2.y);
        float3 N = normalize(vNormal(v0) * bary.x + vNormal(v1) * bary.y + vNormal(v2) * bary.z);
        float2 texUv = vUv(v0) * bary.x + vUv(v1) * bary.y + vUv(v2) * bary.z;
        float3 P = ro + rd * q.CommittedRayT();

        float3 albedo = float3(0.8, 0.8, 0.8);
        if (gn.baseColorTexture >= 0)
            albedo = textures[NonUniformResourceIndex(gn.baseColorTexture)].SampleLevel(samp, texUv, 0).rgb;

        if (dot(N, rd) > 0.0) N = -N; // orient the normal toward the incoming ray (two-sided)
        beta *= albedo;               // Lambertian: cosine-weighted sampling cancels the BRDF/pdf terms

        // Russian roulette after a few bounces to keep paths finite without bias.
        if (bounce >= 3)
        {
            float p = max(beta.x, max(beta.y, beta.z));
            if (rnd(seed) > p) break;
            beta /= max(p, 1e-4);
        }

        ro = P + N * 1e-3;
        rd = cosineSample(N, seed);
    }

    float3 prev = reset ? float3(0, 0, 0) : accumBuffer[flat].xyz;
    float3 acc = prev + L;
    accumBuffer[flat] = float4(acc, 1.0);

    float3 col = acc / (sampleCount + 1.0);
    col = col / (col + 1.0);                 // Reinhard tonemap
    col = pow(max(col, 0.0), 1.0 / 2.2);     // gamma
    outImage[tid.xy] = float4(col, 1.0);
}
