// Software ray tracer (compute) - the cross-backend fallback for examples/raytracing where there
// is no hardware ray query (WebGPU, desktop OpenGL). Each thread shoots one primary ray, brute-
// force intersects it against the scene's triangles (Moeller-Trumbore), then a shadow ray toward
// the directional light to decide lit vs. shadowed. The whole scene (camera basis + light + up to
// kMaxTris triangles with per-triangle normal & albedo) rides in a single constant buffer, so the
// only bindings are the CB and the storage image - identical layout to examples/computeshader, and
// the same image the HARDWARE ray-query path (rt_rayquery.slang) writes, so the two match pixel-wise.
static const int kMaxTris = 64;

struct Tri
{
    float4 v0;     // xyz: vertex 0
    float4 v1;     // xyz: vertex 1
    float4 v2;     // xyz: vertex 2
    float4 nrm;    // xyz: face normal (normalized)
    float4 albedo; // rgb
};

struct Scene
{
    float4 eye;       // xyz: camera position
    float4 camRight;  // right * tan(fovY/2) * aspect
    float4 camUp;     // up    * tan(fovY/2)
    float4 camFwd;    // forward (normalized)
    float4 lightDir;  // xyz: normalized direction TO the light
    float4 params;    // x: triangle count  y: ambient
    Tri    tris[kMaxTris];
};
[[vk::binding(0, 0)]] ConstantBuffer<Scene> g;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outImage;

// Moeller-Trumbore. Returns the forward hit distance in `t` (or false for a miss / behind-origin).
bool intersect(float3 ro, float3 rd, Tri tri, out float t)
{
    t = 0.0;
    float3 e1 = tri.v1.xyz - tri.v0.xyz;
    float3 e2 = tri.v2.xyz - tri.v0.xyz;
    float3 p = cross(rd, e2);
    float det = dot(e1, p);
    if (abs(det) < 1e-7) return false;
    float inv = 1.0 / det;
    float3 tv = ro - tri.v0.xyz;
    float u = dot(tv, p) * inv;
    if (u < 0.0 || u > 1.0) return false;
    float3 q = cross(tv, e1);
    float v = dot(rd, q) * inv;
    if (v < 0.0 || u + v > 1.0) return false;
    t = dot(e2, q) * inv;
    return t > 1e-3;
}

// Any triangle between the surface point and the (infinitely distant) light -> shadowed.
bool occluded(float3 ro, float3 rd, int n)
{
    for (int i = 0; i < n; ++i)
    {
        float t;
        if (intersect(ro, rd, g.tris[i], t)) return true;
    }
    return false;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint w, h;
    outImage.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;

    // pixel -> NDC in [-1,1]; flip Y so row 0 is the top of the image (VRI top-left convention)
    float2 uv = (float2(tid.xy) + 0.5) / float2(w, h);
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    float3 ro = g.eye.xyz;
    float3 rd = normalize(g.camFwd.xyz + ndc.x * g.camRight.xyz + ndc.y * g.camUp.xyz);

    const int n = int(g.params.x);
    const float ambient = g.params.y;

    float bestT = 1e30;
    int hit = -1;
    for (int i = 0; i < n; ++i)
    {
        float t;
        if (intersect(ro, rd, g.tris[i], t) && t < bestT) { bestT = t; hit = i; }
    }

    float3 color;
    if (hit < 0)
    {
        float s = saturate(rd.y * 0.5 + 0.5);
        color = lerp(float3(0.10, 0.12, 0.16), float3(0.45, 0.62, 0.85), s); // sky gradient
    }
    else
    {
        float3 P = ro + rd * bestT;
        float3 N = normalize(g.tris[hit].nrm.xyz);
        float3 L = normalize(g.lightDir.xyz);
        float shadow = occluded(P + N * 2e-3, L, n) ? 0.0 : 1.0;
        float ndl = max(dot(N, L), 0.0);
        color = g.tris[hit].albedo.xyz * (ambient + (1.0 - ambient) * ndl * shadow);
    }
    outImage[tid.xy] = float4(color, 1.0);
}
