// Scene shader for examples/conservativeraster: a pinwheel of three thin sliver blades,
// rasterized into a tiny render target. At that resolution normal rasterization misses most
// of the slivers' pixel centers; conservative rasterization lights every texel they touch.
// Vertices are embedded (no vertex buffer); the spin + aspect come from a push constant.
struct Push { float4 rot; }; // x = cos, y = sin, z = aspect (height/width), w unused
[[vk::push_constant]] ConstantBuffer<Push> pc;

// Three center-anchored slivers, 120 degrees apart, each 0.024 NDC tall at the tip
// (sub-pixel on a ~40x30 target). Distinct colors so the halves read at a glance.
static const float2 kPositions[9] = {
    float2(0.0, 0.0), float2(0.92, 0.012), float2(0.92, -0.012),          // blade 0 (0 deg)
    float2(0.0, 0.0), float2(-0.4496, 0.8028), float2(-0.4704, 0.7908),   // blade 1 (120 deg)
    float2(0.0, 0.0), float2(-0.4704, -0.7908), float2(-0.4496, -0.8028), // blade 2 (240 deg)
};
static const float3 kColors[3] = { float3(1.0, 0.35, 0.3), float3(0.4, 1.0, 0.5), float3(1.0, 0.9, 0.3) };

struct VSOutput
{
    float4 position : SV_Position;
    float3 color : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    float2 v = kPositions[vid];
    float2 p = float2(v.x * pc.rot.x - v.y * pc.rot.y,
                      v.x * pc.rot.y + v.y * pc.rot.x);
    p.x *= pc.rot.z; // keep the pinwheel circular regardless of aspect ratio
    VSOutput o;
    o.position = float4(p, 0.0, 1.0);
    o.color = kColors[vid / 3];
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput i) : SV_Target
{
    return float4(i.color, 1.0);
}
