// Stencil-outline FILL pass (examples/stenciloutline): draw each object lit, and - via the
// pipeline's stencil state (compareOp Always, passOp Replace, ref 1) - stamp stencil = 1 wherever
// the object covers. The OUTLINE pass then redraws each object slightly enlarged and keeps only
// the fragments where stencil != 1 (the rim outside the body). Per-object model is a push constant
// (the same per-draw push path every backend supports, incl. WebGPU's dynamic-offset ring).
struct Push { float4x4 model; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

// Pinned to register(b0): Slang compiles each entry point as its own DXBC blob and could otherwise
// place a cbuffer used in both stages at different registers (it collides with the push constant).
struct Scene
{
    float4x4 viewProj;
    float4   lightDir;     // xyz: normalized direction TO the light, world space
    float4   outlineColor; // used by the outline pass (shared UBO)
};
[[vk::binding(0, 0)]] ConstantBuffer<Scene> u : register(b0);

struct VSInput
{
    float3 position : TEXCOORD0;
    float3 normal : TEXCOORD1;
};
struct VSOutput
{
    float4 position : SV_Position;
    float3 worldNormal : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertexMain(VSInput i)
{
    VSOutput o;
    float4 wp = mul(pc.model, float4(i.position, 1.0));
    o.position = mul(u.viewProj, wp);
    o.worldNormal = mul((float3x3)pc.model, i.normal); // model is translate*rotate (orthonormal 3x3)
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput i) : SV_Target
{
    float3 N = normalize(i.worldNormal);
    float3 L = normalize(u.lightDir.xyz);
    float ndl = max(dot(N, L), 0.0);
    float3 albedo = float3(0.85, 0.55, 0.30);
    float3 color = albedo * (0.25 + 0.75 * ndl);
    return float4(color, 1.0);
}
