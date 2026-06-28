// Stencil-outline OUTLINE pass (examples/stenciloutline): redraw each object with a slightly
// enlarged model (the scale is baked into the push-constant matrix on the CPU) in a flat colour.
// The pipeline's stencil state keeps only fragments where stencil != 1 (compareOp NotEqual, ref 1,
// writeMask 0), i.e. the rim that the enlarged object covers beyond the body the FILL pass stamped.
// Depth test is off (pipeline state) so the outline is never hidden by the object it surrounds.
struct Push { float4x4 model; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

struct Scene
{
    float4x4 viewProj;
    float4   lightDir;
    float4   outlineColor; // rgb: outline colour
};
[[vk::binding(0, 0)]] ConstantBuffer<Scene> u : register(b0);

struct VSInput
{
    float3 position : TEXCOORD0;
    float3 normal : TEXCOORD1; // unused (the outline is a flat colour); kept to match the vertex layout
};
struct VSOutput
{
    float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertexMain(VSInput i)
{
    VSOutput o;
    o.position = mul(u.viewProj, mul(pc.model, float4(i.position, 1.0)));
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput i) : SV_Target
{
    return float4(u.outlineColor.rgb, 1.0);
}
