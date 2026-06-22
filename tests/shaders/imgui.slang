// Dear ImGui UI shader for examples/common/imgui_vri.h: textured + per-vertex-colored 2D.
// ImGui emits vertices in display coordinates, so the vertex stage applies an orthographic
// transform (scale + translate) supplied as a push constant. Vertex layout = ImDrawVert
// (pos float2, uv float2, col RGBA8_UNORM). Sequential TEXCOORD semantics keep the attribute
// locations consistent across SPIR-V / WGSL / DXBC.
struct VSInput
{
    float2 pos : TEXCOORD0;
    float2 uv  : TEXCOORD1;
    float4 col : TEXCOORD2; // from RGBA8_UNORM -> normalized
};
struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 col : TEXCOORD1;
};
struct Ortho { float2 scale; float2 translate; };

[[vk::push_constant]] ConstantBuffer<Ortho> pc;
[[vk::binding(0, 0)]] Texture2D<float4> uTex;
[[vk::binding(1, 0)]] SamplerState uSmp;

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput o;
    o.position = float4(input.pos * pc.scale + pc.translate, 0.0, 1.0);
    o.uv = input.uv;
    o.col = input.col;
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return input.col * uTex.Sample(uSmp, input.uv);
}
