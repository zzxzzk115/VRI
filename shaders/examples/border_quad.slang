// Quad for examples/bordercolor: UVs span [-1, 2], so only the middle ninth of the quad
// shows the texture - everything around it is ClampToBorder territory and takes the
// sampler's border color (preset or custom). Vertices are embedded (no vertex buffer).
struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

static const float2 kCorners[6] = {
    float2(-0.85, -0.85), float2(0.85, -0.85), float2(-0.85, 0.85),
    float2(0.85, -0.85),  float2(0.85, 0.85),  float2(-0.85, 0.85),
};

[[vk::binding(0, 0)]] Texture2D<float4> uTex;
[[vk::binding(1, 0)]] SamplerState uSmp;

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    float2 p = kCorners[vid];
    VSOutput o;
    o.position = float4(p, 0.0, 1.0);
    float2 q = p / 0.85 * 0.5 + 0.5; // quad -> [0,1]
    o.uv = q * 3.0 - 1.0;            // [0,1] -> [-1,2]: a one-texture-wide border ring
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return uTex.Sample(uSmp, input.uv);
}
