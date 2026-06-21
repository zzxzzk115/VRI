// Custom border color: the triangle samples a texture far OUTSIDE [0,1] with
// ClampToBorder addressing, so every fragment returns the sampler's border color.
// With a custom green border the triangle reads back green (not the texture's blue,
// nor a preset black/white). Separate texture + sampler (VRI's descriptor model).
struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

static const float2 kPositions[3] = { float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };

[[vk::binding(0, 0)]] Texture2D<float4> uTex;
[[vk::binding(1, 0)]] SamplerState uSmp;

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    VSOutput o;
    o.position = float4(kPositions[vid], 0.0, 1.0);
    o.uv = float2(5.0, 5.0); // far outside -> ClampToBorder returns the border color
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return uTex.Sample(uSmp, input.uv);
}
