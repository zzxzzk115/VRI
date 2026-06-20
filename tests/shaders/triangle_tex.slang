// Triangle that samples a texture through a SEPARATE texture + sampler (VRI's
// descriptor model). Validates the sampled-texture path across backends; on GL the
// separate texture/sampler are fused into one GLSL combined sampler by SPIRV-Cross.
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
    o.uv = float2(0.5, 0.5); // solid texture -> sample anywhere
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return uTex.Sample(uSmp, input.uv);
}
