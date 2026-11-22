// triangle_tex.slang, sampling layer 0 of a Texture2DArray: a 2DArray texture of one layer must still be an array
// texture to the shader (GL used to create it as a plain 2D texture, which a sampler2DArray cannot sample).
struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

static const float2 kPositions[3] = { float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };

[[vk::binding(0, 0)]] Texture2DArray<float4> uTex;
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
    return uTex.Sample(uSmp, float3(input.uv, 0.0));
}
