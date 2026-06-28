// Textured cube: transform by an MVP (constant buffer), sample a 2D texture. Used by
// examples/cube. Slang's default matrix layout is column-major (glm-compatible), so a
// glm-style MVP uploaded as-is works with mul(mvp, pos).
struct VSInput
{
    float3 position : POSITION;
    float2 uv : TEXCOORD0;
};
struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};
struct Camera { float4x4 mvp; };

[[vk::binding(0, 0)]] ConstantBuffer<Camera> uCam;
[[vk::binding(1, 0)]] Texture2D<float4> uTex;
[[vk::binding(2, 0)]] SamplerState uSmp;

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput o;
    o.position = mul(uCam.mvp, float4(input.position, 1.0));
    o.uv = input.uv;
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return uTex.Sample(uSmp, input.uv);
}
