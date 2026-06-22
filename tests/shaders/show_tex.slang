// Generic fullscreen-triangle texture blit: sample a 2D texture across the screen. Used by
// examples/computeshader to display the compute result; reusable by any "show this texture"
// pass. No vertex buffer (position + uv from SV_VertexID).
struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    float2 p = float2((vid << 1) & 2, vid & 2);
    VSOutput o;
    o.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = p;
    return o;
}

[[vk::binding(0, 0)]] Texture2D<float4> uTex;
[[vk::binding(1, 0)]] SamplerState uSmp;

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return uTex.Sample(uSmp, input.uv);
}
