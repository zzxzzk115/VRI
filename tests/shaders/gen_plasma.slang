// WebGL2 fallback for examples/computeshader: generate the same plasma directly in the
// fragment shader (WebGL2 has no compute). Mirrors compute_plasma.slang's Plasma() so the
// picture is identical to the compute path - explicit degradation, not a different result.
struct Params
{
    float time;
    uint  width;
    uint  height;
    uint  pad;
};

float3 Plasma(float2 uv, float t)
{
    float v = sin(uv.x * 10.0 + t)
            + sin(uv.y * 10.0 + t * 1.3)
            + sin((uv.x + uv.y) * 10.0 + t * 0.7)
            + sin(length(uv - 0.5) * 16.0 - t * 1.5);
    return 0.5 + 0.5 * cos(float3(0.0, 2.0, 4.0) + v * 1.2);
}

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

[[vk::binding(0, 0)]] ConstantBuffer<Params> uParams;

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return float4(Plasma(input.uv, uParams.time), 1.0);
}
