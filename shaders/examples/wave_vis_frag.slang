// Fragment fallback for examples/waveops, used where there is no compute at all (WebGL2):
// the same three animated bands painted directly by a fullscreen fragment shader with an
// assumed stripe width of 32, so the wasm pixel self-check always has a colorful frame.
struct Params
{
    float time;
    uint width, height, pad;
};
[[vk::binding(0, 0)]] ConstantBuffer<Params> uParams;

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

float3 Hue(float h)
{
    return float3(0.5 + 0.5 * sin(h * 6.2832),
                  0.5 + 0.5 * sin(h * 6.2832 + 2.094),
                  0.5 + 0.5 * sin(h * 6.2832 + 4.189));
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    const float kFakeWave = 32.0;
    float x = input.uv.x * float(uParams.width);
    float lane = x - kFakeWave * floor(x / kFakeWave); // x mod 32
    float t = uParams.time;

    float3 color;
    if (input.uv.y < 0.3333)
        color = Hue(lane / kFakeWave);
    else if (input.uv.y < 0.6667)
    {
        float ramp = frac(lane / kFakeWave + t * 0.25);
        color = float3(ramp, ramp * 0.8, 1.0 - ramp) * 0.9;
    }
    else
    {
        float waveStart = x - lane;
        float bits = 0.0;
        for (float i = 0.0; i < kFakeWave; i += 1.0)
            if (sin((waveStart + i) * 0.11 + t * 2.0 + i * 0.31) > 0.0)
                bits += 1.0;
        color = Hue(bits / kFakeWave * 0.7 + 0.15) * 0.85;
    }

    float fy = frac(input.uv.y * 3.0);
    if (fy < 0.015)
        color = float3(0.05, 0.05, 0.08);
    return float4(color, 1.0);
}
