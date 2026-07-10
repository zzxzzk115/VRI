// Deliberately expensive fullscreen fragment shader for examples/vrs: a high-frequency
// animated interference pattern with a heavy per-pixel loop. Coarse shading rates make
// the blockiness directly visible on the fine pattern, and the GPU-time readout in the
// example shows the cost drop. Time + loop count come from a push constant.
struct Push { float4 p; }; // x = time, y = aspect (w/h), z = loop count, w unused
[[vk::push_constant]] ConstantBuffer<Push> pc;

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

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    float2 uv = input.uv;
    uv.x *= pc.p.y;
    float t = pc.p.x;

    // many-octave interference: fine detail everywhere, so coarse shading is obvious
    float v = 0.0;
    int loops = int(pc.p.z + 0.5);
    for (int i = 1; i <= loops; ++i)
    {
        float fi = float(i);
        v += sin(uv.x * fi * 9.0 + t * (0.7 + fi * 0.11))
           * cos(uv.y * fi * 7.0 - t * (0.5 + fi * 0.07)) / fi;
    }
    v *= 0.6;
    float3 color = float3(0.5 + 0.5 * sin(v * 6.2832),
                          0.5 + 0.5 * sin(v * 6.2832 + 2.094),
                          0.5 + 0.5 * sin(v * 6.2832 + 4.189));
    // fine checker overlay: the sharpest detail VRS will visibly coarsen
    float2 g = frac(uv * 96.0);
    float checker = (g.x < 0.5) != (g.y < 0.5) ? 1.0 : 0.82;
    return float4(color * checker, 1.0);
}
