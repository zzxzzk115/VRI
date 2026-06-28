// Offscreen MRT pass for examples/offscreen: render a textured cube to TWO color targets at
// once - albedo (sampled texture) into SV_Target0 and a screen-derived face normal into
// SV_Target1 (a tiny G-buffer). composite.slang then samples both. Same MVP convention as
// cube.slang (Slang column-major + mul(mvp, pos), with the host uploading Transpose(mvp)).
struct VSInput
{
    float3 position : POSITION;
    float2 uv : TEXCOORD0;
};
struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float3 opos : TEXCOORD1; // object-space position, for a derivative-based face normal
};
struct Camera { float4x4 mvp; };

[[vk::binding(0, 0)]] ConstantBuffer<Camera> uCam;
[[vk::binding(1, 0)]] Texture2D<float4> uTex;
[[vk::binding(2, 0)]] SamplerState uSmp;

struct MRTOutput
{
    float4 albedo : SV_Target0;
    float4 normal : SV_Target1;
};

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput o;
    o.position = mul(uCam.mvp, float4(input.position, 1.0));
    o.uv = input.uv;
    o.opos = input.position;
    return o;
}

[shader("fragment")]
MRTOutput fragmentMain(VSOutput input)
{
    MRTOutput o;
    o.albedo = uTex.Sample(uSmp, input.uv);
    // Flat per-face normal from screen-space derivatives of the object position - no normal
    // attribute needed, and it makes the second target visibly different from the albedo.
    float3 n = normalize(cross(ddx(input.opos), ddy(input.opos)));
    o.normal = float4(0.5 + 0.5 * n, 1.0);
    return o;
}
