// Scene shader for examples/queries: simple diffuse-shaded objects. The view-projection
// sits in a constant buffer; each draw supplies its model matrix + color as push constants
// (the example tints a sphere when last frame's occlusion query said it was hidden).
struct Camera
{
    float4x4 viewProj;
};
[[vk::binding(0, 0)]] ConstantBuffer<Camera> uCam;

struct Push
{
    float4x4 model;
    float4 color;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

struct VSInput
{
    float3 pos : TEXCOORD0;
    float3 normal : TEXCOORD1;
};
struct VSOutput
{
    float4 position : SV_Position;
    float3 worldN : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertexMain(VSInput i)
{
    VSOutput o;
    float4 world = mul(pc.model, float4(i.pos, 1.0));
    o.position = mul(uCam.viewProj, world);
    o.worldN = mul(pc.model, float4(i.normal, 0.0)).xyz;
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput v) : SV_Target
{
    const float3 lightDir = normalize(float3(0.35, 0.8, 0.5));
    float diff = max(dot(normalize(v.worldN), lightDir), 0.0);
    return float4(pc.color.rgb * (0.25 + 0.75 * diff), 1.0);
}
