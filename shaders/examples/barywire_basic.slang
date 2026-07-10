// Fallback for examples/barycentrics, used where VriDeviceDesc::hasFragmentShaderBarycentric
// is false (Metal, WebGPU, GL): the same mesh with plain diffuse shading - no barycentrics,
// so no wireframe modes.
struct Camera
{
    float4x4 mvp;
    float4x4 model;
    float4 misc;
};
[[vk::binding(0, 0)]] ConstantBuffer<Camera> uCam;

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
    o.position = mul(uCam.mvp, float4(i.pos, 1.0));
    o.worldN = mul(uCam.model, float4(i.normal, 0.0)).xyz;
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput v) : SV_Target
{
    const float3 lightDir = normalize(float3(0.45, 0.8, 0.45));
    float diff = max(dot(normalize(v.worldN), lightDir), 0.0);
    return float4(float3(0.3, 0.45, 0.8) * (0.2 + 0.8 * diff), 1.0);
}
