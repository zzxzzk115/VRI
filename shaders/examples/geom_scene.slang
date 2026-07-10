// Lit scene pass for examples/geometryshader: a Blinn-Phong shaded mesh (the normal
// visualization is a second pass, geom_normals.slang). Matrices are uploaded transposed
// so mul(M, v) matches the column-major CPU data (same as examples/cube).
struct Camera
{
    float4x4 mvp;
    float4x4 model;
    float4 misc; // x = normal length (used by geom_normals; kept identical for one shared CB)
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
    float3 worldP : TEXCOORD1;
};

[shader("vertex")]
VSOutput vertexMain(VSInput i)
{
    VSOutput o;
    o.position = mul(uCam.mvp, float4(i.pos, 1.0));
    o.worldN = mul(uCam.model, float4(i.normal, 0.0)).xyz; // rigid transform: no inverse-transpose needed
    o.worldP = mul(uCam.model, float4(i.pos, 1.0)).xyz;
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput i) : SV_Target
{
    const float3 lightDir = normalize(float3(0.45, 0.8, 0.45));
    const float3 viewPos = float3(0.0, 0.6, 2.6); // matches the CPU-side camera
    float3 n = normalize(i.worldN);
    float3 v = normalize(viewPos - i.worldP);
    float3 h = normalize(lightDir + v);
    float diff = max(dot(n, lightDir), 0.0);
    float spec = pow(max(dot(n, h), 0.0), 48.0);
    float3 base = float3(0.35, 0.5, 0.85);
    float3 color = base * (0.15 + 0.85 * diff) + float3(0.9, 0.9, 1.0) * spec * 0.6;
    return float4(color, 1.0);
}
