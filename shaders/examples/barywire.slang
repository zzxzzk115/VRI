// Fragment-shader barycentrics for examples/barycentrics: the fragment reads the
// rasterizer-generated SV_Barycentrics and renders one of three modes - raw barycentric
// RGB, a solid wireframe from the distance to the nearest edge (the classic use: crisp
// antialiased lines with no line primitives), or shading with the wireframe overlaid.
// Vulkan: VK_KHR_fragment_shader_barycentric; D3D12: SM6.1 (DXIL - no DXBC exists for
// this file); no WGSL either - the fallback path is barywire_basic.slang.
struct Camera
{
    float4x4 mvp;
    float4x4 model;
    float4 misc; // x = mode (0 raw, 1 wireframe, 2 shaded + wire)
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
float4 fragmentMain(VSOutput v, float3 bary : SV_Barycentrics) : SV_Target
{
    // screen-space-antialiased distance to the nearest triangle edge
    float3 fw = fwidth(bary);
    float3 smoothed = smoothstep(float3(0.0), fw * 1.6, bary);
    float wire = 1.0 - min(min(smoothed.x, smoothed.y), smoothed.z);

    const float3 lightDir = normalize(float3(0.45, 0.8, 0.45));
    float diff = max(dot(normalize(v.worldN), lightDir), 0.0);
    float3 shaded = float3(0.3, 0.45, 0.8) * (0.2 + 0.8 * diff);

    int mode = int(uCam.misc.x + 0.5);
    float3 color = bary;                                             // 0: raw barycentric RGB
    if (mode == 1)
        color = lerp(float3(0.07, 0.08, 0.1), float3(0.9, 0.95, 1.0), wire); // 1: solid wireframe
    else if (mode == 2)
        color = lerp(shaded, float3(1.0, 0.85, 0.3), wire * 0.85);   // 2: shaded + wire overlay
    return float4(color, 1.0);
}
