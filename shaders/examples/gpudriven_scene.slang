// Scene shader for examples/gpudriven: cubes fetch their world position/scale + color from
// a PER-INSTANCE vertex stream - the compacted buffer the culling compute pass wrote. The
// indirect records' baseInstance offsets that stream, so no SV_InstanceID semantics are
// needed (portable across Vulkan / D3D12 / GL indirect draws).
struct Camera
{
    float4x4 viewProj;
};
[[vk::binding(0, 0)]] ConstantBuffer<Camera> uCam;

struct VSInput
{
    float3 pos : TEXCOORD0;      // per-vertex (stream 0)
    float3 normal : TEXCOORD1;   // per-vertex (stream 0)
    float4 posScale : TEXCOORD2; // per-instance (stream 1)
    float4 color : TEXCOORD3;    // per-instance (stream 1)
};
struct VSOutput
{
    float4 position : SV_Position;
    float3 normal : TEXCOORD0;
    float3 color : TEXCOORD1;
};

[shader("vertex")]
VSOutput vertexMain(VSInput i)
{
    float3 world = i.pos * i.posScale.w + i.posScale.xyz;
    VSOutput o;
    o.position = mul(uCam.viewProj, float4(world, 1.0));
    o.normal = i.normal;
    o.color = i.color.rgb;
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput v) : SV_Target
{
    const float3 lightDir = normalize(float3(0.4, 0.85, 0.35));
    float diff = max(dot(normalize(v.normal), lightDir), 0.0);
    return float4(v.color * (0.25 + 0.75 * diff), 1.0);
}
