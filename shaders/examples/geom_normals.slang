// Normal-visualization pass for examples/geometryshader: the vertex stage passes the
// object-space position + normal through, and the GEOMETRY stage turns each triangle
// corner into a line segment from the surface point to point + normal * length, colored
// yellow at the base and red at the tip. Runs only where VriDeviceDesc::hasGeometryShader
// (Vulkan / D3D12 / desktop GL); no WGSL output exists for this file - WebGPU has no
// geometry stage - so the example must not include a _wgsl.h for it.
struct Camera
{
    float4x4 mvp;
    float4x4 model;
    float4 misc; // x = normal length
};
[[vk::binding(0, 0)]] ConstantBuffer<Camera> uCam;

struct VSOut
{
    float4 position : SV_Position; // object-space (w=1); projected in the GS
    float3 normal : TEXCOORD0;
};
struct GSOut
{
    float4 position : SV_Position;
    float3 color : TEXCOORD0;
};

struct VSInput
{
    float3 pos : TEXCOORD0;
    float3 normal : TEXCOORD1;
};

[shader("vertex")]
VSOut vertexMain(VSInput i)
{
    VSOut o;
    o.position = float4(i.pos, 1.0);
    o.normal = i.normal;
    return o;
}

[shader("geometry")]
[maxvertexcount(6)]
void geometryMain(triangle VSOut input[3], inout LineStream<GSOut> stream)
{
    for (int i = 0; i < 3; ++i)
    {
        float3 p = input[i].position.xyz;
        float3 tip = p + input[i].normal * uCam.misc.x;

        GSOut a;
        a.position = mul(uCam.mvp, float4(p, 1.0));
        a.color = float3(1.0, 0.9, 0.2); // base: yellow
        stream.Append(a);

        GSOut b;
        b.position = mul(uCam.mvp, float4(tip, 1.0));
        b.color = float3(1.0, 0.25, 0.2); // tip: red
        stream.Append(b);

        stream.RestartStrip();
    }
}

[shader("fragment")]
float4 fragmentMain(GSOut i) : SV_Target
{
    return float4(i.color, 1.0);
}
