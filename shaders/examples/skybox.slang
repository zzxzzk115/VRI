// Skybox: draw a cube centred on the camera and sample a CUBEMAP by the interpolated
// per-vertex direction (the cube's local position IS the look direction). The view matrix
// has its translation stripped on the CPU so the box stays wrapped around the eye; depth
// test/write are off (the box is the background, everything else draws over it). Used by
// examples/cubemap. The sample direction is a geometry-interpolated varying, independent of
// each backend's clip-space Y flip, so the result is identical on every backend.
struct Scene { float4x4 viewProj; };
[[vk::binding(0, 0)]] ConstantBuffer<Scene> u : register(b0);
[[vk::binding(1, 0)]] TextureCube<float4> sky;
[[vk::binding(2, 0)]] SamplerState smp;

struct VSInput { float3 position : TEXCOORD0; };
struct VSOutput
{
    float4 position : SV_Position;
    float3 dir : TEXCOORD0; // world-space look direction (cube local position)
};

[shader("vertex")]
VSOutput vertexMain(VSInput i)
{
    VSOutput o;
    o.position = mul(u.viewProj, float4(i.position, 1.0));
    o.dir = i.position;
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput i) : SV_Target
{
    return sky.Sample(smp, normalize(i.dir));
}
