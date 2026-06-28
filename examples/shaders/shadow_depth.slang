// Shadow pass (examples/shadowmapping): render the scene's depth from the light's point of
// view into a depth texture. Each object supplies its model matrix via a per-draw push constant;
// the vertex stage applies lightViewProj * model, and the fragment stage is empty so the pipeline
// writes only depth (colorNum == 0) on every backend. Matrices are uploaded transposed and
// combined with mul(M, v) (same convention as cube.slang).
struct Push { float4x4 model; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

struct Scene
{
    float4x4 camViewProj;
    float4x4 lightViewProj;
    float4   lightDir; // xyz: normalized direction TO the light, world space
    float4   params;   // x: depthBias  y: 1/shadowSize  z: clipZeroToOne  w: ambient
};
[[vk::binding(0, 0)]] ConstantBuffer<Scene> u : register(b0); // pin so the VS register matches the layout

struct VSInput
{
    float3 position : TEXCOORD0;
    float3 normal : TEXCOORD1;
};

[shader("vertex")]
float4 vertexMain(VSInput i) : SV_Position
{
    return mul(u.lightViewProj, mul(pc.model, float4(i.position, 1.0)));
}

[shader("fragment")]
void fragmentMain() {}
