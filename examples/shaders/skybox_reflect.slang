// Reflective object (examples/cubemap): a "chrome" sphere that samples the same CUBEMAP by
// the reflection of the eye->surface vector about the surface normal, so the surrounding
// skybox appears mirrored on it. A Fresnel term brightens grazing angles for a glassy rim.
//
// The Scene constant buffer is read in BOTH stages (viewProj/model in the vertex shader,
// cameraPos in the fragment shader); Slang compiles each entry point as its own DXBC blob and
// could otherwise assign the shared cbuffer different registers, so it is pinned to register(b0)
// to keep it identical across stages (the same trick examples/shadowmapping needs).
struct Scene
{
    float4x4 viewProj;
    float4x4 model;
    float4   cameraPos; // xyz: eye position, world space
};
[[vk::binding(0, 0)]] ConstantBuffer<Scene> u : register(b0);
[[vk::binding(1, 0)]] TextureCube<float4> env;
[[vk::binding(2, 0)]] SamplerState smp;

struct VSInput
{
    float3 position : TEXCOORD0;
    float3 normal : TEXCOORD1; // unused: see fragmentMain (the normal is recovered from worldPos)
};
struct VSOutput
{
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertexMain(VSInput i)
{
    VSOutput o;
    float4 wp = mul(u.model, float4(i.position, 1.0));
    o.position = mul(u.viewProj, wp);
    o.worldPos = wp.xyz;
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput i) : SV_Target
{
    // The object is a unit sphere centred at the origin whose model is rotate*uniform-scale (no
    // translation), so the world-space surface normal is exactly the normalized world position - no
    // normal matrix (or interpolated normal attribute) needed.
    float3 N = normalize(i.worldPos);
    float3 V = normalize(i.worldPos - u.cameraPos.xyz); // eye -> surface
    float3 R = reflect(V, N);
    // The cubemap has a single mip; sample LOD 0 explicitly. A curved mirror gives the reflection
    // direction huge screen-space derivatives, and implicit-LOD .Sample() then reads beyond the one
    // mip the texture has (visible as crackle on D3D12); SampleLevel(...,0) is exact on every backend.
    float3 col = env.SampleLevel(smp, R, 0.0).rgb;
    float fresnel = pow(1.0 - max(dot(N, -V), 0.0), 4.0); // brighter rim at grazing angles
    col = lerp(col, float3(1.0, 1.0, 1.0), fresnel * 0.6);
    return float4(col, 1.0);
}
