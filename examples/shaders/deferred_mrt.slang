// Deferred shading - geometry (G-buffer) pass for examples/deferred. Renders the scene's
// opaque geometry once into THREE color targets at once (a "fat" G-buffer): world-space
// position into SV_Target0, the surface normal into SV_Target1, and albedo+specular into
// SV_Target2. The lighting pass (deferred_light.slang) then reads these back and shades
// each pixel for every light, so shading cost is decoupled from geometry complexity. Same
// per-instance scheme as examples/pbrbasic (per-object data via a vertex stream, not push
// constants, so it runs on every backend) and the same column-major MVP convention as
// cube.slang (Slang mul(viewProj, pos) with the host uploading the transpose).
struct VSInput
{
    float3 position : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 instA : TEXCOORD3; // xyz = world-space offset, w = uv tiling factor
    float4 instB : TEXCOORD4; // rgb = albedo tint,        w = specular intensity
};
struct VSOutput
{
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float3 tint : TEXCOORD3;
    float spec : TEXCOORD4;
};
struct Camera { float4x4 viewProj; };

[[vk::binding(0, 0)]] ConstantBuffer<Camera> uCam;
[[vk::binding(1, 0)]] Texture2D<float4> uTex;
[[vk::binding(2, 0)]] SamplerState uSmp;

// The three G-buffer render targets, filled in one pass.
struct GBuffer
{
    float4 position : SV_Target0;   // world-space position (stored in an RGBA16F target)
    float4 normal : SV_Target1;     // encoded normal 0.5+0.5*N (RGBA8 - decoded in the light pass)
    float4 albedoSpec : SV_Target2; // rgb = albedo, a = specular intensity (RGBA8)
};

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    float3 world = input.position + input.instA.xyz;
    VSOutput o;
    o.position = mul(uCam.viewProj, float4(world, 1.0));
    o.worldPos = world;
    o.normal = input.normal; // instances only translate, so object normal == world normal
    o.uv = input.uv * input.instA.w;
    o.tint = input.instB.xyz;
    o.spec = input.instB.w;
    return o;
}

[shader("fragment")]
GBuffer fragmentMain(VSOutput input)
{
    float4 tex = uTex.Sample(uSmp, input.uv);
    float3 N = normalize(input.normal);

    GBuffer o;
    o.position = float4(input.worldPos, 1.0);
    o.normal = float4(0.5 + 0.5 * N, 1.0);
    o.albedoSpec = float4(tex.rgb * input.tint, input.spec);
    return o;
}
