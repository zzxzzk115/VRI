// Main pass (examples/shadowmapping): render the scene from the camera, lit by one directional
// light, with PCF shadows sampled from the light's depth map via a COMPARISON sampler
// (SampleCmpLevelZero). Each object's model comes from a per-draw push constant. The shadow map
// is a render-to-texture sampled like examples/offscreen (uv = ndc.xy * 0.5 + 0.5, no Y flip).
// The light-space depth is remapped to [0,1] on WebGL2 (no glClipControl -> window depth is
// z*0.5+0.5) via params.z; every other backend already has zero-to-one clip depth.
//
// The constant buffer is pinned to register(b0): Slang compiles each entry point as its own DXBC
// blob and can assign a cbuffer used in both stages to different registers, so pinning keeps it
// identical (b0) across the vertex + fragment shaders (else it collides with the push constant).
struct Push { float4x4 model; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

struct Scene
{
    float4x4 camViewProj;
    float4x4 lightViewProj;
    float4   lightDir; // xyz: normalized direction TO the light, world space
    float4   params;   // x: depthBias  y: 1/shadowSize  z: clipZeroToOne  w: ambient
};
[[vk::binding(0, 0)]] ConstantBuffer<Scene> u : register(b0);
[[vk::binding(1, 0)]] Texture2D<float> shadowMap;            // depth texture (sampleType = Depth)
[[vk::binding(2, 0)]] SamplerComparisonState shadowSampler; // comparison sampler (PCF)

struct VSInput
{
    float3 position : TEXCOORD0;
    float3 normal : TEXCOORD1;
};
struct VSOutput
{
    float4 position : SV_Position;
    float3 worldNormal : TEXCOORD0;
    float4 lightClip : TEXCOORD1; // position in the light's clip space
};

[shader("vertex")]
VSOutput vertexMain(VSInput i)
{
    VSOutput o;
    float4 wp = mul(pc.model, float4(i.position, 1.0));
    o.position = mul(u.camViewProj, wp);
    o.lightClip = mul(u.lightViewProj, wp);
    o.worldNormal = mul((float3x3)pc.model, i.normal); // models are translate*rotate (orthonormal)
    return o;
}

// 3x3 PCF: average nine hardware depth comparisons around the projected texel.
float samplePcf(float2 uv, float ref)
{
    float texel = u.params.y;
    float sum = 0.0;
    [unroll]
    for (int y = -1; y <= 1; ++y)
        [unroll]
        for (int x = -1; x <= 1; ++x)
            sum += shadowMap.SampleCmpLevelZero(shadowSampler, uv + float2(x, y) * texel, ref);
    return sum / 9.0;
}

[shader("fragment")]
float4 fragmentMain(VSOutput i) : SV_Target
{
    float3 ndc = i.lightClip.xyz / i.lightClip.w;
    float2 uv = ndc.xy * 0.5 + 0.5;                          // render-to-texture sample (no Y flip)
    float ref = (u.params.z > 0.5) ? ndc.z : (ndc.z * 0.5 + 0.5);
    ref -= u.params.x;                                       // constant depth bias (anti-acne)

    float shadow = 1.0;
    if (uv.x == saturate(uv.x) && uv.y == saturate(uv.y) && ndc.z <= 1.0)
        shadow = samplePcf(uv, ref);

    float3 N = normalize(i.worldNormal);
    float3 L = normalize(u.lightDir.xyz);
    float ndl = max(dot(N, L), 0.0);
    float ambient = u.params.w;
    float3 albedo = float3(0.85, 0.85, 0.88);
    float3 color = albedo * (ambient + (1.0 - ambient) * ndl * shadow);
    return float4(color, 1.0);
}
