// PBR metallic-roughness lighting over an instanced grid of spheres (examples/pbrbasic).
// Per-vertex {position, normal} from stream 0; per-instance {objPos+roughness, color+metallic}
// from stream 1; a shared view-projection + camera + 4 lights in a constant buffer. Direct
// BRDF (Cook-Torrance: GGX distribution, Schlick-Smith geometry, Schlick fresnel), ported
// from Sascha Willems' pbrbasic. All vertex inputs use sequential TEXCOORD semantics so the
// assigned locations are 0..3 on every target (Slang's WGSL emitter assigns @location by
// semantic; mixing POSITION with TEXCOORD would reorder them). viewProj is uploaded already
// transposed, so mul(viewProj, v) matches the column-major data (same as examples/instancing).
struct VSInput
{
    float3 position : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float4 instA : TEXCOORD2; // xyz = object position, w = roughness
    float4 instB : TEXCOORD3; // xyz = base color,      w = metallic
};
struct VSOutput
{
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 mat : TEXCOORD2;   // x = roughness, y = metallic
    float3 color : TEXCOORD3; // base color
};

struct Ubo
{
    float4x4 viewProj;
    float4    camPos;    // xyz
    float4    lights[4]; // xyz = world-space light positions
};
[[vk::binding(0, 0)]] ConstantBuffer<Ubo> uCam;

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    float3 world = input.position + input.instA.xyz;
    VSOutput o;
    o.position = mul(uCam.viewProj, float4(world, 1.0));
    o.worldPos = world;
    o.normal = input.normal;
    o.mat = float2(input.instA.w, input.instB.w);
    o.color = input.instB.xyz;
    return o;
}

static const float PI = 3.14159265359;

// GGX normal distribution.
float D_GGX(float dotNH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = dotNH * dotNH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-Smith geometric shadowing.
float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float GL = dotNL / (dotNL * (1.0 - k) + k);
    float GV = dotNV / (dotNV * (1.0 - k) + k);
    return GL * GV;
}

// Schlick fresnel (F0 from base color, scaled by metallic).
float3 F_Schlick(float cosTheta, float metallic, float3 baseColor)
{
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float3 BRDF(float3 L, float3 V, float3 N, float metallic, float roughness, float3 baseColor)
{
    float3 H = normalize(V + L);
    float dotNV = clamp(dot(N, V), 0.0, 1.0);
    float dotNL = clamp(dot(N, L), 0.0, 1.0);
    float dotNH = clamp(dot(N, H), 0.0, 1.0);

    float3 color = float3(0.0, 0.0, 0.0);
    if (dotNL > 0.0)
    {
        float rr = max(0.05, roughness);
        float D = D_GGX(dotNH, roughness);
        float G = G_SchlicksmithGGX(dotNL, dotNV, rr);
        float3 F = F_Schlick(dotNV, metallic, baseColor);
        float3 spec = D * F * G / (4.0 * dotNL * dotNV);
        color += spec * dotNL; // white lights
    }
    return color;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    float3 N = normalize(input.normal);
    float3 V = normalize(uCam.camPos.xyz - input.worldPos);
    float roughness = input.mat.x;
    float metallic = input.mat.y;

    float3 Lo = float3(0.0, 0.0, 0.0);
    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        float3 L = normalize(uCam.lights[i].xyz - input.worldPos);
        Lo += BRDF(L, V, N, metallic, roughness, input.color);
    }

    float3 color = input.color * 0.02 + Lo;
    color = pow(color, float3(0.4545, 0.4545, 0.4545)); // gamma
    return float4(color, 1.0);
}
