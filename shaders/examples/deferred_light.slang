// Deferred shading - lighting (composition) pass for examples/deferred. A single fullscreen
// triangle (no vertex buffer, same trick as composite.slang) reads the G-buffer that
// deferred_mrt.slang produced - world position, normal, albedo+specular - and shades each
// pixel against ALL the lights, accumulating diffuse + specular with a radius-based
// attenuation. This is the point of deferred: lighting runs once per screen pixel, not once
// per (overdrawn) fragment, so adding lights is cheap. Ported from Sascha Willems' deferred
// (deferred.frag), including the debug-target view that lets you inspect each G-buffer plane.
struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    float2 p = float2((vid << 1) & 2, vid & 2); // {0,2}
    VSOutput o;
    // VRI is Y-up clip with a top-left texture origin, so uv.y=0 must land at the TOP of the
    // screen (clip.y=+1) to sample the G-buffer upright - hence the Y is flipped in clip space
    // (clip.y = 1 - 2*uv.y) rather than the plain p*2-1, which would mirror the result.
    o.position = float4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
    o.uv = p;
    return o;
}

static const int kLightCount = 6;

// Parallel float4 arrays rather than an array of a {position,color} struct: an array of
// structs inside a uniform block transpiles (SPIRV-Cross -> ESSL 3.0) to GLSL that ANGLE /
// WebGL2 rejects, so the fragment shader fails to compile and the pipeline can't be created.
// Flat vec4 arrays are the portable form (same approach as pbr.slang's lights[]).
struct Ubo
{
    float4 lightPosition[kLightCount]; // xyz = world position
    float4 lightColor[kLightCount];    // rgb = color, w = radius
    float4 viewPos;                    // xyz = camera world position
    int4 params;                       // x = debug target (0 = composite, 1 = position, 2 = normal, 3 = albedo, 4 = specular)
};

// The constant buffer is at binding 0: on D3D12 a root CBV takes its register from the
// binding number, and Slang assigns the cbuffer to b0 - so the CB must sit at 0 to match
// (textures/samplers map to t#/s# tables that always start at 0, so their binding order is
// what matters, not the absolute number).
[[vk::binding(0, 0)]] ConstantBuffer<Ubo> uUbo;
[[vk::binding(1, 0)]] Texture2D<float4> uPosition;
[[vk::binding(2, 0)]] Texture2D<float4> uNormal;
[[vk::binding(3, 0)]] Texture2D<float4> uAlbedoSpec;
[[vk::binding(4, 0)]] SamplerState uSmp;

// One light's contribution. Taking the light data by value (not indexing the UBO inside) keeps
// the UBO access at the call site, where a literal index stays a compile-time constant - see the
// constant-indexed calls in fragmentMain.
float3 shadeLight(float4 lightPos, float4 lightCol, float3 fragPos, float3 N, float3 V, float4 albedoSpec)
{
    float3 toLight = lightPos.xyz - fragPos;
    float dist = length(toLight);
    float3 L = toLight / max(dist, 1e-4);
    float atten = lightCol.w / (dist * dist + 1.0);

    float NdotL = max(0.0, dot(N, L));
    float3 diff = lightCol.rgb * albedoSpec.rgb * NdotL * atten;

    float3 R = reflect(-L, N);
    float NdotR = max(0.0, dot(R, V));
    float3 spec = lightCol.rgb * albedoSpec.a * pow(NdotR, 16.0) * atten;
    return diff + spec;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    // Sample every G-buffer plane unconditionally up front: WGSL forbids texture sampling in
    // non-uniform control flow, so the debug branch below must run on already-fetched values.
    float3 fragPos = uPosition.Sample(uSmp, input.uv).rgb;
    float3 N = normalize(uNormal.Sample(uSmp, input.uv).rgb * 2.0 - 1.0);
    float4 albedoSpec = uAlbedoSpec.Sample(uSmp, input.uv);

    int debug = uUbo.params.x;
    if (debug == 1) return float4(fragPos * 0.08 + 0.5, 1.0); // position (scaled into [0,1]-ish)
    if (debug == 2) return float4(N * 0.5 + 0.5, 1.0);        // normal
    if (debug == 3) return float4(albedoSpec.rgb, 1.0);       // albedo
    if (debug == 4) return float4(albedoSpec.aaa, 1.0);       // specular

    float3 V = normalize(uUbo.viewPos.xyz - fragPos);

    // Accumulate all six lights with COMPILE-TIME-CONSTANT UBO indices. A runtime loop here
    // (even with [unroll]) gets re-rolled by SPIRV-Cross into dynamic uUbo.lightPosition[i]
    // indexing, which ANGLE/WebGL2's D3D11 backend rejects (HLSL can't dynamically index the
    // std140-wrapped cbuffer struct-array) - so it must stay unrolled with literal indices.
    float3 color = albedoSpec.rgb * 0.1; // ambient term
    color += shadeLight(uUbo.lightPosition[0], uUbo.lightColor[0], fragPos, N, V, albedoSpec);
    color += shadeLight(uUbo.lightPosition[1], uUbo.lightColor[1], fragPos, N, V, albedoSpec);
    color += shadeLight(uUbo.lightPosition[2], uUbo.lightColor[2], fragPos, N, V, albedoSpec);
    color += shadeLight(uUbo.lightPosition[3], uUbo.lightColor[3], fragPos, N, V, albedoSpec);
    color += shadeLight(uUbo.lightPosition[4], uUbo.lightColor[4], fragPos, N, V, albedoSpec);
    color += shadeLight(uUbo.lightPosition[5], uUbo.lightColor[5], fragPos, N, V, albedoSpec);
    return float4(color, 1.0);
}
