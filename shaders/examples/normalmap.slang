// Tangent-space normal mapping (examples/normalmapping): a textured quad whose per-pixel
// normal comes from a normal map, lit by an orbiting point light (Blinn-Phong-ish). The
// quad's TBN is constant (model is identity), so the vertex stage just forwards it. Vertex
// inputs use sequential TEXCOORD semantics so locations are 0..3 on every target; mvp is
// uploaded transposed (mul(mvp, p) matches the column-major data, like the other examples).
struct VSInput
{
    float3 position : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float3 tangent : TEXCOORD2;
    float2 uv : TEXCOORD3;
};
struct VSOutput
{
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD0;
    float2 uv : TEXCOORD1;
    float3 T : TEXCOORD2;
    float3 B : TEXCOORD3;
    float3 N : TEXCOORD4;
};

struct Ubo
{
    float4x4 mvp;       // transposed proj * view (model is identity)
    float4    lightPos; // xyz, world space
    float4    camPos;   // xyz, world space
};
[[vk::binding(0, 0)]] ConstantBuffer<Ubo> u;
[[vk::binding(1, 0)]] Texture2D<float4> colorTex;
[[vk::binding(2, 0)]] Texture2D<float4> normalTex;
[[vk::binding(3, 0)]] SamplerState samp;

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput o;
    o.position = mul(u.mvp, float4(input.position, 1.0));
    o.worldPos = input.position;
    o.uv = input.uv;
    o.N = input.normal;
    o.T = input.tangent;
    o.B = cross(input.normal, input.tangent);
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    float3 tn = normalTex.Sample(samp, input.uv).rgb * 2.0 - 1.0; // tangent-space normal
    float3 N = normalize(input.N);
    float3 T = normalize(input.T);
    float3 B = normalize(input.B);
    float3 n = normalize(T * tn.x + B * tn.y + N * tn.z);          // -> world space

    float3 L = normalize(u.lightPos.xyz - input.worldPos);
    float3 V = normalize(u.camPos.xyz - input.worldPos);
    float3 H = normalize(L + V);

    float3 albedo = colorTex.Sample(samp, input.uv).rgb;
    float diff = max(dot(n, L), 0.0);
    float spec = pow(max(dot(n, H), 0.0), 32.0);

    float3 color = albedo * (0.15 + diff) + float3(0.6, 0.6, 0.6) * spec;
    return float4(color, 1.0);
}
