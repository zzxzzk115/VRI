// Triangle whose color comes from a uniform buffer at set=0, binding=0.
// Validates the descriptor-set path with Slang-authored shaders.
struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

struct Tint
{
    float4 color;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Tint> uTint;

static const float2 kPositions[3] = { float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    VSOutput o;
    o.position = float4(kPositions[vid], 0.0, 1.0);
    o.color = float3(0.0, 0.0, 0.0);
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return uTint.color;
}
