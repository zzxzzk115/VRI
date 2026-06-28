// Bindless textures (descriptor indexing): a runtime-sized Texture2D array sampled
// at an index supplied by a UBO. The array is the last binding in the set
// (VARIABLE_DESCRIPTOR_COUNT) and only a few slots are populated (PARTIALLY_BOUND).
// SPIR-V is what the Vulkan bindless test consumes.
struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

struct Params { uint index; };

static const float2 kPositions[3] = { float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };

[[vk::binding(0, 0)]] ConstantBuffer<Params> uParams;
[[vk::binding(1, 0)]] SamplerState uSmp;
[[vk::binding(2, 0)]] Texture2D<float4> uTex[]; // bindless: runtime-sized array

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    VSOutput o;
    o.position = float4(kPositions[vid], 0.0, 1.0);
    o.uv = float2(0.5, 0.5); // solid textures -> sample anywhere
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return uTex[uParams.index].Sample(uSmp, input.uv);
}
