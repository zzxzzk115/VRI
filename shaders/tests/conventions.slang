#include "coordinates.slang"

struct Push
{
    float4 clipTransform;
    float4 authoring; // clip Y sign, UV Y scale/bias, authored depth
    float4 uvTransform;
    float4 tint;
};

[[vk::push_constant]] Push pc;
[[vk::binding(0, 0)]] Texture2D<float4> colorTexture;
[[vk::binding(1, 0)]] SamplerState colorSampler;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

static const float2 positions[3] = {float2(0, -0.8), float2(0.8, 0.8), float2(-0.8, 0.8)};
static const float2 texcoords[3] = {float2(0.5, 1), float2(1, 0), float2(0, 0)};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    VertexOutput output;
    float4 position = float4(positions[vertexId].x, positions[vertexId].y * pc.authoring.x, pc.authoring.w, 1);
    output.position = vriTransformClipPosition(position, pc.clipTransform.xyz);
    float2 uv = texcoords[vertexId];
    uv.y = uv.y * pc.authoring.y + pc.authoring.z;
    output.uv = vriTransformUv(uv, pc.uvTransform.xy);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    return colorTexture.Sample(colorSampler, input.uv) * pc.tint;
}
