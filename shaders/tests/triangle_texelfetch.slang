// Triangle whose fragment shader reads an INTEGER texture via texelFetch (.Load) with NO
// sampler - VRI's separate-image, no-sampler path (e.g. an r8ui palette-index texture).
// On GL this image has no (texture, sampler) pair to fuse, so SPIRV-Cross must synthesise a
// dummy sampler before build_combined_image_samplers(); this shader is that path's regression
// guard (the transpile otherwise aborts with "texelFetch without sampler").
struct VSOutput
{
    float4 position : SV_Position;
};

static const float2 kPositions[3] = { float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };

// The SPIR-V sampled-image format must match the r8ui view or Slang defaults it to r32ui and
// the fetch is undefined (Vulkan format validation).
[[vk::image_format("r8ui")]] [[vk::binding(0, 0)]] Texture2D<uint> uIndex;

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    VSOutput o;
    o.position = float4(kPositions[vid], 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    uint v = uIndex.Load(int3(0, 0, 0)); // integer texel fetch, no sampler
    return float4(float(v) / 255.0, 0.0, 0.0, 1.0);
}
