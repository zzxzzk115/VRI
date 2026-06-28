// Descriptor indexing (bindless): sample from a fixed-size array of textures, choosing the
// element per-fragment with a non-uniform index carried as a vertex attribute. Used by
// examples/descriptorindexing on backends that report hasBindless (Vulkan, D3D12). The array
// is fully populated (kTexCount entries) and indexed with NonUniformResourceIndex so faces in
// one draw can pick different textures. WebGPU/WebGL2/GL can't do this and use arraytex.slang.
// Sequential TEXCOORD semantics keep vertex locations consistent across SPIR-V/DXBC.
static const uint kTexCount = 16;

struct VSInput
{
    float3 position : TEXCOORD0;
    float2 uv : TEXCOORD1;
    int    texIndex : TEXCOORD2; // integer vertex attribute (R32_SINT) - works on every backend
};
struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    nointerpolation int texIndex : TEXCOORD1;
};
struct Camera { float4x4 mvp; };

[[vk::binding(0, 0)]] ConstantBuffer<Camera> uCam;
[[vk::binding(1, 0)]] SamplerState uSmp;
[[vk::binding(2, 0)]] Texture2D<float4> uTex[kTexCount]; // bindless: fixed-size, fully populated

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput o;
    o.position = mul(uCam.mvp, float4(input.position, 1.0));
    o.uv = input.uv;
    o.texIndex = input.texIndex;
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return uTex[NonUniformResourceIndex(input.texIndex)].Sample(uSmp, input.uv);
}
