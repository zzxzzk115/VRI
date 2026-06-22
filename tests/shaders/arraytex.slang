// Fallback for examples/descriptorindexing on backends without descriptor indexing
// (WebGPU/WebGL2/GL): the N textures are packed into a single 2D array texture and the
// per-vertex index selects the array LAYER instead of a descriptor. Same picture, different
// mechanism - explicit degradation. Same vertex layout as descriptorindexing.slang.
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
[[vk::binding(2, 0)]] Texture2DArray<float4> uTexArray;

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
    return uTexArray.Sample(uSmp, float3(input.uv, float(input.texIndex)));
}
