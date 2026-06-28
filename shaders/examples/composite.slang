// Display pass for examples/offscreen: a fullscreen triangle that samples the two offscreen
// MRT targets side by side - albedo on the left half, the face-normal G-buffer on the right -
// so one frame proves both render targets were written and then sampled. No vertex buffer.
struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    float2 p = float2((vid << 1) & 2, vid & 2);
    VSOutput o;
    o.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = p;
    return o;
}

[[vk::binding(0, 0)]] Texture2D<float4> uAlbedo;
[[vk::binding(1, 0)]] Texture2D<float4> uNormal;
[[vk::binding(2, 0)]] SamplerState uSmp;

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    // Sample BOTH targets unconditionally, then pick - WGSL forbids texture sampling (which
    // needs implicit derivatives) inside non-uniform control flow, so branching around the
    // sample compiles on SPIR-V/HLSL/GLSL but is rejected by WebGPU.
    float4 albedo = uAlbedo.Sample(uSmp, float2(input.uv.x * 2.0, input.uv.y));
    float4 normal = uNormal.Sample(uSmp, float2((input.uv.x - 0.5) * 2.0, input.uv.y));
    return input.uv.x < 0.5 ? albedo : normal;
}
