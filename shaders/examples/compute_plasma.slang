// Compute pass for examples/computeshader: write an animated plasma into a storage image.
// Paired with show_tex.slang (which samples the result). The same Plasma() is mirrored in
// gen_plasma.slang for the WebGL2 fragment fallback.
struct Params
{
    float time;
    uint  width;
    uint  height;
    uint  pad;
};

float3 Plasma(float2 uv, float t)
{
    float v = sin(uv.x * 10.0 + t)
            + sin(uv.y * 10.0 + t * 1.3)
            + sin((uv.x + uv.y) * 10.0 + t * 0.7)
            + sin(length(uv - 0.5) * 16.0 - t * 1.5);
    return 0.5 + 0.5 * cos(float3(0.0, 2.0, 4.0) + v * 1.2);
}

[[vk::binding(0, 0)]] ConstantBuffer<Params> uParams;
// Explicit storage format: WGSL requires an explicit format on storage textures, and Slang
// otherwise defaults to rgba32float/read_write - which mismatches this RGBA8 target and is
// invalid for read_write in WebGPU. "rgba8" pins it to rgba8unorm to match the texture + the
// backend's storage bind-group layout.
[[vk::binding(1, 0)]] [format("rgba8")] RWTexture2D<float4> uOut;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= uParams.width || id.y >= uParams.height)
        return;
    float2 uv = (float2(id.xy) + 0.5) / float2(uParams.width, uParams.height);
    uOut[id.xy] = float4(Plasma(uv, uParams.time), 1.0);
}
