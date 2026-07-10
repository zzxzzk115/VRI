// No-wave compute fallback for examples/waveops, used where hasComputeShader is true but
// hasShaderWaveOps is false (WebGPU backend): the same three bands, driven by the thread id
// with an assumed stripe width of 32 instead of real wave intrinsics. The picture looks
// similar; the point is the example says so instead of silently pretending.
struct Params
{
    float time;
    uint width, height, pad;
};
[[vk::binding(0, 0)]] ConstantBuffer<Params> uParams;
// Explicit storage format: WGSL requires one, and Slang otherwise defaults to
// rgba32float, which mismatches this RGBA8 target (see compute_plasma.slang).
[[vk::binding(1, 0)]] [format("rgba8")] RWTexture2D<float4> uImage;

float3 Hue(float h)
{
    return float3(0.5 + 0.5 * sin(h * 6.2832),
                  0.5 + 0.5 * sin(h * 6.2832 + 2.094),
                  0.5 + 0.5 * sin(h * 6.2832 + 4.189));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= uParams.width || tid.y >= uParams.height)
        return;

    const uint kFakeWave = 32;
    uint laneIdx = tid.x % kFakeWave;
    // flipped like wave_vis.slang: screen order = lane index | prefix sum | ballot count
    float band = float(uParams.height - 1 - tid.y) / float(uParams.height);
    float t = uParams.time;

    float3 color;
    if (band < 0.3333)
    {
        color = Hue(float(laneIdx) / float(kFakeWave));
    }
    else if (band < 0.6667)
    {
        float ramp = frac(float(laneIdx) / float(kFakeWave) + t * 0.25);
        color = float3(ramp, ramp * 0.8, 1.0 - ramp) * 0.9;
    }
    else
    {
        // approximate the ballot count: how many lanes of this "wave" pass the predicate
        uint waveStart = tid.x - laneIdx;
        uint bits = 0;
        for (uint i = 0; i < kFakeWave; ++i)
            if (sin(float(waveStart + i) * 0.11 + t * 2.0 + float(i) * 0.31) > 0.0)
                ++bits;
        color = Hue(float(bits) / float(kFakeWave) * 0.7 + 0.15) * 0.85;
    }

    float fy = frac(float(tid.y) * 3.0 / float(uParams.height));
    if (fy < 0.015)
        color = float3(0.05, 0.05, 0.08);

    uImage[tid.xy] = float4(color, 1.0);
}
