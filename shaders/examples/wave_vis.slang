// Wave/subgroup visualization for examples/waveops (compute, wave intrinsics). Each 64-wide
// workgroup covers a horizontal pixel run; the image shows three bands driven by real wave
// intrinsics, so the hardware subgroup size is literally visible as the stripe period:
//   top:    hue ramp from WaveGetLaneIndex()/WaveGetLaneCount() - one rainbow per wave
//   middle: WavePrefixSum(1) brightness ramp, scrolled by time
//   bottom: WaveActiveCountBits of an animated per-lane predicate - a solid color per wave
//           that "counts" how many of its lanes are active this frame
// Maps to Vulkan subgroups (SPIR-V), SM6.0 wave intrinsics (DXIL cs_6_0), and WGSL
// 'enable subgroups'. Fallbacks: wave_vis_basic.slang (compute, no waves) and
// wave_vis_frag.slang (fragment, WebGL2).
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

    uint laneIdx = WaveGetLaneIndex();
    uint laneCount = WaveGetLaneCount();
    // The fullscreen display pass samples the storage image flipped vertically (see the
    // examples' composite convention), so key the bands off the flipped y: on SCREEN the
    // order reads top-to-bottom as lane index | prefix sum | ballot count.
    float band = float(uParams.height - 1 - tid.y) / float(uParams.height);
    float t = uParams.time;

    float3 color;
    if (band < 0.3333)
    {
        color = Hue(float(laneIdx) / float(laneCount)); // one rainbow per wave
    }
    else if (band < 0.6667)
    {
        float ramp = frac(float(WavePrefixSum(1u)) / float(laneCount) + t * 0.25);
        color = float3(ramp, ramp * 0.8, 1.0 - ramp) * 0.9;
    }
    else
    {
        bool active = sin(float(tid.x) * 0.11 + t * 2.0 + float(laneIdx) * 0.31) > 0.0;
        uint bits = WaveActiveCountBits(active);
        color = Hue(float(bits) / float(laneCount) * 0.7 + 0.15) * 0.85;
    }

    // thin separators between the bands
    float fy = frac(float(tid.y) * 3.0 / float(uParams.height));
    if (fy < 0.015)
        color = float3(0.05, 0.05, 0.08);

    uImage[tid.xy] = float4(color, 1.0);
}
