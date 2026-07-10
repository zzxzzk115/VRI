// GPU culling pass for examples/gpudriven: one thread per instance frustum-tests its
// bounding sphere against the (freezable) culling view-projection and emits GPU-driven
// draws. Two modes, picked by the example from VriDeviceDesc::hasDrawIndirectCount:
//   compact (misc0.y = 1): survivors append a compacted instance + indirect record via an
//     atomic counter; CmdDrawIndexedIndirectCount consumes counter + records.
//   slot    (misc0.y = 0): every instance keeps its own record slot; culled slots get
//     indexNum = 0 (a draw that draws nothing) so a plain CmdDrawIndexedIndirect with
//     drawNum = total renders only the survivors.
// The pass also plots a top-down minimap (visible = green, culled = dim) into a storage
// texture the example clears each frame with CmdClearStorageTexture.
struct Instance
{
    float4 posScale; // xyz = world position, w = uniform scale
    float4 color;
};

struct Params
{
    float4x4 cullViewProj;
    float4 misc0; // x = instance total, y = compact mode (0/1), z = minimap size (0 = off), w = field extent
    float4 misc1; // x = bounding radius at scale 1
};
[[vk::binding(0, 0)]] ConstantBuffer<Params> uParams;
[[vk::binding(1, 0)]] StructuredBuffer<Instance> uInstances;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> uRecords;  // 5 uints per VriDrawIndexedDesc record
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> uCount;    // [0] = visible count / indirect draw count
[[vk::binding(4, 0)]] RWStructuredBuffer<Instance> uVisible; // per-instance vertex stream for the draw
[[vk::binding(5, 0)]] [format("rgba8")] RWTexture2D<float4> uMinimap;

static const uint kCubeIndexNum = 36;

void WriteRecord(uint slot, uint indexNum, uint baseInstance)
{
    uRecords[slot * 5 + 0] = indexNum;     // indexNum
    uRecords[slot * 5 + 1] = 1;            // instanceNum
    uRecords[slot * 5 + 2] = 0;            // baseIndex
    uRecords[slot * 5 + 3] = 0;            // vertexOffset
    uRecords[slot * 5 + 4] = baseInstance; // baseInstance -> offsets the per-instance vertex stream
}

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint total = uint(uParams.misc0.x + 0.5);
    if (tid.x >= total)
        return;

    Instance inst = uInstances[tid.x];
    float4 clip = mul(uParams.cullViewProj, float4(inst.posScale.xyz, 1.0));
    // conservative sphere-vs-frustum in clip space (0..w depth): grow w by the projected radius
    float r = uParams.misc1.x * inst.posScale.w;
    float grow = clip.w + r * 2.0;
    bool visible = clip.w > -r
                && abs(clip.x) <= grow
                && abs(clip.y) <= grow
                && clip.z >= -r * 2.0
                && clip.z <= grow;

    if (uParams.misc0.y >= 0.5)
    {
        // compact mode: append survivors; CmdDrawIndexedIndirectCount reads uCount[0]
        if (visible)
        {
            uint slot;
            InterlockedAdd(uCount[0], 1, slot);
            uVisible[slot] = inst;
            WriteRecord(slot, kCubeIndexNum, slot);
        }
    }
    else
    {
        // slot mode (no hasDrawIndirectCount): every instance owns record #i; culled draws are empty
        uVisible[tid.x] = inst;
        WriteRecord(tid.x, visible ? kCubeIndexNum : 0, tid.x);
        if (visible)
        {
            uint unused;
            InterlockedAdd(uCount[0], 1, unused); // UI counter only
        }
    }

    // top-down minimap dot (xz plane); the example cleared the texture this frame
    uint mapSize = uint(uParams.misc0.z + 0.5);
    if (mapSize > 0)
    {
        float2 uv = inst.posScale.xz / uParams.misc0.w * 0.5 + 0.5;
        uint2 px = uint2(saturate(uv) * float(mapSize - 1));
        uMinimap[px] = visible ? float4(0.3, 1.0, 0.45, 1.0) : float4(0.25, 0.27, 0.33, 1.0);
    }
}
