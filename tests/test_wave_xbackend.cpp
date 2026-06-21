// Subgroup / wave operations (Vulkan subgroups + D3D12 SM6.0 wave intrinsics) via the
// same VRI compute path. Each lane adds 1 to WaveActiveSum, so the result equals the
// wave's active-lane count; for a full wave that's the wave size, which must match the
// device-reported VriDeviceDesc::subgroupSize. Verifies wave ops work AND the reported
// size is consistent. Self-skips where wave ops are unavailable.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/wave_spv.h"  // g_waveSpv     (Vulkan)
#include "shaders/wave_dxil.h" // g_waveDxilCS  (D3D12)

namespace
{
    constexpr uint32_t kCount = 64; // numthreads(64) -> one or more full waves

    void RunWave(VriGraphicsAPI api, const void* cs, size_t csSize)
    {
        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = api; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success) { MESSAGE("device unavailable - skipping"); return; }
        struct Guard { VriDevice* d; ~Guard() { vriDestroyDevice(d); } } guard{dev};

        VriCoreInterface c{};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        const VriDeviceDesc* dd = c.GetDeviceDesc(dev);
        if (dd->hasShaderWaveOps == VRI_FALSE || dd->subgroupSize == 0) { MESSAGE("no wave ops - skipping"); return; }
        const uint32_t waveSize = dd->subgroupSize;

        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        VriBufferDesc sbd{}; sbd.size = kCount * sizeof(uint32_t); sbd.usage = VriBufferUsage_StorageBuffer | VriBufferUsage_TransferSrc; sbd.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* storage = nullptr; REQUIRE(c.CreateBuffer(dev, &sbd, &storage) == VriResult_Success);
        VriBufferDesc rbd{}; rbd.size = sbd.size; rbd.usage = VriBufferUsage_TransferDst; rbd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr; REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

        VriDescriptorRangeDesc range{}; range.baseRegister = 0; range.descriptorNum = 1; range.descriptorType = VriDescriptorType_StorageBuffer; range.shaderStages = VriShaderStage_Compute;
        VriDescriptorSetDesc setDesc{}; setDesc.registerSpace = 0; setDesc.ranges = &range; setDesc.rangeNum = 1;
        VriPipelineLayoutDesc ld{}; ld.descriptorSets = &setDesc; ld.descriptorSetNum = 1;
        VriPipelineLayout* layout = nullptr; REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriComputePipelineDesc cpd{};
        cpd.pipelineLayout = layout;
        cpd.shader.stage = VriShaderStage_Compute; cpd.shader.bytecode = cs; cpd.shader.bytecodeSize = csSize; cpd.shader.entryPointName = "computeMain";
        VriPipeline* pipeline = nullptr; REQUIRE(c.CreateComputePipeline(dev, &cpd, &pipeline) == VriResult_Success);

        VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 1; pdsc.storageBufferMaxNum = 1;
        VriDescriptorPool* pool = nullptr; REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
        VriDescriptorSet* set = nullptr; REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);
        VriBufferViewDesc bv{}; bv.buffer = storage; bv.viewType = VriDescriptorType_StorageBuffer; bv.offset = 0; bv.size = sbd.size;
        VriDescriptor* sbView = nullptr; REQUIRE(c.CreateBufferView(dev, &bv, &sbView) == VriResult_Success);
        const VriDescriptor* descs[1] = {sbView};
        VriDescriptorRangeUpdateDesc upd{}; upd.descriptors = descs; upd.descriptorNum = 1; upd.baseDescriptor = 0;
        c.UpdateDescriptorRanges(set, 0, 1, &upd);

        VriCommandAllocator* alloc = nullptr; REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr; REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr; REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        c.CmdSetPipeline(cmd, pipeline);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, set);
        VriDispatchDesc disp{}; disp.x = 1; disp.y = 1; disp.z = 1;
        c.CmdDispatch(cmd, &disp);
        {
            VriBufferBarrierDesc bb{};
            bb.buffer = storage;
            bb.before.access = VriAccess_ShaderResourceStorageWrite; bb.before.stages = VriPipelineStage_ComputeShader;
            bb.after.access = VriAccess_CopySourceRead; bb.after.stages = VriPipelineStage_Transfer;
            VriBarrierGroupDesc g{}; g.buffers = &bb; g.bufferNum = 1; c.CmdBarrier(cmd, &g);
        }
        VriBufferCopyDesc copy{}; copy.size = sbd.size;
        c.CmdCopyBuffer(cmd, readback, storage, &copy);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

        VriFenceSubmitDesc signal{}; signal.fence = fence; signal.value = 1; signal.stages = VriPipelineStage_AllCommands;
        VriQueueSubmitDesc submit{}; submit.commandBuffers = &cmd; submit.commandBufferNum = 1; submit.signalFences = &signal; submit.signalFenceNum = 1;
        c.QueueSubmit(queue, &submit);
        c.Wait(fence, 1);

        const uint32_t* out = static_cast<const uint32_t*>(c.MapBuffer(readback, 0, rbd.size));
        const uint32_t observed = out[0]; // WaveActiveSum(1) for lane 0's wave
        c.UnmapBuffer(readback);

        CHECK(waveSize >= 1);
        CHECK(observed == waveSize); // wave ops work AND match the reported subgroup size

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyDescriptorPool(pool);
        c.DestroyDescriptor(sbView); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(readback); c.DestroyBuffer(storage);
    }
} // namespace

TEST_CASE("Vulkan: subgroup/wave operations") { RunWave(VriGraphicsAPI_Vulkan, g_waveSpv, sizeof(g_waveSpv)); }
TEST_CASE("D3D12: subgroup/wave operations")  { RunWave(VriGraphicsAPI_D3D12, g_waveDxilCS, sizeof(g_waveDxilCS)); }
