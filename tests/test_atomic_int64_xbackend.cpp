// 64-bit atomics on a storage buffer (VriFeature_ShaderAtomicInt64): a compute shader takes a 64-bit
// InterlockedMax and InterlockedAdd from 256 invocations; the readback must hold both full 64-bit results.
// Vulkan grants the feature from shaderInt64 + shaderBufferInt64Atomics, D3D12 from Shader Model 6.6 +
// Int64ShaderOps; a device without them self-skips. Metal does not implement the feature and refuses a strict
// request.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <initializer_list>
#include <string>

#include "shaders/tests/atomic_int64_dxil.h" // g_atomicInt64DxilCS (Shader Model 6.6)
#include "shaders/tests/atomic_int64_spv.h"  // g_atomicInt64Spv

namespace
{
    constexpr uint64_t kExpectedMax   = (0xFFFFFFF0ull << 32) | 200u;
    constexpr uint64_t kExpectedCount = 256ull * 0x100000001ull;

    void
    Barrier(VriCoreInterface& c, VriCommandBuffer* cmd, VriBuffer* buffer, VriAccessStage before, VriAccessStage after)
    {
        VriBufferBarrierDesc b {};
        b.buffer = buffer;
        b.before = before;
        b.after  = after;
        VriBarrierGroupDesc g {};
        g.buffers   = &b;
        g.bufferNum = 1;
        c.CmdBarrier(cmd, &g);
    }

    void Run(VriGraphicsAPI api, const char* name, const void* bytecode, size_t bytecodeSize)
    {
        VriDeviceCreationDesc dc  = vriDefaultDeviceCreationDesc();
        dc.coordinates.clipSpaceY = VriClipSpaceY_Up;
        dc.graphicsAPI            = api;
        dc.enableValidation       = VRI_TRUE;
        dc.bestEffort             = VRI_TRUE;
        dc.enabledFeatures        = VriFeature_ShaderAtomicInt64;
        VriDevice* dev            = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << std::string(name) << "] unavailable - skipped");
            return;
        }
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        const VriDeviceDesc* desc = c.GetDeviceDesc(dev);
        CHECK((desc->hasShaderAtomicInt64 != VRI_FALSE) ==
              ((desc->enabledFeatures & VriFeature_ShaderAtomicInt64) != 0));
        if (desc->hasShaderAtomicInt64 == VRI_FALSE)
        {
            MESSAGE("[" << std::string(name) << "] 64-bit buffer atomics unsupported - skipped");
            vriDestroyDevice(dev);
            return;
        }
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        constexpr uint64_t bytes = 2 * sizeof(uint64_t);
        VriBufferDesc      sbd {};
        sbd.size           = bytes;
        sbd.usage          = VriBufferUsage_StorageBuffer | VriBufferUsage_TransferSrc | VriBufferUsage_TransferDst;
        sbd.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* storage = nullptr;
        REQUIRE(c.CreateBuffer(dev, &sbd, &storage) == VriResult_Success);
        VriBufferDesc rbd {};
        rbd.size            = bytes;
        rbd.usage           = VriBufferUsage_TransferDst;
        rbd.memoryLocation  = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

        VriDescriptorRangeDesc range {};
        range.descriptorNum  = 1;
        range.descriptorType = VriDescriptorType_StorageBuffer;
        range.shaderStages   = VriShaderStage_Compute;
        VriDescriptorSetDesc setDesc {};
        setDesc.ranges   = &range;
        setDesc.rangeNum = 1;
        VriPipelineLayoutDesc ld {};
        ld.descriptorSets         = &setDesc;
        ld.descriptorSetNum       = 1;
        VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);
        VriComputePipelineDesc cpd {};
        cpd.pipelineLayout        = layout;
        cpd.shader.stage          = VriShaderStage_Compute;
        cpd.shader.bytecode       = bytecode;
        cpd.shader.bytecodeSize   = bytecodeSize;
        cpd.shader.entryPointName = "computeMain";
        VriPipeline* pipeline     = nullptr;
        REQUIRE(c.CreateComputePipeline(dev, &cpd, &pipeline) == VriResult_Success);

        VriDescriptorPoolDesc pdsc {};
        pdsc.descriptorSetMaxNum = 1;
        pdsc.storageBufferMaxNum = 1;
        VriDescriptorPool* pool  = nullptr;
        REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
        VriDescriptorSet* set = nullptr;
        REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);
        VriBufferViewDesc bv {};
        bv.buffer           = storage;
        bv.viewType         = VriDescriptorType_StorageBuffer;
        bv.size             = bytes;
        VriDescriptor* view = nullptr;
        REQUIRE(c.CreateBufferView(dev, &bv, &view) == VriResult_Success);
        const VriDescriptor*         descs[1] = {view};
        VriDescriptorRangeUpdateDesc upd {};
        upd.descriptors   = descs;
        upd.descriptorNum = 1;
        c.UpdateDescriptorRanges(set, 0, 1, &upd);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        // Start from zero: the maximum and the sum both depend on it.
        c.CmdClearStorageBuffer(cmd, storage, 0, 0, 0);
        Barrier(c,
                cmd,
                storage,
                {VriAccess_CopyDestinationWrite, VriPipelineStage_Transfer},
                {VriAccess_ShaderResourceStorageRead | VriAccess_ShaderResourceStorageWrite,
                 VriPipelineStage_ComputeShader});
        c.CmdSetPipeline(cmd, pipeline);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, set);
        VriDispatchDesc disp {};
        disp.x = 4; // 4 x 64 invocations
        disp.y = 1;
        disp.z = 1;
        c.CmdDispatch(cmd, &disp);
        Barrier(c,
                cmd,
                storage,
                {VriAccess_ShaderResourceStorageWrite, VriPipelineStage_ComputeShader},
                {VriAccess_CopySourceRead, VriPipelineStage_Transfer});
        VriBufferCopyDesc copy {};
        copy.size = bytes;
        c.CmdCopyBuffer(cmd, readback, storage, &copy);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

        VriFenceSubmitDesc signal {};
        signal.fence = fence;
        signal.value = 1;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        c.QueueSubmit(queue, &submit);
        c.Wait(fence, 1);

        const auto* values = static_cast<const uint64_t*>(c.MapBuffer(readback, 0, bytes));
        REQUIRE(values != nullptr);
        CHECK(values[0] == kExpectedMax);
        CHECK(values[1] == kExpectedCount);
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyDescriptor(view);
        c.DestroyDescriptorPool(pool);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(storage);
        c.DestroyBuffer(readback);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("64-bit atomics: InterlockedMax and InterlockedAdd on uint64_t (Vulkan)")
{
    Run(VriGraphicsAPI_Vulkan, "Vulkan", g_atomicInt64Spv, sizeof(g_atomicInt64Spv));
}

TEST_CASE("64-bit atomics: InterlockedMax and InterlockedAdd on uint64_t (D3D12)")
{
    Run(VriGraphicsAPI_D3D12, "D3D12", g_atomicInt64DxilCS, sizeof(g_atomicInt64DxilCS));
}

TEST_CASE("64-bit atomics: a backend that does not implement them refuses the request")
{
    for (const VriGraphicsAPI api : {VriGraphicsAPI_Metal})
    {
        VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI           = api;
        dc.bestEffort            = VRI_FALSE;
        dc.enabledFeatures       = VriFeature_ShaderAtomicInt64;
        VriDevice*      dev      = nullptr;
        const VriResult result   = vriCreateDevice(&dc, &dev);
        if (result == VriResult_Success)
        {
            VriCoreInterface c {};
            REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
            // Only a backend that really grants it may succeed.
            CHECK(c.GetDeviceDesc(dev)->hasShaderAtomicInt64 != VRI_FALSE);
            vriDestroyDevice(dev);
        }
    }
}
