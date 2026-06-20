// Cross-backend compute parity: a compute shader fills a storage buffer with
// out[i] = i + 100 over one 64-wide workgroup; the buffer is read back and every
// element checked. Runs on whichever backends report hasComputeShader (Vulkan,
// desktop OpenGL); backends without compute (WebGPU path not wired yet, WebGL2)
// report false and are skipped - the explicit-capability contract.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/compute_fill_spv.h"  // g_computeFillSpv  (Vulkan + OpenGL via SPIRV-Cross)
#include "shaders/compute_fill_wgsl.h" // g_computeFillWgsl (WebGPU)

namespace
{
    constexpr uint32_t kCount = 64;

    bool RunCompute(VriGraphicsAPI api, const void* shader, size_t shaderSize, bool& ran, bool& hasCompute)
    {
        ran = false; hasCompute = false;
        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = api; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;
        ran = true;

        VriCoreInterface c{};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);

        hasCompute = c.GetDeviceDesc(dev)->hasComputeShader != VRI_FALSE;
        if (!hasCompute)
        {
            // Contract: unsupported -> CreateComputePipeline returns Unsupported.
            VriPipelineLayoutDesc ld0{}; VriPipelineLayout* l0 = nullptr; c.CreatePipelineLayout(dev, &ld0, &l0);
            VriComputePipelineDesc cpd{}; cpd.pipelineLayout = l0;
            cpd.shader.stage = VriShaderStage_Compute; cpd.shader.bytecode = shader; cpd.shader.bytecodeSize = shaderSize; cpd.shader.entryPointName = "computeMain";
            VriPipeline* p0 = nullptr;
            const bool unsupported = c.CreateComputePipeline(dev, &cpd, &p0) == VriResult_Unsupported;
            if (l0) c.DestroyPipelineLayout(l0);
            vriDestroyDevice(dev);
            return unsupported;
        }

        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        // storage buffer (written by the shader) + host readback
        VriBufferDesc sbd{};
        sbd.size = kCount * sizeof(uint32_t); sbd.usage = VriBufferUsage_StorageBuffer | VriBufferUsage_TransferSrc; sbd.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* storage = nullptr;
        REQUIRE(c.CreateBuffer(dev, &sbd, &storage) == VriResult_Success);
        VriBufferDesc rbd{};
        rbd.size = kCount * sizeof(uint32_t); rbd.usage = VriBufferUsage_TransferDst; rbd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

        // layout: set 0 binding 0 = storage buffer (compute stage)
        VriDescriptorRangeDesc range{}; range.baseRegister = 0; range.descriptorNum = 1; range.descriptorType = VriDescriptorType_StorageBuffer; range.shaderStages = VriShaderStage_Compute;
        VriDescriptorSetDesc setDesc{}; setDesc.registerSpace = 0; setDesc.ranges = &range; setDesc.rangeNum = 1;
        VriPipelineLayoutDesc ld{}; ld.descriptorSets = &setDesc; ld.descriptorSetNum = 1;
        VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriComputePipelineDesc cpd{};
        cpd.pipelineLayout = layout;
        cpd.shader.stage = VriShaderStage_Compute; cpd.shader.bytecode = shader; cpd.shader.bytecodeSize = shaderSize; cpd.shader.entryPointName = "computeMain";
        VriPipeline* pipeline = nullptr;
        REQUIRE(c.CreateComputePipeline(dev, &cpd, &pipeline) == VriResult_Success);

        VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 1; pdsc.storageBufferMaxNum = 1;
        VriDescriptorPool* pool = nullptr;
        REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
        VriDescriptorSet* set = nullptr;
        REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);
        VriBufferViewDesc bv{}; bv.buffer = storage; bv.viewType = VriDescriptorType_StorageBuffer; bv.offset = 0; bv.size = sbd.size;
        VriDescriptor* sbView = nullptr;
        REQUIRE(c.CreateBufferView(dev, &bv, &sbView) == VriResult_Success);
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
        VriDispatchDesc disp{}; disp.x = 1; disp.y = 1; disp.z = 1; // one 64-wide group
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

        VriFenceSubmitDesc signal{}; signal.fence = fence; signal.value = 1;
        VriQueueSubmitDesc submit{}; submit.commandBuffers = &cmd; submit.commandBufferNum = 1; submit.signalFences = &signal; submit.signalFenceNum = 1;
        c.QueueSubmit(queue, &submit);
        c.Wait(fence, 1);

        const uint32_t* px = static_cast<const uint32_t*>(c.MapBuffer(readback, 0, rbd.size));
        REQUIRE(px != nullptr);
        bool allOk = true;
        for (uint32_t i = 0; i < kCount; ++i)
            if (px[i] != i + 100u) { allOk = false; break; }
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyDescriptor(sbView); c.DestroyDescriptorPool(pool);
        c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(storage); c.DestroyBuffer(readback);
        vriDestroyDevice(dev);
        return allOk;
    }
} // namespace

TEST_CASE("compute parity: a compute shader fills a storage buffer (out[i] = i + 100)")
{
    bool ran = false, hasCompute = false;
    const bool vk = RunCompute(VriGraphicsAPI_Vulkan, g_computeFillSpv, sizeof(g_computeFillSpv), ran, hasCompute);
    if (ran) { CHECK(vk); MESSAGE("Vulkan compute=", hasCompute); } else { MESSAGE("Vulkan unavailable - skipped"); }

    const bool wgpu = RunCompute(VriGraphicsAPI_WebGPU, g_computeFillWgsl, sizeof(g_computeFillWgsl), ran, hasCompute);
    if (ran) { CHECK(wgpu); MESSAGE("WebGPU compute=", hasCompute); } else { MESSAGE("WebGPU unavailable - skipped"); }

    const bool gl = RunCompute(VriGraphicsAPI_OpenGL, g_computeFillSpv, sizeof(g_computeFillSpv), ran, hasCompute);
    if (ran) { CHECK(gl); MESSAGE("OpenGL compute=", hasCompute); } else { MESSAGE("OpenGL unavailable - skipped"); }
}
