// Async compute parity: split work across two queues synchronized by a timeline fence.
//   - The COMPUTE queue runs a compute shader that fills a storage buffer (out[i] = i + 100) and
//     signals the fence to value 1.
//   - A SECOND queue (Graphics or Transfer, see below) WAITS for fence == 1, then copies the buffer
//     to a host-readback buffer and signals value 2.
// The CPU waits for value 2 and checks every element. A correct result proves cross-queue submission
// and timeline-fence synchronization work: if the copy did not wait for the compute write, it would
// read stale/garbage data. Backends that give Compute its own queue (Vulkan: a second queue in the
// graphics family; Metal: a dedicated MTLCommandQueue; D3D12: the COMPUTE engine) run the two passes
// on distinct queues; where only one queue exists the timeline fence still orders them.
//
// We run it twice per backend, varying which queue performs the copy:
//   - Graphics (DIRECT engine on D3D12)  -> exercises COMPUTE -> GRAPHICS cross-queue sync.
//   - Transfer (COPY/DMA engine on D3D12) -> exercises COMPUTE -> TRANSFER sync and the dedicated
//     copy engine end-to-end (a buffer copy needs no explicit barrier - it promotes from COMMON).
// Skips where compute is absent.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <string>

#include "shaders/compute_fill_spv.h"  // g_computeFillSpv    (Vulkan / OpenGL / Metal via SPIR-V)
#include "shaders/compute_fill_dxbc.h" // g_computeFillDxbcCS (Direct3D 12)

namespace
{
    constexpr uint32_t kCount = 64;

    // copyQueueType selects which queue copies the compute result back: Graphics or Transfer.
    bool RunAsyncCompute(VriGraphicsAPI api, const void* shader, size_t shaderSize, VriQueueType copyQueueType, bool& ran, bool& hasCompute)
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
        if (!hasCompute) { vriDestroyDevice(dev); return true; } // nothing to run; not a failure

        VriQueue* copyQueue = nullptr; REQUIRE(c.GetQueue(dev, copyQueueType, 0, &copyQueue) == VriResult_Success);
        VriQueue* compQueue = nullptr; REQUIRE(c.GetQueue(dev, VriQueueType_Compute, 0, &compQueue) == VriResult_Success);

        // storage buffer (compute writes) + host readback
        VriBufferDesc sbd{}; sbd.size = kCount * sizeof(uint32_t); sbd.usage = VriBufferUsage_StorageBuffer | VriBufferUsage_TransferSrc; sbd.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* storage = nullptr; REQUIRE(c.CreateBuffer(dev, &sbd, &storage) == VriResult_Success);
        VriBufferDesc rbd{}; rbd.size = kCount * sizeof(uint32_t); rbd.usage = VriBufferUsage_TransferDst; rbd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr; REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

        VriDescriptorRangeDesc range{}; range.baseRegister = 0; range.descriptorNum = 1; range.descriptorType = VriDescriptorType_StorageBuffer; range.shaderStages = VriShaderStage_Compute;
        VriDescriptorSetDesc setDesc{}; setDesc.registerSpace = 0; setDesc.ranges = &range; setDesc.rangeNum = 1;
        VriPipelineLayoutDesc ld{}; ld.descriptorSets = &setDesc; ld.descriptorSetNum = 1;
        VriPipelineLayout* layout = nullptr; REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriComputePipelineDesc cpd{}; cpd.pipelineLayout = layout;
        cpd.shader.stage = VriShaderStage_Compute; cpd.shader.bytecode = shader; cpd.shader.bytecodeSize = shaderSize; cpd.shader.entryPointName = "computeMain";
        VriPipeline* pipeline = nullptr; REQUIRE(c.CreateComputePipeline(dev, &cpd, &pipeline) == VriResult_Success);

        VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 1; pdsc.storageBufferMaxNum = 1;
        VriDescriptorPool* pool = nullptr; REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
        VriDescriptorSet* set = nullptr; REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);
        VriBufferViewDesc bv{}; bv.buffer = storage; bv.viewType = VriDescriptorType_StorageBuffer; bv.offset = 0; bv.size = sbd.size;
        VriDescriptor* sbView = nullptr; REQUIRE(c.CreateBufferView(dev, &bv, &sbView) == VriResult_Success);
        const VriDescriptor* descs[1] = {sbView};
        VriDescriptorRangeUpdateDesc upd{}; upd.descriptors = descs; upd.descriptorNum = 1; upd.baseDescriptor = 0;
        c.UpdateDescriptorRanges(set, 0, 1, &upd);

        VriFence* fence = nullptr; REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        // ---- pass 1: COMPUTE queue dispatches + signals fence value 1 ----
        VriCommandAllocator* compAlloc = nullptr; REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Compute, &compAlloc) == VriResult_Success);
        VriCommandBuffer* compCmd = nullptr; REQUIRE(c.CreateCommandBuffer(compAlloc, &compCmd) == VriResult_Success);
        REQUIRE(c.BeginCommandBuffer(compCmd) == VriResult_Success);
        c.CmdSetPipeline(compCmd, pipeline);
        c.CmdSetPipelineLayout(compCmd, layout);
        c.CmdSetDescriptorSet(compCmd, 0, set);
        VriDispatchDesc disp{}; disp.x = 1; disp.y = 1; disp.z = 1; // one 64-wide group
        c.CmdDispatch(compCmd, &disp);
        {
            VriBufferBarrierDesc bb{}; bb.buffer = storage;
            bb.before.access = VriAccess_ShaderResourceStorageWrite; bb.before.stages = VriPipelineStage_ComputeShader;
            bb.after.access = VriAccess_CopySourceRead; bb.after.stages = VriPipelineStage_Transfer;
            VriBarrierGroupDesc g{}; g.buffers = &bb; g.bufferNum = 1; c.CmdBarrier(compCmd, &g);
        }
        REQUIRE(c.EndCommandBuffer(compCmd) == VriResult_Success);
        VriFenceSubmitDesc sig1{}; sig1.fence = fence; sig1.value = 1; sig1.stages = VriPipelineStage_AllCommands;
        VriQueueSubmitDesc sub1{}; sub1.commandBuffers = &compCmd; sub1.commandBufferNum = 1; sub1.signalFences = &sig1; sub1.signalFenceNum = 1;
        c.QueueSubmit(compQueue, &sub1);

        // ---- pass 2: copy queue (Graphics or Transfer) waits for fence value 1, copies, signals value 2 ----
        VriCommandAllocator* copyAlloc = nullptr; REQUIRE(c.CreateCommandAllocator(dev, copyQueueType, &copyAlloc) == VriResult_Success);
        VriCommandBuffer* copyCmd = nullptr; REQUIRE(c.CreateCommandBuffer(copyAlloc, &copyCmd) == VriResult_Success);
        REQUIRE(c.BeginCommandBuffer(copyCmd) == VriResult_Success);
        VriBufferCopyDesc copy{}; copy.size = sbd.size;
        c.CmdCopyBuffer(copyCmd, readback, storage, &copy);
        REQUIRE(c.EndCommandBuffer(copyCmd) == VriResult_Success);
        VriFenceSubmitDesc wait1{}; wait1.fence = fence; wait1.value = 1; wait1.stages = VriPipelineStage_Transfer;
        VriFenceSubmitDesc sig2{}; sig2.fence = fence; sig2.value = 2; sig2.stages = VriPipelineStage_AllCommands;
        VriQueueSubmitDesc sub2{}; sub2.waitFences = &wait1; sub2.waitFenceNum = 1; sub2.commandBuffers = &copyCmd; sub2.commandBufferNum = 1; sub2.signalFences = &sig2; sub2.signalFenceNum = 1;
        c.QueueSubmit(copyQueue, &sub2);

        c.Wait(fence, 2);

        const uint32_t* px = static_cast<const uint32_t*>(c.MapBuffer(readback, 0, rbd.size));
        REQUIRE(px != nullptr);
        bool allOk = true;
        for (uint32_t i = 0; i < kCount; ++i)
            if (px[i] != i + 100u) { allOk = false; break; }
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(copyAlloc); c.DestroyCommandAllocator(compAlloc);
        c.DestroyDescriptor(sbView); c.DestroyDescriptorPool(pool);
        c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(storage); c.DestroyBuffer(readback);
        vriDestroyDevice(dev);
        return allOk;
    }
} // namespace

TEST_CASE("async compute: compute queue fills a buffer; a graphics- or transfer-queue copy waits on a timeline fence")
{
    struct Backend { VriGraphicsAPI api; const void* shader; size_t size; const char* name; };
    const Backend backends[] = {
        {VriGraphicsAPI_Vulkan, g_computeFillSpv,     sizeof(g_computeFillSpv),     "Vulkan"},
        {VriGraphicsAPI_Metal,  g_computeFillSpv,     sizeof(g_computeFillSpv),     "Metal"},
        {VriGraphicsAPI_OpenGL, g_computeFillSpv,     sizeof(g_computeFillSpv),     "OpenGL"},
        {VriGraphicsAPI_D3D12,  g_computeFillDxbcCS,  sizeof(g_computeFillDxbcCS),  "D3D12"},
    };
    // Vary the copy queue so both the COMPUTE->GRAPHICS and COMPUTE->TRANSFER (dedicated copy engine)
    // sync paths are exercised on every available backend.
    struct CopyQueue { VriQueueType type; const char* name; };
    const CopyQueue copyQueues[] = {{VriQueueType_Graphics, "graphics"}, {VriQueueType_Transfer, "transfer"}};

    for (const Backend& b : backends)
        for (const CopyQueue& cq : copyQueues)
        {
            bool ran = false, hasCompute = false;
            const bool ok = RunAsyncCompute(b.api, b.shader, b.size, cq.type, ran, hasCompute);
            // doctest stringifies a bare `const char*` as a pointer and MESSAGE's `*`-binding rejects
            // a `+` expression, so build the line as one std::string and pass that single variable.
            const std::string msg = ran
                ? std::string(b.name) + " async-compute via " + cq.name + " copy, compute=" + (hasCompute ? "yes" : "no")
                : std::string(b.name) + " unavailable - skipped";
            if (ran) CHECK(ok);
            MESSAGE(msg);
        }
}
