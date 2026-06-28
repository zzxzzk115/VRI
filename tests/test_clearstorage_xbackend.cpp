// Clear storage buffer (VriCoreInterface::CmdClearStorageBuffer): fill a storage buffer with a
// repeated uint32 outside a render pass, then read it back and confirm every element is the value.
// Gated on VriDeviceDesc::hasClearStorageBuffer (Vulkan + desktop OpenGL 4.3; D3D12/WebGPU/Metal
// have no direct buffer fill and report it unsupported, so the test self-skips there).
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

namespace
{
    constexpr uint32_t kCount = 64;
    constexpr uint32_t kValue = 0xABCD1234u;

    void Check(VriGraphicsAPI api, const char* name)
    {
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        VriDevice* dev      = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << name << "] unavailable - skipped");
            return;
        }
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        if (c.GetDeviceDesc(dev)->hasClearStorageBuffer == VRI_FALSE)
        {
            MESSAGE("[" << name << "] storage-buffer clear unsupported - skipped");
            vriDestroyDevice(dev);
            return;
        }
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        const uint64_t bytes = static_cast<uint64_t>(kCount) * sizeof(uint32_t);
        VriBufferDesc  bd {};
        bd.size           = bytes;
        bd.usage          = VriBufferUsage_StorageBuffer | VriBufferUsage_TransferSrc | VriBufferUsage_TransferDst;
        bd.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* buffer = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &buffer) == VriResult_Success);
        VriBufferDesc rbd {};
        rbd.size            = bytes;
        rbd.usage           = VriBufferUsage_TransferDst;
        rbd.memoryLocation  = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        c.CmdClearStorageBuffer(cmd, buffer, 0, 0, kValue); // size 0 = whole buffer
        // The clear is a transfer-style write; make it visible to the readback copy.
        VriBufferBarrierDesc bb {};
        bb.buffer        = buffer;
        bb.before.access = VriAccess_CopyDestinationWrite;
        bb.before.stages = VriPipelineStage_Transfer;
        bb.after.access  = VriAccess_CopySourceRead;
        bb.after.stages  = VriPipelineStage_Transfer;
        VriBarrierGroupDesc g {};
        g.buffers   = &bb;
        g.bufferNum = 1;
        c.CmdBarrier(cmd, &g);
        VriBufferCopyDesc cp {};
        cp.size = bytes;
        c.CmdCopyBuffer(cmd, readback, buffer, &cp);
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

        const uint32_t* px     = static_cast<const uint32_t*>(c.MapBuffer(readback, 0, bytes));
        bool            allSet = px != nullptr;
        for (uint32_t i = 0; i < kCount && allSet; ++i)
            allSet = px[i] == kValue;
        c.UnmapBuffer(readback);
        CHECK(allSet); // every element holds the cleared value

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(buffer);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("Clear storage buffer: fill with a uint32 value")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
}
