// Several Vulkan devices in one process: each runs on its own instance, with or without the
// validation layer, and every call must go through that device's own entry points. Creates a
// validated and an unvalidated device in both orders, retires the first, and has the second fill
// and read back a buffer.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

namespace
{
    constexpr uint32_t kCount = 64;

    VriDevice* Create(bool validation)
    {
        VriDeviceCreationDesc dc  = vriDefaultDeviceCreationDesc();
        dc.coordinates.clipSpaceY = VriClipSpaceY_Up;
        dc.graphicsAPI            = VriGraphicsAPI_Vulkan;
        dc.enableValidation       = validation ? VRI_TRUE : VRI_FALSE;
        dc.bestEffort             = VRI_TRUE;
        VriDevice* dev            = nullptr;
        return vriCreateDevice(&dc, &dev) == VriResult_Success ? dev : nullptr;
    }

    // Clears a device buffer to `value` on the GPU, copies it to host memory and checks it.
    bool FillAndReadBack(VriDevice* dev, uint32_t value)
    {
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
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
        c.CmdClearStorageBuffer(cmd, buffer, 0, 0, value);
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
        REQUIRE(c.QueueSubmit(queue, &submit) == VriResult_Success);
        REQUIRE(c.WaitForFence(fence, 1, UINT64_MAX) == VriResult_Success);

        const uint32_t* data = static_cast<const uint32_t*>(c.MapBuffer(readback, 0, bytes));
        bool            ok   = data != nullptr;
        for (uint32_t i = 0; i < kCount && ok; ++i)
            ok = data[i] == value;
        c.UnmapBuffer(readback);

        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(buffer);
        return ok;
    }

    void Check(bool firstValidated)
    {
        VriDevice* first = Create(firstValidated);
        if (!first)
        {
            MESSAGE("Vulkan unavailable - skipped");
            return;
        }
        VriDevice* second = Create(!firstValidated);
        REQUIRE(second != nullptr);
        CHECK(FillAndReadBack(first, 0x11111111u));
        CHECK(FillAndReadBack(second, 0x22222222u));
        // The first device's instance is gone; the second keeps working on its own entry points.
        vriDestroyDevice(first);
        CHECK(FillAndReadBack(second, 0x33333333u));
        vriDestroyDevice(second);
    }
} // namespace

TEST_CASE("Vulkan: a validated and an unvalidated device coexist, created in either order")
{
    SUBCASE("validated first") { Check(true); }
    SUBCASE("unvalidated first") { Check(false); }
}
