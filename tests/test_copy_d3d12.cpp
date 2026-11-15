// D3D12 buffer copies into a buffer that is no longer in COMMON: a constant buffer refreshed by a
// copy every frame (example-asynccompute) is in VERTEX_AND_CONSTANT_BUFFER when the next copy
// arrives. The copy must transition it; the debug layer reports a copy into a read state.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace
{
    void VRI_CALL CountErrors(void* user, VriMessageSeverity severity, const char* message)
    {
        if (severity == VriMessageSeverity_Error)
        {
            ++*static_cast<int*>(user);
            MESSAGE(std::string(message));
        }
    }
} // namespace

TEST_CASE("D3D12: a buffer copy transitions a destination that is in a read state")
{
    int                   errors = 0;
    VriCallbackInterface  callback {&errors, CountErrors};
    VriDeviceCreationDesc dc  = vriDefaultDeviceCreationDesc();
    dc.coordinates.clipSpaceY = VriClipSpaceY_Up;
    dc.graphicsAPI            = VriGraphicsAPI_D3D12;
    dc.enableValidation       = VRI_TRUE;
    dc.bestEffort             = VRI_TRUE;
    dc.callbackInterface      = &callback;
    VriDevice* dev            = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    auto makeBuffer = [&](VriBufferUsageFlags usage, VriMemoryLocation location) {
        VriBufferDesc bd {};
        bd.size           = 256;
        bd.usage          = usage;
        bd.memoryLocation = location;
        VriBuffer* b      = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &b) == VriResult_Success);
        return b;
    };
    VriBuffer* first  = makeBuffer(VriBufferUsage_TransferSrc, VriMemoryLocation_HostUpload);
    VriBuffer* second = makeBuffer(VriBufferUsage_TransferSrc, VriMemoryLocation_HostUpload);
    VriBuffer* target =
        makeBuffer(VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst | VriBufferUsage_TransferSrc,
                   VriMemoryLocation_Device);
    VriBuffer*     readback = makeBuffer(VriBufferUsage_TransferDst, VriMemoryLocation_HostReadback);
    const uint32_t one = 1, two = 2;
    std::memcpy(c.MapBuffer(first, 0, 4), &one, 4);
    c.UnmapBuffer(first);
    std::memcpy(c.MapBuffer(second, 0, 4), &two, 4);
    c.UnmapBuffer(second);

    VriCommandAllocator* alloc = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    auto barrier = [&](VriAccessFlags before, VriAccessFlags after) {
        VriBufferBarrierDesc bb {};
        bb.buffer        = target;
        bb.before.access = before;
        bb.after.access  = after;
        VriBarrierGroupDesc g {};
        g.buffers   = &bb;
        g.bufferNum = 1;
        c.CmdBarrier(cmd, &g);
    };
    VriBufferCopyDesc copy {};
    copy.size = 4;
    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
    c.CmdCopyBuffer(cmd, target, first, &copy);
    barrier(VriAccess_CopyDestinationWrite, VriAccess_ConstantBufferRead);
    c.CmdCopyBuffer(cmd, target, second, &copy); // target is in VERTEX_AND_CONSTANT_BUFFER here
    barrier(VriAccess_CopyDestinationWrite, VriAccess_CopySourceRead);
    c.CmdCopyBuffer(cmd, readback, target, &copy);
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
    c.Wait(fence, 1);

    uint32_t value = 0;
    std::memcpy(&value, c.MapBuffer(readback, 0, 4), 4);
    c.UnmapBuffer(readback);
    CHECK(value == 2);
    CHECK(errors == 0);

    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    c.DestroyBuffer(first);
    c.DestroyBuffer(second);
    c.DestroyBuffer(target);
    c.DestroyBuffer(readback);
    vriDestroyDevice(dev);
}
