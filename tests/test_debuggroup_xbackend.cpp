// Debug-group marker smoke test. CmdBeginDebugGroup / CmdEndDebugGroup annotate a
// command stream for GPU debuggers (PIX, RenderDoc, Xcode). Their visual effect can
// only be confirmed inside a capture tool, but the wiring has a hard, CI-checkable
// contract: recording markers - including nested ones - around real work must never
// corrupt the stream. A mis-encoded marker (e.g. a wrong BeginEvent blob size on
// D3D12) or an unbalanced begin/end can crash submission or scramble the commands
// around it. So we wrap - and nest - a GPU buffer copy in debug groups, submit it on
// every available backend, and assert the copy still lands byte-for-byte with no error
// diagnostic. This pins the D3D12 PIX-marker path (BeginEvent/EndEvent) and guards the
// Metal implementation, while the other backends' no-op markers must stay harmless.
//
// Unavailable backends self-skip, so the file is stable under the CI backend matrix.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <string>

namespace
{
    struct MsgState
    {
        int errors = 0;
    };
    void VRI_CALL OnMessage(void* userArg, VriMessageSeverity severity, const char* /*message*/)
    {
        if (severity == VriMessageSeverity_Error)
            static_cast<MsgState*>(userArg)->errors++;
    }

    const char* ApiName(VriGraphicsAPI api)
    {
        switch (api)
        {
            case VriGraphicsAPI_Vulkan:
                return "Vulkan";
            case VriGraphicsAPI_WebGPU:
                return "WebGPU";
            case VriGraphicsAPI_OpenGL:
                return "OpenGL";
            case VriGraphicsAPI_D3D12:
                return "D3D12";
            case VriGraphicsAPI_Metal:
                return "Metal";
            default:
                return "?";
        }
    }

    struct Probe
    {
        bool ran           = false; // backend available and device created
        bool resultCorrect = false; // copy landed byte-for-byte
        bool sawError      = false; // any Error diagnostic during the marked stream
    };

    // Copy a known pattern staging -> device -> readback, with every command wrapped in
    // an outer debug group and the readback copy in a nested inner group.
    Probe RunDebugGroupCopy(VriGraphicsAPI api)
    {
        Probe                p;
        MsgState             msg;
        VriCallbackInterface cb {};
        cb.userArg         = &msg;
        cb.MessageCallback = OnMessage;

        VriDeviceCreationDesc dc {};
        dc.graphicsAPI       = api;
        dc.enableValidation  = VRI_TRUE;
        dc.bestEffort        = VRI_TRUE;
        dc.callbackInterface = &cb;
        VriDevice* dev       = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return p; // ran == false -> caller skips
        p.ran = true;

        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        constexpr uint32_t kCount = 64;
        const uint64_t     bytes  = kCount * sizeof(uint32_t);

        VriBufferDesc sb {};
        sb.size            = bytes;
        sb.usage           = VriBufferUsage_TransferSrc;
        sb.memoryLocation  = VriMemoryLocation_HostUpload;
        VriBuffer* staging = nullptr;
        REQUIRE(c.CreateBuffer(dev, &sb, &staging) == VriResult_Success);
        {
            uint32_t* m = static_cast<uint32_t*>(c.MapBuffer(staging, 0, bytes));
            REQUIRE(m != nullptr);
            for (uint32_t i = 0; i < kCount; ++i)
                m[i] = i + 100u;
            c.UnmapBuffer(staging);
        }

        VriBufferDesc db {};
        db.size           = bytes;
        db.usage          = VriBufferUsage_TransferSrc | VriBufferUsage_TransferDst;
        db.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* device = nullptr;
        REQUIRE(c.CreateBuffer(dev, &db, &device) == VriResult_Success);

        VriBufferDesc rb {};
        rb.size             = bytes;
        rb.usage            = VriBufferUsage_TransferDst;
        rb.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rb, &readback) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        const int errBefore = msg.errors;
        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);

        c.CmdBeginDebugGroup(cmd, "vri.test.debuggroup");
        VriBufferCopyDesc up {};
        up.size = bytes;
        c.CmdCopyBuffer(cmd, device, staging, &up);
        {
            VriBufferBarrierDesc bb {};
            bb.buffer        = device;
            bb.before.access = VriAccess_CopyDestinationWrite;
            bb.before.stages = VriPipelineStage_Transfer;
            bb.after.access  = VriAccess_CopySourceRead;
            bb.after.stages  = VriPipelineStage_Transfer;
            VriBarrierGroupDesc g {};
            g.buffers   = &bb;
            g.bufferNum = 1;
            c.CmdBarrier(cmd, &g);
        }
        c.CmdBeginDebugGroup(cmd, "vri.test.debuggroup.nested-readback");
        VriBufferCopyDesc down {};
        down.size = bytes;
        c.CmdCopyBuffer(cmd, readback, device, &down);
        c.CmdEndDebugGroup(cmd); // nested
        c.CmdEndDebugGroup(cmd); // outer

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
        p.sawError = msg.errors > errBefore;

        const uint32_t* px = static_cast<const uint32_t*>(c.MapBuffer(readback, 0, bytes));
        REQUIRE(px != nullptr);
        bool allOk = true;
        for (uint32_t i = 0; i < kCount; ++i)
            if (px[i] != i + 100u)
            {
                allOk = false;
                break;
            }
        p.resultCorrect = allOk;
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(device);
        c.DestroyBuffer(staging);
        vriDestroyDevice(dev);
        return p;
    }

    void Check(VriGraphicsAPI api)
    {
        const std::string tag   = std::string(ApiName(api)) + " debug-group copy";
        const Probe       probe = RunDebugGroupCopy(api);
        if (!probe.ran)
        {
            const std::string m = tag + " unavailable - skipped";
            MESSAGE(m);
            return;
        }
        INFO(tag << ": resultCorrect=" << probe.resultCorrect << " sawError=" << probe.sawError);
        CHECK(probe.resultCorrect); // markers must not corrupt the wrapped work
        CHECK(probe.sawError == false);
    }
} // namespace

TEST_CASE("debug-group: nested Begin/End markers wrap GPU work without corrupting the stream")
{
    Check(VriGraphicsAPI_Vulkan);
    Check(VriGraphicsAPI_WebGPU);
    Check(VriGraphicsAPI_OpenGL);
    Check(VriGraphicsAPI_D3D12);
    Check(VriGraphicsAPI_Metal);
}
