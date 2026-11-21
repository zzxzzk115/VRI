// A submission that waits on a fence an earlier submission signals, shared by the desktop suite and the
// browser pages. Submission A copies a pattern into a device buffer and signals F = 1; submission B waits on
// F = 1, copies that buffer into a readback buffer and signals G = 1. Both submits must succeed, and the
// readback must hold the pattern once G reaches 1.
#pragma once

#include <vri/vri.h>

#include <cstdint>
#include <cstring>
#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

namespace vri::test
{
    struct OrderedWaitResult
    {
        VriResult submitA          = VriResult_Failure;
        VriResult submitB          = VriResult_Failure;
        uint64_t  fenceAfterSubmit = 0; // F, read right after B's QueueSubmit returned
        bool      dataOk           = false;
    };

    // Returns 0 when the setup worked (the outcome is in `result`), else the failing line.
    inline int RunOrderedWait(VriDevice* device, const VriCoreInterface& c, OrderedWaitResult& result)
    {
        constexpr uint64_t   kBytes   = 256;
        const VriDeviceDesc* desc     = c.GetDeviceDesc(device);
        VriQueue*            graphics = nullptr;
        VriQueue*            other    = nullptr;
        if (c.GetQueue(device, VriQueueType_Graphics, 0, &graphics) != VriResult_Success)
            return __LINE__;
        // A on the compute queue where there is one, as a renderer splitting work across queues would.
        if (desc->queueCount[VriQueueType_Compute] == 0 ||
            c.GetQueue(device, VriQueueType_Compute, 0, &other) != VriResult_Success)
            other = graphics;

        VriBufferDesc bd {};
        bd.size             = kBytes;
        bd.usage            = VriBufferUsage_TransferSrc;
        bd.memoryLocation   = VriMemoryLocation_HostUpload;
        VriBuffer* upload   = nullptr;
        VriBuffer* middle   = nullptr;
        VriBuffer* readback = nullptr;
        if (c.CreateBuffer(device, &bd, &upload) != VriResult_Success)
            return __LINE__;
        bd.usage          = VriBufferUsage_TransferSrc | VriBufferUsage_TransferDst;
        bd.memoryLocation = VriMemoryLocation_Device;
        if (c.CreateBuffer(device, &bd, &middle) != VriResult_Success)
            return __LINE__;
        bd.usage          = VriBufferUsage_TransferDst;
        bd.memoryLocation = VriMemoryLocation_HostReadback;
        if (c.CreateBuffer(device, &bd, &readback) != VriResult_Success)
            return __LINE__;
        uint8_t pattern[kBytes];
        for (uint32_t i = 0; i < kBytes; ++i)
            pattern[i] = static_cast<uint8_t>(i * 7 + 3);
        std::memcpy(c.MapBuffer(upload, 0, kBytes), pattern, kBytes);
        c.UnmapBuffer(upload);

        VriFence* f = nullptr;
        VriFence* g = nullptr;
        if (c.CreateFence(device, 0, &f) != VriResult_Success || c.CreateFence(device, 0, &g) != VriResult_Success)
            return __LINE__;
        const VriQueueType   otherType = other == graphics ? VriQueueType_Graphics : VriQueueType_Compute;
        VriCommandAllocator* allocA    = nullptr;
        VriCommandAllocator* allocB    = nullptr;
        VriCommandBuffer*    cmdA      = nullptr;
        VriCommandBuffer*    cmdB      = nullptr;
        if (c.CreateCommandAllocator(device, otherType, &allocA) != VriResult_Success ||
            c.CreateCommandAllocator(device, VriQueueType_Graphics, &allocB) != VriResult_Success ||
            c.CreateCommandBuffer(allocA, &cmdA) != VriResult_Success ||
            c.CreateCommandBuffer(allocB, &cmdB) != VriResult_Success)
            return __LINE__;
        VriBufferCopyDesc copy {};
        copy.size = kBytes;
        c.BeginCommandBuffer(cmdA);
        c.CmdCopyBuffer(cmdA, middle, upload, &copy);
        c.EndCommandBuffer(cmdA);
        c.BeginCommandBuffer(cmdB);
        VriBufferBarrierDesc bb {};
        bb.buffer        = middle;
        bb.before.access = VriAccess_CopyDestinationWrite;
        bb.before.stages = VriPipelineStage_Transfer;
        bb.after.access  = VriAccess_CopySourceRead;
        bb.after.stages  = VriPipelineStage_Transfer;
        VriBarrierGroupDesc group {};
        group.buffers   = &bb;
        group.bufferNum = 1;
        c.CmdBarrier(cmdB, &group);
        c.CmdCopyBuffer(cmdB, readback, middle, &copy);
        c.EndCommandBuffer(cmdB);

        VriFenceSubmitDesc signalF {f, 1, VriPipelineStage_AllCommands};
        VriQueueSubmitDesc a {};
        a.commandBuffers   = &cmdA;
        a.commandBufferNum = 1;
        a.signalFences     = &signalF;
        a.signalFenceNum   = 1;
        result.submitA     = c.QueueSubmit(other, &a);

        VriFenceSubmitDesc waitF {f, 1, VriPipelineStage_Transfer};
        VriFenceSubmitDesc signalG {g, 1, VriPipelineStage_AllCommands};
        VriQueueSubmitDesc b {};
        b.waitFences            = &waitF;
        b.waitFenceNum          = 1;
        b.commandBuffers        = &cmdB;
        b.commandBufferNum      = 1;
        b.signalFences          = &signalG;
        b.signalFenceNum        = 1;
        result.submitB          = c.QueueSubmit(graphics, &b);
        result.fenceAfterSubmit = c.GetFenceValue(f);

        if (result.submitA == VriResult_Success && result.submitB == VriResult_Success)
        {
#if defined(__EMSCRIPTEN__)
            // The fences advance from the browser event loop: yield until G arrives.
            for (int i = 0; i < 10000 && c.GetFenceValue(g) < 1; ++i)
                emscripten_sleep(1);
#else
            c.WaitForFence(g, 1, UINT64_MAX);
#endif
            if (c.GetFenceValue(g) >= 1 && c.GetFenceValue(f) >= 1)
            {
                const void* data = c.MapBuffer(readback, 0, kBytes);
                result.dataOk    = data && std::memcmp(data, pattern, kBytes) == 0;
                if (data)
                    c.UnmapBuffer(readback);
            }
        }

        c.DeviceWaitIdle(device);
        c.DestroyCommandAllocator(allocB);
        c.DestroyCommandAllocator(allocA);
        c.DestroyFence(g);
        c.DestroyFence(f);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(middle);
        c.DestroyBuffer(upload);
        return 0;
    }

    // A wait on a value no submission signals cannot be ordered on a single queue: QueueSubmit refuses it.
    inline VriResult SubmitUnsignaledWait(VriDevice* device, const VriCoreInterface& c)
    {
        VriQueue* queue = nullptr;
        VriFence* f     = nullptr;
        c.GetQueue(device, VriQueueType_Graphics, 0, &queue);
        if (c.CreateFence(device, 0, &f) != VriResult_Success)
            return VriResult_Failure;
        VriFenceSubmitDesc wait {f, 1, VriPipelineStage_AllCommands};
        VriQueueSubmitDesc submit {};
        submit.waitFences   = &wait;
        submit.waitFenceNum = 1;
        const VriResult r   = c.QueueSubmit(queue, &submit);
        c.DestroyFence(f);
        return r;
    }
} // namespace vri::test
