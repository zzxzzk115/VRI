// MapBufferAsync round trip shared by the desktop suite and the browser pages: copy a known pattern
// into a readback buffer on the GPU, then read it through MapBufferAsync, whole and from an offset.
#pragma once

#include <vri/vri.h>

#include <cstdint>
#include <cstring>
#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

namespace vri::test
{
    struct MapOutcome
    {
        int       calls  = 0;
        VriResult result = VriResult_Failure;
        void*     data   = nullptr;
    };

    inline void VRI_CALL RecordMap(void* userData, VriResult result, void* data)
    {
        auto* outcome = static_cast<MapOutcome*>(userData);
        ++outcome->calls;
        outcome->result = result;
        outcome->data   = data;
    }

    // Map through MapBufferAsync. `deferred`: the callback must not have run when the call returns
    // (browser WebGPU), and this waits for it; otherwise it must already have run, exactly once.
    inline int MapAsync(const VriCoreInterface& c,
                        VriBuffer*              buffer,
                        uint64_t                offset,
                        uint64_t                size,
                        bool                    deferred,
                        MapOutcome&             outcome)
    {
        if (c.MapBufferAsync(buffer, offset, size, RecordMap, &outcome) != VriResult_Success)
            return __LINE__;
        if (deferred)
        {
            if (outcome.calls != 0)
                return __LINE__;
#if defined(__EMSCRIPTEN__)
            for (int i = 0; i < 10000 && outcome.calls == 0; ++i)
                emscripten_sleep(1);
#endif
        }
        if (outcome.calls != 1 || outcome.result != VriResult_Success || outcome.data == nullptr)
            return __LINE__;
        return 0;
    }

    // Returns 0 on success, else the failing line.
    inline int CheckMapBufferAsync(VriDevice* device, const VriCoreInterface& c, bool deferred)
    {
        constexpr uint32_t kCount = 64;
        constexpr uint64_t kBytes = kCount * sizeof(uint32_t);
        constexpr uint32_t kSkip  = 16; // elements before the offset map's first one

        if (c.MapBufferAsync(nullptr, 0, 0, RecordMap, nullptr) != VriResult_InvalidArgument)
            return __LINE__;

        VriQueue* queue = nullptr;
        if (c.GetQueue(device, VriQueueType_Graphics, 0, &queue) != VriResult_Success)
            return __LINE__;
        VriBufferDesc ud {};
        ud.size           = kBytes;
        ud.usage          = VriBufferUsage_TransferSrc;
        ud.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* upload = nullptr;
        if (c.CreateBuffer(device, &ud, &upload) != VriResult_Success)
            return __LINE__;
        VriBufferDesc rd {};
        rd.size             = kBytes;
        rd.usage            = VriBufferUsage_TransferDst;
        rd.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        if (c.CreateBuffer(device, &rd, &readback) != VriResult_Success)
            return __LINE__;
        if (c.MapBufferAsync(readback, 0, kBytes, nullptr, nullptr) != VriResult_InvalidArgument)
            return __LINE__;

        uint32_t pattern[kCount];
        for (uint32_t i = 0; i < kCount; ++i)
            pattern[i] = 0xC0DE0000u + i;
        // A host-upload buffer maps at once on every backend, the browser included.
        MapOutcome uploadMap;
        if (const int failure = MapAsync(c, upload, 0, kBytes, false, uploadMap))
            return failure;
        std::memcpy(uploadMap.data, pattern, kBytes);
        c.UnmapBuffer(upload);

        VriCommandAllocator* alloc = nullptr;
        VriCommandBuffer*    cmd   = nullptr;
        VriFence*            fence = nullptr;
        if (c.CreateCommandAllocator(device, VriQueueType_Graphics, &alloc) != VriResult_Success ||
            c.CreateCommandBuffer(alloc, &cmd) != VriResult_Success ||
            c.CreateFence(device, 0, &fence) != VriResult_Success)
            return __LINE__;
        c.BeginCommandBuffer(cmd);
        VriBufferCopyDesc copy {};
        copy.size = kBytes;
        c.CmdCopyBuffer(cmd, readback, upload, &copy);
        c.EndCommandBuffer(cmd);
        VriFenceSubmitDesc signal {};
        signal.fence = fence;
        signal.value = 1;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        if (c.QueueSubmit(queue, &submit) != VriResult_Success)
            return __LINE__;
        if (!deferred) // the browser's map itself waits for the copy
            c.Wait(fence, 1);

        int        failure = 0;
        MapOutcome whole;
        failure = MapAsync(c, readback, 0, kBytes, deferred, whole);
        if (!failure && std::memcmp(whole.data, pattern, kBytes) != 0)
            failure = __LINE__;
        if (whole.data)
            c.UnmapBuffer(readback);

        MapOutcome tail;
        if (!failure)
            failure =
                MapAsync(c, readback, kSkip * sizeof(uint32_t), kBytes - kSkip * sizeof(uint32_t), deferred, tail);
        if (!failure && std::memcmp(tail.data, pattern + kSkip, kBytes - kSkip * sizeof(uint32_t)) != 0)
            failure = __LINE__;
        if (tail.data)
            c.UnmapBuffer(readback);

        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(upload);
        return failure;
    }
} // namespace vri::test
