// A fence destroyed while a submission that signals it is still in flight, shared by the desktop suite and the
// WebGPU page. The submission completes later; draining it must not touch the destroyed fence. A second
// submission's fence is then polled to completion, which drains the first submission's record.
#pragma once

#include <vri/vri.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

namespace vri::test
{
    // Returns 0 when the second fence reached its value, else the failing line.
    inline int CheckFenceDestroyedInFlight(VriDevice* device, const VriCoreInterface& c)
    {
        VriQueue* queue = nullptr;
        if (c.GetQueue(device, VriQueueType_Graphics, 0, &queue) != VriResult_Success)
            return __LINE__;
        VriFence* first  = nullptr;
        VriFence* second = nullptr;
        if (c.CreateFence(device, 0, &first) != VriResult_Success)
            return __LINE__;
        VriFenceSubmitDesc signal {first, 1, VriPipelineStage_AllCommands};
        VriQueueSubmitDesc submit {};
        submit.signalFences   = &signal;
        submit.signalFenceNum = 1;
        if (c.QueueSubmit(queue, &submit) != VriResult_Success)
            return __LINE__;
        c.DestroyFence(first); // the submission is still in flight

        if (c.CreateFence(device, 0, &second) != VriResult_Success)
            return __LINE__;
        signal.fence = second;
        if (c.QueueSubmit(queue, &submit) != VriResult_Success)
            return __LINE__;
        int i = 0;
        for (; i < 10000 && c.GetFenceValue(second) < 1; ++i)
        {
#if defined(__EMSCRIPTEN__)
            emscripten_sleep(1);
#else
            c.WaitForFence(second, 1, 1000000);
#endif
        }
        const int failure = c.GetFenceValue(second) >= 1 ? 0 : __LINE__;
        c.DestroyFence(second);
        return failure;
    }
} // namespace vri::test
