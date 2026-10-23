#pragma once

#include "backend_wgpu/objects_wgpu.h"

#include <vri/vri.h>

#include <chrono>
#include <mutex>
#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#else
#include <thread>
#endif

namespace vri::test
{
    // The caller supplies a raw WebGPU device. Inspecting retained records avoids
    // masking the regression with a public fence query or idle wait.
    inline int CheckWebGPUCompletionRetirement(VriDevice* device, const VriCoreInterface& core)
    {
        VriQueue* queue = nullptr;
        VriFence* fence = nullptr;
        if (core.GetQueue(device, VriQueueType_Graphics, 0, &queue) != VriResult_Success ||
            core.CreateFence(device, 0, &fence) != VriResult_Success)
        {
            return __LINE__;
        }
        const int failure = [&]() -> int {
            auto*              nativeQueue = reinterpret_cast<wgpu::QueueWGPU*>(queue);
            auto*              nativeFence = reinterpret_cast<wgpu::FenceWGPU*>(fence);
            VriFenceSubmitDesc signal {fence, 0, VriPipelineStage_AllCommands};
            VriQueueSubmitDesc submit {};
            submit.signalFences   = &signal;
            submit.signalFenceNum = 1;
            for (uint32_t batch = 0; batch < 4; ++batch)
            {
                for (uint32_t i = 0; i < 64; ++i)
                {
                    ++signal.value;
                    if (core.QueueSubmit(queue, &submit) != VriResult_Success)
                    {
                        return __LINE__;
                    }
                }
                const VriQueueSubmitDesc empty {};
                const auto               deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (true)
                {
                    if (core.QueueSubmit(queue, &empty) != VriResult_Success)
                    {
                        return __LINE__;
                    }
                    {
                        std::lock_guard lock(nativeQueue->completionMutex);
                        if (nativeQueue->completions.empty())
                        {
                            if (nativeFence->value.load(std::memory_order_acquire) != signal.value)
                            {
                                return __LINE__;
                            }
                            break;
                        }
                    }
                    if (std::chrono::steady_clock::now() >= deadline)
                    {
                        return __LINE__;
                    }
#if defined(__EMSCRIPTEN__)
                    emscripten_sleep(1);
#else
                    std::this_thread::yield();
#endif
                }
            }
            return 0;
        }();
        // Cleanup runs after success/failure was recorded, so it cannot hide a leak.
        const VriResult idle = core.QueueWaitIdle(queue);
        core.DestroyFence(fence);
        if (failure)
        {
            return failure;
        }
        return idle == VriResult_Success ? 0 : __LINE__;
    }
} // namespace vri::test
