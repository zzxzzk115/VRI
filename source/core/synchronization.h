#pragma once

#include <vri/vri.h>

#include <chrono>
#if !defined(__EMSCRIPTEN__)
#include <algorithm>
#include <thread>
#endif

namespace vri::core
{
    inline bool ValidSubmitPointers(const VriQueueSubmitDesc* desc)
    {
        if (!desc || (desc->commandBufferNum && !desc->commandBuffers) || (desc->waitFenceNum && !desc->waitFences) ||
            (desc->signalFenceNum && !desc->signalFences))
        {
            return false;
        }
        for (uint32_t i = 0; i < desc->commandBufferNum; ++i)
        {
            if (!desc->commandBuffers[i])
            {
                return false;
            }
        }
        for (uint32_t i = 0; i < desc->waitFenceNum; ++i)
        {
            if (!desc->waitFences[i].fence)
            {
                return false;
            }
        }
        for (uint32_t i = 0; i < desc->signalFenceNum; ++i)
        {
            if (!desc->signalFences[i].fence)
            {
                return false;
            }
            for (uint32_t j = 0; j < i; ++j)
            {
                if (desc->signalFences[i].fence == desc->signalFences[j].fence)
                    return false;
            }
        }
        return true;
    }

    // MapBufferAsync for a backend whose mapping completes on the calling thread: map now and report
    // through the callback before returning.
    inline VriResult MapBufferNow(void*(VRI_CALL* map)(VriBuffer*, uint64_t, uint64_t),
                                  VriBuffer*              buffer,
                                  uint64_t                offset,
                                  uint64_t                size,
                                  VriBufferMappedCallback callback,
                                  void*                   userData)
    {
        if (!buffer || !callback)
            return VriResult_InvalidArgument;
        void* data = map(buffer, offset, size);
        callback(userData, data ? VriResult_Success : VriResult_Failure, data);
        return VriResult_Success;
    }

    class WaitDeadline
    {
    public:
        explicit WaitDeadline(uint64_t timeoutNs) : m_timeoutNs(timeoutNs) {}

        uint64_t Remaining() const
        {
            if (m_timeoutNs == UINT64_MAX)
            {
                return UINT64_MAX;
            }
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - m_start)
                    .count();
            const auto elapsedNs = static_cast<uint64_t>(elapsed);
            return elapsedNs >= m_timeoutNs ? 0 : m_timeoutNs - elapsedNs;
        }

#if !defined(__EMSCRIPTEN__)
        // Sleep between native polls, capped by the remaining deadline.
        void Pause() const
        {
            const uint64_t remaining = Remaining();
            if (remaining)
            {
                std::this_thread::sleep_for(std::chrono::nanoseconds(std::min<uint64_t>(remaining, 1000000)));
            }
        }
#endif

    private:
        std::chrono::steady_clock::time_point m_start = std::chrono::steady_clock::now();
        uint64_t                              m_timeoutNs;
    };
} // namespace vri::core
