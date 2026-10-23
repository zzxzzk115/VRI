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
