/*
 * object_registry.h - what a device owns, and what each object cost.
 *
 * GetVideoMemoryInfo answers how much device memory is in use; it never answers what is using it,
 * and that is the question a renderer near its VRAM budget actually needs. Every backend tracks
 * its buffers, textures, acceleration structures and micromaps here so the answer has one shape
 * and one lifetime rule across all of them - a backend that skipped it would leave
 * EnumerateObjects silently partial, which is worse than unsupported.
 *
 * Header-only and backend-agnostic on purpose: the only thing a backend supplies is the size, and
 * where it gets that (VMA's allocation info, D3D12's resource allocation info, Metal's
 * allocatedSize) is the one part that cannot be shared.
 */
#ifndef VRI_CORE_OBJECT_REGISTRY_H
#define VRI_CORE_OBJECT_REGISTRY_H

#include <cstring>
#include <mutex>
#include <unordered_map>

#include "vri/vri_core.h"

namespace vri
{
    class ObjectRegistry
    {
    public:
        void Track(const void* handle, const VriObjectInfo& info)
        {
            if (handle == nullptr)
                return;
            const std::lock_guard<std::mutex> lock {m_mutex};
            VriObjectInfo                     entry = info;
            entry.handle                            = handle;
            // A re-registered handle keeps the name it was given: backends create the object
            // before the caller can name it, but an object recreated in place (a swapchain
            // texture on resize) should not silently lose its label.
            const auto existing = m_objects.find(handle);
            if (existing != m_objects.end() && existing->second.name[0] != '\0' && entry.name[0] == '\0')
                std::memcpy(entry.name, existing->second.name, sizeof entry.name);
            m_objects[handle] = entry;
        }

        void Untrack(const void* handle)
        {
            if (handle == nullptr)
                return;
            const std::lock_guard<std::mutex> lock {m_mutex};
            m_objects.erase(handle);
        }

        // Silently ignores an unknown handle. SetDebugName takes an opaque void* and is documented
        // to accept any VRI object, including the ones with no memory of their own that this does
        // not track; refusing those would make naming an error-prone special case for the caller.
        void SetName(const void* handle, const char* name)
        {
            if (handle == nullptr)
                return;
            const std::lock_guard<std::mutex> lock {m_mutex};
            const auto                        it = m_objects.find(handle);
            if (it == m_objects.end())
                return;
            if (name == nullptr)
            {
                it->second.name[0] = '\0';
                return;
            }
            std::strncpy(it->second.name, name, VRI_OBJECT_NAME_MAX - 1);
            it->second.name[VRI_OBJECT_NAME_MAX - 1] = '\0';
        }

        // Subtract from a tracked object's footprint, and optionally overwrite one of the split
        // fields. For a resource that can give memory back without being destroyed - an
        // acceleration structure releasing its build scratch - a listing that still showed the
        // freed bytes would be worse than not tracking at all.
        void Shrink(const void* handle, uint64_t bytes, uint32_t newHeight)
        {
            if (handle == nullptr)
                return;
            const std::lock_guard<std::mutex> lock {m_mutex};
            const auto                        it = m_objects.find(handle);
            if (it == m_objects.end())
                return;
            it->second.memoryBytes = it->second.memoryBytes > bytes ? it->second.memoryBytes - bytes : 0;
            it->second.height      = newHeight;
        }

        // out == nullptr reports the count. Otherwise writes at most *count entries and updates
        // *count to what was written, so a caller whose array went stale between the two calls
        // truncates rather than overruns.
        [[nodiscard]] VriResult Snapshot(uint32_t* count, VriObjectInfo* out) const
        {
            if (count == nullptr)
                return VriResult_InvalidArgument;
            const std::lock_guard<std::mutex> lock {m_mutex};
            if (out == nullptr)
            {
                *count = static_cast<uint32_t>(m_objects.size());
                return VriResult_Success;
            }
            uint32_t written = 0;
            for (const auto& [handle, info] : m_objects)
            {
                if (written >= *count)
                    break;
                out[written++] = info;
            }
            *count = written;
            return VriResult_Success;
        }

    private:
        mutable std::mutex                             m_mutex;
        std::unordered_map<const void*, VriObjectInfo> m_objects;
    };

    // Fills the fields every backend sets the same way. The caller adds the size and the
    // format/extent fields it has.
    [[nodiscard]] inline VriObjectInfo MakeObjectInfo(const VriObjectType type)
    {
        VriObjectInfo info {};
        info.type   = type;
        info.format = VriFormat_Unknown;
        return info;
    }
} // namespace vri

#endif /* VRI_CORE_OBJECT_REGISTRY_H */
