#pragma once

#include <cstdint>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include <vulkan/vulkan.h>

namespace vri::vk
{
    class DeviceVK;

    // Naming accepts untyped VRI handles, including descriptors whose first member is
    // not a device pointer. Register the native identity when exposing each handle.
    // This table is independent of the memory-accounting registry and validation.
    class DebugObjectsVK
    {
    public:
        struct Object
        {
            DeviceVK*    device = nullptr;
            VkObjectType type   = VK_OBJECT_TYPE_UNKNOWN;
            uint64_t     handle = 0;
        };

        template<typename H>
        static uint64_t NativeHandle(H handle)
        {
            if constexpr (std::is_pointer_v<H>)
                return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
            else
                return static_cast<uint64_t>(handle);
        }

        template<typename H>
        static void Track(const void* object, DeviceVK* device, VkObjectType type, H handle)
        {
            const std::lock_guard lock {Mutex()};
            Objects()[object] = {device, type, NativeHandle(handle)};
        }

        static void Untrack(const void* object)
        {
            const std::lock_guard lock {Mutex()};
            Objects().erase(object);
        }

        static Object Find(const void* object)
        {
            const std::lock_guard lock {Mutex()};
            const auto            it = Objects().find(object);
            return it == Objects().end() ? Object {} : it->second;
        }

    private:
        static std::mutex& Mutex()
        {
            static std::mutex mutex;
            return mutex;
        }
        static std::unordered_map<const void*, Object>& Objects()
        {
            static std::unordered_map<const void*, Object> objects;
            return objects;
        }
    };
} // namespace vri::vk
