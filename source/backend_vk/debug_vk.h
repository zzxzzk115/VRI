#pragma once

#include <cstdint>
#include <mutex>
#include <string>
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
            std::string  name;
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
            Object renamed;
            {
                const std::lock_guard lock {Mutex()};
                auto&                 entry   = Objects()[object];
                const auto            native  = NativeHandle(handle);
                const bool            changed = entry.handle != native || entry.type != type || entry.device != device;
                entry.device                  = device;
                entry.type                    = type;
                entry.handle                  = native;
                if (changed && !entry.name.empty())
                    renamed = entry;
            }
            // Recreating a native object must retain its logical VRI object's label.
            // Native dispatch happens outside the registry lock (tool layers may reenter).
            if (renamed.device)
                ApplyName(renamed);
        }

        static Object SetName(const void* object, const char* name)
        {
            Object named;
            {
                const std::lock_guard lock {Mutex()};
                const auto            it = Objects().find(object);
                if (it == Objects().end())
                    return {};
                it->second.name = name ? name : "";
                named           = it->second;
            }
            ApplyName(named);
            return named;
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
        static void ApplyName(const Object& object);

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
