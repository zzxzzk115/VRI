// query_vk.h - Vulkan GPU query pools (timestamp + occlusion).
#pragma once

#include <vri/vri.h>

namespace vri::vk
{
    // Returns the query function table. Registered by every Vulkan device.
    const VriQueryInterface* GetQueryInterfaceVK();
} // namespace vri::vk
