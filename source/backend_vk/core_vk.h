// CoreVK.h - access to the Vulkan core interface function table.
#pragma once

#include <vri/vri_core.h>

namespace vri::vk
{
    // Returns the singleton, fully-populated core interface table for Vulkan.
    const VriCoreInterface* GetCoreInterfaceVK();
} // namespace vri::vk
