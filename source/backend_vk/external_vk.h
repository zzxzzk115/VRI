// external_vk.h - Vulkan external memory/semaphore export interface
// (VK_KHR_external_memory_{win32,fd} + VK_KHR_external_semaphore_{win32,fd}).
#pragma once

#include <vri/vri.h>

namespace vri::vk
{
    // Returns the external-memory/semaphore export function table. Registered by the device
    // only when VriFeature_ExternalMemory was granted at creation.
    const VriExternalInterface* GetExternalInterfaceVK();
} // namespace vri::vk
