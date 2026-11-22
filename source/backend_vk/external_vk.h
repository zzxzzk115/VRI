// external_vk.h - Vulkan external memory/semaphore export interface
// (VK_KHR_external_memory_{win32,fd} + VK_KHR_external_semaphore_{win32,fd}).
#pragma once

#include "vulkan_vk.h"

#include <vri/vri.h>

namespace vri::vk
{
    // Returns the external-memory/semaphore export function table. Registered by the device
    // only when VriFeature_ExternalMemory was granted at creation.
    const VriExternalInterface* GetExternalInterfaceVK();

    // Whether the device exports, with the platform's handle type, the two things every use of the interface needs:
    // a timeline semaphore (CreateExportableFence's fixed combination) and a buffer of the minimal usage (transfer
    // source and destination; other usage bits can depend on extensions the device did not enable, which makes the
    // query itself invalid). A driver can list the export extensions and export nothing (the Android emulator's).
    // Any other buffer usage, and every texture, is answered when it is created.
    bool ExportsMemoryAndFences(const VolkInstanceTable& vki, VkPhysicalDevice physicalDevice);
} // namespace vri::vk
