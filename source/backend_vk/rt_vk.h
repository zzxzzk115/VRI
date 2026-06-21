// rt_vk.h - Vulkan ray tracing interface (KHR acceleration structure + RT pipeline).
#pragma once

#include <vri/vri.h>

namespace vri::vk
{
    // Returns the ray-tracing function table. Registered by the device only when
    // VriFeature_RayTracing was granted at creation.
    const VriRayTracingInterface* GetRayTracingInterfaceVK();
} // namespace vri::vk
