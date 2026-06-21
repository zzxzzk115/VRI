// vrs_vk.h - Vulkan Variable Rate Shading interface (VK_KHR_fragment_shading_rate).
#pragma once

#include <vri/vri.h>

namespace vri::vk
{
    // Returns the VRS function table. Registered by the device only when
    // VriFeature_VariableShadingRate was granted at creation.
    const VriShadingRateInterface* GetShadingRateInterfaceVK();
} // namespace vri::vk
