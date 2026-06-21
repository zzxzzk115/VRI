// omm_vk.h - Vulkan opacity micromap interface (VK_EXT_opacity_micromap).
#pragma once

#include <vri/vri.h>

namespace vri::vk
{
    // Returns the opacity-micromap function table. Registered by the device only
    // when VriFeature_OpacityMicromap was granted at creation.
    const VriOpacityMicromapInterface* GetOpacityMicromapInterfaceVK();
} // namespace vri::vk
