// interop_vk.h - native-handle interop (the OpenXR / external-API seam).
#pragma once

#include <vri/ext/vri_ext_interop.h>

namespace vri::vk
{
    const VriInteropInterface* GetInteropInterfaceVK();
} // namespace vri::vk
