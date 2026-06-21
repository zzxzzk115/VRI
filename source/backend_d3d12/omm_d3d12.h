// omm_d3d12.h - Direct3D 12 opacity micromap interface (DXR 1.2).
#pragma once

#include <vri/vri.h>

namespace vri::d3d12
{
    // Returns the opacity-micromap function table. Registered by the device only
    // when VriFeature_OpacityMicromap was granted (Raytracing Tier 1.2+).
    const VriOpacityMicromapInterface* GetOpacityMicromapInterfaceD3D12();
} // namespace vri::d3d12
