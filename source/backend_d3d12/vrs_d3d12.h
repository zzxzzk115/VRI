// vrs_d3d12.h - Direct3D 12 Variable Rate Shading interface (RSSetShadingRate).
#pragma once

#include <vri/vri.h>

namespace vri::d3d12
{
    // Returns the VRS function table. Registered by the device only when
    // VriFeature_VariableShadingRate was granted at creation (VRS Tier 1+).
    const VriShadingRateInterface* GetShadingRateInterfaceD3D12();
} // namespace vri::d3d12
