// query_d3d12.h - D3D12 GPU query pools (timestamp + occlusion).
#pragma once

#include <vri/vri.h>

namespace vri::d3d12
{
    // Returns the query function table. Registered by every D3D12 device.
    const VriQueryInterface* GetQueryInterfaceD3D12();
} // namespace vri::d3d12
