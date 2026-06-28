// query_wgpu.h - WebGPU GPU timestamp query pools.
#pragma once

#include <vri/vri.h>

namespace vri::wgpu
{
    // Returns the query function table. Registered by every WebGPU device; CreateQueryPool
    // returns Unsupported when the adapter lacks the timestamp-query feature.
    const VriQueryInterface* GetQueryInterfaceWGPU();
} // namespace vri::wgpu
