// rt_d3d12.h - Direct3D 12 ray tracing interface (DXR: state objects + acceleration structures).
#pragma once

#include <vri/vri.h>

namespace vri::d3d12
{
    // Returns the ray-tracing function table. Registered by the device only when
    // VriFeature_RayTracing was granted (Raytracing Tier 1.0+).
    const VriRayTracingInterface* GetRayTracingInterfaceD3D12();
} // namespace vri::d3d12
