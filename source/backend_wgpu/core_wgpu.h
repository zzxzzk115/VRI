// core_wgpu.h - access to the WebGPU core interface function table.
#pragma once

#include <vri/vri_core.h>

namespace vri::wgpu
{
    const VriCoreInterface* GetCoreInterfaceWGPU();
} // namespace vri::wgpu
