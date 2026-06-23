// factory_mtl.h - the Metal device factory, declared without importing any
// Objective-C / Metal headers so the plain-C++ dispatch TU (vri_entry.cpp) can
// include it. device_mtl.{h,mm} (Objective-C++) define it.
#pragma once

#include <vri/vri.h>

#include "core/device_base.h"

namespace vri::mtl
{
    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult);
} // namespace vri::mtl
