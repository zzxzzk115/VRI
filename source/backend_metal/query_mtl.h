// query_mtl.h - accessor for the native Metal VriQueryInterface (timestamps + occlusion).
#pragma once

#include <vri/vri.h>

namespace vri::mtl
{
    // Returns the query function table. Registered by every Metal device (Apple GPUs have a
    // timestamp counter set + occlusion visibility-result buffers; pipeline statistics are absent).
    const VriQueryInterface* GetQueryInterfaceMTL();
} // namespace vri::mtl
