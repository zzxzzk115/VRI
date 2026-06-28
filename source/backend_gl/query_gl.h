// query_gl.h - OpenGL GPU timestamp query pools (desktop GL only).
#pragma once

#include <vri/vri.h>

namespace vri::gl
{
    // Returns the query function table. Registered by every GL device; CreateQueryPool returns
    // Unsupported on GLES/WebGL (no core timer query) and reports via hasTimestampQueries.
    const VriQueryInterface* GetQueryInterfaceGL();
} // namespace vri::gl
