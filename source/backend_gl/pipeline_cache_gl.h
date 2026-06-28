// pipeline_cache_gl.h - OpenGL pipeline cache, emulated via program binaries.
#pragma once

#include <vri/vri.h>

namespace vri::gl
{
    // Returns the pipeline-cache function table. Registered only on desktop GL contexts that
    // support program binaries (ARB_get_program_binary); GLES/WebGL2 report Unsupported.
    const VriPipelineCacheInterface* GetPipelineCacheInterfaceGL();
} // namespace vri::gl
