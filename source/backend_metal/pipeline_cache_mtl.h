// pipeline_cache_mtl.h - accessor for the native Metal VriPipelineCacheInterface.
#pragma once

#include <vri/vri.h>

namespace vri::mtl
{
    // Returns the pipeline-cache function table (backed by MTLBinaryArchive). Registered by every
    // Metal device; pipelines opt in via VriGraphicsPipelineDesc/VriComputePipelineDesc::pipelineCache.
    const VriPipelineCacheInterface* GetPipelineCacheInterfaceMTL();
} // namespace vri::mtl
