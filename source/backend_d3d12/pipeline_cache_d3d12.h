// pipeline_cache_d3d12.h - Direct3D 12 serializable pipeline cache (ID3D12PipelineLibrary).
#pragma once

#include <vri/vri.h>

namespace vri::d3d12
{
    // Returns the pipeline-cache function table. Registered by every D3D12 device (CreatePipelineCache
    // returns Unsupported on adapters without pipeline-library support).
    const VriPipelineCacheInterface* GetPipelineCacheInterfaceD3D12();
} // namespace vri::d3d12
