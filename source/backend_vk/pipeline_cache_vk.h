// pipeline_cache_vk.h - Vulkan serializable pipeline cache (VkPipelineCache).
#pragma once

#include <vri/vri.h>

namespace vri::vk
{
    // Returns the pipeline-cache function table. Registered by every Vulkan device.
    const VriPipelineCacheInterface* GetPipelineCacheInterfaceVK();
} // namespace vri::vk
