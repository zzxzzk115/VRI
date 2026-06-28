// pipeline_cache_vk.cpp - VriPipelineCacheInterface for Vulkan, backed by VkPipelineCache.
// The cache handle is passed to vkCreate{Graphics,Compute}Pipelines (see core_vk.cpp / PipeCache),
// serialized via vkGetPipelineCacheData, and seeded back via VkPipelineCacheCreateInfo. Vulkan
// validates the blob's header against this device and silently ignores a foreign/stale seed.
// Registered by every Vulkan device (the API is core Vulkan 1.0 - no extension needed).

#include "pipeline_cache_vk.h"
#include "device_vk.h"
#include "objects_vk.h"

namespace vri::vk
{
    namespace
    {
        inline DeviceVK*        Dev(VriDevice* h) { return reinterpret_cast<DeviceVK*>(h); }
        inline PipelineCacheVK* PC(VriPipelineCache* h) { return reinterpret_cast<PipelineCacheVK*>(h); }

        VriResult VRI_CALL CreatePipelineCache(VriDevice*         device,
                                               const void*        initialData,
                                               size_t             initialSize,
                                               VriPipelineCache** out)
        {
            if (!out)
                return VriResult_InvalidArgument;
            DeviceVK*                 d  = Dev(device);
            VkPipelineCacheCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
            ci.initialDataSize           = initialSize;
            ci.pInitialData              = initialSize ? initialData : nullptr;
            VkPipelineCache cache        = VK_NULL_HANDLE;
            if (vkCreatePipelineCache(d->Device(), &ci, nullptr, &cache) != VK_SUCCESS)
                return VriResult_Failure;
            *out = ToHandle(new PipelineCacheVK {d, cache});
            return VriResult_Success;
        }

        void VRI_CALL DestroyPipelineCache(VriPipelineCache* h)
        {
            if (!h)
                return;
            PipelineCacheVK* p = PC(h);
            vkDestroyPipelineCache(p->device->Device(), p->cache, nullptr);
            delete p;
        }

        VriResult VRI_CALL GetPipelineCacheData(VriPipelineCache* h, void* data, size_t* size)
        {
            if (!h || !size)
                return VriResult_InvalidArgument;
            PipelineCacheVK* p = PC(h);
            if (!data) // size query
            {
                size_t sz = 0;
                if (vkGetPipelineCacheData(p->device->Device(), p->cache, &sz, nullptr) != VK_SUCCESS)
                    return VriResult_Failure;
                *size = sz;
                return VriResult_Success;
            }
            // *size is the caller's buffer capacity; vkGetPipelineCacheData writes the byte count back.
            VkResult vr = vkGetPipelineCacheData(p->device->Device(), p->cache, size, data);
            if (vr == VK_INCOMPLETE)
                return VriResult_InvalidArgument; // buffer too small
            return vr == VK_SUCCESS ? VriResult_Success : VriResult_Failure;
        }

        const VriPipelineCacheInterface g_pipelineCacheVK = {
            CreatePipelineCache,
            DestroyPipelineCache,
            GetPipelineCacheData,
        };
    } // namespace

    const VriPipelineCacheInterface* GetPipelineCacheInterfaceVK() { return &g_pipelineCacheVK; }
} // namespace vri::vk
