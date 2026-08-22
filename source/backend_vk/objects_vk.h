// ObjectsVK.h - concrete backend objects behind the opaque VRI handles, plus
// reinterpret_cast helpers. Phase 1 subset (triangle vertical slice).
#pragma once

#include <vector>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <vri/vri.h>

namespace vri::vk
{
    class DeviceVK;

    struct QueueVK
    {
        DeviceVK* device;
        VkQueue   queue;
        uint32_t  familyIndex;
        uint32_t  indexInFamily;
    };

    struct CommandAllocatorVK
    {
        DeviceVK*     device;
        VkCommandPool pool;
        VriQueueType  queueType;
    };

    struct CommandBufferVK
    {
        DeviceVK*       device;
        VkCommandBuffer cmd;
        // recording state
        VriPipelineLayout*  boundLayout;
        VkPipelineBindPoint boundBindPoint;
    };

    struct BufferVK
    {
        DeviceVK*     device;
        VkBuffer      buffer;
        VmaAllocation allocation; // null if not VMA-owned
        uint64_t      size;
        // Set when the buffer owns a dedicated, non-VMA VkDeviceMemory (the exportable
        // external-memory path; see external_vk.cpp). allocation is null in that case.
        VkDeviceMemory dedicatedMemory     = VK_NULL_HANDLE;
        uint64_t       dedicatedMemorySize = 0; // total allocation size (for export sizing)
    };

    struct AccelerationStructureVK
    {
        DeviceVK*                      device;
        VkAccelerationStructureKHR     as;
        VkBuffer                       buffer; // backing store (ACCELERATION_STRUCTURE_STORAGE)
        VmaAllocation                  bufferAlloc;
        VkBuffer                       scratch; // build scratch (STORAGE | SHADER_DEVICE_ADDRESS)
        VmaAllocation                  scratchAlloc;
        VkDeviceAddress                deviceAddress;
        VkDeviceAddress                scratchAddress;
        uint64_t                       scratchSizeBytes = 0; // for the object registry's accounting
        VkAccelerationStructureTypeKHR type;
        // Transient query pool for the compacted-size query (lazily created; freed with the AS).
        VkQueryPool compactedSizePool = VK_NULL_HANDLE;
    };
    inline VriAccelerationStructure* ToHandle(AccelerationStructureVK* a)
    {
        return reinterpret_cast<VriAccelerationStructure*>(a);
    }

    struct MicromapVK
    {
        DeviceVK*                  device;
        VkMicromapEXT              micromap;
        VkBuffer                   buffer; // backing store (MICROMAP_STORAGE)
        VmaAllocation              bufferAlloc;
        VkBuffer                   scratch; // build scratch
        VmaAllocation              scratchAlloc;
        VkDeviceAddress            scratchAddress;
        VkOpacityMicromapFormatEXT format;
        uint32_t                   subdivisionLevel;
        uint32_t                   triangleCount;
    };
    inline VriMicromap* ToHandle(MicromapVK* m) { return reinterpret_cast<VriMicromap*>(m); }

    struct TextureVK
    {
        DeviceVK*     device;
        VkImage       image;
        VmaAllocation allocation; // null if wrapped (not owned)
        VkFormat      format;
        VkExtent3D    extent;
        uint32_t      mipNum;
        uint32_t      layerNum;
        VkImageType   type;
        bool          owned;
        // Set when the image owns a dedicated, non-VMA VkDeviceMemory (exportable path).
        VkDeviceMemory dedicatedMemory     = VK_NULL_HANDLE;
        uint64_t       dedicatedMemorySize = 0;
    };

    // A view (texture/buffer) or a sampler.
    struct DescriptorVK
    {
        enum class Kind
        {
            TextureView,
            BufferView,
            Sampler,
            AccelerationStructure
        } kind;
        DeviceVK*                  device;
        VkImageView                imageView;
        VkBufferView               bufferView; // only for typed/texel buffer views
        VkSampler                  sampler;
        VkAccelerationStructureKHR accel = VK_NULL_HANDLE; // only for Kind::AccelerationStructure
        // texture-view metadata (attachments, transitions)
        VkFormat           format;
        VkImageAspectFlags aspect;
        const TextureVK*   texture;
        // buffer-view metadata (constant/structured/storage buffer bindings)
        const BufferVK* buffer;
        uint64_t        bufferOffset;
        uint64_t        bufferRange;
    };

    // Per-binding metadata kept so UpdateDescriptorRanges can build writes
    // without re-deriving binding/type from the pipeline layout desc.
    struct RangeInfoVK
    {
        uint32_t         binding;
        VkDescriptorType type;
        uint32_t         count;
    };

    struct PipelineLayoutVK
    {
        DeviceVK*                          device;
        VkPipelineLayout                   layout;
        std::vector<VkDescriptorSetLayout> setLayouts;
        // setRanges[set][range] -> binding/type/count
        std::vector<std::vector<RangeInfoVK>> setRanges;
        // Per-set variable (bindless) descriptor count: 0 if the set has no
        // VARIABLE_DESCRIPTOR_COUNT binding, else that binding's max count (used to
        // chain VkDescriptorSetVariableDescriptorCountAllocateInfo at allocation).
        std::vector<uint32_t> setVariableCount;
        // Combined push-constant stage flags + total size (0 = no push range). CmdSetConstants
        // must pass exactly these stages to vkCmdPushConstants (not VK_SHADER_STAGE_ALL).
        VkShaderStageFlags pushStages = 0;
        uint32_t           pushSize   = 0;
    };

    struct DescriptorPoolVK
    {
        DeviceVK*        device;
        VkDescriptorPool pool;
    };

    struct DescriptorSetVK
    {
        DeviceVK*               device;
        VkDescriptorSet         set;
        const PipelineLayoutVK* layout;
        uint32_t                setIndex;
    };

    struct PipelineVK
    {
        DeviceVK*           device;
        VkPipeline          pipeline;
        VkPipelineBindPoint bindPoint;
    };

    struct FenceVK
    {
        DeviceVK*   device;
        VkSemaphore timeline; // timeline semaphore
    };

    struct MemoryVK
    {
        DeviceVK*     device;
        VmaAllocation allocation;
    };

    struct QueryPoolVK
    {
        DeviceVK*   device;
        VkQueryPool pool;
        VkQueryType type;
        uint32_t    count;
    };
    inline VriQueryPool* ToHandle(QueryPoolVK* q) { return reinterpret_cast<VriQueryPool*>(q); }

    struct PipelineCacheVK
    {
        DeviceVK*       device;
        VkPipelineCache cache;
    };
    inline VriPipelineCache* ToHandle(PipelineCacheVK* p) { return reinterpret_cast<VriPipelineCache*>(p); }
    // The VkPipelineCache to seed a create with (VK_NULL_HANDLE when the desc has none).
    inline VkPipelineCache PipeCache(VriPipelineCache* h)
    {
        return h ? reinterpret_cast<PipelineCacheVK*>(h)->cache : VK_NULL_HANDLE;
    }

    // ---- opaque <-> concrete casts -------------------------------------
    template<typename T, typename H>
    inline T* Cast(H* h)
    {
        return reinterpret_cast<T*>(h);
    }
    template<typename T, typename H>
    inline const T* Cast(const H* h)
    {
        return reinterpret_cast<const T*>(h);
    }

    inline VriQueue*            ToHandle(QueueVK* q) { return reinterpret_cast<VriQueue*>(q); }
    inline VriCommandAllocator* ToHandle(CommandAllocatorVK* a) { return reinterpret_cast<VriCommandAllocator*>(a); }
    inline VriCommandBuffer*    ToHandle(CommandBufferVK* c) { return reinterpret_cast<VriCommandBuffer*>(c); }
    inline VriBuffer*           ToHandle(BufferVK* b) { return reinterpret_cast<VriBuffer*>(b); }
    inline VriTexture*          ToHandle(TextureVK* t) { return reinterpret_cast<VriTexture*>(t); }
    inline VriDescriptor*       ToHandle(DescriptorVK* d) { return reinterpret_cast<VriDescriptor*>(d); }
    inline VriPipelineLayout*   ToHandle(PipelineLayoutVK* p) { return reinterpret_cast<VriPipelineLayout*>(p); }
    inline VriPipeline*         ToHandle(PipelineVK* p) { return reinterpret_cast<VriPipeline*>(p); }
    inline VriFence*            ToHandle(FenceVK* f) { return reinterpret_cast<VriFence*>(f); }
    inline VriMemory*           ToHandle(MemoryVK* m) { return reinterpret_cast<VriMemory*>(m); }
    inline VriDescriptorPool*   ToHandle(DescriptorPoolVK* p) { return reinterpret_cast<VriDescriptorPool*>(p); }
    inline VriDescriptorSet*    ToHandle(DescriptorSetVK* s) { return reinterpret_cast<VriDescriptorSet*>(s); }
} // namespace vri::vk
