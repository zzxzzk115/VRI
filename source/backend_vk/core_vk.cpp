// core_vk.cpp - the Vulkan implementation of VriCoreInterface.
//
// Phase 1 (triangle vertical slice): resources, views/samplers, pipeline
// layout, graphics/compute pipelines, timeline fences, command recording, and
// submission are implemented. The explicit memory path and descriptor
// pools/sets are stubbed (clearly marked) and land in Phase 2.

#include "core_vk.h"
#include "conversions_vk.h"
#include "device_vk.h"
#include "objects_vk.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <vector>

namespace vri::vk
{
    namespace
    {
        inline DeviceVK*               Dev(VriDevice* h) { return reinterpret_cast<DeviceVK*>(h); }
        inline const DeviceVK*         Dev(const VriDevice* h) { return reinterpret_cast<const DeviceVK*>(h); }
        inline QueueVK*                Q(VriQueue* h) { return reinterpret_cast<QueueVK*>(h); }
        inline CommandAllocatorVK*     CA(VriCommandAllocator* h) { return reinterpret_cast<CommandAllocatorVK*>(h); }
        inline CommandBufferVK*        CB(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferVK*>(h); }
        inline BufferVK*               Buf(VriBuffer* h) { return reinterpret_cast<BufferVK*>(h); }
        inline const BufferVK*         Buf(const VriBuffer* h) { return reinterpret_cast<const BufferVK*>(h); }
        inline TextureVK*              Tex(VriTexture* h) { return reinterpret_cast<TextureVK*>(h); }
        inline DescriptorVK*           Desc(VriDescriptor* h) { return reinterpret_cast<DescriptorVK*>(h); }
        inline const DescriptorVK*     Desc(const VriDescriptor* h) { return reinterpret_cast<const DescriptorVK*>(h); }
        inline PipelineLayoutVK*       PL(VriPipelineLayout* h) { return reinterpret_cast<PipelineLayoutVK*>(h); }
        inline const PipelineLayoutVK* PL(const VriPipelineLayout* h)
        {
            return reinterpret_cast<const PipelineLayoutVK*>(h);
        }
        inline PipelineVK*       Pipe(VriPipeline* h) { return reinterpret_cast<PipelineVK*>(h); }
        inline FenceVK*          Fen(VriFence* h) { return reinterpret_cast<FenceVK*>(h); }
        inline MemoryVK*         Mem(VriMemory* h) { return reinterpret_cast<MemoryVK*>(h); }
        inline DescriptorPoolVK* DPool(VriDescriptorPool* h) { return reinterpret_cast<DescriptorPoolVK*>(h); }
        inline DescriptorSetVK*  DSet(VriDescriptorSet* h) { return reinterpret_cast<DescriptorSetVK*>(h); }

        void FillVmaCreateInfo(VriMemoryLocation loc, VmaAllocationCreateInfo& aci)
        {
            switch (loc)
            {
                case VriMemoryLocation_DeviceUpload:
                    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                    aci.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
                    break;
                case VriMemoryLocation_HostUpload:
                    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
                    aci.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
                    break;
                case VriMemoryLocation_HostReadback:
                    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
                    aci.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
                    break;
                case VriMemoryLocation_Device:
                default:
                    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                    break;
            }
        }

        // Raw vmaAllocateMemory cannot use VMA_MEMORY_USAGE_AUTO* (it has no
        // resource to infer from), so the explicit path selects by property flags.
        void FillVmaRequiredFlags(VriMemoryLocation loc, VmaAllocationCreateInfo& aci)
        {
            switch (loc)
            {
                case VriMemoryLocation_DeviceUpload:
                    aci.requiredFlags  = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                    aci.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                    break;
                case VriMemoryLocation_HostUpload:
                    aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                    break;
                case VriMemoryLocation_HostReadback:
                    aci.requiredFlags  = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                    aci.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
                    break;
                case VriMemoryLocation_Device:
                default:
                    aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                    break;
            }
        }

        // ---- queries -------------------------------------------------------
        const VriDeviceDesc* VRI_CALL GetDeviceDesc(const VriDevice* device) { return &Dev(device)->Desc(); }

        VriFormatSupportFlags VRI_CALL GetFormatSupport(const VriDevice* device, VriFormat format)
        {
            VkFormatProperties props = {};
            vkGetPhysicalDeviceFormatProperties(Dev(device)->PhysicalDevice(), ToVkFormat(format), &props);
            const VkFormatFeatureFlags f = props.optimalTilingFeatures;

            VriFormatSupportFlags r = VriFormatSupport_None;
            if (f & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
                r |= VriFormatSupport_Texture;
            if (f & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
                r |= VriFormatSupport_StorageTexture;
            if (f & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)
                r |= VriFormatSupport_ColorAttachment;
            if (f & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
                r |= VriFormatSupport_DepthStencil;
            if (f & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT)
                r |= VriFormatSupport_Blend;
            if (props.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)
                r |= VriFormatSupport_VertexBuffer;
            return r;
        }

        VriResult VRI_CALL GetVideoMemoryInfo(const VriDevice*    device,
                                              VriMemoryLocation   location,
                                              VriVideoMemoryInfo* out)
        {
            if (!out)
                return VriResult_InvalidArgument;
            const DeviceVK* d = Dev(device);
            if (!d->HasMemoryBudget())
                return VriResult_Unsupported;
            VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
            VkPhysicalDeviceMemoryProperties2 props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
            props.pNext                             = &budget;
            vkGetPhysicalDeviceMemoryProperties2(d->PhysicalDevice(), &props);
            // Sum the device-local heaps for Device/DeviceUpload, the rest (system) otherwise.
            const bool wantDeviceLocal =
                location == VriMemoryLocation_Device || location == VriMemoryLocation_DeviceUpload;
            uint64_t b = 0, u = 0;
            for (uint32_t i = 0; i < props.memoryProperties.memoryHeapCount; ++i)
            {
                const bool deviceLocal =
                    (props.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
                if (deviceLocal == wantDeviceLocal)
                {
                    b += budget.heapBudget[i];
                    u += budget.heapUsage[i];
                }
            }
            out->budget = b;
            out->usage  = u;
            return VriResult_Success;
        }

        VriResult VRI_CALL GetQueue(VriDevice* device, VriQueueType type, uint32_t /*index*/, VriQueue** outQueue)
        {
            if (type >= VriQueueType_Count)
                return VriResult_InvalidArgument;
            *outQueue = ToHandle(Dev(device)->GetQueue(type));
            return VriResult_Success;
        }

        // ---- command allocation / lifecycle --------------------------------
        VriResult VRI_CALL CreateCommandAllocator(VriDevice* device, VriQueueType type, VriCommandAllocator** out)
        {
            DeviceVK*               d  = Dev(device);
            VkCommandPoolCreateInfo ci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            ci.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            ci.queueFamilyIndex        = d->GetQueue(type)->familyIndex;

            VkCommandPool pool = VK_NULL_HANDLE;
            if (vkCreateCommandPool(d->Device(), &ci, nullptr, &pool) != VK_SUCCESS)
                return VriResult_Failure;

            *out = ToHandle(new CommandAllocatorVK {d, pool, type});
            return VriResult_Success;
        }

        void VRI_CALL ResetCommandAllocator(VriCommandAllocator* allocator)
        {
            CommandAllocatorVK* a = CA(allocator);
            vkResetCommandPool(a->device->Device(), a->pool, 0);
        }

        void VRI_CALL DestroyCommandAllocator(VriCommandAllocator* allocator)
        {
            if (!allocator)
                return;
            CommandAllocatorVK* a = CA(allocator);
            vkDestroyCommandPool(a->device->Device(), a->pool, nullptr);
            delete a;
        }

        VriResult VRI_CALL CreateCommandBuffer(VriCommandAllocator* allocator, VriCommandBuffer** out)
        {
            CommandAllocatorVK*         a  = CA(allocator);
            VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool                 = a->pool;
            ai.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount          = 1;

            VkCommandBuffer cmd = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(a->device->Device(), &ai, &cmd) != VK_SUCCESS)
                return VriResult_Failure;

            *out = ToHandle(new CommandBufferVK {a->device, cmd, nullptr, VK_PIPELINE_BIND_POINT_GRAPHICS});
            return VriResult_Success;
        }

        VriResult VRI_CALL BeginCommandBuffer(VriCommandBuffer* cmd)
        {
            CommandBufferVK* c          = CB(cmd);
            c->boundLayout              = nullptr;
            VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            return vkBeginCommandBuffer(c->cmd, &bi) == VK_SUCCESS ? VriResult_Success : VriResult_Failure;
        }

        VriResult VRI_CALL EndCommandBuffer(VriCommandBuffer* cmd)
        {
            return vkEndCommandBuffer(CB(cmd)->cmd) == VK_SUCCESS ? VriResult_Success : VriResult_Failure;
        }

        // ---- resources -----------------------------------------------------
        VriResult VRI_CALL CreateBuffer(VriDevice* device, const VriBufferDesc* desc, VriBuffer** out)
        {
            DeviceVK*          d   = Dev(device);
            VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size               = desc->size;
            bci.usage              = ToVkBufferUsage(desc->usage);
            bci.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;
            uint32_t families[VriQueueType_Count];
            if (desc->usage & VriBufferUsage_ConcurrentQueues)
            {
                // Cross-queue use without ownership-transfer barriers. Degrades to EXCLUSIVE
                // on single-family devices, where the flag is meaningless.
                if (const uint32_t n = d->DistinctQueueFamilies(families); n > 1)
                {
                    bci.sharingMode           = VK_SHARING_MODE_CONCURRENT;
                    bci.queueFamilyIndexCount = n;
                    bci.pQueueFamilyIndices   = families;
                }
            }

            VkBuffer      buffer = VK_NULL_HANDLE;
            VmaAllocation alloc  = VK_NULL_HANDLE;

            if (desc->memoryLocation == VriMemoryLocation_Undefined)
            {
                // Unbound: caller drives AllocateMemory + BindBufferMemory.
                if (vkCreateBuffer(d->Device(), &bci, nullptr, &buffer) != VK_SUCCESS)
                    return VriResult_Failure;
            }
            else
            {
                VmaAllocationCreateInfo aci = {};
                FillVmaCreateInfo(desc->memoryLocation, aci);
                // AS/micromap build inputs and the SBT have a 256-byte device-address
                // alignment requirement; force the allocation so the address satisfies it.
                const bool needsAlign =
                    desc->usage & (VriBufferUsage_AccelerationBuildInput | VriBufferUsage_MicromapBuildInput |
                                   VriBufferUsage_ShaderBindingTable);
                const VkResult vr =
                    needsAlign ?
                        vmaCreateBufferWithAlignment(d->Allocator(), &bci, &aci, 256, &buffer, &alloc, nullptr) :
                        vmaCreateBuffer(d->Allocator(), &bci, &aci, &buffer, &alloc, nullptr);
                if (vr != VK_SUCCESS)
                    return VriResult_OutOfMemory;
            }

            *out = ToHandle(new BufferVK {d, buffer, alloc, desc->size});
            return VriResult_Success;
        }

        void VRI_CALL DestroyBuffer(VriBuffer* buffer)
        {
            if (!buffer)
                return;
            BufferVK* b = Buf(buffer);
            if (b->allocation)
                vmaDestroyBuffer(b->device->Allocator(), b->buffer, b->allocation);
            else if (b->dedicatedMemory)
            {
                // Exportable external-memory path: dedicated VkDeviceMemory, not VMA-owned.
                vkDestroyBuffer(b->device->Device(), b->buffer, nullptr);
                vkFreeMemory(b->device->Device(), b->dedicatedMemory, nullptr);
            }
            else if (b->buffer)
                vkDestroyBuffer(b->device->Device(), b->buffer, nullptr);
            delete b;
        }

        void* VRI_CALL MapBuffer(VriBuffer* buffer, uint64_t offset, uint64_t /*size*/)
        {
            BufferVK* b = Buf(buffer);
            void*     p = nullptr;
            if (vmaMapMemory(b->device->Allocator(), b->allocation, &p) != VK_SUCCESS)
                return nullptr;
            return static_cast<uint8_t*>(p) + offset;
        }

        void VRI_CALL UnmapBuffer(VriBuffer* buffer)
        {
            BufferVK* b = Buf(buffer);
            vmaUnmapMemory(b->device->Allocator(), b->allocation);
        }

        uint64_t VRI_CALL GetBufferDeviceAddress(const VriBuffer* buffer)
        {
            const BufferVK*           b    = Buf(buffer);
            VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            info.buffer                    = b->buffer;
            return vkGetBufferDeviceAddress(b->device->Device(), &info);
        }

        VriResult VRI_CALL CreateTexture(VriDevice* device, const VriTextureDesc* desc, VriTexture** out)
        {
            DeviceVK*         d   = Dev(device);
            VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.imageType         = ToVkImageType(desc->type);
            ici.format            = ToVkFormat(desc->format);
            ici.extent            = {desc->width, desc->height ? desc->height : 1u, desc->depth ? desc->depth : 1u};
            ici.mipLevels         = desc->mipNum ? desc->mipNum : 1u;
            ici.arrayLayers       = desc->layerNum ? desc->layerNum : 1u;
            ici.samples           = static_cast<VkSampleCountFlagBits>(desc->sampleNum ? desc->sampleNum : 1u);
            ici.tiling            = VK_IMAGE_TILING_OPTIMAL;
            ici.usage             = ToVkImageUsage(desc->usage);
            ici.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
            uint32_t families[VriQueueType_Count];
            if (desc->usage & VriTextureUsage_ConcurrentQueues)
            {
                // See CreateBuffer: concurrent across the distinct families, or EXCLUSIVE when
                // the device only has one.
                if (const uint32_t n = d->DistinctQueueFamilies(families); n > 1)
                {
                    ici.sharingMode           = VK_SHARING_MODE_CONCURRENT;
                    ici.queueFamilyIndexCount = n;
                    ici.pQueueFamilyIndices   = families;
                }
            }
            if (desc->type == VriTextureType_Cube || desc->type == VriTextureType_CubeArray)
                ici.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

            VkImage       image = VK_NULL_HANDLE;
            VmaAllocation alloc = VK_NULL_HANDLE;

            if (desc->memoryLocation == VriMemoryLocation_Undefined)
            {
                if (vkCreateImage(d->Device(), &ici, nullptr, &image) != VK_SUCCESS)
                    return VriResult_Failure;
            }
            else
            {
                VmaAllocationCreateInfo aci = {};
                FillVmaCreateInfo(desc->memoryLocation, aci);
                if (vmaCreateImage(d->Allocator(), &ici, &aci, &image, &alloc, nullptr) != VK_SUCCESS)
                    return VriResult_OutOfMemory;
            }

            TextureVK* t  = new TextureVK {};
            t->device     = d;
            t->image      = image;
            t->allocation = alloc;
            t->format     = ici.format;
            t->extent     = ici.extent;
            t->mipNum     = ici.mipLevels;
            t->layerNum   = ici.arrayLayers;
            t->type       = ici.imageType;
            t->owned      = true;
            *out          = ToHandle(t);
            return VriResult_Success;
        }

        void VRI_CALL DestroyTexture(VriTexture* texture)
        {
            if (!texture)
                return;
            TextureVK* t = Tex(texture);
            if (t->owned && t->allocation)
                vmaDestroyImage(t->device->Allocator(), t->image, t->allocation);
            else if (t->owned && t->dedicatedMemory)
            {
                // Exportable external-memory path: dedicated VkDeviceMemory, not VMA-owned.
                vkDestroyImage(t->device->Device(), t->image, nullptr);
                vkFreeMemory(t->device->Device(), t->dedicatedMemory, nullptr);
            }
            else if (t->owned && t->image)
                vkDestroyImage(t->device->Device(), t->image, nullptr); // unbound (explicit memory)
            delete t;
        }

        // ---- explicit memory -----------------------------------------------
        void VRI_CALL GetBufferMemoryDesc(const VriDevice*     device,
                                          const VriBufferDesc* desc,
                                          VriMemoryLocation    location,
                                          VriMemoryDesc*       out)
        {
            if (!out)
                return;
            VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size               = desc->size;
            bci.usage              = ToVkBufferUsage(desc->usage);
            bci.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;
            // Mirror CreateBuffer's sharing mode: requirements may legally depend on it.
            uint32_t families[VriQueueType_Count];
            if (desc->usage & VriBufferUsage_ConcurrentQueues)
            {
                if (const uint32_t n = Dev(device)->DistinctQueueFamilies(families); n > 1)
                {
                    bci.sharingMode           = VK_SHARING_MODE_CONCURRENT;
                    bci.queueFamilyIndexCount = n;
                    bci.pQueueFamilyIndices   = families;
                }
            }

            // Query via a throwaway VkBuffer + core-1.0 vkGetBufferMemoryRequirements rather than
            // the 1.3 vkGetDeviceBufferMemoryRequirements: MoltenVK reports apiVersion 1.2, where
            // that entry point is null and calling it segfaults. The temp buffer allocates no memory.
            *out         = VriMemoryDesc {};
            VkDevice dev = Dev(device)->Device();
            VkBuffer tmp = VK_NULL_HANDLE;
            if (vkCreateBuffer(dev, &bci, nullptr, &tmp) != VK_SUCCESS)
                return;
            VkMemoryRequirements mr = {};
            vkGetBufferMemoryRequirements(dev, tmp, &mr);
            vkDestroyBuffer(dev, tmp, nullptr);
            out->size           = mr.size;
            out->alignment      = mr.alignment;
            out->location       = location;
            out->memoryTypeMask = mr.memoryTypeBits;
        }

        void VRI_CALL GetTextureMemoryDesc(const VriDevice*      device,
                                           const VriTextureDesc* desc,
                                           VriMemoryLocation     location,
                                           VriMemoryDesc*        out)
        {
            if (!out)
                return;
            VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.imageType         = ToVkImageType(desc->type);
            ici.format            = ToVkFormat(desc->format);
            ici.extent            = {desc->width, desc->height ? desc->height : 1u, desc->depth ? desc->depth : 1u};
            ici.mipLevels         = desc->mipNum ? desc->mipNum : 1u;
            ici.arrayLayers       = desc->layerNum ? desc->layerNum : 1u;
            ici.samples           = static_cast<VkSampleCountFlagBits>(desc->sampleNum ? desc->sampleNum : 1u);
            ici.tiling            = VK_IMAGE_TILING_OPTIMAL;
            ici.usage             = ToVkImageUsage(desc->usage);
            ici.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
            // Mirror CreateTexture's sharing mode: requirements may legally depend on it.
            uint32_t families[VriQueueType_Count];
            if (desc->usage & VriTextureUsage_ConcurrentQueues)
            {
                if (const uint32_t n = Dev(device)->DistinctQueueFamilies(families); n > 1)
                {
                    ici.sharingMode           = VK_SHARING_MODE_CONCURRENT;
                    ici.queueFamilyIndexCount = n;
                    ici.pQueueFamilyIndices   = families;
                }
            }

            // Query via a throwaway VkImage + core-1.0 vkGetImageMemoryRequirements rather than
            // the 1.3 vkGetDeviceImageMemoryRequirements: MoltenVK reports apiVersion 1.2, where
            // that entry point is null and calling it segfaults. The temp image allocates no memory.
            *out         = VriMemoryDesc {};
            VkDevice dev = Dev(device)->Device();
            VkImage  tmp = VK_NULL_HANDLE;
            if (vkCreateImage(dev, &ici, nullptr, &tmp) != VK_SUCCESS)
                return;
            VkMemoryRequirements mr = {};
            vkGetImageMemoryRequirements(dev, tmp, &mr);
            vkDestroyImage(dev, tmp, nullptr);
            out->size           = mr.size;
            out->alignment      = mr.alignment;
            out->location       = location;
            out->memoryTypeMask = mr.memoryTypeBits;
        }

        VriResult VRI_CALL AllocateMemory(VriDevice* device, const VriMemoryDesc* desc, VriMemory** out)
        {
            DeviceVK*            d   = Dev(device);
            VkMemoryRequirements req = {};
            req.size                 = desc->size;
            req.alignment            = desc->alignment ? desc->alignment : 1;
            req.memoryTypeBits       = desc->memoryTypeMask ? desc->memoryTypeMask : ~0u;

            VmaAllocationCreateInfo aci = {};
            FillVmaRequiredFlags(desc->location, aci);

            VmaAllocation alloc = VK_NULL_HANDLE;
            if (vmaAllocateMemory(d->Allocator(), &req, &aci, &alloc, nullptr) != VK_SUCCESS)
                return VriResult_OutOfMemory;

            *out = ToHandle(new MemoryVK {d, alloc});
            return VriResult_Success;
        }

        void VRI_CALL FreeMemory(VriMemory* memory)
        {
            if (!memory)
                return;
            MemoryVK* m = Mem(memory);
            vmaFreeMemory(m->device->Allocator(), m->allocation);
            delete m;
        }

        VriResult VRI_CALL BindBufferMemory(VriDevice* device, VriBuffer* buffer, VriMemory* memory, uint64_t offset)
        {
            BufferVK* b = Buf(buffer);
            MemoryVK* m = Mem(memory);
            return vmaBindBufferMemory2(Dev(device)->Allocator(), m->allocation, offset, b->buffer, nullptr) ==
                           VK_SUCCESS ?
                       VriResult_Success :
                       VriResult_Failure;
        }

        VriResult VRI_CALL BindTextureMemory(VriDevice* device, VriTexture* texture, VriMemory* memory, uint64_t offset)
        {
            TextureVK* t = Tex(texture);
            MemoryVK*  m = Mem(memory);
            return vmaBindImageMemory2(Dev(device)->Allocator(), m->allocation, offset, t->image, nullptr) ==
                           VK_SUCCESS ?
                       VriResult_Success :
                       VriResult_Failure;
        }

        // ---- views & samplers ----------------------------------------------
        VriResult VRI_CALL CreateBufferView(VriDevice* device, const VriBufferViewDesc* desc, VriDescriptor** out)
        {
            DescriptorVK* v = new DescriptorVK {};
            v->kind         = DescriptorVK::Kind::BufferView;
            v->device       = Dev(device);
            v->buffer       = Buf(desc->buffer);
            v->bufferOffset = desc->offset;
            v->bufferRange  = desc->size;
            *out            = ToHandle(v);
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateTextureView(VriDevice* device, const VriTextureViewDesc* desc, VriDescriptor** out)
        {
            DeviceVK*        d = Dev(device);
            const TextureVK* t = reinterpret_cast<const TextureVK*>(desc->texture);

            VkImageViewCreateInfo ci         = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            ci.image                         = t->image;
            ci.viewType                      = ToVkImageViewType(desc->viewType);
            ci.format                        = desc->format == VriFormat_Unknown ? t->format : ToVkFormat(desc->format);
            ci.subresourceRange.aspectMask   = ToVkAspect(desc->aspect);
            ci.subresourceRange.baseMipLevel = desc->baseMip;
            ci.subresourceRange.levelCount   = desc->mipNum ? desc->mipNum : VK_REMAINING_MIP_LEVELS;
            ci.subresourceRange.baseArrayLayer = desc->baseLayer;
            ci.subresourceRange.layerCount     = desc->layerNum ? desc->layerNum : VK_REMAINING_ARRAY_LAYERS;

            VkImageView view = VK_NULL_HANDLE;
            if (vkCreateImageView(d->Device(), &ci, nullptr, &view) != VK_SUCCESS)
                return VriResult_Failure;

            DescriptorVK* v = new DescriptorVK {};
            v->kind         = DescriptorVK::Kind::TextureView;
            v->device       = d;
            v->imageView    = view;
            v->format       = ci.format;
            v->aspect       = ci.subresourceRange.aspectMask;
            v->texture      = t;
            *out            = ToHandle(v);
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateSampler(VriDevice* device, const VriSamplerDesc* desc, VriDescriptor** out)
        {
            DeviceVK* d      = Dev(device);
            auto      toAddr = [](VriAddressMode m) {
                switch (m)
                {
                    case VriAddressMode_MirroredRepeat:
                        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
                    case VriAddressMode_ClampToEdge:
                        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                    case VriAddressMode_ClampToBorder:
                        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                    case VriAddressMode_MirrorClampToEdge:
                        return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
                    default:
                        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
                }
            };

            VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            ci.magFilter           = desc->magFilter == VriFilter_Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
            ci.minFilter           = desc->minFilter == VriFilter_Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
            ci.mipmapMode          = desc->mipmapMode == VriMipmapMode_Linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR :
                                                                                VK_SAMPLER_MIPMAP_MODE_NEAREST;
            ci.addressModeU        = toAddr(desc->addressModeU);
            ci.addressModeV        = toAddr(desc->addressModeV);
            ci.addressModeW        = toAddr(desc->addressModeW);
            ci.mipLodBias          = desc->mipLodBias;
            // Anisotropy is a device feature with a per-device ceiling: honour the request only
            // where it is enabled, and clamp to the limit rather than tripping validation.
            const bool aniso    = desc->anisotropyEnable && d->HasSamplerAnisotropy();
            ci.anisotropyEnable = aniso ? VK_TRUE : VK_FALSE;
            ci.maxAnisotropy = aniso ? std::min(std::max(desc->maxAnisotropy, 1.0f), d->MaxSamplerAnisotropy()) : 1.0f;
            ci.compareEnable = desc->compareEnable ? VK_TRUE : VK_FALSE;
            ci.compareOp     = ToVkCompareOp(desc->compareOp);
            ci.minLod        = desc->minLod;
            ci.maxLod        = desc->maxLod;

            VkSamplerCustomBorderColorCreateInfoEXT custom = {
                VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
            if (desc->useCustomBorderColor)
            {
                ci.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
                custom.format  = VK_FORMAT_UNDEFINED;
                for (int i = 0; i < 4; ++i)
                    custom.customBorderColor.float32[i] = desc->customBorderColor[i];
                ci.pNext = &custom;
            }
            else
            {
                switch (desc->borderColor)
                {
                    case VriBorderColor_FloatOpaqueBlack:
                        ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
                        break;
                    case VriBorderColor_FloatOpaqueWhite:
                        ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
                        break;
                    case VriBorderColor_IntTransparentBlack:
                        ci.borderColor = VK_BORDER_COLOR_INT_TRANSPARENT_BLACK;
                        break;
                    case VriBorderColor_IntOpaqueBlack:
                        ci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
                        break;
                    case VriBorderColor_IntOpaqueWhite:
                        ci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_WHITE;
                        break;
                    default:
                        ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
                        break;
                }
            }

            VkSampler sampler = VK_NULL_HANDLE;
            if (vkCreateSampler(d->Device(), &ci, nullptr, &sampler) != VK_SUCCESS)
                return VriResult_Failure;

            DescriptorVK* v = new DescriptorVK {};
            v->kind         = DescriptorVK::Kind::Sampler;
            v->device       = d;
            v->sampler      = sampler;
            *out            = ToHandle(v);
            return VriResult_Success;
        }

        void VRI_CALL DestroyDescriptor(VriDescriptor* descriptor)
        {
            if (!descriptor)
                return;
            DescriptorVK* v   = Desc(descriptor);
            VkDevice      dev = v->device->Device();
            if (v->imageView)
                vkDestroyImageView(dev, v->imageView, nullptr);
            if (v->bufferView)
                vkDestroyBufferView(dev, v->bufferView, nullptr);
            if (v->sampler)
                vkDestroySampler(dev, v->sampler, nullptr);
            delete v;
        }

        // ---- pipeline layout & pipelines -----------------------------------
        VriResult VRI_CALL CreatePipelineLayout(VriDevice*                   device,
                                                const VriPipelineLayoutDesc* desc,
                                                VriPipelineLayout**          out)
        {
            DeviceVK*         d      = Dev(device);
            PipelineLayoutVK* layout = new PipelineLayoutVK {};
            layout->device           = d;

            for (uint32_t s = 0; s < desc->descriptorSetNum; ++s)
            {
                const VriDescriptorSetDesc&               set = desc->descriptorSets[s];
                std::vector<VkDescriptorSetLayoutBinding> bindings;
                std::vector<RangeInfoVK>                  rangeInfos;
                std::vector<VkDescriptorBindingFlags>     bindingFlags; // descriptor indexing (bindless)
                bindings.reserve(set.rangeNum);
                rangeInfos.reserve(set.rangeNum);
                bindingFlags.reserve(set.rangeNum);
                uint32_t variableCount       = 0; // max count of a VARIABLE_DESCRIPTOR_COUNT binding (0 = none)
                bool     usesUpdateAfterBind = false;
                for (uint32_t r = 0; r < set.rangeNum; ++r)
                {
                    const VriDescriptorRangeDesc& range = set.ranges[r];
                    VkDescriptorSetLayoutBinding  b     = {};
                    b.binding                           = range.baseRegister;
                    b.descriptorType                    = ToVkDescriptorType(range.descriptorType);
                    b.descriptorCount                   = range.descriptorNum;
                    b.stageFlags                        = ToVkShaderStageFlags(range.shaderStages);
                    bindings.push_back(b);
                    rangeInfos.push_back({b.binding, b.descriptorType, b.descriptorCount});

                    VkDescriptorBindingFlags bf = 0;
                    if (range.flags & VriDescriptorRange_PartiallyBound)
                        bf |= VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
                    if (range.flags & VriDescriptorRange_VariableSized)
                    {
                        // MoltenVK can't do VARIABLE_DESCRIPTOR_COUNT with image arrays; on macOS
                        // keep the array its declared fixed size (still partially-bound). [UNVERIFIED]
#if !defined(__APPLE__)
                        bf |= VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
                        variableCount = range.descriptorNum; // VK requires this be the last binding
#endif
                    }
                    if (bf)
                    {
                        bf |= VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
                        usesUpdateAfterBind = true;
                    }
                    bindingFlags.push_back(bf);
                }
                layout->setRanges.push_back(std::move(rangeInfos));
                layout->setVariableCount.push_back(variableCount);

                VkDescriptorSetLayoutBindingFlagsCreateInfo bfci = {
                    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
                bfci.bindingCount  = static_cast<uint32_t>(bindingFlags.size());
                bfci.pBindingFlags = bindingFlags.data();

                VkDescriptorSetLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
                lci.bindingCount                    = static_cast<uint32_t>(bindings.size());
                lci.pBindings                       = bindings.data();
                if (usesUpdateAfterBind)
                {
                    lci.pNext = &bfci; // update-after-bind bindless set
                    lci.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
                }

                VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
                if (vkCreateDescriptorSetLayout(d->Device(), &lci, nullptr, &setLayout) != VK_SUCCESS)
                {
                    for (VkDescriptorSetLayout l : layout->setLayouts)
                        vkDestroyDescriptorSetLayout(d->Device(), l, nullptr);
                    delete layout;
                    return VriResult_Failure;
                }
                layout->setLayouts.push_back(setLayout);
            }

            std::vector<VkPushConstantRange> pushRanges;
            pushRanges.reserve(desc->pushConstantNum);
            for (uint32_t i = 0; i < desc->pushConstantNum; ++i)
            {
                const VriPushConstantDesc& pc = desc->pushConstants[i];
                VkPushConstantRange        r  = {};
                r.stageFlags                  = ToVkShaderStageFlags(pc.shaderStages);
                r.offset = 0; // byte offset into the push block (baseRegister is a register, not a byte offset)
                r.size   = pc.size;
                pushRanges.push_back(r);
                layout->pushStages |= r.stageFlags;
                layout->pushSize = (pc.size > layout->pushSize) ? pc.size : layout->pushSize;
            }

            VkPipelineLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            ci.setLayoutCount             = static_cast<uint32_t>(layout->setLayouts.size());
            ci.pSetLayouts                = layout->setLayouts.data();
            ci.pushConstantRangeCount     = static_cast<uint32_t>(pushRanges.size());
            ci.pPushConstantRanges        = pushRanges.data();

            if (vkCreatePipelineLayout(d->Device(), &ci, nullptr, &layout->layout) != VK_SUCCESS)
            {
                for (VkDescriptorSetLayout l : layout->setLayouts)
                    vkDestroyDescriptorSetLayout(d->Device(), l, nullptr);
                delete layout;
                return VriResult_Failure;
            }

            *out = ToHandle(layout);
            return VriResult_Success;
        }

        void VRI_CALL DestroyPipelineLayout(VriPipelineLayout* layout)
        {
            if (!layout)
                return;
            PipelineLayoutVK* l   = PL(layout);
            VkDevice          dev = l->device->Device();
            for (VkDescriptorSetLayout s : l->setLayouts)
                vkDestroyDescriptorSetLayout(dev, s, nullptr);
            vkDestroyPipelineLayout(dev, l->layout, nullptr);
            delete l;
        }

        VkShaderModule MakeShaderModule(VkDevice dev, const VriShaderDesc& s)
        {
            VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            ci.codeSize                 = s.bytecodeSize;
            ci.pCode                    = static_cast<const uint32_t*>(s.bytecode);
            VkShaderModule m            = VK_NULL_HANDLE;
            vkCreateShaderModule(dev, &ci, nullptr, &m);
            return m;
        }

        VriResult VRI_CALL CreateGraphicsPipeline(VriDevice*                     device,
                                                  const VriGraphicsPipelineDesc* desc,
                                                  VriPipeline**                  out)
        {
            DeviceVK* d   = Dev(device);
            VkDevice  dev = d->Device();

            // shader stages
            std::vector<VkPipelineShaderStageCreateInfo> stages;
            std::vector<VkShaderModule>                  modules;
            stages.reserve(desc->shaderNum);
            modules.reserve(desc->shaderNum);
            bool hasMeshStage = false;
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                const VriShaderDesc& s = desc->shaders[i];
                if (s.stage == VriShaderStage_Mesh || s.stage == VriShaderStage_Task)
                    hasMeshStage = true;
                VkShaderModule m = MakeShaderModule(dev, s);
                modules.push_back(m);
                VkPipelineShaderStageCreateInfo si = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
                si.stage                           = ToVkShaderStage(s.stage);
                si.module                          = m;
                si.pName                           = s.entryPointName ? s.entryPointName : "main";
                stages.push_back(si);
            }
            // A failed shader module (e.g. invalid SPIR-V rejected by spirv-val) must
            // fail the pipeline explicitly, not yield a broken pipeline that faults later.
            for (VkShaderModule m : modules)
            {
                if (m == VK_NULL_HANDLE)
                {
                    for (VkShaderModule mm : modules)
                        if (mm)
                            vkDestroyShaderModule(dev, mm, nullptr);
                    d->ReportError("CreateGraphicsPipeline: shader module creation failed");
                    return VriResult_Failure;
                }
            }

            // vertex input
            std::vector<VkVertexInputBindingDescription>   vkStreams;
            std::vector<VkVertexInputAttributeDescription> vkAttrs;
            for (uint32_t i = 0; i < desc->vertexInput.streamNum; ++i)
            {
                const VriVertexStreamDesc&      s = desc->vertexInput.streams[i];
                VkVertexInputBindingDescription b = {};
                b.binding                         = s.bindingSlot;
                b.stride                          = s.stride;
                b.inputRate = s.stepRate == VriVertexStepRate_PerInstance ? VK_VERTEX_INPUT_RATE_INSTANCE :
                                                                            VK_VERTEX_INPUT_RATE_VERTEX;
                vkStreams.push_back(b);
            }
            for (uint32_t i = 0; i < desc->vertexInput.attributeNum; ++i)
            {
                const VriVertexAttributeDesc&     a  = desc->vertexInput.attributes[i];
                VkVertexInputAttributeDescription va = {};
                va.location                          = i;
                va.binding                           = a.streamIndex;
                va.format                            = ToVkFormat(a.format);
                va.offset                            = a.offset;
                vkAttrs.push_back(va);
            }
            VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vi.vertexBindingDescriptionCount        = static_cast<uint32_t>(vkStreams.size());
            vi.pVertexBindingDescriptions           = vkStreams.data();
            vi.vertexAttributeDescriptionCount      = static_cast<uint32_t>(vkAttrs.size());
            vi.pVertexAttributeDescriptions         = vkAttrs.data();

            VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology                               = ToVkTopology(desc->inputAssembly.topology);
            ia.primitiveRestartEnable                 = desc->inputAssembly.primitiveRestart ? VK_TRUE : VK_FALSE;

            VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vp.viewportCount                     = 1;
            vp.scissorCount                      = 1;

            // Tessellation: patch control points (only consumed with a tess pipeline /
            // PatchList topology, but harmless to fill).
            VkPipelineTessellationStateCreateInfo tess = {VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
            tess.patchControlPoints                    = desc->tessellation.patchControlPoints;

            const VriRasterizationDesc&            rs     = desc->rasterization;
            VkPipelineRasterizationStateCreateInfo raster = {
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            raster.polygonMode      = ToVkPolygonMode(rs.polygonMode);
            raster.cullMode         = ToVkCullMode(rs.cullMode);
            raster.frontFace        = ToVkFrontFace(rs.frontFace);
            raster.depthClampEnable = rs.depthClamp ? VK_TRUE : VK_FALSE;
            raster.lineWidth        = rs.lineWidth > 0.0f ? rs.lineWidth : 1.0f;
            if (rs.depthBias.constant != 0.0f || rs.depthBias.slope != 0.0f)
            {
                raster.depthBiasEnable         = VK_TRUE;
                raster.depthBiasConstantFactor = rs.depthBias.constant;
                raster.depthBiasClamp          = rs.depthBias.clamp;
                raster.depthBiasSlopeFactor    = rs.depthBias.slope;
            }
            VkPipelineRasterizationConservativeStateCreateInfoEXT conservative = {
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT};
            if (rs.conservativeRaster)
            {
                conservative.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
                raster.pNext                               = &conservative;
            }

            VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples =
                static_cast<VkSampleCountFlagBits>(desc->multisample.sampleNum ? desc->multisample.sampleNum : 1u);

            const VriDepthStencilDesc&            dss = desc->depthStencil;
            VkPipelineDepthStencilStateCreateInfo ds  = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            ds.depthTestEnable                        = dss.depthTest ? VK_TRUE : VK_FALSE;
            ds.depthWriteEnable                       = dss.depthWrite ? VK_TRUE : VK_FALSE;
            ds.depthCompareOp                         = ToVkCompareOp(dss.depthCompareOp);
            ds.stencilTestEnable                      = dss.stencilTest ? VK_TRUE : VK_FALSE;
            auto toVkStencil                          = [](const VriStencilOpDesc& s) {
                VkStencilOpState o = {};
                o.failOp           = ToVkStencilOp(s.failOp);
                o.passOp           = ToVkStencilOp(s.passOp);
                o.depthFailOp      = ToVkStencilOp(s.depthFailOp);
                o.compareOp        = ToVkCompareOp(s.compareOp);
                o.compareMask      = s.compareMask;
                o.writeMask        = s.writeMask;
                o.reference        = s.reference;
                return o;
            };
            ds.front = toVkStencil(dss.front);
            ds.back  = toVkStencil(dss.back);

            // color blend + dynamic-rendering formats
            std::vector<VkPipelineColorBlendAttachmentState> blends;
            std::vector<VkFormat>                            colorFormats;
            for (uint32_t i = 0; i < desc->outputMerger.colorNum; ++i)
            {
                const VriColorAttachmentDesc&       c = desc->outputMerger.colors[i];
                VkPipelineColorBlendAttachmentState b = {};
                b.blendEnable                         = c.blend.enable ? VK_TRUE : VK_FALSE;
                b.srcColorBlendFactor                 = ToVkBlendFactor(c.blend.srcColor);
                b.dstColorBlendFactor                 = ToVkBlendFactor(c.blend.dstColor);
                b.colorBlendOp                        = ToVkBlendOp(c.blend.colorOp);
                b.srcAlphaBlendFactor                 = ToVkBlendFactor(c.blend.srcAlpha);
                b.dstAlphaBlendFactor                 = ToVkBlendFactor(c.blend.dstAlpha);
                b.alphaBlendOp                        = ToVkBlendOp(c.blend.alphaOp);
                b.colorWriteMask                      = c.colorWriteMask ? c.colorWriteMask : VriColorWrite_RGBA;
                blends.push_back(b);
                colorFormats.push_back(ToVkFormat(c.format));
            }
            VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cb.attachmentCount                     = static_cast<uint32_t>(blends.size());
            cb.pAttachments                        = blends.data();

            // When VRS is enabled the per-draw shading rate is always dynamic (set via
            // CmdSetShadingRate). Declaring the dynamic state lets a draw set it; CmdBeginRendering
            // primes a default 1x1 rate so a draw that never sets one is still valid (no validation
            // error) and not flagged for "dynamic state used but pipeline didn't declare it".
            VkDynamicState                   dynStates[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dyn          = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dyn.dynamicStateCount                         = 2;
            if (d->EnabledFeatures() & VriFeature_VariableShadingRate)
                dynStates[dyn.dynamicStateCount++] = VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR;
            dyn.pDynamicStates = dynStates;

            VkPipelineRenderingCreateInfo rendering = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            rendering.colorAttachmentCount          = static_cast<uint32_t>(colorFormats.size());
            rendering.pColorAttachmentFormats       = colorFormats.data();
            // Multiview: the pipeline's viewMask must match the render pass's (VkRenderingInfo::viewMask).
            rendering.viewMask = desc->outputMerger.viewMask;
            if (desc->outputMerger.depthStencilFormat != VriFormat_Unknown)
            {
                const VkFormat dsf              = ToVkFormat(desc->outputMerger.depthStencilFormat);
                rendering.depthAttachmentFormat = dsf;
                if (dsf == VK_FORMAT_S8_UINT || dsf == VK_FORMAT_D16_UNORM_S8_UINT ||
                    dsf == VK_FORMAT_D24_UNORM_S8_UINT || dsf == VK_FORMAT_D32_SFLOAT_S8_UINT)
                    rendering.stencilAttachmentFormat = dsf;
            }

            VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            ci.pNext                        = &rendering;
            ci.stageCount                   = static_cast<uint32_t>(stages.size());
            ci.pStages                      = stages.data();
            // Mesh pipelines have no vertex-input/IA stage; the spec says they're
            // ignored but drivers may still dereference them, so pass null.
            ci.pVertexInputState   = hasMeshStage ? nullptr : &vi;
            ci.pInputAssemblyState = hasMeshStage ? nullptr : &ia;
            ci.pTessellationState  = &tess;
            ci.pViewportState      = &vp;
            ci.pRasterizationState = &raster;
            ci.pMultisampleState   = &ms;
            ci.pDepthStencilState  = &ds;
            ci.pColorBlendState    = &cb;
            ci.pDynamicState       = &dyn;
            ci.layout              = desc->pipelineLayout ? PL(desc->pipelineLayout)->layout : VK_NULL_HANDLE;

            VkPipeline pipeline = VK_NULL_HANDLE;
            VkResult   vr = vkCreateGraphicsPipelines(dev, PipeCache(desc->pipelineCache), 1, &ci, nullptr, &pipeline);

            for (VkShaderModule m : modules)
                vkDestroyShaderModule(dev, m, nullptr);

            if (vr != VK_SUCCESS)
                return VriResult_Failure;

            *out = ToHandle(new PipelineVK {d, pipeline, VK_PIPELINE_BIND_POINT_GRAPHICS});
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateComputePipeline(VriDevice*                    device,
                                                 const VriComputePipelineDesc* desc,
                                                 VriPipeline**                 out)
        {
            DeviceVK*      d   = Dev(device);
            VkDevice       dev = d->Device();
            VkShaderModule m   = MakeShaderModule(dev, desc->shader);

            VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage.sType                 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            ci.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.module                = m;
            ci.stage.pName                 = desc->shader.entryPointName ? desc->shader.entryPointName : "main";
            ci.layout                      = desc->pipelineLayout ? PL(desc->pipelineLayout)->layout : VK_NULL_HANDLE;

            VkPipeline pipeline = VK_NULL_HANDLE;
            VkResult   vr = vkCreateComputePipelines(dev, PipeCache(desc->pipelineCache), 1, &ci, nullptr, &pipeline);
            vkDestroyShaderModule(dev, m, nullptr);
            if (vr != VK_SUCCESS)
                return VriResult_Failure;

            *out = ToHandle(new PipelineVK {d, pipeline, VK_PIPELINE_BIND_POINT_COMPUTE});
            return VriResult_Success;
        }

        void VRI_CALL DestroyPipeline(VriPipeline* pipeline)
        {
            if (!pipeline)
                return;
            PipelineVK* p = Pipe(pipeline);
            vkDestroyPipeline(p->device->Device(), p->pipeline, nullptr);
            delete p;
        }

        // ---- descriptor pools / sets ---------------------------------------
        VriResult VRI_CALL CreateDescriptorPool(VriDevice*                   device,
                                                const VriDescriptorPoolDesc* desc,
                                                VriDescriptorPool**          out)
        {
            DeviceVK*                         d = Dev(device);
            std::vector<VkDescriptorPoolSize> sizes;
            auto                              add = [&](VkDescriptorType t, uint32_t n) {
                if (n)
                    sizes.push_back({t, n});
            };
            add(VK_DESCRIPTOR_TYPE_SAMPLER, desc->samplerMaxNum);
            add(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, desc->textureMaxNum);
            add(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, desc->storageTextureMaxNum);
            add(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, desc->constantBufferMaxNum);
            add(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, desc->structuredBufferMaxNum + desc->storageBufferMaxNum);
            add(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, desc->accelerationStructureMaxNum);
            if (sizes.empty())
                return VriResult_InvalidArgument;

            VkDescriptorPoolCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            ci.maxSets                    = desc->descriptorSetMaxNum;
            ci.poolSizeCount              = static_cast<uint32_t>(sizes.size());
            ci.pPoolSizes                 = sizes.data();
            // Bindless sets are update-after-bind; the pool must allow it. Only when the
            // device granted the feature (the flag is illegal otherwise).
            if (d->EnabledFeatures() & VriFeature_Bindless)
                ci.flags |= VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;

            VkDescriptorPool pool = VK_NULL_HANDLE;
            if (vkCreateDescriptorPool(d->Device(), &ci, nullptr, &pool) != VK_SUCCESS)
                return VriResult_Failure;

            *out = ToHandle(new DescriptorPoolVK {d, pool});
            return VriResult_Success;
        }

        void VRI_CALL ResetDescriptorPool(VriDescriptorPool* pool)
        {
            DescriptorPoolVK* p = DPool(pool);
            vkResetDescriptorPool(p->device->Device(), p->pool, 0);
        }

        void VRI_CALL DestroyDescriptorPool(VriDescriptorPool* pool)
        {
            if (!pool)
                return;
            DescriptorPoolVK* p = DPool(pool);
            vkDestroyDescriptorPool(p->device->Device(), p->pool, nullptr);
            delete p;
        }

        VriResult VRI_CALL AllocateDescriptorSets(VriDescriptorPool*       pool,
                                                  const VriPipelineLayout* layout,
                                                  uint32_t                 setIndex,
                                                  VriDescriptorSet**       outSets,
                                                  uint32_t                 setNum)
        {
            DescriptorPoolVK*       p = DPool(pool);
            const PipelineLayoutVK* l = PL(layout);
            if (setIndex >= l->setLayouts.size())
                return VriResult_InvalidArgument;

            std::vector<VkDescriptorSetLayout> layouts(setNum, l->setLayouts[setIndex]);
            VkDescriptorSetAllocateInfo        ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool                     = p->pool;
            ai.descriptorSetCount                 = setNum;
            ai.pSetLayouts                        = layouts.data();

            // Bindless set: tell VK the actual size of the VARIABLE_DESCRIPTOR_COUNT
            // binding (we allocate the layout's max for each set).
            const uint32_t        varCount = setIndex < l->setVariableCount.size() ? l->setVariableCount[setIndex] : 0;
            std::vector<uint32_t> counts(setNum, varCount);
            VkDescriptorSetVariableDescriptorCountAllocateInfo vci = {
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO};
            if (varCount > 0)
            {
                vci.descriptorSetCount = setNum;
                vci.pDescriptorCounts  = counts.data();
                ai.pNext               = &vci;
            }

            std::vector<VkDescriptorSet> sets(setNum);
            if (vkAllocateDescriptorSets(p->device->Device(), &ai, sets.data()) != VK_SUCCESS)
                return VriResult_Failure;

            for (uint32_t i = 0; i < setNum; ++i)
                outSets[i] = ToHandle(new DescriptorSetVK {p->device, sets[i], l, setIndex});
            return VriResult_Success;
        }

        void VRI_CALL UpdateDescriptorRanges(VriDescriptorSet*                   set,
                                             uint32_t                            baseRange,
                                             uint32_t                            rangeNum,
                                             const VriDescriptorRangeUpdateDesc* updates)
        {
            DescriptorSetVK*                s      = DSet(set);
            const std::vector<RangeInfoVK>& ranges = s->layout->setRanges[s->setIndex];

            std::vector<VkWriteDescriptorSet> writes;
            // One CONTIGUOUS array per write (VK reads descriptorCount elements from
            // pImageInfo/pBufferInfo); the deque keeps each vector's address stable.
            std::deque<std::vector<VkDescriptorImageInfo>>           imageInfoLists;
            std::deque<std::vector<VkDescriptorBufferInfo>>          bufferInfoLists;
            std::deque<std::vector<VkAccelerationStructureKHR>>      accelLists; // stable across push_back
            std::deque<VkWriteDescriptorSetAccelerationStructureKHR> accelWrites;
            writes.reserve(rangeNum);

            for (uint32_t r = 0; r < rangeNum; ++r)
            {
                const VriDescriptorRangeUpdateDesc& u       = updates[r];
                const RangeInfoVK&                  info    = ranges[baseRange + r];
                const bool                          isImage = info.type == VK_DESCRIPTOR_TYPE_SAMPLER ||
                                     info.type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
                                     info.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                const bool isAccel = info.type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

                VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                w.dstSet               = s->set;
                w.dstBinding           = info.binding;
                w.dstArrayElement      = u.baseDescriptor;
                w.descriptorCount      = u.descriptorNum;
                w.descriptorType       = info.type;

                if (isAccel)
                {
                    accelLists.emplace_back();
                    std::vector<VkAccelerationStructureKHR>& list = accelLists.back();
                    for (uint32_t k = 0; k < u.descriptorNum; ++k)
                        list.push_back(Desc(u.descriptors[k])->accel);
                    VkWriteDescriptorSetAccelerationStructureKHR aw = {
                        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
                    aw.accelerationStructureCount = static_cast<uint32_t>(list.size());
                    aw.pAccelerationStructures    = list.data();
                    accelWrites.push_back(aw);
                    w.pNext = &accelWrites.back();
                }
                else if (isImage)
                {
                    imageInfoLists.emplace_back();
                    std::vector<VkDescriptorImageInfo>& ii = imageInfoLists.back();
                    ii.reserve(u.descriptorNum);
                    for (uint32_t k = 0; k < u.descriptorNum; ++k)
                    {
                        const DescriptorVK*   d = Desc(u.descriptors[k]);
                        VkDescriptorImageInfo e = {};
                        e.sampler               = d->sampler;
                        e.imageView             = d->imageView;
                        e.imageLayout           = info.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ?
                                                      VK_IMAGE_LAYOUT_GENERAL :
                                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        ii.push_back(e);
                    }
                    w.pImageInfo = ii.data();
                }
                else
                {
                    bufferInfoLists.emplace_back();
                    std::vector<VkDescriptorBufferInfo>& bi = bufferInfoLists.back();
                    bi.reserve(u.descriptorNum);
                    for (uint32_t k = 0; k < u.descriptorNum; ++k)
                    {
                        const DescriptorVK*    d = Desc(u.descriptors[k]);
                        VkDescriptorBufferInfo e = {};
                        e.buffer                 = d->buffer ? d->buffer->buffer : VK_NULL_HANDLE;
                        e.offset                 = d->bufferOffset;
                        e.range                  = d->bufferRange ? d->bufferRange : VK_WHOLE_SIZE;
                        bi.push_back(e);
                    }
                    w.pBufferInfo = bi.data();
                }
                writes.push_back(w);
            }

            vkUpdateDescriptorSets(
                s->device->Device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        // ---- synchronization (timeline) ------------------------------------
        VriResult VRI_CALL CreateFence(VriDevice* device, uint64_t initialValue, VriFence** out)
        {
            DeviceVK*                 d    = Dev(device);
            VkSemaphoreTypeCreateInfo type = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            type.semaphoreType             = VK_SEMAPHORE_TYPE_TIMELINE;
            type.initialValue              = initialValue;
            VkSemaphoreCreateInfo ci       = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            ci.pNext                       = &type;

            VkSemaphore sem = VK_NULL_HANDLE;
            if (vkCreateSemaphore(d->Device(), &ci, nullptr, &sem) != VK_SUCCESS)
                return VriResult_Failure;

            *out = ToHandle(new FenceVK {d, sem});
            return VriResult_Success;
        }

        void VRI_CALL DestroyFence(VriFence* fence)
        {
            if (!fence)
                return;
            FenceVK* f = Fen(fence);
            vkDestroySemaphore(f->device->Device(), f->timeline, nullptr);
            delete f;
        }

        uint64_t VRI_CALL GetFenceValue(VriFence* fence)
        {
            FenceVK* f     = Fen(fence);
            uint64_t value = 0;
            vkGetSemaphoreCounterValue(f->device->Device(), f->timeline, &value);
            return value;
        }

        void VRI_CALL Wait(VriFence* fence, uint64_t value)
        {
            FenceVK*            f  = Fen(fence);
            VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wi.semaphoreCount      = 1;
            wi.pSemaphores         = &f->timeline;
            wi.pValues             = &value;
            vkWaitSemaphores(f->device->Device(), &wi, UINT64_MAX);
        }

        // ---- command recording ---------------------------------------------
        void VRI_CALL CmdBeginRendering(VriCommandBuffer* cmd, const VriAttachmentsDesc* a)
        {
            CommandBufferVK*                       c = CB(cmd);
            std::vector<VkRenderingAttachmentInfo> colors;
            colors.reserve(a->colorNum);
            for (uint32_t i = 0; i < a->colorNum; ++i)
            {
                const VriAttachmentDesc&  src  = a->colors[i];
                const DescriptorVK*       view = Desc(src.view);
                VkRenderingAttachmentInfo ai   = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                ai.imageView                   = view->imageView;
                ai.imageLayout                 = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                ai.loadOp                      = src.loadOp == VriAttachmentLoadOp_Clear ? VK_ATTACHMENT_LOAD_OP_CLEAR :
                                                 src.loadOp == VriAttachmentLoadOp_DontCare ? VK_ATTACHMENT_LOAD_OP_DONT_CARE :
                                                                                              VK_ATTACHMENT_LOAD_OP_LOAD;
                ai.storeOp = src.storeOp == VriAttachmentStoreOp_DontCare ? VK_ATTACHMENT_STORE_OP_DONT_CARE :
                                                                            VK_ATTACHMENT_STORE_OP_STORE;
                std::memcpy(ai.clearValue.color.float32, src.clearValue.color.f32, sizeof(float) * 4);
                if (src.resolveView) // MSAA resolve target
                {
                    ai.resolveImageView   = Desc(src.resolveView)->imageView;
                    ai.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    ai.resolveMode        = VK_RESOLVE_MODE_AVERAGE_BIT;
                }
                colors.push_back(ai);
            }

            VkRenderingInfo ri      = {VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea.offset    = {a->renderArea.x, a->renderArea.y};
            ri.renderArea.extent    = {a->renderArea.width, a->renderArea.height};
            ri.layerCount           = a->layerNum ? a->layerNum : 1u;
            ri.viewMask             = a->viewMask;
            ri.colorAttachmentCount = static_cast<uint32_t>(colors.size());
            ri.pColorAttachments    = colors.data();

            VkRenderingAttachmentInfo depth   = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            VkRenderingAttachmentInfo stencil = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            if (a->depth && a->depth->view)
            {
                const DescriptorVK* view = Desc(a->depth->view);
                depth.imageView          = view->imageView;
                depth.imageLayout        = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                depth.loadOp             = a->depth->loadOp == VriAttachmentLoadOp_Clear ? VK_ATTACHMENT_LOAD_OP_CLEAR :
                                                                                           VK_ATTACHMENT_LOAD_OP_LOAD;
                depth.storeOp            = VK_ATTACHMENT_STORE_OP_STORE;
                depth.clearValue.depthStencil.depth   = a->depth->clearValue.depthStencil.depth;
                depth.clearValue.depthStencil.stencil = a->depth->clearValue.depthStencil.stencil;
                ri.pDepthAttachment                   = &depth;
                // Attach the stencil aspect too when the format carries stencil.
                const VkFormat f = view->format;
                if (f == VK_FORMAT_S8_UINT || f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT ||
                    f == VK_FORMAT_D32_SFLOAT_S8_UINT)
                {
                    stencil               = depth;
                    ri.pStencilAttachment = &stencil;
                }
            }

            c->device->Ext().CmdBeginRendering(c->cmd, &ri);

            // Pipelines declare the fragment-shading-rate dynamic state when VRS is enabled, so the
            // rate must be set before any draw. Prime the no-op 1x1 rate here; CmdSetShadingRate can
            // still override it within the pass. This keeps non-VRS draws valid without a per-pipeline
            // opt-in (and avoids the "dynamic state set but pipeline didn't declare it" warning).
            if ((c->device->EnabledFeatures() & VriFeature_VariableShadingRate) &&
                c->device->Ext().CmdSetFragmentShadingRate)
            {
                const VkExtent2D                         rate    = {1, 1};
                const VkFragmentShadingRateCombinerOpKHR keep[2] = {VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
                                                                    VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR};
                c->device->Ext().CmdSetFragmentShadingRate(c->cmd, &rate, keep);
            }
        }

        void VRI_CALL CmdEndRendering(VriCommandBuffer* cmd) { CB(cmd)->device->Ext().CmdEndRendering(CB(cmd)->cmd); }

        void VRI_CALL CmdSetViewports(VriCommandBuffer* cmd, const VriViewport* vps, uint32_t num)
        {
            // VRI standardizes on Y-up clip space (D3D12/WebGPU/Metal convention).
            // Vulkan's clip Y points down, so flip it with a negative-height
            // viewport (VK_KHR_maintenance1, core 1.1) -- no shader change needed.
            std::vector<VkViewport> v(num);
            for (uint32_t i = 0; i < num; ++i)
            {
                v[i].x        = vps[i].x;
                v[i].y        = vps[i].y + vps[i].height;
                v[i].width    = vps[i].width;
                v[i].height   = -vps[i].height;
                v[i].minDepth = vps[i].minDepth;
                v[i].maxDepth = vps[i].maxDepth;
            }
            vkCmdSetViewport(CB(cmd)->cmd, 0, num, v.data());
        }

        void VRI_CALL CmdSetScissors(VriCommandBuffer* cmd, const VriRect* rects, uint32_t num)
        {
            std::vector<VkRect2D> r(num);
            for (uint32_t i = 0; i < num; ++i)
                r[i] = {{rects[i].x, rects[i].y}, {rects[i].width, rects[i].height}};
            vkCmdSetScissor(CB(cmd)->cmd, 0, num, r.data());
        }

        void VRI_CALL CmdSetPipelineLayout(VriCommandBuffer* cmd, VriPipelineLayout* layout)
        {
            CB(cmd)->boundLayout = layout;
        }

        void VRI_CALL CmdSetPipeline(VriCommandBuffer* cmd, VriPipeline* pipeline)
        {
            CommandBufferVK* c = CB(cmd);
            PipelineVK*      p = Pipe(pipeline);
            c->boundBindPoint  = p->bindPoint;
            vkCmdBindPipeline(c->cmd, p->bindPoint, p->pipeline);
        }

        void VRI_CALL CmdSetDescriptorSet(VriCommandBuffer* cmd, uint32_t setIndex, const VriDescriptorSet* set)
        {
            CommandBufferVK* c = CB(cmd);
            if (!c->boundLayout || !set)
                return;
            const DescriptorSetVK* s  = reinterpret_cast<const DescriptorSetVK*>(set);
            VkDescriptorSet        vs = s->set;
            vkCmdBindDescriptorSets(
                c->cmd, c->boundBindPoint, PL(c->boundLayout)->layout, setIndex, 1, &vs, 0, nullptr);
        }

        void VRI_CALL CmdSetConstants(VriCommandBuffer* cmd, uint32_t /*index*/, const void* data, uint32_t size)
        {
            CommandBufferVK* c = CB(cmd);
            if (!c->boundLayout)
                return;
            const PipelineLayoutVK* pl = PL(c->boundLayout);
            if (!pl->pushStages)
                return; // layout declares no push-constant range
            vkCmdPushConstants(c->cmd, pl->layout, pl->pushStages, 0, size, data);
        }

        void VRI_CALL CmdSetVertexBuffers(VriCommandBuffer*             cmd,
                                          uint32_t                      baseSlot,
                                          const VriVertexBufferBinding* bindings,
                                          uint32_t                      num)
        {
            std::vector<VkBuffer>     buffers(num);
            std::vector<VkDeviceSize> offsets(num);
            for (uint32_t i = 0; i < num; ++i)
            {
                buffers[i] = Buf(bindings[i].buffer)->buffer;
                offsets[i] = bindings[i].offset;
            }
            vkCmdBindVertexBuffers(CB(cmd)->cmd, baseSlot, num, buffers.data(), offsets.data());
        }

        void VRI_CALL CmdSetIndexBuffer(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, VriIndexType type)
        {
            vkCmdBindIndexBuffer(CB(cmd)->cmd,
                                 Buf(buffer)->buffer,
                                 offset,
                                 type == VriIndexType_UInt16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        }

        void VRI_CALL CmdDraw(VriCommandBuffer* cmd, const VriDrawDesc* d)
        {
            vkCmdDraw(CB(cmd)->cmd, d->vertexNum, d->instanceNum, d->baseVertex, d->baseInstance);
        }

        void VRI_CALL CmdDrawIndexed(VriCommandBuffer* cmd, const VriDrawIndexedDesc* d)
        {
            vkCmdDrawIndexed(CB(cmd)->cmd, d->indexNum, d->instanceNum, d->baseIndex, d->vertexOffset, d->baseInstance);
        }

        void VRI_CALL
        CmdDrawIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint32_t drawNum, uint32_t stride)
        {
            vkCmdDrawIndirect(CB(cmd)->cmd, Buf(buffer)->buffer, offset, drawNum, stride);
        }
        void VRI_CALL CmdDrawIndexedIndirect(VriCommandBuffer* cmd,
                                             VriBuffer*        buffer,
                                             uint64_t          offset,
                                             uint32_t          drawNum,
                                             uint32_t          stride)
        {
            vkCmdDrawIndexedIndirect(CB(cmd)->cmd, Buf(buffer)->buffer, offset, drawNum, stride);
        }
        // vkCmdDraw[Indexed]IndirectCount is core Vulkan 1.2 (VRI requests 1.3) - always available.
        void VRI_CALL CmdDrawIndirectCount(VriCommandBuffer* cmd,
                                           VriBuffer*        buffer,
                                           uint64_t          offset,
                                           VriBuffer*        countBuffer,
                                           uint64_t          countOffset,
                                           uint32_t          maxDrawNum,
                                           uint32_t          stride)
        {
            vkCmdDrawIndirectCount(
                CB(cmd)->cmd, Buf(buffer)->buffer, offset, Buf(countBuffer)->buffer, countOffset, maxDrawNum, stride);
        }
        void VRI_CALL CmdDrawIndexedIndirectCount(VriCommandBuffer* cmd,
                                                  VriBuffer*        buffer,
                                                  uint64_t          offset,
                                                  VriBuffer*        countBuffer,
                                                  uint64_t          countOffset,
                                                  uint32_t          maxDrawNum,
                                                  uint32_t          stride)
        {
            vkCmdDrawIndexedIndirectCount(
                CB(cmd)->cmd, Buf(buffer)->buffer, offset, Buf(countBuffer)->buffer, countOffset, maxDrawNum, stride);
        }

        void VRI_CALL CmdDispatch(VriCommandBuffer* cmd, const VriDispatchDesc* d)
        {
            vkCmdDispatch(CB(cmd)->cmd, d->x, d->y, d->z);
        }

        void VRI_CALL CmdDispatchIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset)
        {
            vkCmdDispatchIndirect(CB(cmd)->cmd, Buf(buffer)->buffer, offset);
        }

        void VRI_CALL CmdBarrier(VriCommandBuffer* cmd, const VriBarrierGroupDesc* group)
        {
            CommandBufferVK*                    c = CB(cmd);
            std::vector<VkBufferMemoryBarrier2> bufBarriers;
            std::vector<VkImageMemoryBarrier2>  imgBarriers;

            for (uint32_t i = 0; i < group->bufferNum; ++i)
            {
                const VriBufferBarrierDesc& b  = group->buffers[i];
                VkBufferMemoryBarrier2      vb = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
                vb.srcStageMask                = ToVkStages(b.before.stages);
                vb.srcAccessMask               = ToVkAccess(b.before.access);
                vb.dstStageMask                = ToVkStages(b.after.stages);
                vb.dstAccessMask               = ToVkAccess(b.after.access);
                vb.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
                vb.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
                vb.buffer                      = Buf(b.buffer)->buffer;
                vb.offset                      = 0;
                vb.size                        = VK_WHOLE_SIZE;
                bufBarriers.push_back(vb);
            }

            for (uint32_t i = 0; i < group->textureNum; ++i)
            {
                const VriTextureBarrierDesc& t     = group->textures[i];
                const TextureVK*             tex   = reinterpret_cast<const TextureVK*>(t.texture);
                VkImageMemoryBarrier2        vi    = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                vi.srcStageMask                    = ToVkStages(t.before.stages);
                vi.srcAccessMask                   = ToVkAccess(t.before.access);
                vi.dstStageMask                    = ToVkStages(t.after.stages);
                vi.dstAccessMask                   = ToVkAccess(t.after.access);
                vi.oldLayout                       = ToVkLayout(t.before.layout);
                vi.newLayout                       = ToVkLayout(t.after.layout);
                vi.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
                vi.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
                vi.image                           = tex->image;
                vi.subresourceRange.aspectMask     = t.aspect ? ToVkAspect(t.aspect) : VK_IMAGE_ASPECT_COLOR_BIT;
                vi.subresourceRange.baseMipLevel   = t.baseMip;
                vi.subresourceRange.levelCount     = t.mipNum ? t.mipNum : VK_REMAINING_MIP_LEVELS;
                vi.subresourceRange.baseArrayLayer = t.baseLayer;
                vi.subresourceRange.layerCount     = t.layerNum ? t.layerNum : VK_REMAINING_ARRAY_LAYERS;
                imgBarriers.push_back(vi);
            }

            VkDependencyInfo dep         = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.bufferMemoryBarrierCount = static_cast<uint32_t>(bufBarriers.size());
            dep.pBufferMemoryBarriers    = bufBarriers.data();
            dep.imageMemoryBarrierCount  = static_cast<uint32_t>(imgBarriers.size());
            dep.pImageMemoryBarriers     = imgBarriers.data();
            c->device->Ext().CmdPipelineBarrier2(c->cmd, &dep);
        }

        void VRI_CALL
        CmdClearStorageBuffer(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint64_t size, uint32_t value)
        {
            vkCmdFillBuffer(CB(cmd)->cmd, Buf(buffer)->buffer, offset, size ? size : VK_WHOLE_SIZE, value);
        }
        void VRI_CALL CmdClearStorageTexture(VriCommandBuffer* cmd, VriTexture* texture, const VriClearColor* value)
        {
            VkClearColorValue cv;
            std::memcpy(&cv, value, sizeof(cv)); // VriClearColor + VkClearColorValue are both a 16-byte union
            VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS};
            // vkCmdClearColorImage is a transfer-domain op, so the caller barriers the texture to the
            // CopyDestination state/layout (matching the storage-buffer clear's transfer contract).
            vkCmdClearColorImage(
                CB(cmd)->cmd, Tex(texture)->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &range);
        }
        void VRI_CALL CmdCopyBuffer(VriCommandBuffer*        cmd,
                                    VriBuffer*               dst,
                                    VriBuffer*               src,
                                    const VriBufferCopyDesc* region)
        {
            VkBufferCopy r = {region->srcOffset, region->dstOffset, region->size};
            vkCmdCopyBuffer(CB(cmd)->cmd, Buf(src)->buffer, Buf(dst)->buffer, 1, &r);
        }

        VkBufferImageCopy MakeBufferImageCopy(const VriBufferTextureCopyDesc* r, const TextureVK* tex)
        {
            VkBufferImageCopy c = {};
            c.bufferOffset      = r->bufferOffset;
            c.bufferRowLength   = r->bufferRowLength;
            c.bufferImageHeight = r->bufferImageHeight;
            c.imageSubresource.aspectMask =
                r->texture.aspect ? ToVkAspect(r->texture.aspect) : VK_IMAGE_ASPECT_COLOR_BIT;
            c.imageSubresource.mipLevel       = r->texture.mip;
            c.imageSubresource.baseArrayLayer = r->texture.baseLayer;
            c.imageSubresource.layerCount     = r->texture.layerNum ? r->texture.layerNum : 1u;
            c.imageOffset                     = {r->texture.x, r->texture.y, r->texture.z};
            c.imageExtent                     = {r->texture.width ? r->texture.width : tex->extent.width,
                             r->texture.height ? r->texture.height : tex->extent.height,
                             r->texture.depth ? r->texture.depth : tex->extent.depth};
            return c;
        }

        void VRI_CALL CmdCopyTexture(VriCommandBuffer*         cmd,
                                     VriTexture*               dst,
                                     VriTexture*               src,
                                     const VriTextureCopyDesc* region)
        {
            const TextureVK* s          = reinterpret_cast<const TextureVK*>(src);
            const TextureVK* d          = reinterpret_cast<const TextureVK*>(dst);
            VkImageCopy      c          = {};
            c.srcSubresource.aspectMask = region ? ToVkAspect(region->src.aspect) : VK_IMAGE_ASPECT_COLOR_BIT;
            c.srcSubresource.mipLevel   = region ? region->src.mip : 0;
            c.srcSubresource.layerCount = 1;
            c.dstSubresource.aspectMask = region ? ToVkAspect(region->dst.aspect) : VK_IMAGE_ASPECT_COLOR_BIT;
            c.dstSubresource.mipLevel   = region ? region->dst.mip : 0;
            c.dstSubresource.layerCount = 1;
            c.extent                    = s->extent;
            vkCmdCopyImage(CB(cmd)->cmd,
                           s->image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           d->image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &c);
        }

        void VRI_CALL CmdUploadBufferToTexture(VriCommandBuffer*               cmd,
                                               VriTexture*                     dst,
                                               VriBuffer*                      src,
                                               const VriBufferTextureCopyDesc* region)
        {
            const TextureVK*  t = reinterpret_cast<const TextureVK*>(dst);
            VkBufferImageCopy c = MakeBufferImageCopy(region, t);
            vkCmdCopyBufferToImage(
                CB(cmd)->cmd, Buf(src)->buffer, t->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        }

        void VRI_CALL CmdReadbackTextureToBuffer(VriCommandBuffer*               cmd,
                                                 VriBuffer*                      dst,
                                                 VriTexture*                     src,
                                                 const VriBufferTextureCopyDesc* region)
        {
            const TextureVK*  t = reinterpret_cast<const TextureVK*>(src);
            VkBufferImageCopy c = MakeBufferImageCopy(region, t);
            vkCmdCopyImageToBuffer(
                CB(cmd)->cmd, t->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, Buf(dst)->buffer, 1, &c);
        }

        void VRI_CALL CmdBeginDebugGroup(VriCommandBuffer*, const char*) {}
        void VRI_CALL CmdEndDebugGroup(VriCommandBuffer*) {}

        // ---- submission ----------------------------------------------------
        void VRI_CALL QueueSubmit(VriQueue* queue, const VriQueueSubmitDesc* submit)
        {
            QueueVK* q = Q(queue);

            std::vector<VkSemaphoreSubmitInfo> waits(submit->waitFenceNum);
            for (uint32_t i = 0; i < submit->waitFenceNum; ++i)
            {
                const VriFenceSubmitDesc& w = submit->waitFences[i];
                waits[i]                    = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
                waits[i].semaphore          = Fen(w.fence)->timeline;
                waits[i].value              = w.value;
                waits[i].stageMask          = ToVkStages(w.stages ? w.stages : VriPipelineStage_AllCommands);
            }

            std::vector<VkCommandBufferSubmitInfo> cmds(submit->commandBufferNum);
            for (uint32_t i = 0; i < submit->commandBufferNum; ++i)
            {
                cmds[i]               = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
                cmds[i].commandBuffer = CB(submit->commandBuffers[i])->cmd;
            }

            std::vector<VkSemaphoreSubmitInfo> signals(submit->signalFenceNum);
            for (uint32_t i = 0; i < submit->signalFenceNum; ++i)
            {
                const VriFenceSubmitDesc& s = submit->signalFences[i];
                signals[i]                  = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
                signals[i].semaphore        = Fen(s.fence)->timeline;
                signals[i].value            = s.value;
                signals[i].stageMask        = ToVkStages(s.stages ? s.stages : VriPipelineStage_AllCommands);
            }

            VkSubmitInfo2 si            = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
            si.waitSemaphoreInfoCount   = static_cast<uint32_t>(waits.size());
            si.pWaitSemaphoreInfos      = waits.data();
            si.commandBufferInfoCount   = static_cast<uint32_t>(cmds.size());
            si.pCommandBufferInfos      = cmds.data();
            si.signalSemaphoreInfoCount = static_cast<uint32_t>(signals.size());
            si.pSignalSemaphoreInfos    = signals.data();

            q->device->Ext().QueueSubmit2(q->queue, 1, &si, VK_NULL_HANDLE);
        }

        void VRI_CALL QueueWaitIdle(VriQueue* queue) { vkQueueWaitIdle(Q(queue)->queue); }
        void VRI_CALL DeviceWaitIdle(VriDevice* device) { vkDeviceWaitIdle(Dev(device)->Device()); }
        void VRI_CALL SetDebugName(void*, const char*) {}

        const VriCoreInterface g_coreVK = {
            GetDeviceDesc,
            GetFormatSupport,
            GetQueue,
            CreateCommandAllocator,
            ResetCommandAllocator,
            DestroyCommandAllocator,
            CreateCommandBuffer,
            BeginCommandBuffer,
            EndCommandBuffer,
            CreateBuffer,
            DestroyBuffer,
            MapBuffer,
            UnmapBuffer,
            GetBufferDeviceAddress,
            CreateTexture,
            DestroyTexture,
            GetBufferMemoryDesc,
            GetTextureMemoryDesc,
            AllocateMemory,
            FreeMemory,
            BindBufferMemory,
            BindTextureMemory,
            CreateBufferView,
            CreateTextureView,
            CreateSampler,
            DestroyDescriptor,
            CreatePipelineLayout,
            DestroyPipelineLayout,
            CreateGraphicsPipeline,
            CreateComputePipeline,
            DestroyPipeline,
            CreateDescriptorPool,
            ResetDescriptorPool,
            DestroyDescriptorPool,
            AllocateDescriptorSets,
            UpdateDescriptorRanges,
            CreateFence,
            DestroyFence,
            GetFenceValue,
            Wait,
            CmdBeginRendering,
            CmdEndRendering,
            CmdSetViewports,
            CmdSetScissors,
            CmdSetPipelineLayout,
            CmdSetPipeline,
            CmdSetDescriptorSet,
            CmdSetConstants,
            CmdSetVertexBuffers,
            CmdSetIndexBuffer,
            CmdDraw,
            CmdDrawIndexed,
            CmdDrawIndirect,
            CmdDrawIndexedIndirect,
            CmdDrawIndirectCount,
            CmdDrawIndexedIndirectCount,
            CmdDispatch,
            CmdDispatchIndirect,
            CmdBarrier,
            CmdCopyBuffer,
            CmdCopyTexture,
            CmdUploadBufferToTexture,
            CmdReadbackTextureToBuffer,
            CmdBeginDebugGroup,
            CmdEndDebugGroup,
            QueueSubmit,
            QueueWaitIdle,
            DeviceWaitIdle,
            SetDebugName,
            GetVideoMemoryInfo,
            CmdClearStorageBuffer,
            CmdClearStorageTexture,
        };
    } // namespace

    const VriCoreInterface* GetCoreInterfaceVK() { return &g_coreVK; }
} // namespace vri::vk
