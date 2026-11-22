// external_vk.cpp - VriExternalInterface for Vulkan: export OS handles for buffer/texture
// backing memory and for timeline fences, so an external consumer (CUDA, OptiX, NVENC,
// another process / graphics API) can import the same memory and timeline. VRI itself does
// NOT depend on CUDA -- it only produces the handles the consumer imports.
//
// Win32 vs fd: this TU owns the platform-specific Vulkan external-handle code (defining
// VK_USE_PLATFORM_WIN32_KHR before including Vulkan, mirroring swapchain_vk.cpp) so the
// win32 types never leak into the broadly-included device_vk.h. Device-level export entry
// points are resolved via vkGetDeviceProcAddr (the loader does not export them statically).

#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "external_vk.h"
#include "conversions_vk.h"
#include "device_vk.h"
#include "objects_vk.h"

#include "core/resource_desc_checks.h"

#include <cstdio>

namespace vri::vk
{
    namespace
    {
        inline DeviceVK*  Dev(VriDevice* h) { return reinterpret_cast<DeviceVK*>(h); }
        inline BufferVK*  Buf(VriBuffer* h) { return reinterpret_cast<BufferVK*>(h); }
        inline TextureVK* Tex(VriTexture* h) { return reinterpret_cast<TextureVK*>(h); }
        inline FenceVK*   Fen(VriFence* h) { return reinterpret_cast<FenceVK*>(h); }

        // The OPAQUE_* handle types are platform-bound: Win32 on Windows, fd elsewhere.
        // Reject a request that does not match the platform the backend was built for.
        bool HandleTypeOk(VriExternalHandleType t)
        {
#if defined(_WIN32)
            return t == VriExternalHandleType_OpaqueWin32;
#else
            return t == VriExternalHandleType_OpaqueFd;
#endif
        }

        VkExternalMemoryHandleTypeFlagBits MemHandleBit()
        {
#if defined(_WIN32)
            return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
            return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
        }

        VkExternalSemaphoreHandleTypeFlagBits SemHandleBit()
        {
#if defined(_WIN32)
            return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
            return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
        }

        const char* HandleName()
        {
#if defined(_WIN32)
            return "OPAQUE_WIN32";
#else
            return "OPAQUE_FD";
#endif
        }

        // Whether a buffer of `usage` can be exported with the platform's handle type.
        bool BufferExportable(const VolkInstanceTable& vki, VkPhysicalDevice pd, VkBufferUsageFlags usage)
        {
            VkPhysicalDeviceExternalBufferInfo info = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
            info.usage                              = usage;
            info.handleType                         = MemHandleBit();
            VkExternalBufferProperties properties   = {VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
            vki.vkGetPhysicalDeviceExternalBufferProperties(pd, &info, &properties);
            return (properties.externalMemoryProperties.externalMemoryFeatures &
                    VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0;
        }

        // Whether an image created with `ici` can be exported with the platform's handle type.
        bool ImageExportable(const VolkInstanceTable& vki, VkPhysicalDevice pd, const VkImageCreateInfo& ici)
        {
            VkPhysicalDeviceExternalImageFormatInfo external = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
            external.handleType                        = MemHandleBit();
            VkPhysicalDeviceImageFormatInfo2 info      = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
            info.pNext                                 = &external;
            info.format                                = ici.format;
            info.type                                  = ici.imageType;
            info.tiling                                = ici.tiling;
            info.usage                                 = ici.usage;
            info.flags                                 = ici.flags;
            VkExternalImageFormatProperties exportable = {VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
            VkImageFormatProperties2        properties = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
            properties.pNext                           = &exportable;
            if (vki.vkGetPhysicalDeviceImageFormatProperties2(pd, &info, &properties) != VK_SUCCESS)
                return false;
            return (exportable.externalMemoryProperties.externalMemoryFeatures &
                    VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0;
        }

        // Desired memory property flags for a memory location. Exportable resources are
        // typically device-local (the common CUDA-interop case); host-visible locations are
        // honored for staging-style sharing.
        VkMemoryPropertyFlags PropsForLocation(VriMemoryLocation loc)
        {
            switch (loc)
            {
                case VriMemoryLocation_HostUpload:
                case VriMemoryLocation_HostReadback:
                    return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                case VriMemoryLocation_DeviceUpload:
                    return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
                case VriMemoryLocation_Device:
                case VriMemoryLocation_Undefined:
                default:
                    return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            }
        }

        // Find a memory type satisfying typeBits + required props; fall back to DEVICE_LOCAL
        // then to any matching type so a missing exact match still allocates.
        uint32_t FindMemoryType(DeviceVK* d, uint32_t typeBits, VkMemoryPropertyFlags required)
        {
            VkPhysicalDeviceMemoryProperties mp;
            d->Vki().vkGetPhysicalDeviceMemoryProperties(d->PhysicalDevice(), &mp);
            for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
                if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & required) == required)
                    return i;
            for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
                if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                    return i;
            for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
                if (typeBits & (1u << i))
                    return i;
            return 0;
        }

        VriResult VRI_CALL CreateExportableBuffer(VriDevice*            device,
                                                  const VriBufferDesc*  desc,
                                                  VriExternalHandleType handleType,
                                                  VriBuffer**           out)
        {
            if (!desc || !out || !HandleTypeOk(handleType))
                return VriResult_InvalidArgument;
            DeviceVK* d = Dev(device);
            if (core::RejectDesc(*d, core::CheckBufferDesc("CreateExportableBuffer", *desc)))
            {
                *out = nullptr;
                return VriResult_InvalidArgument;
            }

            VkExternalMemoryBufferCreateInfo extBci = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
            extBci.handleTypes                      = MemHandleBit();

            const VkBufferUsageFlags vkUsage = ToVkBufferUsage(desc->usage);
            if (!BufferExportable(d->Vki(), d->PhysicalDevice(), vkUsage))
            {
                char msg[160];
                std::snprintf(msg,
                              sizeof(msg),
                              "CreateExportableBuffer: this device cannot export a buffer of usage 0x%x as an %s "
                              "handle",
                              static_cast<unsigned>(desc->usage),
                              HandleName());
                d->ReportError(msg);
                *out = nullptr;
                return VriResult_Unsupported;
            }
            VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.pNext              = &extBci;
            bci.size               = desc->size;
            bci.usage              = vkUsage;
            bci.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

            VkBuffer buffer = VK_NULL_HANDLE;
            if (d->Vkd().vkCreateBuffer(d->Device(), &bci, nullptr, &buffer) != VK_SUCCESS)
                return VriResult_Failure;

            VkMemoryRequirements req = {};
            d->Vkd().vkGetBufferMemoryRequirements(d->Device(), buffer, &req);

            // Chain: alloc <- exportInfo <- dedicatedInfo [<- flagsInfo if device address].
            VkExportMemoryAllocateInfo exportInfo   = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
            exportInfo.handleTypes                  = MemHandleBit();
            VkMemoryDedicatedAllocateInfo dedicated = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
            dedicated.buffer                        = buffer;
            dedicated.pNext                         = &exportInfo;
            VkMemoryAllocateFlagsInfo flagsInfo     = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            if (vkUsage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
            {
                flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
                flagsInfo.pNext = &dedicated;
            }

            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.pNext          = (vkUsage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ? static_cast<void*>(&flagsInfo) :
                                                                                        static_cast<void*>(&dedicated);
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, PropsForLocation(desc->memoryLocation));

            VkDeviceMemory memory = VK_NULL_HANDLE;
            if (d->Vkd().vkAllocateMemory(d->Device(), &ai, nullptr, &memory) != VK_SUCCESS)
            {
                d->Vkd().vkDestroyBuffer(d->Device(), buffer, nullptr);
                return VriResult_OutOfMemory;
            }
            if (d->Vkd().vkBindBufferMemory(d->Device(), buffer, memory, 0) != VK_SUCCESS)
            {
                d->Vkd().vkFreeMemory(d->Device(), memory, nullptr);
                d->Vkd().vkDestroyBuffer(d->Device(), buffer, nullptr);
                return VriResult_Failure;
            }

            BufferVK* b            = new BufferVK {d, buffer, VK_NULL_HANDLE, desc->size};
            b->dedicatedMemory     = memory;
            b->dedicatedMemorySize = req.size;
            *out                   = ToHandle(b);
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateExportableTexture(VriDevice*            device,
                                                   const VriTextureDesc* desc,
                                                   VriExternalHandleType handleType,
                                                   VriTexture**          out)
        {
            if (!desc || !out || !HandleTypeOk(handleType))
                return VriResult_InvalidArgument;
            DeviceVK* d = Dev(device);
            if (core::RejectDesc(*d, core::CheckTextureDesc("CreateExportableTexture", *desc)))
            {
                *out = nullptr;
                return VriResult_InvalidArgument;
            }

            VkExternalMemoryImageCreateInfo extIci = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
            extIci.handleTypes                     = MemHandleBit();

            VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.pNext             = &extIci;
            ici.imageType         = ToVkImageType(desc->type);
            ici.format            = ToVkFormat(desc->format);
            ici.extent            = {desc->width, desc->height, desc->depth};
            ici.mipLevels         = desc->mipNum;
            ici.arrayLayers       = desc->layerNum;
            ici.samples           = static_cast<VkSampleCountFlagBits>(desc->sampleNum);
            ici.tiling            = VK_IMAGE_TILING_OPTIMAL;
            ici.usage             = ToVkImageUsage(desc->usage);
            ici.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
            if (desc->type == VriTextureType_Cube || desc->type == VriTextureType_CubeArray)
                ici.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
            if (!ImageExportable(d->Vki(), d->PhysicalDevice(), ici))
            {
                char msg[192];
                std::snprintf(msg,
                              sizeof(msg),
                              "CreateExportableTexture: this device cannot export a texture of VriFormat %d, type %d "
                              "and usage 0x%x as an %s handle",
                              static_cast<int>(desc->format),
                              static_cast<int>(desc->type),
                              static_cast<unsigned>(desc->usage),
                              HandleName());
                d->ReportError(msg);
                *out = nullptr;
                return VriResult_Unsupported;
            }

            VkImage image = VK_NULL_HANDLE;
            if (d->Vkd().vkCreateImage(d->Device(), &ici, nullptr, &image) != VK_SUCCESS)
                return VriResult_Failure;

            VkMemoryRequirements req = {};
            d->Vkd().vkGetImageMemoryRequirements(d->Device(), image, &req);

            VkExportMemoryAllocateInfo exportInfo   = {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
            exportInfo.handleTypes                  = MemHandleBit();
            VkMemoryDedicatedAllocateInfo dedicated = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
            dedicated.image                         = image;
            dedicated.pNext                         = &exportInfo;

            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.pNext                = &dedicated;
            ai.allocationSize       = req.size;
            ai.memoryTypeIndex      = FindMemoryType(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

            VkDeviceMemory memory = VK_NULL_HANDLE;
            if (d->Vkd().vkAllocateMemory(d->Device(), &ai, nullptr, &memory) != VK_SUCCESS)
            {
                d->Vkd().vkDestroyImage(d->Device(), image, nullptr);
                return VriResult_OutOfMemory;
            }
            if (d->Vkd().vkBindImageMemory(d->Device(), image, memory, 0) != VK_SUCCESS)
            {
                d->Vkd().vkFreeMemory(d->Device(), memory, nullptr);
                d->Vkd().vkDestroyImage(d->Device(), image, nullptr);
                return VriResult_Failure;
            }

            TextureVK* t           = new TextureVK {};
            t->device              = d;
            t->image               = image;
            t->allocation          = VK_NULL_HANDLE;
            t->format              = ici.format;
            t->extent              = ici.extent;
            t->mipNum              = ici.mipLevels;
            t->layerNum            = ici.arrayLayers;
            t->type                = ici.imageType;
            t->owned               = true;
            t->dedicatedMemory     = memory;
            t->dedicatedMemorySize = req.size;
            *out                   = ToHandle(t);
            return VriResult_Success;
        }

        // Export the OS handle for a dedicated VkDeviceMemory.
        VriResult ExportMemory(DeviceVK*              d,
                               VkDeviceMemory         memory,
                               uint64_t               size,
                               VriExternalHandleType  handleType,
                               VriExternalMemoryInfo* out)
        {
            if (!memory || !out || !HandleTypeOk(handleType))
                return VriResult_InvalidArgument;
#if defined(_WIN32)
            auto getWin32 = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
                d->Vki().vkGetDeviceProcAddr(d->Device(), "vkGetMemoryWin32HandleKHR"));
            if (!getWin32)
                return VriResult_Unsupported;
            VkMemoryGetWin32HandleInfoKHR gi = {VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
            gi.memory                        = memory;
            gi.handleType                    = MemHandleBit();
            HANDLE h                         = nullptr;
            if (getWin32(d->Device(), &gi, &h) != VK_SUCCESS)
                return VriResult_Failure;
            out->handle = h;
#else
            auto getFd =
                reinterpret_cast<PFN_vkGetMemoryFdKHR>(d->Vki().vkGetDeviceProcAddr(d->Device(), "vkGetMemoryFdKHR"));
            if (!getFd)
                return VriResult_Unsupported;
            VkMemoryGetFdInfoKHR gi = {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
            gi.memory               = memory;
            gi.handleType           = MemHandleBit();
            int fd                  = -1;
            if (getFd(d->Device(), &gi, &fd) != VK_SUCCESS)
                return VriResult_Failure;
            out->handle = reinterpret_cast<void*>(static_cast<intptr_t>(fd));
#endif
            out->size      = size;
            out->dedicated = VRI_TRUE;
            return VriResult_Success;
        }

        VriResult VRI_CALL GetBufferMemoryHandle(VriDevice*             device,
                                                 VriBuffer*             buffer,
                                                 VriExternalHandleType  handleType,
                                                 VriExternalMemoryInfo* out)
        {
            if (!buffer)
                return VriResult_InvalidArgument;
            BufferVK* b = Buf(buffer);
            return ExportMemory(Dev(device), b->dedicatedMemory, b->dedicatedMemorySize, handleType, out);
        }

        VriResult VRI_CALL GetTextureMemoryHandle(VriDevice*             device,
                                                  VriTexture*            texture,
                                                  VriExternalHandleType  handleType,
                                                  VriExternalMemoryInfo* out)
        {
            if (!texture)
                return VriResult_InvalidArgument;
            TextureVK* t = Tex(texture);
            return ExportMemory(Dev(device), t->dedicatedMemory, t->dedicatedMemorySize, handleType, out);
        }

        VriResult VRI_CALL CreateExportableFence(VriDevice*            device,
                                                 uint64_t              initialValue,
                                                 VriExternalHandleType handleType,
                                                 VriFence**            out)
        {
            if (!out || !HandleTypeOk(handleType))
                return VriResult_InvalidArgument;
            DeviceVK* d = Dev(device);

            VkExportSemaphoreCreateInfo exportInfo = {VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
            exportInfo.handleTypes                 = SemHandleBit();
            VkSemaphoreTypeCreateInfo type         = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            type.semaphoreType                     = VK_SEMAPHORE_TYPE_TIMELINE;
            type.initialValue                      = initialValue;
            type.pNext                             = &exportInfo;
            VkSemaphoreCreateInfo ci               = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            ci.pNext                               = &type;

            VkSemaphore sem = VK_NULL_HANDLE;
            if (d->Vkd().vkCreateSemaphore(d->Device(), &ci, nullptr, &sem) != VK_SUCCESS)
                return VriResult_Failure;
            *out = ToHandle(new FenceVK {d, sem});
            return VriResult_Success;
        }

        VriResult VRI_CALL GetFenceHandle(VriDevice*            device,
                                          VriFence*             fence,
                                          VriExternalHandleType handleType,
                                          void**                outHandle)
        {
            if (!fence || !outHandle || !HandleTypeOk(handleType))
                return VriResult_InvalidArgument;
            DeviceVK* d = Dev(device);
            FenceVK*  f = Fen(fence);
#if defined(_WIN32)
            auto getWin32 = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
                d->Vki().vkGetDeviceProcAddr(d->Device(), "vkGetSemaphoreWin32HandleKHR"));
            if (!getWin32)
                return VriResult_Unsupported;
            VkSemaphoreGetWin32HandleInfoKHR gi = {VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
            gi.semaphore                        = f->timeline;
            gi.handleType                       = SemHandleBit();
            HANDLE h                            = nullptr;
            if (getWin32(d->Device(), &gi, &h) != VK_SUCCESS)
                return VriResult_Failure;
            *outHandle = h;
#else
            auto getFd = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
                d->Vki().vkGetDeviceProcAddr(d->Device(), "vkGetSemaphoreFdKHR"));
            if (!getFd)
                return VriResult_Unsupported;
            VkSemaphoreGetFdInfoKHR gi = {VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
            gi.semaphore               = f->timeline;
            gi.handleType              = SemHandleBit();
            int fd                     = -1;
            if (getFd(d->Device(), &gi, &fd) != VK_SUCCESS)
                return VriResult_Failure;
            *outHandle = reinterpret_cast<void*>(static_cast<intptr_t>(fd));
#endif
            return VriResult_Success;
        }

        const VriExternalInterface g_externalVK = {
            CreateExportableBuffer,
            CreateExportableTexture,
            GetBufferMemoryHandle,
            GetTextureMemoryHandle,
            CreateExportableFence,
            GetFenceHandle,
        };
    } // namespace

    const VriExternalInterface* GetExternalInterfaceVK() { return &g_externalVK; }

    bool ExportsMemoryAndFences(const VolkInstanceTable& vki, VkPhysicalDevice physicalDevice)
    {
        VkSemaphoreTypeCreateInfo timeline         = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        timeline.semaphoreType                     = VK_SEMAPHORE_TYPE_TIMELINE;
        VkPhysicalDeviceExternalSemaphoreInfo info = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};
        info.pNext                                 = &timeline;
        info.handleType                            = SemHandleBit();
        VkExternalSemaphoreProperties semaphore    = {VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
        vki.vkGetPhysicalDeviceExternalSemaphoreProperties(physicalDevice, &info, &semaphore);
        if ((semaphore.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) == 0)
            return false;
        return BufferExportable(
            vki, physicalDevice, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }
} // namespace vri::vk
