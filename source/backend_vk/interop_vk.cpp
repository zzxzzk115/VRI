// interop_vk.cpp - VriInteropInterface for Vulkan: expose native handles and
// wrap externally-created textures (e.g. OpenXR swapchain images).

#include "interop_vk.h"
#include "conversions_vk.h"
#include "device_vk.h"
#include "objects_vk.h"

namespace vri::vk
{
    namespace
    {
        inline DeviceVK*       Dev(VriDevice* h)        { return reinterpret_cast<DeviceVK*>(h); }
        inline const DeviceVK* Dev(const VriDevice* h)  { return reinterpret_cast<const DeviceVK*>(h); }

        VriResult VRI_CALL GetDeviceNativeHandles(const VriDevice* device, VriDeviceNativeHandles* out)
        {
            if (!out) return VriResult_InvalidArgument;
            DeviceVK* d = const_cast<DeviceVK*>(Dev(device));
            const QueueVK* gfx = d->GetQueue(VriQueueType_Graphics);

            *out = VriDeviceNativeHandles{};
            out->api = VriGraphicsAPI_Vulkan;
            out->u.vulkan.instance = d->Instance();
            out->u.vulkan.physicalDevice = d->PhysicalDevice();
            out->u.vulkan.device = d->Device();
            out->u.vulkan.graphicsQueue = gfx->queue;
            out->u.vulkan.graphicsQueueFamilyIndex = gfx->familyIndex;
            out->u.vulkan.graphicsQueueIndex = gfx->indexInFamily;
            return VriResult_Success;
        }

        void* VRI_CALL GetQueueNativeHandle(const VriQueue* queue)
        {
            return reinterpret_cast<const QueueVK*>(queue)->queue;
        }

        void* VRI_CALL GetTextureNativeHandle(const VriTexture* texture)
        {
            return reinterpret_cast<const TextureVK*>(texture)->image;
        }

        void* VRI_CALL GetBufferNativeHandle(const VriBuffer* buffer)
        {
            return reinterpret_cast<const BufferVK*>(buffer)->buffer;
        }

        VriResult VRI_CALL WrapTexture(VriDevice* device, const VriWrapTextureDesc* desc, VriTexture** out)
        {
            if (!desc || !desc->nativeTexture) return VriResult_InvalidArgument;
            DeviceVK* d = Dev(device);

            TextureVK* t = new TextureVK{};
            t->device = d;
            t->image = static_cast<VkImage>(desc->nativeTexture);
            t->allocation = VK_NULL_HANDLE;
            t->format = ToVkFormat(desc->desc.format);
            t->extent = {desc->desc.width, desc->desc.height ? desc->desc.height : 1u, desc->desc.depth ? desc->desc.depth : 1u};
            t->mipNum = desc->desc.mipNum ? desc->desc.mipNum : 1u;
            t->layerNum = desc->desc.layerNum ? desc->desc.layerNum : 1u;
            t->type = ToVkImageType(desc->desc.type);
            t->owned = false; // borrowed: DestroyTexture won't free it
            *out = ToHandle(t);
            return VriResult_Success;
        }

        const VriInteropInterface g_interopVK = {
            GetDeviceNativeHandles,
            GetQueueNativeHandle,
            GetTextureNativeHandle,
            GetBufferNativeHandle,
            WrapTexture,
        };
    } // namespace

    const VriInteropInterface* GetInteropInterfaceVK() { return &g_interopVK; }
} // namespace vri::vk
