/*
 * vri_ext_interop.h - native-handle interop, queried via
 * vriGetInterface(device, VRI_INTERFACE_INTEROP, ...).
 *
 * This is the seam for layers that must share the underlying API objects with
 * VRI -- most importantly OpenXR, which needs the native instance/physical
 * device/device/queue to build its graphics binding, and whose swapchain
 * images (native textures) must be wrapped back into VriTexture for rendering.
 */
#ifndef VRI_EXT_INTEROP_H
#define VRI_EXT_INTEROP_H

#include "../vri_base.h"
#include "../vri_handles.h"
#include "../vri_enums.h"
#include "../vri_resource.h"

VRI_EXTERN_C_BEGIN

/* Native device-level objects, tagged by graphics API. void* avoids leaking
 * native headers into this C-clean header; cast to the real types per API. */
typedef struct VriDeviceNativeHandles
{
    VriGraphicsAPI api;
    union
    {
        struct
        {
            void*    instance;          /* VkInstance */
            void*    physicalDevice;    /* VkPhysicalDevice */
            void*    device;            /* VkDevice */
            void*    graphicsQueue;     /* VkQueue */
            uint32_t graphicsQueueFamilyIndex;
            uint32_t graphicsQueueIndex;
        } vulkan;
        struct { void* device; void* immediateContext; } d3d11;
        struct { void* device; void* commandQueue; }      d3d12;
        struct { void* device; void* commandQueue; }      metal;
        struct { void* device; void* adapter; }           webgpu;
    } u;
} VriDeviceNativeHandles;

/* Wrap an externally-created native texture (e.g. an OpenXR swapchain image)
 * as a VriTexture. The desc must describe the native object accurately. */
typedef struct VriWrapTextureDesc
{
    void*          nativeTexture; /* VkImage / ID3D11Texture2D* / id<MTLTexture> ... */
    VriTextureDesc desc;
} VriWrapTextureDesc;

/* Vulkan instance/device creation hooks, for OpenXR's XR_KHR_vulkan_enable2 (required by runtimes
 * such as the Meta XR Simulator, which must interpose Vulkan creation). Point
 * VriDeviceCreationDesc::nativeCreateInfo at this when graphicsAPI == Vulkan. VRI still builds the
 * VkInstanceCreateInfo / VkDeviceCreateInfo itself -- so it knows exactly which extensions and
 * features it enabled -- but routes the actual create call through these hooks, which forward to
 * xrCreateVulkanInstanceKHR / xrCreateVulkanDeviceKHR. void* keeps this header free of the Vulkan
 * headers; the callee casts. Each hook returns a VkResult (0 == VK_SUCCESS).
 *
 * Ordering note: VRI creates the instance first, then reads `physicalDevice`. A hook implementation
 * that learns the device from OpenXR (xrGetVulkanGraphicsDevice2KHR, which needs the VkInstance)
 * should fill `physicalDevice` from inside `createInstance` before returning. NULL = VRI picks. */
typedef struct VriVulkanCreateHooks
{
    VriGraphicsAPI api; /* must be VriGraphicsAPI_Vulkan */
    /* (userData, const VkInstanceCreateInfo* createInfo, VkInstance* outInstance) -> VkResult */
    int32_t (VRI_CALL *createInstance)(void* userData, const void* createInfo, void* outInstance);
    /* (userData, VkPhysicalDevice, const VkDeviceCreateInfo* createInfo, VkDevice* outDevice) -> VkResult */
    int32_t (VRI_CALL *createDevice)(void* userData, void* physicalDevice, const void* createInfo, void* outDevice);
    void* userData;
    void* physicalDevice; /* VkPhysicalDevice OpenXR mandates (xrGetVulkanGraphicsDevice2KHR); NULL = VRI picks */
} VriVulkanCreateHooks;

typedef struct VriInteropInterface
{
    VriResult (VRI_CALL *GetDeviceNativeHandles)(const VriDevice* device, VriDeviceNativeHandles* out);
    void*     (VRI_CALL *GetQueueNativeHandle)(const VriQueue* queue);
    void*     (VRI_CALL *GetTextureNativeHandle)(const VriTexture* texture);
    void*     (VRI_CALL *GetBufferNativeHandle)(const VriBuffer* buffer);

    /* Create a VriTexture that borrows a native texture (does not own it). */
    VriResult (VRI_CALL *WrapTexture)(VriDevice* device, const VriWrapTextureDesc* desc, VriTexture** outTexture);
} VriInteropInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_INTEROP_H */
