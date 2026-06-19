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
