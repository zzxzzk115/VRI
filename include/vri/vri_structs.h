/*
 * vri_structs.h - POD descriptor structs used by the entry points and the
 * core interface. Phase-0 stub: creation/allocation structs only.
 */
#ifndef VRI_STRUCTS_H
#define VRI_STRUCTS_H

#include "vri_base.h"
#include "vri_enums.h"

VRI_EXTERN_C_BEGIN

/* Optional host allocation callbacks (NRI-style). NULL => default malloc/free. */
typedef struct VriAllocationCallbacks
{
    void* userArg;
    void* (VRI_CALL *Allocate)(void* userArg, size_t size, size_t alignment);
    void* (VRI_CALL *Reallocate)(void* userArg, void* ptr, size_t size, size_t alignment);
    void  (VRI_CALL *Free)(void* userArg, void* ptr);
} VriAllocationCallbacks;

typedef enum VriMessageSeverity
{
    VriMessageSeverity_Info = 0,
    VriMessageSeverity_Warning,
    VriMessageSeverity_Error,
    VriMessageSeverity_MaxEnum = 0x7fffffff
} VriMessageSeverity;

typedef struct VriCallbackInterface
{
    void* userArg;
    void  (VRI_CALL *MessageCallback)(void* userArg, VriMessageSeverity severity, const char* message);
} VriCallbackInterface;

/* Parameters for vriCreateDevice. */
typedef struct VriDeviceCreationDesc
{
    VriGraphicsAPI                graphicsAPI;
    uint32_t                      adapterIndex;     /* index into vriEnumerateAdapters */
    VriBool                       enableValidation;

    /* bitset of optional features to enable at creation (see VriFeatureBits). */
    uint64_t                      enabledFeatures;
    /* if VRI_TRUE: proceed with whatever subset is granted; else fail hard. */
    VriBool                       bestEffort;

    /* Extra native-API extensions to require at instance/device creation.
       Required by interop layers such as OpenXR, which dictate the extension
       set (xrGetVulkanInstanceExtensionsKHR / xrGetVulkanDeviceExtensionsKHR)
       and the physical device to use. Ignored by backends without extensions. */
    const char* const*            requiredInstanceExtensions;
    uint32_t                      requiredInstanceExtensionNum;
    const char* const*            requiredDeviceExtensions;
    uint32_t                      requiredDeviceExtensionNum;

    /* Optional backend-specific creation chain, e.g. importing an existing
       native instance/device/adapter for OpenXR or another host. NULL for
       normal creation; cast per backend (see ext/vri_ext_interop.h). */
    const void*                   nativeCreateInfo;

    /* Optional native display connection shared between the device's GL/EGL context
       and window presentation. Required by the native OpenGL ES backend on Wayland:
       pass the app's wl_display*, since EGL needs the same connection for the context
       and the window surface. NULL (the default) -> the backend picks a default /
       headless display. Ignored by backends that don't need it (X11 window IDs, and
       Vulkan/D3D12/Metal/desktop-GL, work without it). */
    const void*                   nativeDisplay;

    const VriAllocationCallbacks* allocationCallbacks; /* optional */
    const VriCallbackInterface*   callbackInterface;   /* optional */
} VriDeviceCreationDesc;

/* Optional features requestable at device creation. */
typedef enum VriFeatureBits
{
    VriFeature_None         = 0,
    VriFeature_RayTracing   = 1ull << 0,
    VriFeature_MeshShader   = 1ull << 1,
    VriFeature_LowLatency   = 1ull << 2,
    VriFeature_Bindless     = 1ull << 3,
    VriFeature_VariableShadingRate = 1ull << 4,
    VriFeature_OpacityMicromap     = 1ull << 5,
    VriFeature_RayQuery            = 1ull << 6, /* inline ray tracing (no RT pipeline/SBT) */
    VriFeature_ExternalMemory      = 1ull << 7  /* external memory/semaphore export (CUDA/OptiX/cross-API/-process interop) */
} VriFeatureBits;

VRI_EXTERN_C_END

#endif /* VRI_STRUCTS_H */
