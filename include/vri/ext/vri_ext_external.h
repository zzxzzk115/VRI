/*
 * vri_ext_external.h - external memory & semaphore export ("CUDA interop seam").
 *
 * Queried via vriGetInterface(device, VRI_INTERFACE_EXTERNAL, ...). Registered only by
 * backends that support it (Vulkan VK_KHR_external_memory_{win32,fd} +
 * VK_KHR_external_semaphore_{win32,fd}). Enable with VriFeature_ExternalMemory at device
 * creation and check VriDeviceDesc::hasExternalMemory.
 *
 * VRI does NOT depend on CUDA. This interface exports an OS handle (Win32 HANDLE / POSIX
 * fd) for a buffer/texture's backing memory and for a timeline fence, which the
 * application then imports into CUDA (cudaImportExternalMemory / cudaImportExternalSemaphore),
 * OptiX, NVENC, another process, or another graphics API. The same memory/timeline is then
 * shared, so a CUDA kernel can read/write a VRI buffer and synchronize against the graphics
 * queue.
 *
 * Exportable resources must be DEDICATED allocations (one VkDeviceMemory per resource), so
 * they are created through this interface's own create-functions rather than the core
 * CreateBuffer/CreateTexture (which pool via VMA and cannot be exported). Destroy them with
 * the core DestroyBuffer/DestroyTexture/DestroyFence as usual.
 *
 * Handle ownership: GetBufferMemoryHandle/GetTextureMemoryHandle/GetFenceHandle return a
 * handle the CALLER owns and must release after importing it (CloseHandle on Win32, close()
 * on the fd). Vulkan duplicates on each query, so the exported handle's lifetime is
 * independent of the VriBuffer/VriTexture/VriFence (destroying those frees the underlying
 * VkDeviceMemory/VkSemaphore but not an already-exported handle).
 */
#ifndef VRI_EXT_EXTERNAL_H
#define VRI_EXT_EXTERNAL_H

#include "../vri_base.h"
#include "../vri_handles.h"
#include "../vri_resource.h"

VRI_EXTERN_C_BEGIN

/* OS handle flavor to export. Mirrors VK_EXTERNAL_*_HANDLE_TYPE_OPAQUE_* and maps to
 * CUDA's cudaExternalMemoryHandleType / cudaExternalSemaphoreHandleType. */
typedef enum VriExternalHandleType
{
    VriExternalHandleType_OpaqueWin32 = 1, /* Win32 HANDLE (default on Windows) */
    VriExternalHandleType_OpaqueFd    = 2, /* POSIX fd (default on Linux) */
    VriExternalHandleType_MaxEnum     = 0x7fffffff
} VriExternalHandleType;

/* Everything cudaExternalMemoryHandleDesc needs to import the exported allocation. */
typedef struct VriExternalMemoryInfo
{
    void*    handle;    /* Win32 HANDLE (caller CloseHandle) or fd (caller close()) */
    uint64_t size;      /* total backing allocation size -> cudaExternalMemoryHandleDesc.size */
    VriBool  dedicated; /* whether the allocation is dedicated -> CUDA DEDICATED flag (always true here) */
} VriExternalMemoryInfo;

typedef struct VriExternalInterface
{
    /* Create an exportable buffer/texture (dedicated, non-pooled allocation). The desc's
     * memoryLocation is honored for the heap; handleType picks the exportable flavor. */
    VriResult (VRI_CALL *CreateExportableBuffer)(VriDevice* device, const VriBufferDesc* desc,
                                                 VriExternalHandleType handleType, VriBuffer** outBuffer);
    VriResult (VRI_CALL *CreateExportableTexture)(VriDevice* device, const VriTextureDesc* desc,
                                                  VriExternalHandleType handleType, VriTexture** outTexture);

    /* Export the OS handle + sizing for an exportable buffer/texture's backing memory. */
    VriResult (VRI_CALL *GetBufferMemoryHandle)(VriDevice* device, VriBuffer* buffer,
                                                VriExternalHandleType handleType, VriExternalMemoryInfo* out);
    VriResult (VRI_CALL *GetTextureMemoryHandle)(VriDevice* device, VriTexture* texture,
                                                 VriExternalHandleType handleType, VriExternalMemoryInfo* out);

    /* Create an exportable timeline fence (for CUDA <-> graphics-queue synchronization). */
    VriResult (VRI_CALL *CreateExportableFence)(VriDevice* device, uint64_t initialValue,
                                                VriExternalHandleType handleType, VriFence** outFence);
    /* Export the OS handle for an exportable fence's timeline semaphore. */
    VriResult (VRI_CALL *GetFenceHandle)(VriDevice* device, VriFence* fence,
                                         VriExternalHandleType handleType, void** outHandle);
} VriExternalInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_EXTERNAL_H */
