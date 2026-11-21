/*
 * vri_core.h - global entry points + the core function-table interface.
 *
 * Dispatch model (NRI-style, two stages):
 *   1. vriCreateDevice() instantiates a backend device behind an opaque handle.
 *   2. vriGetInterface() copies a pre-filled function table (the core interface
 *      or an optional extension interface) into a caller-provided struct.
 */
#ifndef VRI_CORE_H
#define VRI_CORE_H

#include "vri_base.h"
#include "vri_command.h"
#include "vri_descriptor.h"
#include "vri_device_desc.h"
#include "vri_enums.h"
#include "vri_flags.h"
#include "vri_format.h"
#include "vri_handles.h"
#include "vri_pipeline.h"
#include "vri_resource.h"
#include "vri_structs.h"
#include "vri_sync.h"

VRI_EXTERN_C_BEGIN

/* ---- global entry points ---------------------------------------------- */

VRI_API uint32_t VRI_CALL vriEnumerateAdapters(VriGraphicsAPI  api,
                                               VriAdapterDesc* outAdapters,
                                               uint32_t        adapterCapacity);

VRI_API VriResult VRI_CALL vriCreateDevice(const VriDeviceCreationDesc* desc, VriDevice** outDevice);

/*
 * Create a device and hand it to `callback` instead of returning it. The browser's WebGPU gives out
 * adapters and devices only from its event loop, so a program linked without -sASYNCIFY creates its
 * WebGPU device this way (vriCreateDevice reports VriResult_Unsupported for it there).
 *
 * `callback` runs exactly once: from the browser event loop while a browser WebGPU device is being
 * requested, otherwise before vriCreateDeviceAsync returns. `device` is NULL unless `result` is
 * VriResult_Success; the callback then owns it (vriDestroyDevice). VriGraphicsAPI_Auto goes through
 * the same backend order as vriCreateDevice, moving on when a WebGPU request fails. `*desc` is
 * copied; what it points to must stay valid until the callback has run. Returns
 * VriResult_InvalidArgument, without calling `callback`, when `desc` or `callback` is NULL.
 */
typedef void(VRI_CALL* VriDeviceCreatedCallback)(void* userData, VriResult result, VriDevice* device);

VRI_API VriResult VRI_CALL vriCreateDeviceAsync(const VriDeviceCreationDesc* desc,
                                                VriDeviceCreatedCallback     callback,
                                                void*                        userData);

VRI_API void VRI_CALL vriDestroyDevice(VriDevice* device);

/*
 * Copy a named interface's function table into `out`. `name` is one of the
 * VRI_INTERFACE_* identifiers; `size` must equal sizeof the target struct
 * (guards ABI drift). Returns VriResult_Unsupported if the device's backend
 * does not provide the interface.
 */
VRI_API VriResult VRI_CALL vriGetInterface(const VriDevice* device, const char* name, size_t size, void* out);

#define VRI_INTERFACE_CORE "VriCoreInterface"
#define VRI_INTERFACE_SWAPCHAIN "VriSwapChainInterface"
#define VRI_INTERFACE_INTEROP "VriInteropInterface"
#define VRI_INTERFACE_RAYTRACING "VriRayTracingInterface"
#define VRI_INTERFACE_MESHSHADER "VriMeshShaderInterface"
#define VRI_INTERFACE_VRS "VriShadingRateInterface"
#define VRI_INTERFACE_OMM "VriOpacityMicromapInterface"
#define VRI_INTERFACE_EXTERNAL "VriExternalInterface"
#define VRI_INTERFACE_COOPERATIVE_VECTOR "VriCooperativeVectorInterface"
#define VRI_INTERFACE_QUERY "VriQueryInterface"
#define VRI_INTERFACE_IMGUI "VriImguiInterface"
#define VRI_INTERFACE_PIPELINE_CACHE "VriPipelineCacheInterface"

/* ---- core interface (function table) ---------------------------------- */

/*
 * Receives the result of MapBufferAsync. `data` points at the mapped range (the buffer's `offset`)
 * when `result` is VriResult_Success and is NULL otherwise; it stays valid until UnmapBuffer.
 */
typedef void(VRI_CALL* VriBufferMappedCallback)(void* userData, VriResult result, void* data);

typedef struct VriCoreInterface
{
    /* queries */
    const VriDeviceDesc*(VRI_CALL* GetDeviceDesc)(const VriDevice* device);
    VriFormatSupportFlags(VRI_CALL* GetFormatSupport)(const VriDevice* device, VriFormat format);
    VriResult(VRI_CALL* GetQueue)(VriDevice* device, VriQueueType type, uint32_t index, VriQueue** outQueue);

    /* command allocation / recording lifecycle */
    VriResult(VRI_CALL* CreateCommandAllocator)(VriDevice*            device,
                                                VriQueueType          type,
                                                VriCommandAllocator** outAllocator);
    void(VRI_CALL* ResetCommandAllocator)(VriCommandAllocator* allocator);
    void(VRI_CALL* DestroyCommandAllocator)(VriCommandAllocator* allocator);
    VriResult(VRI_CALL* CreateCommandBuffer)(VriCommandAllocator* allocator, VriCommandBuffer** outCommandBuffer);
    VriResult(VRI_CALL* BeginCommandBuffer)(VriCommandBuffer* cmd);
    VriResult(VRI_CALL* EndCommandBuffer)(VriCommandBuffer* cmd);

    /* resources */
    VriResult(VRI_CALL* CreateBuffer)(VriDevice* device, const VriBufferDesc* desc, VriBuffer** outBuffer);
    void(VRI_CALL* DestroyBuffer)(VriBuffer* buffer);
    void*(VRI_CALL* MapBuffer)(VriBuffer* buffer, uint64_t offset, uint64_t size);
    void(VRI_CALL* UnmapBuffer)(VriBuffer* buffer);
    uint64_t(VRI_CALL* GetBufferDeviceAddress)(const VriBuffer* buffer);
    VriResult(VRI_CALL* CreateTexture)(VriDevice* device, const VriTextureDesc* desc, VriTexture** outTexture);
    void(VRI_CALL* DestroyTexture)(VriTexture* texture);

    /* explicit memory */
    void(VRI_CALL* GetBufferMemoryDesc)(const VriDevice*     device,
                                        const VriBufferDesc* desc,
                                        VriMemoryLocation    location,
                                        VriMemoryDesc*       outMemoryDesc);
    void(VRI_CALL* GetTextureMemoryDesc)(const VriDevice*      device,
                                         const VriTextureDesc* desc,
                                         VriMemoryLocation     location,
                                         VriMemoryDesc*        outMemoryDesc);
    VriResult(VRI_CALL* AllocateMemory)(VriDevice* device, const VriMemoryDesc* desc, VriMemory** outMemory);
    void(VRI_CALL* FreeMemory)(VriMemory* memory);
    VriResult(VRI_CALL* BindBufferMemory)(VriDevice* device, VriBuffer* buffer, VriMemory* memory, uint64_t offset);
    VriResult(VRI_CALL* BindTextureMemory)(VriDevice* device, VriTexture* texture, VriMemory* memory, uint64_t offset);

    /* views & samplers (each returns an opaque VriDescriptor) */
    VriResult(VRI_CALL* CreateBufferView)(VriDevice*               device,
                                          const VriBufferViewDesc* desc,
                                          VriDescriptor**          outDescriptor);
    VriResult(VRI_CALL* CreateTextureView)(VriDevice*                device,
                                           const VriTextureViewDesc* desc,
                                           VriDescriptor**           outDescriptor);
    VriResult(VRI_CALL* CreateSampler)(VriDevice* device, const VriSamplerDesc* desc, VriDescriptor** outDescriptor);
    void(VRI_CALL* DestroyDescriptor)(VriDescriptor* descriptor);

    /* pipeline layout & pipelines */
    VriResult(VRI_CALL* CreatePipelineLayout)(VriDevice*                   device,
                                              const VriPipelineLayoutDesc* desc,
                                              VriPipelineLayout**          outLayout);
    void(VRI_CALL* DestroyPipelineLayout)(VriPipelineLayout* layout);
    VriResult(VRI_CALL* CreateGraphicsPipeline)(VriDevice*                     device,
                                                const VriGraphicsPipelineDesc* desc,
                                                VriPipeline**                  outPipeline);
    VriResult(VRI_CALL* CreateComputePipeline)(VriDevice*                    device,
                                               const VriComputePipelineDesc* desc,
                                               VriPipeline**                 outPipeline);
    void(VRI_CALL* DestroyPipeline)(VriPipeline* pipeline);

    /* descriptor pools & sets */
    VriResult(VRI_CALL* CreateDescriptorPool)(VriDevice*                   device,
                                              const VriDescriptorPoolDesc* desc,
                                              VriDescriptorPool**          outPool);
    void(VRI_CALL* ResetDescriptorPool)(VriDescriptorPool* pool);
    void(VRI_CALL* DestroyDescriptorPool)(VriDescriptorPool* pool);
    VriResult(VRI_CALL* AllocateDescriptorSets)(VriDescriptorPool*       pool,
                                                const VriPipelineLayout* layout,
                                                uint32_t                 setIndex,
                                                VriDescriptorSet**       outSets,
                                                uint32_t                 setNum);
    void(VRI_CALL* UpdateDescriptorRanges)(VriDescriptorSet*                   set,
                                           uint32_t                            baseRange,
                                           uint32_t                            rangeNum,
                                           const VriDescriptorRangeUpdateDesc* updates);

    /* synchronization (timeline fences) */
    VriResult(VRI_CALL* CreateFence)(VriDevice* device, uint64_t initialValue, VriFence** outFence);
    void(VRI_CALL* DestroyFence)(VriFence* fence);
    uint64_t(VRI_CALL* GetFenceValue)(VriFence* fence);
    void(VRI_CALL* Wait)(VriFence* fence, uint64_t value);

    /* command recording */
    void(VRI_CALL* CmdBeginRendering)(VriCommandBuffer* cmd, const VriAttachmentsDesc* attachments);
    void(VRI_CALL* CmdEndRendering)(VriCommandBuffer* cmd);
    void(VRI_CALL* CmdSetViewports)(VriCommandBuffer* cmd, const VriViewport* viewports, uint32_t viewportNum);
    void(VRI_CALL* CmdSetScissors)(VriCommandBuffer* cmd, const VriRect* scissors, uint32_t scissorNum);
    void(VRI_CALL* CmdSetPipelineLayout)(VriCommandBuffer* cmd, VriPipelineLayout* layout);
    void(VRI_CALL* CmdSetPipeline)(VriCommandBuffer* cmd, VriPipeline* pipeline);
    void(VRI_CALL* CmdSetDescriptorSet)(VriCommandBuffer* cmd, uint32_t setIndex, const VriDescriptorSet* set);
    void(VRI_CALL* CmdSetConstants)(VriCommandBuffer* cmd, uint32_t pushConstantIndex, const void* data, uint32_t size);
    void(VRI_CALL* CmdSetVertexBuffers)(VriCommandBuffer*             cmd,
                                        uint32_t                      baseSlot,
                                        const VriVertexBufferBinding* bindings,
                                        uint32_t                      bindingNum);
    void(VRI_CALL* CmdSetIndexBuffer)(VriCommandBuffer* cmd,
                                      VriBuffer*        buffer,
                                      uint64_t          offset,
                                      VriIndexType      indexType);
    void(VRI_CALL* CmdDraw)(VriCommandBuffer* cmd, const VriDrawDesc* draw);
    void(VRI_CALL* CmdDrawIndexed)(VriCommandBuffer* cmd, const VriDrawIndexedDesc* draw);
    /* Indirect draws read their arguments from `buffer` (one VriDrawDesc / VriDrawIndexedDesc record
     * per draw, `stride` bytes apart). The *Count variants take the draw count from countBuffer (a
     * uint32 at countOffset), clamped to maxDrawNum - for GPU-driven rendering where the GPU decides
     * how many draws to issue. The *Count variants require VriDeviceDesc::hasDrawIndirectCount. */
    void(VRI_CALL* CmdDrawIndirect)(VriCommandBuffer* cmd,
                                    VriBuffer*        buffer,
                                    uint64_t          offset,
                                    uint32_t          drawNum,
                                    uint32_t          stride);
    void(VRI_CALL* CmdDrawIndexedIndirect)(VriCommandBuffer* cmd,
                                           VriBuffer*        buffer,
                                           uint64_t          offset,
                                           uint32_t          drawNum,
                                           uint32_t          stride);
    void(VRI_CALL* CmdDrawIndirectCount)(VriCommandBuffer* cmd,
                                         VriBuffer*        buffer,
                                         uint64_t          offset,
                                         VriBuffer*        countBuffer,
                                         uint64_t          countOffset,
                                         uint32_t          maxDrawNum,
                                         uint32_t          stride);
    void(VRI_CALL* CmdDrawIndexedIndirectCount)(VriCommandBuffer* cmd,
                                                VriBuffer*        buffer,
                                                uint64_t          offset,
                                                VriBuffer*        countBuffer,
                                                uint64_t          countOffset,
                                                uint32_t          maxDrawNum,
                                                uint32_t          stride);
    void(VRI_CALL* CmdDispatch)(VriCommandBuffer* cmd, const VriDispatchDesc* dispatch);
    void(VRI_CALL* CmdDispatchIndirect)(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset);
    void(VRI_CALL* CmdBarrier)(VriCommandBuffer* cmd, const VriBarrierGroupDesc* barrierGroup);
    void(VRI_CALL* CmdCopyBuffer)(VriCommandBuffer*        cmd,
                                  VriBuffer*               dst,
                                  VriBuffer*               src,
                                  const VriBufferCopyDesc* region);
    void(VRI_CALL* CmdCopyTexture)(VriCommandBuffer*         cmd,
                                   VriTexture*               dst,
                                   VriTexture*               src,
                                   const VriTextureCopyDesc* region);
    void(VRI_CALL* CmdUploadBufferToTexture)(VriCommandBuffer*               cmd,
                                             VriTexture*                     dst,
                                             VriBuffer*                      src,
                                             const VriBufferTextureCopyDesc* region);
    void(VRI_CALL* CmdReadbackTextureToBuffer)(VriCommandBuffer*               cmd,
                                               VriBuffer*                      dst,
                                               VriTexture*                     src,
                                               const VriBufferTextureCopyDesc* region);
    void(VRI_CALL* CmdBeginDebugGroup)(VriCommandBuffer* cmd, const char* name);
    void(VRI_CALL* CmdEndDebugGroup)(VriCommandBuffer* cmd);

    /* submission */
    VriResult(VRI_CALL* QueueSubmit)(VriQueue* queue, const VriQueueSubmitDesc* submit);
    VriResult(VRI_CALL* QueueWaitIdle)(VriQueue* queue);
    VriResult(VRI_CALL* DeviceWaitIdle)(VriDevice* device);
    void(VRI_CALL* SetDebugName)(void* object, const char* name);

    /* Live video-memory budget/usage for a memory location (Device = VRAM, others = system).
     * Returns VriResult_Unsupported where the backend can't report it (e.g. WebGPU, OpenGL). */
    VriResult(VRI_CALL* GetVideoMemoryInfo)(const VriDevice*    device,
                                            VriMemoryLocation   location,
                                            VriVideoMemoryInfo* out);

    /* Fill a storage (UAV/SSBO) buffer region with a repeated uint32, outside a render pass - e.g.
     * to reset an atomic counter or accumulation buffer before a compute pass. `size` 0 = to the end.
     * The caller barriers the buffer (CopyDestinationWrite-style) around the clear. Backends without a
     * direct buffer fill (Direct3D 12, WebGPU, Metal) diagnose rather than silently no-op. */
    void(VRI_CALL* CmdClearStorageBuffer)(VriCommandBuffer* cmd,
                                          VriBuffer*        buffer,
                                          uint64_t          offset,
                                          uint64_t          size,
                                          uint32_t          value);

    /* Clear a storage (UAV/image) texture's mip 0 (all layers) to a color value, outside a render
     * pass. `value` is interpreted per the texture's format - fill the f32 (float/normalized) or
     * u32 / i32 (integer) union member that matches it. Gated by VriDeviceDesc::hasClearStorageTexture
     * where implemented. Like CmdClearStorageBuffer this is a
     * transfer-domain op: barrier the texture to CopyDestination before, and to its next use after. */
    void(VRI_CALL* CmdClearStorageTexture)(VriCommandBuffer* cmd, VriTexture* texture, const VriClearColor* value);

    /* Snapshot every buffer, texture, acceleration structure and micromap this device currently
     * owns, with the memory each one actually holds. GetVideoMemoryInfo answers how much is in
     * use; this answers what is using it, which is the question a renderer near its VRAM budget
     * needs. Names come from SetDebugName.
     *
     * Call with `out` NULL to read the count, then again with an array of at least that many.
     * `*count` is updated to the number written. The snapshot is taken under the device's lock and
     * is a plain copy, so it stays valid afterwards. Returns VriResult_Unsupported on backends
     * that do not track objects. */
    VriResult(VRI_CALL* EnumerateObjects)(const VriDevice* device, uint32_t* count, VriObjectInfo* out);
    /* Bounded host wait on the same timeline as Wait(). Zero polls; UINT64_MAX
     * waits indefinitely. Timeout never changes the fence's completion value. */
    VriResult(VRI_CALL* WaitForFence)(VriFence* fence, uint64_t value, uint64_t timeoutNanoseconds);

    /* Map a buffer range like MapBuffer and hand the pointer to `callback` instead of returning it.
     * Mapping a GPU buffer in the browser's WebGPU completes only from the browser event loop, so a
     * program linked without -sASYNCIFY reads buffers back this way (MapBuffer reports an error for
     * them there). The browser maps the buffer once the GPU work already submitted that uses it has
     * finished; elsewhere, as for MapBuffer, wait for that work (a fence) before mapping.
     *
     * `callback` runs exactly once: from the browser event loop for a browser WebGPU buffer other
     * than a host-upload one, otherwise before MapBufferAsync returns, with the result MapBuffer
     * gives. Release the mapping with UnmapBuffer. Destroying the buffer while its browser map is in
     * flight cancels the map: `callback` runs inside DestroyBuffer with VriResult_Failure and NULL
     * data, and nothing more arrives from the browser. UnmapBuffer during that map aborts it:
     * `callback` runs inside UnmapBuffer with VriResult_Failure, and the device reports an error.
     * The device must outlive the buffer. Returns VriResult_InvalidArgument, without calling
     * `callback`, when `buffer` or `callback` is NULL. */
    VriResult(VRI_CALL* MapBufferAsync)(VriBuffer*              buffer,
                                        uint64_t                offset,
                                        uint64_t                size,
                                        VriBufferMappedCallback callback,
                                        void*                   userData);
} VriCoreInterface;

VRI_EXTERN_C_END

#endif /* VRI_CORE_H */
