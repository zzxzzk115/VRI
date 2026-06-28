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
#include "vri_handles.h"
#include "vri_enums.h"
#include "vri_flags.h"
#include "vri_format.h"
#include "vri_structs.h"
#include "vri_device_desc.h"
#include "vri_resource.h"
#include "vri_descriptor.h"
#include "vri_pipeline.h"
#include "vri_command.h"
#include "vri_sync.h"

VRI_EXTERN_C_BEGIN

/* ---- global entry points ---------------------------------------------- */

VRI_API uint32_t VRI_CALL vriEnumerateAdapters(VriGraphicsAPI api,
                                               VriAdapterDesc* outAdapters,
                                               uint32_t adapterCapacity);

VRI_API VriResult VRI_CALL vriCreateDevice(const VriDeviceCreationDesc* desc,
                                           VriDevice** outDevice);

VRI_API void VRI_CALL vriDestroyDevice(VriDevice* device);

/*
 * Copy a named interface's function table into `out`. `name` is one of the
 * VRI_INTERFACE_* identifiers; `size` must equal sizeof the target struct
 * (guards ABI drift). Returns VriResult_Unsupported if the device's backend
 * does not provide the interface.
 */
VRI_API VriResult VRI_CALL vriGetInterface(const VriDevice* device,
                                           const char* name,
                                           size_t size,
                                           void* out);

#define VRI_INTERFACE_CORE       "VriCoreInterface"
#define VRI_INTERFACE_SWAPCHAIN  "VriSwapChainInterface"
#define VRI_INTERFACE_INTEROP    "VriInteropInterface"
#define VRI_INTERFACE_RAYTRACING "VriRayTracingInterface"
#define VRI_INTERFACE_MESHSHADER "VriMeshShaderInterface"
#define VRI_INTERFACE_VRS        "VriShadingRateInterface"
#define VRI_INTERFACE_OMM        "VriOpacityMicromapInterface"
#define VRI_INTERFACE_EXTERNAL   "VriExternalInterface"

/* ---- core interface (function table) ---------------------------------- */

typedef struct VriCoreInterface
{
    /* queries */
    const VriDeviceDesc*  (VRI_CALL *GetDeviceDesc)(const VriDevice* device);
    VriFormatSupportFlags (VRI_CALL *GetFormatSupport)(const VriDevice* device, VriFormat format);
    VriResult             (VRI_CALL *GetQueue)(VriDevice* device, VriQueueType type, uint32_t index, VriQueue** outQueue);

    /* command allocation / recording lifecycle */
    VriResult (VRI_CALL *CreateCommandAllocator)(VriDevice* device, VriQueueType type, VriCommandAllocator** outAllocator);
    void      (VRI_CALL *ResetCommandAllocator)(VriCommandAllocator* allocator);
    void      (VRI_CALL *DestroyCommandAllocator)(VriCommandAllocator* allocator);
    VriResult (VRI_CALL *CreateCommandBuffer)(VriCommandAllocator* allocator, VriCommandBuffer** outCommandBuffer);
    VriResult (VRI_CALL *BeginCommandBuffer)(VriCommandBuffer* cmd);
    VriResult (VRI_CALL *EndCommandBuffer)(VriCommandBuffer* cmd);

    /* resources */
    VriResult (VRI_CALL *CreateBuffer)(VriDevice* device, const VriBufferDesc* desc, VriBuffer** outBuffer);
    void      (VRI_CALL *DestroyBuffer)(VriBuffer* buffer);
    void*     (VRI_CALL *MapBuffer)(VriBuffer* buffer, uint64_t offset, uint64_t size);
    void      (VRI_CALL *UnmapBuffer)(VriBuffer* buffer);
    uint64_t  (VRI_CALL *GetBufferDeviceAddress)(const VriBuffer* buffer);
    VriResult (VRI_CALL *CreateTexture)(VriDevice* device, const VriTextureDesc* desc, VriTexture** outTexture);
    void      (VRI_CALL *DestroyTexture)(VriTexture* texture);

    /* explicit memory */
    void      (VRI_CALL *GetBufferMemoryDesc)(const VriDevice* device, const VriBufferDesc* desc, VriMemoryLocation location, VriMemoryDesc* outMemoryDesc);
    void      (VRI_CALL *GetTextureMemoryDesc)(const VriDevice* device, const VriTextureDesc* desc, VriMemoryLocation location, VriMemoryDesc* outMemoryDesc);
    VriResult (VRI_CALL *AllocateMemory)(VriDevice* device, const VriMemoryDesc* desc, VriMemory** outMemory);
    void      (VRI_CALL *FreeMemory)(VriMemory* memory);
    VriResult (VRI_CALL *BindBufferMemory)(VriDevice* device, VriBuffer* buffer, VriMemory* memory, uint64_t offset);
    VriResult (VRI_CALL *BindTextureMemory)(VriDevice* device, VriTexture* texture, VriMemory* memory, uint64_t offset);

    /* views & samplers (each returns an opaque VriDescriptor) */
    VriResult (VRI_CALL *CreateBufferView)(VriDevice* device, const VriBufferViewDesc* desc, VriDescriptor** outDescriptor);
    VriResult (VRI_CALL *CreateTextureView)(VriDevice* device, const VriTextureViewDesc* desc, VriDescriptor** outDescriptor);
    VriResult (VRI_CALL *CreateSampler)(VriDevice* device, const VriSamplerDesc* desc, VriDescriptor** outDescriptor);
    void      (VRI_CALL *DestroyDescriptor)(VriDescriptor* descriptor);

    /* pipeline layout & pipelines */
    VriResult (VRI_CALL *CreatePipelineLayout)(VriDevice* device, const VriPipelineLayoutDesc* desc, VriPipelineLayout** outLayout);
    void      (VRI_CALL *DestroyPipelineLayout)(VriPipelineLayout* layout);
    VriResult (VRI_CALL *CreateGraphicsPipeline)(VriDevice* device, const VriGraphicsPipelineDesc* desc, VriPipeline** outPipeline);
    VriResult (VRI_CALL *CreateComputePipeline)(VriDevice* device, const VriComputePipelineDesc* desc, VriPipeline** outPipeline);
    void      (VRI_CALL *DestroyPipeline)(VriPipeline* pipeline);

    /* descriptor pools & sets */
    VriResult (VRI_CALL *CreateDescriptorPool)(VriDevice* device, const VriDescriptorPoolDesc* desc, VriDescriptorPool** outPool);
    void      (VRI_CALL *ResetDescriptorPool)(VriDescriptorPool* pool);
    void      (VRI_CALL *DestroyDescriptorPool)(VriDescriptorPool* pool);
    VriResult (VRI_CALL *AllocateDescriptorSets)(VriDescriptorPool* pool, const VriPipelineLayout* layout, uint32_t setIndex, VriDescriptorSet** outSets, uint32_t setNum);
    void      (VRI_CALL *UpdateDescriptorRanges)(VriDescriptorSet* set, uint32_t baseRange, uint32_t rangeNum, const VriDescriptorRangeUpdateDesc* updates);

    /* synchronization (timeline fences) */
    VriResult (VRI_CALL *CreateFence)(VriDevice* device, uint64_t initialValue, VriFence** outFence);
    void      (VRI_CALL *DestroyFence)(VriFence* fence);
    uint64_t  (VRI_CALL *GetFenceValue)(VriFence* fence);
    void      (VRI_CALL *Wait)(VriFence* fence, uint64_t value);

    /* command recording */
    void (VRI_CALL *CmdBeginRendering)(VriCommandBuffer* cmd, const VriAttachmentsDesc* attachments);
    void (VRI_CALL *CmdEndRendering)(VriCommandBuffer* cmd);
    void (VRI_CALL *CmdSetViewports)(VriCommandBuffer* cmd, const VriViewport* viewports, uint32_t viewportNum);
    void (VRI_CALL *CmdSetScissors)(VriCommandBuffer* cmd, const VriRect* scissors, uint32_t scissorNum);
    void (VRI_CALL *CmdSetPipelineLayout)(VriCommandBuffer* cmd, VriPipelineLayout* layout);
    void (VRI_CALL *CmdSetPipeline)(VriCommandBuffer* cmd, VriPipeline* pipeline);
    void (VRI_CALL *CmdSetDescriptorSet)(VriCommandBuffer* cmd, uint32_t setIndex, const VriDescriptorSet* set);
    void (VRI_CALL *CmdSetConstants)(VriCommandBuffer* cmd, uint32_t pushConstantIndex, const void* data, uint32_t size);
    void (VRI_CALL *CmdSetVertexBuffers)(VriCommandBuffer* cmd, uint32_t baseSlot, const VriVertexBufferBinding* bindings, uint32_t bindingNum);
    void (VRI_CALL *CmdSetIndexBuffer)(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, VriIndexType indexType);
    void (VRI_CALL *CmdDraw)(VriCommandBuffer* cmd, const VriDrawDesc* draw);
    void (VRI_CALL *CmdDrawIndexed)(VriCommandBuffer* cmd, const VriDrawIndexedDesc* draw);
    void (VRI_CALL *CmdDrawIndirect)(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint32_t drawNum, uint32_t stride);
    void (VRI_CALL *CmdDispatch)(VriCommandBuffer* cmd, const VriDispatchDesc* dispatch);
    void (VRI_CALL *CmdDispatchIndirect)(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset);
    void (VRI_CALL *CmdBarrier)(VriCommandBuffer* cmd, const VriBarrierGroupDesc* barrierGroup);
    void (VRI_CALL *CmdCopyBuffer)(VriCommandBuffer* cmd, VriBuffer* dst, VriBuffer* src, const VriBufferCopyDesc* region);
    void (VRI_CALL *CmdCopyTexture)(VriCommandBuffer* cmd, VriTexture* dst, VriTexture* src, const VriTextureCopyDesc* region);
    void (VRI_CALL *CmdUploadBufferToTexture)(VriCommandBuffer* cmd, VriTexture* dst, VriBuffer* src, const VriBufferTextureCopyDesc* region);
    void (VRI_CALL *CmdReadbackTextureToBuffer)(VriCommandBuffer* cmd, VriBuffer* dst, VriTexture* src, const VriBufferTextureCopyDesc* region);
    void (VRI_CALL *CmdBeginDebugGroup)(VriCommandBuffer* cmd, const char* name);
    void (VRI_CALL *CmdEndDebugGroup)(VriCommandBuffer* cmd);

    /* submission */
    void (VRI_CALL *QueueSubmit)(VriQueue* queue, const VriQueueSubmitDesc* submit);
    void (VRI_CALL *QueueWaitIdle)(VriQueue* queue);
    void (VRI_CALL *DeviceWaitIdle)(VriDevice* device);
    void (VRI_CALL *SetDebugName)(void* object, const char* name);
} VriCoreInterface;

VRI_EXTERN_C_END

#endif /* VRI_CORE_H */
