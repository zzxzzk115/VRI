// core_d3d12.cpp - the D3D12 core function table.
//
// Phase 0: device queries (GetDeviceDesc/GetQueue) are real so a device can be
// created and introspected; the resource/pipeline/command/submission entry points
// are safe stubs (Unsupported / no-op) and get filled in as the backend is built
// up. Every pointer in the table is non-null so the memcpy'd table is safe to hold.
#include "core_d3d12.h"

#include "device_d3d12.h"
#include "objects_d3d12.h"

namespace vri::d3d12
{
    namespace
    {
        DeviceD3D12* Dev(VriDevice* d) { return reinterpret_cast<DeviceD3D12*>(d); }
        const DeviceD3D12* Dev(const VriDevice* d) { return reinterpret_cast<const DeviceD3D12*>(d); }

        // ---- queries (implemented) -------------------------------------
        const VriDeviceDesc* VRI_CALL GetDeviceDesc(const VriDevice* device) { return &Dev(device)->Desc(); }
        VriFormatSupportFlags VRI_CALL GetFormatSupport(const VriDevice*, VriFormat)
        {
            return VriFormatSupport_Texture | VriFormatSupport_ColorAttachment | VriFormatSupport_Blend | VriFormatSupport_VertexBuffer;
        }
        VriResult VRI_CALL GetQueue(VriDevice* device, VriQueueType type, uint32_t, VriQueue** outQueue)
        {
            if (type >= VriQueueType_Count) return VriResult_InvalidArgument;
            *outQueue = ToHandle(Dev(device)->GetQueue(type));
            return VriResult_Success;
        }

        // ---- not-yet-implemented (Phase 0 stubs) -----------------------
        VriResult VRI_CALL CreateCommandAllocator(VriDevice*, VriQueueType, VriCommandAllocator**) { return VriResult_Unsupported; }
        void      VRI_CALL ResetCommandAllocator(VriCommandAllocator*) {}
        void      VRI_CALL DestroyCommandAllocator(VriCommandAllocator*) {}
        VriResult VRI_CALL CreateCommandBuffer(VriCommandAllocator*, VriCommandBuffer**) { return VriResult_Unsupported; }
        VriResult VRI_CALL BeginCommandBuffer(VriCommandBuffer*) { return VriResult_Unsupported; }
        VriResult VRI_CALL EndCommandBuffer(VriCommandBuffer*) { return VriResult_Unsupported; }

        VriResult VRI_CALL CreateBuffer(VriDevice*, const VriBufferDesc*, VriBuffer**) { return VriResult_Unsupported; }
        void      VRI_CALL DestroyBuffer(VriBuffer*) {}
        void*     VRI_CALL MapBuffer(VriBuffer*, uint64_t, uint64_t) { return nullptr; }
        void      VRI_CALL UnmapBuffer(VriBuffer*) {}
        uint64_t  VRI_CALL GetBufferDeviceAddress(const VriBuffer*) { return 0; }
        VriResult VRI_CALL CreateTexture(VriDevice*, const VriTextureDesc*, VriTexture**) { return VriResult_Unsupported; }
        void      VRI_CALL DestroyTexture(VriTexture*) {}

        void      VRI_CALL GetBufferMemoryDesc(const VriDevice*, const VriBufferDesc*, VriMemoryLocation, VriMemoryDesc* o) { if (o) *o = VriMemoryDesc{}; }
        void      VRI_CALL GetTextureMemoryDesc(const VriDevice*, const VriTextureDesc*, VriMemoryLocation, VriMemoryDesc* o) { if (o) *o = VriMemoryDesc{}; }
        VriResult VRI_CALL AllocateMemory(VriDevice*, const VriMemoryDesc*, VriMemory**) { return VriResult_Unsupported; }
        void      VRI_CALL FreeMemory(VriMemory*) {}
        VriResult VRI_CALL BindBufferMemory(VriDevice*, VriBuffer*, VriMemory*, uint64_t) { return VriResult_Unsupported; }
        VriResult VRI_CALL BindTextureMemory(VriDevice*, VriTexture*, VriMemory*, uint64_t) { return VriResult_Unsupported; }

        VriResult VRI_CALL CreateBufferView(VriDevice*, const VriBufferViewDesc*, VriDescriptor**) { return VriResult_Unsupported; }
        VriResult VRI_CALL CreateTextureView(VriDevice*, const VriTextureViewDesc*, VriDescriptor**) { return VriResult_Unsupported; }
        VriResult VRI_CALL CreateSampler(VriDevice*, const VriSamplerDesc*, VriDescriptor**) { return VriResult_Unsupported; }
        void      VRI_CALL DestroyDescriptor(VriDescriptor*) {}

        VriResult VRI_CALL CreatePipelineLayout(VriDevice*, const VriPipelineLayoutDesc*, VriPipelineLayout**) { return VriResult_Unsupported; }
        void      VRI_CALL DestroyPipelineLayout(VriPipelineLayout*) {}
        VriResult VRI_CALL CreateGraphicsPipeline(VriDevice*, const VriGraphicsPipelineDesc*, VriPipeline**) { return VriResult_Unsupported; }
        VriResult VRI_CALL CreateComputePipeline(VriDevice*, const VriComputePipelineDesc*, VriPipeline**) { return VriResult_Unsupported; }
        void      VRI_CALL DestroyPipeline(VriPipeline*) {}

        VriResult VRI_CALL CreateDescriptorPool(VriDevice*, const VriDescriptorPoolDesc*, VriDescriptorPool**) { return VriResult_Unsupported; }
        void      VRI_CALL ResetDescriptorPool(VriDescriptorPool*) {}
        void      VRI_CALL DestroyDescriptorPool(VriDescriptorPool*) {}
        VriResult VRI_CALL AllocateDescriptorSets(VriDescriptorPool*, const VriPipelineLayout*, uint32_t, VriDescriptorSet**, uint32_t) { return VriResult_Unsupported; }
        void      VRI_CALL UpdateDescriptorRanges(VriDescriptorSet*, uint32_t, uint32_t, const VriDescriptorRangeUpdateDesc*) {}

        VriResult VRI_CALL CreateFence(VriDevice*, uint64_t, VriFence**) { return VriResult_Unsupported; }
        void      VRI_CALL DestroyFence(VriFence*) {}
        uint64_t  VRI_CALL GetFenceValue(VriFence*) { return 0; }
        void      VRI_CALL Wait(VriFence*, uint64_t) {}

        void VRI_CALL CmdBeginRendering(VriCommandBuffer*, const VriAttachmentsDesc*) {}
        void VRI_CALL CmdEndRendering(VriCommandBuffer*) {}
        void VRI_CALL CmdSetViewports(VriCommandBuffer*, const VriViewport*, uint32_t) {}
        void VRI_CALL CmdSetScissors(VriCommandBuffer*, const VriRect*, uint32_t) {}
        void VRI_CALL CmdSetPipelineLayout(VriCommandBuffer*, VriPipelineLayout*) {}
        void VRI_CALL CmdSetPipeline(VriCommandBuffer*, VriPipeline*) {}
        void VRI_CALL CmdSetDescriptorSet(VriCommandBuffer*, uint32_t, const VriDescriptorSet*) {}
        void VRI_CALL CmdSetConstants(VriCommandBuffer*, uint32_t, const void*, uint32_t) {}
        void VRI_CALL CmdSetVertexBuffers(VriCommandBuffer*, uint32_t, const VriVertexBufferBinding*, uint32_t) {}
        void VRI_CALL CmdSetIndexBuffer(VriCommandBuffer*, VriBuffer*, uint64_t, VriIndexType) {}
        void VRI_CALL CmdDraw(VriCommandBuffer*, const VriDrawDesc*) {}
        void VRI_CALL CmdDrawIndexed(VriCommandBuffer*, const VriDrawIndexedDesc*) {}
        void VRI_CALL CmdDrawIndirect(VriCommandBuffer*, VriBuffer*, uint64_t, uint32_t, uint32_t) {}
        void VRI_CALL CmdDispatch(VriCommandBuffer*, const VriDispatchDesc*) {}
        void VRI_CALL CmdDispatchIndirect(VriCommandBuffer*, VriBuffer*, uint64_t) {}
        void VRI_CALL CmdBarrier(VriCommandBuffer*, const VriBarrierGroupDesc*) {}
        void VRI_CALL CmdCopyBuffer(VriCommandBuffer*, VriBuffer*, VriBuffer*, const VriBufferCopyDesc*) {}
        void VRI_CALL CmdCopyTexture(VriCommandBuffer*, VriTexture*, VriTexture*, const VriTextureCopyDesc*) {}
        void VRI_CALL CmdUploadBufferToTexture(VriCommandBuffer*, VriTexture*, VriBuffer*, const VriBufferTextureCopyDesc*) {}
        void VRI_CALL CmdReadbackTextureToBuffer(VriCommandBuffer*, VriBuffer*, VriTexture*, const VriBufferTextureCopyDesc*) {}
        void VRI_CALL CmdBeginDebugGroup(VriCommandBuffer*, const char*) {}
        void VRI_CALL CmdEndDebugGroup(VriCommandBuffer*) {}

        void VRI_CALL QueueSubmit(VriQueue*, const VriQueueSubmitDesc*) {}
        void VRI_CALL QueueWaitIdle(VriQueue*) {}
        void VRI_CALL DeviceWaitIdle(VriDevice*) {}
        void VRI_CALL SetDebugName(void*, const char*) {}
    } // namespace

    const VriCoreInterface* GetCoreInterfaceD3D12()
    {
        static const VriCoreInterface t = {
            GetDeviceDesc, GetFormatSupport, GetQueue,
            CreateCommandAllocator, ResetCommandAllocator, DestroyCommandAllocator, CreateCommandBuffer, BeginCommandBuffer, EndCommandBuffer,
            CreateBuffer, DestroyBuffer, MapBuffer, UnmapBuffer, GetBufferDeviceAddress, CreateTexture, DestroyTexture,
            GetBufferMemoryDesc, GetTextureMemoryDesc, AllocateMemory, FreeMemory, BindBufferMemory, BindTextureMemory,
            CreateBufferView, CreateTextureView, CreateSampler, DestroyDescriptor,
            CreatePipelineLayout, DestroyPipelineLayout, CreateGraphicsPipeline, CreateComputePipeline, DestroyPipeline,
            CreateDescriptorPool, ResetDescriptorPool, DestroyDescriptorPool, AllocateDescriptorSets, UpdateDescriptorRanges,
            CreateFence, DestroyFence, GetFenceValue, Wait,
            CmdBeginRendering, CmdEndRendering, CmdSetViewports, CmdSetScissors, CmdSetPipelineLayout, CmdSetPipeline,
            CmdSetDescriptorSet, CmdSetConstants, CmdSetVertexBuffers, CmdSetIndexBuffer,
            CmdDraw, CmdDrawIndexed, CmdDrawIndirect, CmdDispatch, CmdDispatchIndirect, CmdBarrier,
            CmdCopyBuffer, CmdCopyTexture, CmdUploadBufferToTexture, CmdReadbackTextureToBuffer, CmdBeginDebugGroup, CmdEndDebugGroup,
            QueueSubmit, QueueWaitIdle, DeviceWaitIdle, SetDebugName,
        };
        return &t;
    }
} // namespace vri::d3d12
