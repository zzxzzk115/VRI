// core_d3d12.cpp - the D3D12 core function table.
//
// Phase 1 implements the clear+readback path: resources (committed default/upload/
// readback heaps), RTV views, command allocator/list, resource-state barriers,
// render-pass clears, texture->buffer readback, fences and submission. Pipelines,
// descriptor sets and draws are stubbed (Unsupported/no-op) until Phase 2.
#include "core_d3d12.h"

#include "conversions_d3d12.h"
#include "device_d3d12.h"
#include "objects_d3d12.h"

#include <d3dcompiler.h> // D3DReflect (VS input-signature reflection for the input layout)

#include <cstring>
#include <string>
#include <vector>

namespace vri::d3d12
{
    namespace
    {
        // The VriCoreInterface implementation is split across the section includes below to keep
        // this file navigable. They are #included into THIS anonymous namespace - one translation
        // unit - so this is a pure code move with no linkage or behavior change. Order matters
        // (later sections use earlier helpers/types), so keep clang-format from reordering them.
        // clang-format off
#include "core_d3d12_common.inc"
#include "core_d3d12_resource.inc"
#include "core_d3d12_pipeline.inc"
#include "core_d3d12_command.inc"
        // clang-format on
    } // namespace

    const VriCoreInterface* GetCoreInterfaceD3D12()
    {
        static const VriCoreInterface t = {
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
        return &t;
    }
} // namespace vri::d3d12
