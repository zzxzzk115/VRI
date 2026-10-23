// D3D12 core implementation and function table.

#include "core_d3d12.h"

#include "conversions_d3d12.h"
#include "device_d3d12.h"
#include "objects_d3d12.h"

#include "core/synchronization.h"

#include <d3dcompiler.h> // D3DReflect (VS input-signature reflection for the input layout)

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace vri::d3d12
{
    namespace
    {
        // The VriCoreInterface implementation is split across the section includes below to keep
        // this file navigable. They are #included into THIS anonymous namespace - one translation
        // unit - so this is a pure code move with no linkage or behavior change. Order matters,
        // so keep clang-format from reordering them.
        // clang-format off
#include "core_d3d12_common.inc"
#include "core_d3d12_resource.inc"
#include "core_d3d12_command.inc"
#include "core_d3d12_memory.inc"
#include "core_d3d12_pipeline.inc"
#include "core_d3d12_desccmd.inc"
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
            EnumerateObjects,
            WaitForFence,
        };
        return &t;
    }
} // namespace vri::d3d12
