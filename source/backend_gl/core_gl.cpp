// OpenGL/GLES/WebGL core implementation.

#include "core_gl.h"
#include "conversions_gl.h"
#include "device_gl.h"
#include "objects_gl.h"

#include "core/synchronization.h"

#include "gl_loader.h"

#include <spirv_cross/spirv_glsl.hpp>

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif

namespace vri::gl
{
    namespace
    {
        // The VriCoreInterface implementation is split across the section includes below to keep
        // this file navigable. They are #included into THIS anonymous namespace - one translation
        // unit - so this is a pure code move with no linkage or behavior change. Order matters
        // (later sections use earlier helpers/types), so keep clang-format from reordering them.
        // clang-format off
#include "core_gl_common.inc"
#include "core_gl_resource.inc"
#include "core_gl_pipeline.inc"
#include "core_gl_command.inc"
        // clang-format on

        VriCoreInterface MakeTable()
        {
            VriCoreInterface t            = {};
            t.GetDeviceDesc               = GetDeviceDesc;
            t.GetFormatSupport            = GetFormatSupport;
            t.GetQueue                    = GetQueue;
            t.CreateCommandAllocator      = CreateCommandAllocator;
            t.ResetCommandAllocator       = ResetCommandAllocator;
            t.DestroyCommandAllocator     = DestroyCommandAllocator;
            t.CreateCommandBuffer         = CreateCommandBuffer;
            t.BeginCommandBuffer          = BeginCommandBuffer;
            t.EndCommandBuffer            = EndCommandBuffer;
            t.CreateBuffer                = CreateBuffer;
            t.DestroyBuffer               = DestroyBuffer;
            t.MapBuffer                   = MapBuffer;
            t.UnmapBuffer                 = UnmapBuffer;
            t.GetBufferDeviceAddress      = GetBufferDeviceAddress;
            t.CreateTexture               = CreateTexture;
            t.DestroyTexture              = DestroyTexture;
            t.GetBufferMemoryDesc         = GetBufferMemoryDesc;
            t.GetTextureMemoryDesc        = GetTextureMemoryDesc;
            t.AllocateMemory              = AllocateMemory;
            t.FreeMemory                  = FreeMemory;
            t.BindBufferMemory            = BindBufferMemory;
            t.BindTextureMemory           = BindTextureMemory;
            t.CreateBufferView            = CreateBufferView;
            t.CreateTextureView           = CreateTextureView;
            t.CreateSampler               = CreateSampler;
            t.DestroyDescriptor           = DestroyDescriptor;
            t.CreatePipelineLayout        = CreatePipelineLayout;
            t.DestroyPipelineLayout       = DestroyPipelineLayout;
            t.CreateGraphicsPipeline      = CreateGraphicsPipeline;
            t.CreateComputePipeline       = CreateComputePipeline;
            t.DestroyPipeline             = DestroyPipeline;
            t.CreateDescriptorPool        = CreateDescriptorPool;
            t.ResetDescriptorPool         = ResetDescriptorPool;
            t.DestroyDescriptorPool       = DestroyDescriptorPool;
            t.AllocateDescriptorSets      = AllocateDescriptorSets;
            t.UpdateDescriptorRanges      = UpdateDescriptorRanges;
            t.CreateFence                 = CreateFence;
            t.DestroyFence                = DestroyFence;
            t.GetFenceValue               = GetFenceValue;
            t.Wait                        = Wait;
            t.WaitForFence                = WaitForFence;
            t.CmdBeginRendering           = CmdBeginRendering;
            t.CmdEndRendering             = CmdEndRendering;
            t.CmdSetViewports             = CmdSetViewports;
            t.CmdSetScissors              = CmdSetScissors;
            t.CmdSetPipelineLayout        = CmdSetPipelineLayout;
            t.CmdSetPipeline              = CmdSetPipeline;
            t.CmdSetDescriptorSet         = CmdSetDescriptorSet;
            t.CmdSetConstants             = CmdSetConstants;
            t.CmdSetVertexBuffers         = CmdSetVertexBuffers;
            t.CmdSetIndexBuffer           = CmdSetIndexBuffer;
            t.CmdDraw                     = CmdDraw;
            t.CmdDrawIndexed              = CmdDrawIndexed;
            t.CmdDrawIndirect             = CmdDrawIndirect;
            t.CmdDrawIndexedIndirect      = CmdDrawIndexedIndirect;
            t.CmdDrawIndirectCount        = CmdDrawIndirectCount;
            t.CmdDrawIndexedIndirectCount = CmdDrawIndexedIndirectCount;
            t.CmdDispatch                 = CmdDispatch;
            t.CmdDispatchIndirect         = CmdDispatchIndirect;
            t.CmdBarrier                  = CmdBarrier;
            t.CmdCopyBuffer               = CmdCopyBuffer;
            t.CmdCopyTexture              = CmdCopyTexture;
            t.CmdUploadBufferToTexture    = CmdUploadBufferToTexture;
            t.CmdReadbackTextureToBuffer  = CmdReadbackTextureToBuffer;
            t.CmdBeginDebugGroup          = CmdBeginDebugGroup;
            t.CmdEndDebugGroup            = CmdEndDebugGroup;
            t.QueueSubmit                 = QueueSubmit;
            t.QueueWaitIdle               = QueueWaitIdle;
            t.DeviceWaitIdle              = DeviceWaitIdle;
            t.SetDebugName                = SetDebugName;
            t.EnumerateObjects            = EnumerateObjects;
            t.GetVideoMemoryInfo          = GetVideoMemoryInfo;
            t.CmdClearStorageBuffer       = CmdClearStorageBuffer;
            t.CmdClearStorageTexture      = CmdClearStorageTexture;
            return t;
        }

        const VriCoreInterface g_coreGL = MakeTable();
    } // namespace

    const VriCoreInterface* GetCoreInterfaceGL() { return &g_coreGL; }
} // namespace vri::gl
