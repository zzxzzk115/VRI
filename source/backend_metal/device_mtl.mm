// device_mtl.mm - native Metal device creation, capability query, interface table.

#include "device_mtl.h"
#include "core_mtl.h"
#include "swapchain_mtl.h"
#include "rt_mtl.h"
#include "meshshader_mtl.h"
#include "query_mtl.h"
#include "pipeline_cache_mtl.h"

#include "core/imgui_vri.h" // backend-agnostic built-in ImGui renderer (SPIR-V transpiled to MSL)

#include <cstdio>
#include <cstring>

namespace vri::mtl
{
    DeviceMTL::~DeviceMTL()
    {
        // QueueMTL objects are members (not heap). Compute/Transfer own a dedicated MTLCommandQueue
        // (Graphics aliases m_queue); release those, then m_queue.
        for (int t = 0; t < VriQueueType_Count; ++t)
            if (m_queueObjs[t].queue && m_queueObjs[t].queue != m_queue) [m_queueObjs[t].queue release];
        if (m_queue) [m_queue release];
        if (m_device) [m_device release];
    }

    void DeviceMTL::ReportError(const char* message) const { Diagnostic(VriMessageSeverity_Error, message); }

    void DeviceMTL::Diagnostic(VriMessageSeverity severity, const char* message) const
    {
        if (m_callback.MessageCallback)
            m_callback.MessageCallback(m_callback.userArg, severity, message);
        else
            std::fprintf(stderr, "[VRI/Metal] %s\n", message);
    }

    VriResult DeviceMTL::Init(const VriDeviceCreationDesc& desc)
    {
        if (desc.callbackInterface)
            m_callback = *desc.callbackInterface;

        @autoreleasepool
        {
            // MTLCreateSystemDefaultDevice picks the system's preferred GPU (the only
            // device on Apple Silicon). adapterIndex selection across MTLCopyAllDevices
            // is a desktop-Intel nicety we skip for the arm64 MVP.
            m_device = MTLCreateSystemDefaultDevice();
            if (!m_device)
            {
                ReportError("MTLCreateSystemDefaultDevice returned nil");
                return VriResult_Unsupported;
            }
            [m_device retain]; // outlive the autorelease pool

            m_queue = [m_device newCommandQueue]; // +1 owned
            if (!m_queue)
            {
                ReportError("newCommandQueue failed");
                return VriResult_Failure;
            }
        }

        // Graphics uses the primary queue; Compute and Transfer each get their own MTLCommandQueue
        // so work submitted to them runs concurrently with graphics (async compute / async transfer).
        // Cross-queue ordering is via MTLSharedEvent timeline fences (see QueueSubmit).
        for (int t = 0; t < VriQueueType_Count; ++t)
        {
            m_queueObjs[t].device = this;
            m_queueObjs[t].queue = (t == VriQueueType_Graphics) ? m_queue : [m_device newCommandQueue];
            if (!m_queueObjs[t].queue) { ReportError("newCommandQueue (per-type) failed"); return VriResult_Failure; }
        }

        FillDeviceDesc();
        FillRegistry();
        return VriResult_Success;
    }

    void DeviceMTL::FillDeviceDesc()
    {
        m_desc = {};
        m_desc.graphicsAPI = VriGraphicsAPI_Metal;

        const char* name = [[m_device name] UTF8String];
        std::strncpy(m_desc.adapter.name, name ? name : "Metal Device", sizeof(m_desc.adapter.name) - 1);
        // Apple GPUs are integrated (unified memory); recommendedMaxWorkingSetSize ~ VRAM budget.
        m_desc.adapter.type = [m_device hasUnifiedMemory] ? VriAdapterType_Integrated : VriAdapterType_Discrete;
        m_desc.adapter.videoMemorySize = [m_device recommendedMaxWorkingSetSize];

        m_desc.apiVersionMajor = 3; // Metal 3 baseline on supported macOS
        m_desc.apiVersionMinor = 0;
        m_desc.viewportMaxNum = 16;
        m_desc.attachmentColorMaxNum = 8;
        m_desc.attachmentMaxDim = 16384;
        m_desc.texture1DMaxDim = 16384;
        m_desc.texture2DMaxDim = 16384;
        m_desc.texture3DMaxDim = 2048;
        m_desc.textureArrayLayerMaxNum = 2048;
        m_desc.bufferMaxSize = [m_device maxBufferLength];
        // Keep Vulkan-compatible alignments (callers align uploads/UBOs to these).
        m_desc.uploadBufferTextureRowAlignment = 256;
        m_desc.constantBufferOffsetAlignment = 256;
        m_desc.storageBufferOffsetAlignment = 32;

        for (int t = 0; t < VriQueueType_Count; ++t)
            m_desc.queueCount[t] = 1;
        m_desc.hasComputeShader = VRI_TRUE;
        m_desc.hasGeometryShader = VRI_FALSE; // Metal has no geometry stage
        m_desc.hasTessellation = VRI_FALSE;   // Metal tessellation differs from VK; not in MVP
        m_desc.hasCustomBorderColor = VRI_TRUE;
        // Inline ray query runs in a compute/fragment shader over MTLAccelerationStructure.
        // (hasRayTracing stays FALSE: Metal has no DXR-style RT pipeline / SBT.)
        m_desc.hasRayQuery = [m_device supportsRaytracing] ? VRI_TRUE : VRI_FALSE;
        // Mesh/object shaders require the Metal 3 family.
        m_desc.hasMeshShader = [m_device supportsFamily:MTLGPUFamilyMetal3] ? VRI_TRUE : VRI_FALSE;

        // GPU queries: timestamps need the standard counter set + stage-boundary sampling; occlusion
        // (visibility result buffers) is always available. Apple GPUs expose no statistic counter set.
        m_desc.hasTimestampQueries = ([m_device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary] &&
                                      HasTimestampCounterSet()) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasPipelineStatistics      = VRI_FALSE;
        m_desc.hasCalibratedTimestamps    = VRI_TRUE; // [MTLDevice sampleTimestamps:gpuTimestamp:]

        // Multiview (single-pass stereo): SPIRV-Cross emits the instanced form (instances *= viewCount,
        // each view writes its array slice via [[render_target_array_index]]), which every Apple GPU
        // supports. maxViewCount is the viewMask bit width (the array-layer count is the real ceiling).
        m_desc.hasMultiview = [m_device supportsFamily:MTLGPUFamilyApple1] ? VRI_TRUE : VRI_FALSE;
        m_desc.maxViewCount = m_desc.hasMultiview ? 32 : 0;
        // Apple Silicon GPU timestamps are already in nanoseconds (same domain as sampleTimestamps).
        m_desc.timestampPeriodNanoseconds = m_desc.hasTimestampQueries ? 1.0f : 0.0f;
    }

    bool DeviceMTL::HasTimestampCounterSet() const
    {
        for (id<MTLCounterSet> cs in [m_device counterSets])
            if ([[cs name] isEqualToString:MTLCommonCounterSetTimestamp])
                return true;
        return false;
    }

    void DeviceMTL::FillRegistry()
    {
        m_registry.Register(VRI_INTERFACE_CORE, GetCoreInterfaceMTL(), sizeof(VriCoreInterface));
        m_registry.Register(VRI_INTERFACE_SWAPCHAIN, GetSwapChainInterfaceMTL(), sizeof(VriSwapChainInterface));
        m_registry.Register(VRI_INTERFACE_QUERY, GetQueryInterfaceMTL(), sizeof(VriQueryInterface));
        m_registry.Register(
            VRI_INTERFACE_PIPELINE_CACHE, GetPipelineCacheInterfaceMTL(), sizeof(VriPipelineCacheInterface));
        m_registry.Register(VRI_INTERFACE_IMGUI, core::GetImguiInterface(), sizeof(VriImguiInterface));
        if (m_desc.hasRayQuery)
            m_registry.Register(VRI_INTERFACE_RAYTRACING, GetRayTracingInterfaceMTL(), sizeof(VriRayTracingInterface));
        if (m_desc.hasMeshShader)
            m_registry.Register(VRI_INTERFACE_MESHSHADER, GetMeshShaderInterfaceMTL(), sizeof(VriMeshShaderInterface));
    }

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult)
    {
        DeviceMTL* device = new DeviceMTL();
        outResult = device->Init(desc);
        if (outResult != VriResult_Success)
        {
            delete device;
            return nullptr;
        }
        return device;
    }
} // namespace vri::mtl
