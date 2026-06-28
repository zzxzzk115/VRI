#include "device_d3d12.h"
#include "core_d3d12.h"
#include "external_d3d12.h"
#include "meshshader_d3d12.h"
#include "omm_d3d12.h"
#include "query_d3d12.h"
#include "rt_d3d12.h"
#include "swapchain_d3d12.h"
#include "vrs_d3d12.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vri::d3d12
{
    DeviceD3D12::~DeviceD3D12()
    {
        // Unregister the debug-message callback before any D3D12 object is released: object
        // teardown itself emits debug-layer messages, which would otherwise invoke the
        // callback with a now-dangling `this`.
        if (m_infoQueue && m_msgCookie)
            m_infoQueue->UnregisterMessageCallback(m_msgCookie);
        for (auto& q : m_queues)
            if (q.idleEvent)
                CloseHandle(q.idleEvent);
    }

    void DeviceD3D12::ReportError(const char* message) const { Diagnostic(VriMessageSeverity_Error, message); }

    void DeviceD3D12::Diagnostic(VriMessageSeverity severity, const char* message) const
    {
        if (m_callback.MessageCallback)
            m_callback.MessageCallback(m_callback.userArg, severity, message);
        else
            std::fprintf(stderr, "[VRI/D3D12] %s\n", message);
    }

    namespace
    {
        // D3D12 debug-layer messages -> the app callback (ID3D12InfoQueue1, Win10 SDK 20348+).
        void CALLBACK D3D12MessageCallback(D3D12_MESSAGE_CATEGORY /*category*/,
                                           D3D12_MESSAGE_SEVERITY severity,
                                           D3D12_MESSAGE_ID /*id*/,
                                           LPCSTR description,
                                           void*  context)
        {
            if (severity == D3D12_MESSAGE_SEVERITY_INFO || severity == D3D12_MESSAGE_SEVERITY_MESSAGE)
                return; // info/message is too chatty
            const auto* dev = static_cast<const DeviceD3D12*>(context);
            if (!dev)
                return;
            const VriMessageSeverity s =
                severity == D3D12_MESSAGE_SEVERITY_WARNING ? VriMessageSeverity_Warning : VriMessageSeverity_Error;
            dev->Diagnostic(s, description ? description : "<null>");
        }
    } // namespace

    VriResult DeviceD3D12::Init(const VriDeviceCreationDesc& desc)
    {
        if (desc.callbackInterface)
            m_callback = *desc.callbackInterface;

        UINT factoryFlags = 0;
        if (desc.enableValidation)
        {
            ComPtr<ID3D12Debug> dbg;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg))))
            {
                dbg->EnableDebugLayer();
                factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
            }
        }
        if (FAILED(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&m_factory))))
        {
            ReportError("CreateDXGIFactory2 failed");
            return VriResult_Failure;
        }

        // Pick the first hardware adapter that supports feature level 11_0; fall back
        // to the WARP software adapter so headless/CI machines without a GPU still work.
        ComPtr<IDXGIAdapter1> chosen;
        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; m_factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i)
        {
            DXGI_ADAPTER_DESC1 ad = {};
            adapter->GetDesc1(&ad);
            if (ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            {
                adapter.Reset();
                continue;
            }
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device))))
            {
                chosen = adapter;
                break;
            }
            adapter.Reset();
        }
        if (!m_device)
        {
            if (SUCCEEDED(m_factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) &&
                SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device))))
            {
                chosen = adapter;
            }
        }
        if (!m_device)
        {
            ReportError("D3D12CreateDevice failed (no FL11_0 hardware or WARP adapter)");
            return VriResult_Unsupported;
        }

        if (desc.enableValidation)
        {
            // Forward debug-layer messages to the app callback (ID3D12InfoQueue1; Win10 SDK
            // 20348+). If unavailable the debug layer still prints to the VS output window.
            if (SUCCEEDED(m_device->QueryInterface(IID_PPV_ARGS(&m_infoQueue))))
                m_infoQueue->RegisterMessageCallback(
                    D3D12MessageCallback, D3D12_MESSAGE_CALLBACK_FLAG_NONE, this, &m_msgCookie);
        }

        // One queue per VriQueueType, each on its own D3D12 engine so Compute/Transfer overlap
        // graphics. Cross-queue ordering is the caller's via timeline fences (QueueSubmit).
        static constexpr D3D12_COMMAND_LIST_TYPE kQueueListType[VriQueueType_Count] = {
            D3D12_COMMAND_LIST_TYPE_DIRECT,  // Graphics
            D3D12_COMMAND_LIST_TYPE_COMPUTE, // Compute  (async compute engine)
            D3D12_COMMAND_LIST_TYPE_COPY,    // Transfer (DMA copy engine)
        };
        for (int t = 0; t < VriQueueType_Count; ++t)
        {
            D3D12_COMMAND_QUEUE_DESC qd = {};
            qd.Type                     = kQueueListType[t];
            if (FAILED(m_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_queues[t].queue))))
            {
                ReportError("CreateCommandQueue failed");
                return VriResult_Failure;
            }
            m_queues[t].device = this;
            m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_queues[t].idleFence)); // for *WaitIdle
            m_queues[t].idleEvent = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        }

        if (VriResult r = CreateDescriptorHeaps(); r != VriResult_Success)
            return r;
        if (VriResult r = NegotiateFeatures(desc); r != VriResult_Success)
            return r;

        FillDeviceDesc(chosen.Get());
        FillRegistry();
        return VriResult_Success;
    }

    VriResult DeviceD3D12::CreateDescriptorHeaps()
    {
        D3D12_DESCRIPTOR_HEAP_DESC rtv = {};
        rtv.Type                       = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtv.NumDescriptors             = kRtvHeapSize;
        rtv.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_NONE; // CPU-only
        if (FAILED(m_device->CreateDescriptorHeap(&rtv, IID_PPV_ARGS(&m_rtvHeap))))
        {
            ReportError("CreateDescriptorHeap (RTV) failed");
            return VriResult_Failure;
        }
        m_rtvSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        D3D12_DESCRIPTOR_HEAP_DESC dsv = {};
        dsv.Type                       = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dsv.NumDescriptors             = kDsvHeapSize;
        dsv.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        if (FAILED(m_device->CreateDescriptorHeap(&dsv, IID_PPV_ARGS(&m_dsvHeap))))
        {
            ReportError("CreateDescriptorHeap (DSV) failed");
            return VriResult_Failure;
        }
        m_dsvSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        return VriResult_Success;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE DeviceD3D12::AllocRtv()
    {
        D3D12_CPU_DESCRIPTOR_HANDLE h = m_rtvHeap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += static_cast<SIZE_T>(m_rtvNext % kRtvHeapSize) * m_rtvSize; // ring (Phase 1: few RTVs)
        ++m_rtvNext;
        return h;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE DeviceD3D12::AllocDsv()
    {
        D3D12_CPU_DESCRIPTOR_HANDLE h = m_dsvHeap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += static_cast<SIZE_T>(m_dsvNext % kDsvHeapSize) * m_dsvSize;
        ++m_dsvNext;
        return h;
    }

    // D3D12 has no per-feature device-creation flag (features are pure capabilities),
    // so "enabling" == checking support and granting. Degradation is explicit:
    // unsupported + !bestEffort fails creation; otherwise the feature is simply not
    // granted and its interface is not registered.
    VriResult DeviceD3D12::NegotiateFeatures(const VriDeviceCreationDesc& desc)
    {
        const uint64_t requested = desc.enabledFeatures;
        uint64_t       granted   = 0;
        auto           fail      = [&](const char* what) -> bool {
            if (desc.bestEffort == VRI_FALSE)
            {
                ReportError(what);
                return true;
            }
            return false;
        };

        if (requested & VriFeature_VariableShadingRate)
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS6 o6 = {};
            const bool ok = SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS6, &o6, sizeof(o6))) &&
                            o6.VariableShadingRateTier >= D3D12_VARIABLE_SHADING_RATE_TIER_1;
            if (ok)
                granted |= VriFeature_VariableShadingRate;
            else if (fail("variable rate shading not supported"))
                return VriResult_Unsupported;
        }

        if (requested & VriFeature_MeshShader)
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS7 o7 = {};
            const bool ok = SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &o7, sizeof(o7))) &&
                            o7.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1;
            if (ok)
                granted |= VriFeature_MeshShader;
            else if (fail("mesh shaders not supported"))
                return VriResult_Unsupported;
        }

        if (requested & VriFeature_RayTracing)
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5 = {};
            const bool ok = SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5))) &&
                            o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0;
            if (ok)
                granted |= VriFeature_RayTracing;
            else if (fail("ray tracing not supported"))
                return VriResult_Unsupported;
        }

        if (requested & VriFeature_RayQuery)
        {
            // Ray query = DXR 1.1 inline ray tracing (RaytracingTier 1.1+), no state object.
            D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5 = {};
            const bool ok = SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5))) &&
                            o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1;
            if (ok)
                granted |= VriFeature_RayQuery;
            else if (fail("ray query not supported (needs Raytracing Tier 1.1)"))
                return VriResult_Unsupported;
        }

        if (requested & VriFeature_OpacityMicromap)
        {
            // OMM extends ray-tracing geometry and is gated by Raytracing Tier 1.2 (DXR 1.2).
            D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5 = {};
            const bool                        ok = (granted & VriFeature_RayTracing) &&
                            SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5))) &&
                            o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_2;
            if (ok)
                granted |= VriFeature_OpacityMicromap;
            else if (fail("opacity micromap not supported (needs ray tracing + Raytracing Tier 1.2 / DXR 1.2 runtime)"))
                return VriResult_Unsupported;
        }
        if (requested & VriFeature_Bindless)
        {
            // Descriptor indexing = dynamic (incl. non-uniform) indexing into descriptor
            // tables; Resource Binding Tier 2 guarantees it for SRV/sampler tables.
            D3D12_FEATURE_DATA_D3D12_OPTIONS o = {};
            const bool ok = SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o, sizeof(o))) &&
                            o.ResourceBindingTier >= D3D12_RESOURCE_BINDING_TIER_2;
            if (ok)
                granted |= VriFeature_Bindless;
            else if (fail("bindless (descriptor indexing) not supported (needs Resource Binding Tier 2)"))
                return VriResult_Unsupported;
        }
        if ((requested & VriFeature_LowLatency) && fail("low-latency not implemented on D3D12"))
            return VriResult_Unsupported;

        // External memory/fence export: D3D12 shared handles (CreateSharedHandle) are always
        // available on Windows, so grant whenever requested.
        if (requested & VriFeature_ExternalMemory)
            granted |= VriFeature_ExternalMemory;

        m_enabledFeatures = granted;
        return VriResult_Success;
    }

    void DeviceD3D12::FillDeviceDesc(IDXGIAdapter1* adapter)
    {
        m_desc             = {};
        m_desc.graphicsAPI = VriGraphicsAPI_D3D12;
        if (adapter)
        {
            DXGI_ADAPTER_DESC1 ad = {};
            adapter->GetDesc1(&ad);
            std::wcstombs(m_desc.adapter.name, ad.Description, sizeof(m_desc.adapter.name) - 1);
            m_desc.adapter.videoMemorySize  = ad.DedicatedVideoMemory;
            m_desc.adapter.sharedMemorySize = ad.SharedSystemMemory;
            m_desc.adapter.deviceId         = ad.DeviceId;
            m_desc.adapter.vendorId         = ad.VendorId;
            if (ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                m_desc.adapter.type = VriAdapterType_Software;
            else
                m_desc.adapter.type = ad.DedicatedVideoMemory > 0 ? VriAdapterType_Discrete : VriAdapterType_Integrated;
        }
        m_desc.apiVersionMajor = 12;
        m_desc.apiVersionMinor = 0;

        // Direct3D feature level 11_0+ guarantees (Tier-1 minimums VRI relies on).
        m_desc.viewportMaxNum          = D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE; // 16
        m_desc.attachmentColorMaxNum   = D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT;                   // 8
        m_desc.attachmentMaxDim        = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;                     // 16384
        m_desc.texture1DMaxDim         = D3D12_REQ_TEXTURE1D_U_DIMENSION;
        m_desc.texture2DMaxDim         = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
        m_desc.texture3DMaxDim         = D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
        m_desc.textureArrayLayerMaxNum = D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION;
        m_desc.bufferMaxSize           = 1ull << 30;
        // Alignments kept Vulkan-compatible across backends.
        m_desc.uploadBufferTextureRowAlignment = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;             // 256
        m_desc.constantBufferOffsetAlignment   = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT; // 256
        m_desc.storageBufferOffsetAlignment    = 32;

        for (int t = 0; t < VriQueueType_Count; ++t)
            m_desc.queueCount[t] = 1;
        m_desc.hasTessellation   = VRI_TRUE;
        m_desc.hasGeometryShader = VRI_TRUE;
        m_desc.hasComputeShader  = VRI_TRUE;

        D3D12_FEATURE_DATA_D3D12_OPTIONS o0 = {};
        if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o0, sizeof(o0))))
            m_desc.hasConservativeRaster =
                o0.ConservativeRasterizationTier >= D3D12_CONSERVATIVE_RASTERIZATION_TIER_1 ? VRI_TRUE : VRI_FALSE;
        D3D12_FEATURE_DATA_D3D12_OPTIONS3 o3 = {};
        if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &o3, sizeof(o3))))
            m_desc.hasFragmentShaderBarycentric = o3.BarycentricsSupported ? VRI_TRUE : VRI_FALSE;
        m_desc.hasCustomBorderColor          = VRI_TRUE; // D3D12 sampler border color is always a float4
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 o1 = {};
        if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &o1, sizeof(o1))))
        {
            m_desc.hasShaderWaveOps = o1.WaveOps ? VRI_TRUE : VRI_FALSE;
            m_desc.subgroupSize     = o1.WaveLaneCountMin; // SM6.0 wave width
        }

        // Timestamp queries are always available on the direct/compute queues; derive the tick
        // scale from the graphics queue's frequency (ticks/sec -> nanoseconds/tick).
        m_desc.hasTimestampQueries = VRI_TRUE;
        UINT64 freq                = 0;
        if (m_queues[VriQueueType_Graphics].queue &&
            SUCCEEDED(m_queues[VriQueueType_Graphics].queue->GetTimestampFrequency(&freq)) && freq != 0)
            m_desc.timestampPeriodNanoseconds = 1.0e9f / static_cast<float>(freq);
        else
            m_desc.hasTimestampQueries = VRI_FALSE;

        m_desc.enabledFeatures        = m_enabledFeatures;
        m_desc.hasVariableShadingRate = (m_enabledFeatures & VriFeature_VariableShadingRate) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasMeshShader          = (m_enabledFeatures & VriFeature_MeshShader) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasRayTracing          = (m_enabledFeatures & VriFeature_RayTracing) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasOpacityMicromap     = (m_enabledFeatures & VriFeature_OpacityMicromap) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasRayQuery            = (m_enabledFeatures & VriFeature_RayQuery) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasBindless            = (m_enabledFeatures & VriFeature_Bindless) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasExternalMemory      = (m_enabledFeatures & VriFeature_ExternalMemory) ? VRI_TRUE : VRI_FALSE;
        if (m_enabledFeatures & (VriFeature_RayTracing | VriFeature_RayQuery))
        {
            // DXR shader-table layout constants (mirror the Vulkan RT props fields).
            m_desc.rtShaderGroupHandleSize      = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;         // 32
            m_desc.rtShaderGroupBaseAlignment   = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;  // 64
            m_desc.rtShaderGroupHandleAlignment = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT; // 32
        }
    }

    ID3D12CommandSignature* DeviceD3D12::DispatchMeshSignature()
    {
        if (!m_dispatchMeshSig)
        {
            D3D12_INDIRECT_ARGUMENT_DESC arg = {};
            arg.Type                         = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
            D3D12_COMMAND_SIGNATURE_DESC sd  = {};
            sd.ByteStride                    = 3 * sizeof(uint32_t); // x, y, z
            sd.NumArgumentDescs              = 1;
            sd.pArgumentDescs                = &arg;
            m_device->CreateCommandSignature(&sd, nullptr, IID_PPV_ARGS(&m_dispatchMeshSig));
        }
        return m_dispatchMeshSig.Get();
    }

    void DeviceD3D12::FillRegistry()
    {
        m_registry.Register(VRI_INTERFACE_CORE, GetCoreInterfaceD3D12(), sizeof(VriCoreInterface));
        m_registry.Register(VRI_INTERFACE_SWAPCHAIN,
                            GetSwapChainInterfaceD3D12(),
                            sizeof(VriSwapChainInterface)); // DXGI flip-model present
        m_registry.Register(VRI_INTERFACE_QUERY, GetQueryInterfaceD3D12(), sizeof(VriQueryInterface));
        if (m_enabledFeatures & VriFeature_VariableShadingRate)
            m_registry.Register(VRI_INTERFACE_VRS, GetShadingRateInterfaceD3D12(), sizeof(VriShadingRateInterface));
        if (m_enabledFeatures & VriFeature_MeshShader)
            m_registry.Register(
                VRI_INTERFACE_MESHSHADER, GetMeshShaderInterfaceD3D12(), sizeof(VriMeshShaderInterface));
        // RT interface also creates acceleration structures, which ray query needs.
        if (m_enabledFeatures & (VriFeature_RayTracing | VriFeature_RayQuery))
            m_registry.Register(
                VRI_INTERFACE_RAYTRACING, GetRayTracingInterfaceD3D12(), sizeof(VriRayTracingInterface));
        if (m_enabledFeatures & VriFeature_OpacityMicromap)
            m_registry.Register(
                VRI_INTERFACE_OMM, GetOpacityMicromapInterfaceD3D12(), sizeof(VriOpacityMicromapInterface));
        if (m_enabledFeatures & VriFeature_ExternalMemory)
            m_registry.Register(VRI_INTERFACE_EXTERNAL, GetExternalInterfaceD3D12(), sizeof(VriExternalInterface));
    }

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult)
    {
        DeviceD3D12* device = new DeviceD3D12();
        outResult           = device->Init(desc);
        if (outResult != VriResult_Success)
        {
            delete device;
            return nullptr;
        }
        return device;
    }
} // namespace vri::d3d12
