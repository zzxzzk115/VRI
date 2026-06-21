#include "device_d3d12.h"
#include "core_d3d12.h"
#include "swapchain_d3d12.h"
#include "vrs_d3d12.h"
#include "meshshader_d3d12.h"
#include "rt_d3d12.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vri::d3d12
{
    DeviceD3D12::~DeviceD3D12()
    {
        if (m_queue.idleEvent) CloseHandle(m_queue.idleEvent);
    }

    void DeviceD3D12::ReportError(const char* message) const
    {
        if (m_callback.MessageCallback)
            m_callback.MessageCallback(m_callback.userArg, VriMessageSeverity_Error, message);
        else
            std::fprintf(stderr, "[VRI/D3D12] %s\n", message);
    }

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
            if (ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter.Reset(); continue; }
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device))))
            { chosen = adapter; break; }
            adapter.Reset();
        }
        if (!m_device)
        {
            if (SUCCEEDED(m_factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) &&
                SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device))))
            { chosen = adapter; }
        }
        if (!m_device)
        {
            ReportError("D3D12CreateDevice failed (no FL11_0 hardware or WARP adapter)");
            return VriResult_Unsupported;
        }

        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(m_device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m_queue.queue))))
        {
            ReportError("CreateCommandQueue failed");
            return VriResult_Failure;
        }
        m_queue.device = this;
        m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_queue.idleFence)); // for *WaitIdle
        m_queue.idleEvent = CreateEventA(nullptr, FALSE, FALSE, nullptr);

        if (VriResult r = CreateDescriptorHeaps(); r != VriResult_Success) return r;
        if (VriResult r = NegotiateFeatures(desc); r != VriResult_Success) return r;

        FillDeviceDesc(chosen.Get());
        FillRegistry();
        return VriResult_Success;
    }

    VriResult DeviceD3D12::CreateDescriptorHeaps()
    {
        D3D12_DESCRIPTOR_HEAP_DESC rtv = {};
        rtv.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtv.NumDescriptors = kRtvHeapSize;
        rtv.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE; // CPU-only
        if (FAILED(m_device->CreateDescriptorHeap(&rtv, IID_PPV_ARGS(&m_rtvHeap))))
        {
            ReportError("CreateDescriptorHeap (RTV) failed");
            return VriResult_Failure;
        }
        m_rtvSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

        D3D12_DESCRIPTOR_HEAP_DESC dsv = {};
        dsv.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dsv.NumDescriptors = kDsvHeapSize;
        dsv.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
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
        uint64_t granted = 0;
        auto fail = [&](const char* what) -> bool {
            if (desc.bestEffort == VRI_FALSE) { ReportError(what); return true; }
            return false;
        };

        if (requested & VriFeature_VariableShadingRate)
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS6 o6 = {};
            const bool ok = SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS6, &o6, sizeof(o6))) &&
                            o6.VariableShadingRateTier >= D3D12_VARIABLE_SHADING_RATE_TIER_1;
            if (ok) granted |= VriFeature_VariableShadingRate;
            else if (fail("variable rate shading not supported")) return VriResult_Unsupported;
        }

        if (requested & VriFeature_MeshShader)
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS7 o7 = {};
            const bool ok = SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &o7, sizeof(o7))) &&
                            o7.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1;
            if (ok) granted |= VriFeature_MeshShader;
            else if (fail("mesh shaders not supported")) return VriResult_Unsupported;
        }

        if (requested & VriFeature_RayTracing)
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5 = {};
            const bool ok = SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5))) &&
                            o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_0;
            if (ok) granted |= VriFeature_RayTracing;
            else if (fail("ray tracing not supported")) return VriResult_Unsupported;
        }

        if ((requested & VriFeature_OpacityMicromap) && fail("opacity micromap not supported on D3D12"))
            return VriResult_Unsupported;
        if ((requested & VriFeature_Bindless) && fail("bindless not implemented on D3D12"))
            return VriResult_Unsupported;
        if ((requested & VriFeature_LowLatency) && fail("low-latency not implemented on D3D12"))
            return VriResult_Unsupported;

        m_enabledFeatures = granted;
        return VriResult_Success;
    }

    void DeviceD3D12::FillDeviceDesc(IDXGIAdapter1* adapter)
    {
        m_desc = {};
        m_desc.graphicsAPI = VriGraphicsAPI_D3D12;
        if (adapter)
        {
            DXGI_ADAPTER_DESC1 ad = {};
            adapter->GetDesc1(&ad);
            std::wcstombs(m_desc.adapter.name, ad.Description, sizeof(m_desc.adapter.name) - 1);
            m_desc.adapter.videoMemorySize = ad.DedicatedVideoMemory;
            m_desc.adapter.sharedMemorySize = ad.SharedSystemMemory;
            m_desc.adapter.deviceId = ad.DeviceId;
            m_desc.adapter.vendorId = ad.VendorId;
            if (ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) m_desc.adapter.type = VriAdapterType_Software;
            else m_desc.adapter.type = ad.DedicatedVideoMemory > 0 ? VriAdapterType_Discrete : VriAdapterType_Integrated;
        }
        m_desc.apiVersionMajor = 12;
        m_desc.apiVersionMinor = 0;

        // Direct3D feature level 11_0+ guarantees (Tier-1 minimums VRI relies on).
        m_desc.viewportMaxNum = D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE; // 16
        m_desc.attachmentColorMaxNum = D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT;            // 8
        m_desc.attachmentMaxDim = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;                   // 16384
        m_desc.texture1DMaxDim = D3D12_REQ_TEXTURE1D_U_DIMENSION;
        m_desc.texture2DMaxDim = D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
        m_desc.texture3DMaxDim = D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
        m_desc.textureArrayLayerMaxNum = D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION;
        m_desc.bufferMaxSize = 1ull << 30;
        // Alignments kept Vulkan-compatible across backends.
        m_desc.uploadBufferTextureRowAlignment = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;  // 256
        m_desc.constantBufferOffsetAlignment = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT; // 256
        m_desc.storageBufferOffsetAlignment = 32;

        for (int t = 0; t < VriQueueType_Count; ++t)
            m_desc.queueCount[t] = 1;
        m_desc.hasTessellation = VRI_TRUE;
        m_desc.hasGeometryShader = VRI_TRUE;
        m_desc.hasComputeShader = VRI_TRUE;

        m_desc.enabledFeatures = m_enabledFeatures;
        m_desc.hasVariableShadingRate = (m_enabledFeatures & VriFeature_VariableShadingRate) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasMeshShader = (m_enabledFeatures & VriFeature_MeshShader) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasRayTracing = (m_enabledFeatures & VriFeature_RayTracing) ? VRI_TRUE : VRI_FALSE;
        if (m_enabledFeatures & VriFeature_RayTracing)
        {
            // DXR shader-table layout constants (mirror the Vulkan RT props fields).
            m_desc.rtShaderGroupHandleSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;        // 32
            m_desc.rtShaderGroupBaseAlignment = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT; // 64
            m_desc.rtShaderGroupHandleAlignment = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT; // 32
        }
    }

    ID3D12CommandSignature* DeviceD3D12::DispatchMeshSignature()
    {
        if (!m_dispatchMeshSig)
        {
            D3D12_INDIRECT_ARGUMENT_DESC arg = {};
            arg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
            D3D12_COMMAND_SIGNATURE_DESC sd = {};
            sd.ByteStride = 3 * sizeof(uint32_t); // x, y, z
            sd.NumArgumentDescs = 1;
            sd.pArgumentDescs = &arg;
            m_device->CreateCommandSignature(&sd, nullptr, IID_PPV_ARGS(&m_dispatchMeshSig));
        }
        return m_dispatchMeshSig.Get();
    }

    void DeviceD3D12::FillRegistry()
    {
        m_registry.Register(VRI_INTERFACE_CORE, GetCoreInterfaceD3D12(), sizeof(VriCoreInterface));
        m_registry.Register(VRI_INTERFACE_SWAPCHAIN, GetSwapChainInterfaceD3D12(), sizeof(VriSwapChainInterface)); // DXGI flip-model present
        if (m_enabledFeatures & VriFeature_VariableShadingRate)
            m_registry.Register(VRI_INTERFACE_VRS, GetShadingRateInterfaceD3D12(), sizeof(VriShadingRateInterface));
        if (m_enabledFeatures & VriFeature_MeshShader)
            m_registry.Register(VRI_INTERFACE_MESHSHADER, GetMeshShaderInterfaceD3D12(), sizeof(VriMeshShaderInterface));
        if (m_enabledFeatures & VriFeature_RayTracing)
            m_registry.Register(VRI_INTERFACE_RAYTRACING, GetRayTracingInterfaceD3D12(), sizeof(VriRayTracingInterface));
    }

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult)
    {
        DeviceD3D12* device = new DeviceD3D12();
        outResult = device->Init(desc);
        if (outResult != VriResult_Success)
        {
            delete device;
            return nullptr;
        }
        return device;
    }
} // namespace vri::d3d12
