#include "device_wgpu.h"
#include "core_wgpu.h"
#include "query_wgpu.h"
#include "swapchain_wgpu.h"
#include "wgpu_native.h" // webgpu.h + native poll helpers (browser yields via ASYNCIFY)

#include "core/imgui_vri.h" // backend-agnostic built-in ImGui renderer

#include <cstdio>
#include <cstring>

namespace vri::wgpu
{
    namespace
    {
        // Copy a WGPUStringView into a fixed C buffer.
        void CopyStringView(char* dst, size_t dstSize, WGPUStringView sv)
        {
            if (!sv.data || dstSize == 0)
            {
                if (dstSize)
                    dst[0] = '\0';
                return;
            }
            size_t n = sv.length;
            if (n == WGPU_STRLEN)
                n = std::strlen(sv.data);
            if (n >= dstSize)
                n = dstSize - 1;
            std::memcpy(dst, sv.data, n);
            dst[n] = '\0';
        }

        struct AdapterRequest
        {
            bool        done    = false;
            WGPUAdapter adapter = nullptr;
        };

        void OnAdapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView, void* ud1, void*)
        {
            auto* r = static_cast<AdapterRequest*>(ud1);
            r->done = true;
            if (status == WGPURequestAdapterStatus_Success)
                r->adapter = adapter;
        }

        struct DeviceRequest
        {
            bool       done   = false;
            WGPUDevice device = nullptr;
        };

        void OnDevice(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView, void* ud1, void*)
        {
            auto* r = static_cast<DeviceRequest*>(ud1);
            r->done = true;
            if (status == WGPURequestDeviceStatus_Success)
                r->device = device;
        }

        // WebGPU validation errors are otherwise silent (they go to the browser devtools
        // console, which emrun does not forward). Route them to the app's message callback
        // (userdata1 = the device) so every backend surfaces native diagnostics uniformly -
        // matching VRI's "explicit, never silent" stance.
        void OnUncapturedError(const WGPUDevice*, WGPUErrorType type, WGPUStringView message, void* ud1, void*)
        {
            char buf[1024];
            std::snprintf(buf,
                          sizeof(buf),
                          "uncaptured error (type %d): %.*s",
                          static_cast<int>(type),
                          static_cast<int>(message.length),
                          message.data ? message.data : "");
            if (const auto* dev = static_cast<const DeviceWGPU*>(ud1))
                dev->Diagnostic(VriMessageSeverity_Error, buf);
            else
                std::fprintf(stderr, "[VRI/WGPU] %s\n", buf);
        }
    } // namespace

    DeviceWGPU::~DeviceWGPU()
    {
        if (m_device)
            PollDevice(m_device);
        // Release the queue too (wgpuDeviceGetQueue returns an owned ref that holds the
        // device alive); without this the device + its memory leak across create/destroy
        // cycles, exhausting limited adapters (CI software WebGPU OOMs).
        if (m_queue)
            wgpuQueueRelease(m_queue);
        if (m_device)
            wgpuDeviceRelease(m_device);
        if (m_adapter)
            wgpuAdapterRelease(m_adapter);
        if (m_instance)
            wgpuInstanceRelease(m_instance);
    }

    void DeviceWGPU::ReportError(const char* message) const { Diagnostic(VriMessageSeverity_Error, message); }
    void DeviceWGPU::ReportWarning(const char* message) const { Diagnostic(VriMessageSeverity_Warning, message); }

    void DeviceWGPU::Diagnostic(VriMessageSeverity severity, const char* message) const
    {
        if (m_callback.MessageCallback)
            m_callback.MessageCallback(m_callback.userArg, severity, message);
        else
            std::fprintf(stderr, "[VRI/WGPU] %s\n", message);
    }

    VriResult DeviceWGPU::Init(const VriDeviceCreationDesc& desc)
    {
        if (desc.callbackInterface)
            m_callback = *desc.callbackInterface;

        WGPUInstanceDescriptor instanceDesc = {};
        m_instance                          = wgpuCreateInstance(&instanceDesc);
        if (!m_instance)
        {
            ReportError("wgpuCreateInstance failed");
            return VriResult_Failure;
        }

        // request adapter (async; pumped via ProcessEvents)
        AdapterRequest            areq;
        WGPURequestAdapterOptions aopts    = {};
        aopts.powerPreference              = WGPUPowerPreference_HighPerformance;
        WGPURequestAdapterCallbackInfo acb = {};
        acb.mode                           = kCallbackMode;
        acb.callback                       = OnAdapter;
        acb.userdata1                      = &areq;
        wgpuInstanceRequestAdapter(m_instance, &aopts, acb);
        for (int i = 0; i < 100000 && !areq.done; ++i)
            PumpInstance(m_instance);
        if (!areq.adapter)
        {
            ReportError("no WebGPU adapter");
            return VriResult_Failure;
        }
        m_adapter = areq.adapter;

        // Opt into timestamp queries when the adapter supports them (for the query interface).
        // VRI's CmdWriteTimestamp records at arbitrary encoder points, which wgpu-native gates
        // behind the native TimestampQueryInsideEncoders feature; the browser (emdawnwebgpu) has
        // no such feature (timestamps are pass-scoped there), so timestamps are native-only.
        const bool hasTsQuery = wgpuAdapterHasFeature(m_adapter, WGPUFeatureName_TimestampQuery) != 0;
#if !defined(__EMSCRIPTEN__)
        const bool hasTsInside =
            wgpuAdapterHasFeature(m_adapter,
                                  static_cast<WGPUFeatureName>(WGPUNativeFeature_TimestampQueryInsideEncoders)) != 0;
#else
        const bool hasTsInside = false;
#endif
        m_hasTimestamp = hasTsQuery && hasTsInside;

        // request device
        DeviceRequest        dreq;
        WGPUDeviceDescriptor deviceDesc                  = {};
        deviceDesc.uncapturedErrorCallbackInfo.callback  = OnUncapturedError;
        deviceDesc.uncapturedErrorCallbackInfo.userdata1 = this;
        WGPUFeatureName requiredFeatures[2]              = {};
        uint32_t        featureCount                     = 0;
        if (m_hasTimestamp)
        {
            requiredFeatures[featureCount++] = WGPUFeatureName_TimestampQuery;
#if !defined(__EMSCRIPTEN__)
            requiredFeatures[featureCount++] =
                static_cast<WGPUFeatureName>(WGPUNativeFeature_TimestampQueryInsideEncoders);
#endif
            deviceDesc.requiredFeatures     = requiredFeatures;
            deviceDesc.requiredFeatureCount = featureCount;
        }
        WGPURequestDeviceCallbackInfo dcb = {};
        dcb.mode                          = kCallbackMode;
        dcb.callback                      = OnDevice;
        dcb.userdata1                     = &dreq;
        wgpuAdapterRequestDevice(m_adapter, &deviceDesc, dcb);
        for (int i = 0; i < 100000 && !dreq.done; ++i)
            PumpInstance(m_instance);
        if (!dreq.device)
        {
            ReportError("wgpuAdapterRequestDevice failed");
            return VriResult_Failure;
        }
        m_device          = dreq.device;
        m_queue           = wgpuDeviceGetQueue(m_device);
        m_queueObj.device = this;
        m_queueObj.queue  = m_queue;

        FillDeviceDesc();
        FillRegistry();
        return VriResult_Success;
    }

    void DeviceWGPU::FillDeviceDesc()
    {
        m_desc             = {};
        m_desc.graphicsAPI = VriGraphicsAPI_WebGPU;

        WGPUAdapterInfo info = {};
        wgpuAdapterGetInfo(m_adapter, &info);
        CopyStringView(m_desc.adapter.name, sizeof(m_desc.adapter.name), info.device);
        m_desc.adapter.deviceId = info.deviceID;
        m_desc.adapter.vendorId = info.vendorID;
        switch (info.adapterType)
        {
            case WGPUAdapterType_DiscreteGPU:
                m_desc.adapter.type = VriAdapterType_Discrete;
                break;
            case WGPUAdapterType_IntegratedGPU:
                m_desc.adapter.type = VriAdapterType_Integrated;
                break;
            case WGPUAdapterType_CPU:
                m_desc.adapter.type = VriAdapterType_Software;
                break;
            default:
                m_desc.adapter.type = VriAdapterType_Unknown;
                break;
        }
        wgpuAdapterInfoFreeMembers(info);

        WGPULimits limits = {};
        if (wgpuAdapterGetLimits(m_adapter, &limits) == WGPUStatus_Success)
        {
            m_desc.texture1DMaxDim               = limits.maxTextureDimension1D;
            m_desc.texture2DMaxDim               = limits.maxTextureDimension2D;
            m_desc.texture3DMaxDim               = limits.maxTextureDimension3D;
            m_desc.textureArrayLayerMaxNum       = limits.maxTextureArrayLayers;
            m_desc.attachmentColorMaxNum         = limits.maxColorAttachments;
            m_desc.bufferMaxSize                 = limits.maxBufferSize;
            m_desc.constantBufferOffsetAlignment = limits.minUniformBufferOffsetAlignment;
            m_desc.storageBufferOffsetAlignment  = limits.minStorageBufferOffsetAlignment;
        }
        m_desc.viewportMaxNum = 1;
        for (int t = 0; t < VriQueueType_Count; ++t)
            m_desc.queueCount[t] = 1;
        // WebGPU compute is wired (CreateComputePipeline + compute pass / CmdDispatch),
        // on both native (wgpu-native) and the browser (emdawnwebgpu).
        m_desc.hasComputeShader = VRI_TRUE;
        // WebGPU has no geometry or tessellation stages.
        m_desc.hasGeometryShader = VRI_FALSE;
        m_desc.hasTessellation   = VRI_FALSE;
        // Timestamp query values are already in nanoseconds (period = 1).
        m_desc.hasTimestampQueries        = m_hasTimestamp ? VRI_TRUE : VRI_FALSE;
        m_desc.timestampPeriodNanoseconds = m_hasTimestamp ? 1.0f : 0.0f;
    }

    void DeviceWGPU::FillRegistry()
    {
        m_registry.Register(VRI_INTERFACE_CORE, GetCoreInterfaceWGPU(), sizeof(VriCoreInterface));
        m_registry.Register(VRI_INTERFACE_SWAPCHAIN, GetSwapChainInterfaceWGPU(), sizeof(VriSwapChainInterface));
        m_registry.Register(VRI_INTERFACE_QUERY, GetQueryInterfaceWGPU(), sizeof(VriQueryInterface));
        m_registry.Register(VRI_INTERFACE_IMGUI, core::GetImguiInterface(), sizeof(VriImguiInterface));
    }

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult)
    {
        DeviceWGPU* device = new DeviceWGPU();
        outResult          = device->Init(desc);
        if (outResult != VriResult_Success)
        {
            delete device;
            return nullptr;
        }
        return device;
    }
} // namespace vri::wgpu
