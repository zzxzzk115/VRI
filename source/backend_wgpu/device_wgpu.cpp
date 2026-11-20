#include "device_wgpu.h"
#include "core_wgpu.h"
#include "query_wgpu.h"
#include "swapchain_wgpu.h"
#include "wgpu_native.h" // webgpu.h + native poll helpers (browser yields via ASYNCIFY)

#include "core/imgui_vri.h" // backend-agnostic built-in ImGui renderer

#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>

namespace vri::wgpu
{
    namespace
    {
        // The API a WebGPU adapter is implemented on.
        VriGraphicsAPI ToNativeAPI(WGPUBackendType backend)
        {
            switch (backend)
            {
                case WGPUBackendType_WebGPU:
                    return VriGraphicsAPI_WebGPU;
                case WGPUBackendType_D3D11:
                    return VriGraphicsAPI_D3D11;
                case WGPUBackendType_D3D12:
                    return VriGraphicsAPI_D3D12;
                case WGPUBackendType_Metal:
                    return VriGraphicsAPI_Metal;
                case WGPUBackendType_Vulkan:
                    return VriGraphicsAPI_Vulkan;
                case WGPUBackendType_OpenGL:
                    return VriGraphicsAPI_OpenGL;
                case WGPUBackendType_OpenGLES:
                    return VriGraphicsAPI_OpenGLES;
                default: // Undefined, Null
                    return VriGraphicsAPI_None;
            }
        }

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
        if (!CanWait())
        {
            ReportWarning("a browser WebGPU device is created asynchronously: use vriCreateDeviceAsync, or link the "
                          "program with -sASYNCIFY");
            return VriResult_Unsupported;
        }

        struct Pending
        {
            bool      done   = false;
            VriResult result = VriResult_Failure;
        } pending;
        InitAsync(
            desc,
            [](void* userData, VriResult result) {
                auto* p   = static_cast<Pending*>(userData);
                p->result = result;
                p->done   = true;
            },
            &pending);
        for (int i = 0; i < 100000 && !pending.done; ++i)
            PumpInstance(m_instance);
        if (!pending.done)
        {
            ReportError("the WebGPU adapter or device request did not complete");
            return VriResult_Failure;
        }
        return pending.result;
    }

    void DeviceWGPU::InitAsync(const VriDeviceCreationDesc& desc, InitCallback done, void* userData)
    {
        if (desc.callbackInterface)
            m_callback = *desc.callbackInterface;
        m_initDone     = done;
        m_initUserData = userData;
        m_coordinates  = desc.coordinates;

        WGPUInstanceDescriptor instanceDesc = {};
        m_instance                          = wgpuCreateInstance(&instanceDesc);
        if (!m_instance)
        {
            ReportError("wgpuCreateInstance failed");
            FinishInit(VriResult_Failure);
            return;
        }

        WGPURequestAdapterOptions aopts    = {};
        aopts.powerPreference              = WGPUPowerPreference_HighPerformance;
        WGPURequestAdapterCallbackInfo acb = {};
        acb.mode                           = kCallbackMode;
        acb.callback                       = OnAdapter;
        acb.userdata1                      = this;
        wgpuInstanceRequestAdapter(m_instance, &aopts, acb);
    }

    void DeviceWGPU::OnAdapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView, void* ud1, void*)
    {
        auto* self = static_cast<DeviceWGPU*>(ud1);
        if (status != WGPURequestAdapterStatus_Success || !adapter)
        {
            self->ReportError("no WebGPU adapter");
            self->FinishInit(VriResult_Failure);
            return;
        }
        self->m_adapter = adapter;
        self->RequestDevice();
    }

    void DeviceWGPU::RequestDevice()
    {
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

        WGPUDeviceDescriptor deviceDesc                  = {};
        deviceDesc.uncapturedErrorCallbackInfo.callback  = OnUncapturedError;
        deviceDesc.uncapturedErrorCallbackInfo.userdata1 = this;
        // At most: two timestamp features, adapter-specific formats, float32-filterable, three compression families.
        WGPUFeatureName requiredFeatures[7] = {};
        uint32_t        featureCount        = 0;
        if (m_hasTimestamp)
        {
            requiredFeatures[featureCount++] = WGPUFeatureName_TimestampQuery;
#if !defined(__EMSCRIPTEN__)
            requiredFeatures[featureCount++] =
                static_cast<WGPUFeatureName>(WGPUNativeFeature_TimestampQueryInsideEncoders);
#endif
        }
#if !defined(__EMSCRIPTEN__)
        // wgpu-native keeps read-write storage textures, even the r32 formats core WebGPU allows, behind this
        // native feature; the browser has them in core.
        const auto adapterFormats =
            static_cast<WGPUFeatureName>(WGPUNativeFeature_TextureAdapterSpecificFormatFeatures);
        if (wgpuAdapterHasFeature(m_adapter, adapterFormats) != 0)
            requiredFeatures[featureCount++] = adapterFormats;
#endif
        // 32-bit float textures are filterable where the adapter allows it; without the feature they are only
        // loaded, or sampled with a non-filtering sampler.
        m_hasFloat32Filterable = wgpuAdapterHasFeature(m_adapter, WGPUFeatureName_Float32Filterable) != 0;
        if (m_hasFloat32Filterable)
            requiredFeatures[featureCount++] = WGPUFeatureName_Float32Filterable;
        // Every compressed family the adapter has: desktop GPUs offer BC, mobile and Apple GPUs ETC2 and ASTC.
        for (const WGPUFeatureName compression : {WGPUFeatureName_TextureCompressionBC,
                                                  WGPUFeatureName_TextureCompressionETC2,
                                                  WGPUFeatureName_TextureCompressionASTC})
        {
            if (wgpuAdapterHasFeature(m_adapter, compression) != 0)
                requiredFeatures[featureCount++] = compression;
        }
        deviceDesc.requiredFeatures       = featureCount ? requiredFeatures : nullptr;
        deviceDesc.requiredFeatureCount   = featureCount;
        WGPURequestDeviceCallbackInfo dcb = {};
        dcb.mode                          = kCallbackMode;
        dcb.callback                      = OnDevice;
        dcb.userdata1                     = this;
        wgpuAdapterRequestDevice(m_adapter, &deviceDesc, dcb);
    }

    void DeviceWGPU::OnDevice(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView, void* ud1, void*)
    {
        auto* self = static_cast<DeviceWGPU*>(ud1);
        if (status != WGPURequestDeviceStatus_Success || !device)
        {
            self->ReportError("wgpuAdapterRequestDevice failed");
            self->FinishInit(VriResult_Failure);
            return;
        }
        self->m_device          = device;
        self->m_queue           = wgpuDeviceGetQueue(device);
        self->m_queueObj.device = self;
        self->m_queueObj.queue  = self->m_queue;

        self->FillDeviceDesc();
        self->m_desc.coordinates = self->m_coordinates;
        self->FillRegistry();
        self->FinishInit(VriResult_Success);
    }

    void DeviceWGPU::FinishInit(VriResult result)
    {
        // Last statement: the callback may delete this device.
        const InitCallback done = std::exchange(m_initDone, nullptr);
        done(m_initUserData, result);
    }

    void DeviceWGPU::FillDeviceDesc()
    {
        m_desc             = {};
        m_desc.graphicsAPI = VriGraphicsAPI_WebGPU;
        // Buffer/texture copy layouts require aligned rows in both transfer directions.
        m_desc.uploadBufferTextureRowAlignment = 256;

        WGPUAdapterInfo info = {};
        wgpuAdapterGetInfo(m_adapter, &info);
        CopyStringView(m_desc.adapter.name, sizeof(m_desc.adapter.name), info.device);
        m_desc.adapter.deviceId = info.deviceID;
        m_desc.adapter.vendorId = info.vendorID;
        m_desc.nativeAPI        = ToNativeAPI(info.backendType);
        m_desc.nativeValidation = VRI_TRUE; // WebGPU validates every call; errors reach OnUncapturedError
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

        // The device was created with the core default limits, which are lower than the adapter's.
        WGPULimits limits = {};
        if (wgpuDeviceGetLimits(m_device, &limits) == WGPUStatus_Success)
        {
            m_desc.texture1DMaxDim         = limits.maxTextureDimension1D;
            m_desc.texture2DMaxDim         = limits.maxTextureDimension2D;
            m_desc.texture3DMaxDim         = limits.maxTextureDimension3D;
            m_desc.textureArrayLayerMaxNum = limits.maxTextureArrayLayers;
            // wgpu-native leaves this field 0; such a device has the core default of 8.
            m_desc.attachmentColorMaxNum         = limits.maxColorAttachments != 0 ? limits.maxColorAttachments : 8;
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
        // CmdClearStorageTexture copies from a buffer: the formats WebGPU stores, on CopyDst textures.
        m_desc.hasClearStorageTexture = VRI_TRUE;
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

    void CreateDeviceAsync(const VriDeviceCreationDesc& desc, DeviceCreatedCallback done, void* userData)
    {
#if defined(__EMSCRIPTEN__)
        struct Creation
        {
            DeviceWGPU*           device;
            DeviceCreatedCallback done;
            void*                 userData;
        };
        auto* creation = new Creation {new DeviceWGPU(), done, userData};
        creation->device->InitAsync(
            desc,
            [](void* ud, VriResult result) {
                const std::unique_ptr<Creation> c(static_cast<Creation*>(ud));
                if (result != VriResult_Success)
                {
                    delete c->device;
                    c->done(c->userData, result, nullptr);
                    return;
                }
                c->done(c->userData, VriResult_Success, c->device);
            },
            creation);
#else
        // wgpu-native runs its callbacks only while the instance's events are pumped, which the
        // synchronous Init does.
        VriResult         result = VriResult_Failure;
        core::DeviceBase* device = CreateDevice(desc, result);
        done(userData, result, device);
#endif
    }
} // namespace vri::wgpu
