#include "device_wgpu.h"
#include "core_wgpu.h"
#include "swapchain_wgpu.h"

#include <webgpu/wgpu.h> // wgpu-native extensions (wgpuDevicePoll)

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
                if (dstSize) dst[0] = '\0';
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
            bool        done = false;
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
            bool       done = false;
            WGPUDevice device = nullptr;
        };

        void OnDevice(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView, void* ud1, void*)
        {
            auto* r = static_cast<DeviceRequest*>(ud1);
            r->done = true;
            if (status == WGPURequestDeviceStatus_Success)
                r->device = device;
        }
    } // namespace

    DeviceWGPU::~DeviceWGPU()
    {
        if (m_device)
        {
            wgpuDevicePoll(m_device, /*wait*/ true, nullptr);
            wgpuDeviceRelease(m_device);
        }
        if (m_adapter)
            wgpuAdapterRelease(m_adapter);
        if (m_instance)
            wgpuInstanceRelease(m_instance);
    }

    void DeviceWGPU::ReportError(const char* message) const
    {
        if (m_callback.MessageCallback)
            m_callback.MessageCallback(m_callback.userArg, VriMessageSeverity_Error, message);
        else
            std::fprintf(stderr, "[VRI/WGPU] %s\n", message);
    }

    VriResult DeviceWGPU::Init(const VriDeviceCreationDesc& desc)
    {
        if (desc.callbackInterface)
            m_callback = *desc.callbackInterface;

        WGPUInstanceDescriptor instanceDesc = {};
        m_instance = wgpuCreateInstance(&instanceDesc);
        if (!m_instance)
        {
            ReportError("wgpuCreateInstance failed");
            return VriResult_Failure;
        }

        // request adapter (async; pumped via ProcessEvents)
        AdapterRequest areq;
        WGPURequestAdapterOptions aopts = {};
        aopts.powerPreference = WGPUPowerPreference_HighPerformance;
        WGPURequestAdapterCallbackInfo acb = {};
        acb.mode = WGPUCallbackMode_AllowProcessEvents;
        acb.callback = OnAdapter;
        acb.userdata1 = &areq;
        wgpuInstanceRequestAdapter(m_instance, &aopts, acb);
        for (int i = 0; i < 100000 && !areq.done; ++i)
            wgpuInstanceProcessEvents(m_instance);
        if (!areq.adapter)
        {
            ReportError("no WebGPU adapter");
            return VriResult_Failure;
        }
        m_adapter = areq.adapter;

        // request device
        DeviceRequest dreq;
        WGPUDeviceDescriptor deviceDesc = {};
        WGPURequestDeviceCallbackInfo dcb = {};
        dcb.mode = WGPUCallbackMode_AllowProcessEvents;
        dcb.callback = OnDevice;
        dcb.userdata1 = &dreq;
        wgpuAdapterRequestDevice(m_adapter, &deviceDesc, dcb);
        for (int i = 0; i < 100000 && !dreq.done; ++i)
            wgpuInstanceProcessEvents(m_instance);
        if (!dreq.device)
        {
            ReportError("wgpuAdapterRequestDevice failed");
            return VriResult_Failure;
        }
        m_device = dreq.device;
        m_queue = wgpuDeviceGetQueue(m_device);
        m_queueObj.device = this;
        m_queueObj.queue = m_queue;

        FillDeviceDesc();
        FillRegistry();
        return VriResult_Success;
    }

    void DeviceWGPU::FillDeviceDesc()
    {
        m_desc = {};
        m_desc.graphicsAPI = VriGraphicsAPI_WebGPU;

        WGPUAdapterInfo info = {};
        wgpuAdapterGetInfo(m_adapter, &info);
        CopyStringView(m_desc.adapter.name, sizeof(m_desc.adapter.name), info.device);
        m_desc.adapter.deviceId = info.deviceID;
        m_desc.adapter.vendorId = info.vendorID;
        switch (info.adapterType)
        {
            case WGPUAdapterType_DiscreteGPU:   m_desc.adapter.type = VriAdapterType_Discrete; break;
            case WGPUAdapterType_IntegratedGPU: m_desc.adapter.type = VriAdapterType_Integrated; break;
            case WGPUAdapterType_CPU:           m_desc.adapter.type = VriAdapterType_Software; break;
            default:                            m_desc.adapter.type = VriAdapterType_Unknown; break;
        }
        wgpuAdapterInfoFreeMembers(info);

        WGPULimits limits = {};
        if (wgpuAdapterGetLimits(m_adapter, &limits) == WGPUStatus_Success)
        {
            m_desc.texture1DMaxDim = limits.maxTextureDimension1D;
            m_desc.texture2DMaxDim = limits.maxTextureDimension2D;
            m_desc.texture3DMaxDim = limits.maxTextureDimension3D;
            m_desc.textureArrayLayerMaxNum = limits.maxTextureArrayLayers;
            m_desc.attachmentColorMaxNum = limits.maxColorAttachments;
            m_desc.bufferMaxSize = limits.maxBufferSize;
            m_desc.constantBufferOffsetAlignment = limits.minUniformBufferOffsetAlignment;
            m_desc.storageBufferOffsetAlignment = limits.minStorageBufferOffsetAlignment;
        }
        m_desc.viewportMaxNum = 1;
        for (int t = 0; t < VriQueueType_Count; ++t)
            m_desc.queueCount[t] = 1;
        m_desc.hasComputeShader = VRI_TRUE;
    }

    void DeviceWGPU::FillRegistry()
    {
        m_registry.Register(VRI_INTERFACE_CORE, GetCoreInterfaceWGPU(), sizeof(VriCoreInterface));
        m_registry.Register(VRI_INTERFACE_SWAPCHAIN, GetSwapChainInterfaceWGPU(), sizeof(VriSwapChainInterface));
    }

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult)
    {
        DeviceWGPU* device = new DeviceWGPU();
        outResult = device->Init(desc);
        if (outResult != VriResult_Success)
        {
            delete device;
            return nullptr;
        }
        return device;
    }
} // namespace vri::wgpu
