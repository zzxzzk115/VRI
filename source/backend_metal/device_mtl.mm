// device_mtl.mm - native Metal device creation, capability query, interface table.

#include "device_mtl.h"
#include "core_mtl.h"
#include "swapchain_mtl.h"

#include <cstdio>
#include <cstring>

namespace vri::mtl
{
    DeviceMTL::~DeviceMTL()
    {
        // QueueMTL objects are members (not heap), nothing to free there.
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

        for (int t = 0; t < VriQueueType_Count; ++t)
        {
            m_queueObjs[t].device = this;
            m_queueObjs[t].queue = m_queue; // Metal has a single unified queue; all VRI queue types share it
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
    }

    void DeviceMTL::FillRegistry()
    {
        m_registry.Register(VRI_INTERFACE_CORE, GetCoreInterfaceMTL(), sizeof(VriCoreInterface));
        m_registry.Register(VRI_INTERFACE_SWAPCHAIN, GetSwapChainInterfaceMTL(), sizeof(VriSwapChainInterface));
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
