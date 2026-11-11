// device_mtl.h - the native Metal VriDevice implementation + factory.
#pragma once

#import <Metal/Metal.h>

#include <vri/vri.h>

#include <mutex>

#include "core/device_base.h"
#include "objects_mtl.h"

namespace vri::mtl
{
    class DeviceMTL final : public core::DeviceBase
    {
    public:
        ~DeviceMTL() override;

        VriResult Init(const VriDeviceCreationDesc& desc);

        id<MTLDevice>       Device() const { return m_device; }
        MTLSharedEventListener* EventListener() const { return m_eventListener; }
        id<MTLCommandQueue> Queue() const { return m_queue; }
        QueueMTL*           GetQueue(VriQueueType type) { return &m_queueObjs[type]; }
        const VriDeviceDesc& Desc() const { return m_desc; }

        void ReportError(const char* message) const;
        void Diagnostic(VriMessageSeverity severity, const char* message) const;
        id<MTLComputePipelineState> ClearBufferPipeline();
        id<MTLComputePipelineState> ClearTexturePipeline(uint32_t dimension, uint32_t component);

    private:
        void FillDeviceDesc();
        void FillRegistry();
        bool HasTimestampCounterSet() const;
        bool SupportsArgumentBufferBindless() const;

        MTLSharedEventListener* m_eventListener = nil;
        id<MTLDevice>        m_device = nil;
        id<MTLCommandQueue>  m_queue = nil;
        id<MTLComputePipelineState> m_clearBufferPipeline = nil;
        id<MTLComputePipelineState> m_clearTexturePipelines[5][3] = {};
        std::mutex           m_clearTextureMutex;
        QueueMTL             m_queueObjs[VriQueueType_Count] = {};
        VriDeviceDesc        m_desc = {};
        uint64_t             m_enabledFeatures = 0;
        VriCallbackInterface m_callback = {};
    };

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult);
} // namespace vri::mtl
