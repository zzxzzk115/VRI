// device_mtl.h - the native Metal VriDevice implementation + factory.
#pragma once

#import <Metal/Metal.h>

#include <vri/vri.h>

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
        id<MTLCommandQueue> Queue() const { return m_queue; }
        QueueMTL*           GetQueue(VriQueueType type) { return &m_queueObjs[type]; }
        const VriDeviceDesc& Desc() const { return m_desc; }

        void ReportError(const char* message) const;
        void Diagnostic(VriMessageSeverity severity, const char* message) const;

    private:
        void FillDeviceDesc();
        void FillRegistry();

        id<MTLDevice>        m_device = nil;
        id<MTLCommandQueue>  m_queue = nil;
        QueueMTL             m_queueObjs[VriQueueType_Count] = {};
        VriDeviceDesc        m_desc = {};
        VriCallbackInterface m_callback = {};
    };

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult);
} // namespace vri::mtl
