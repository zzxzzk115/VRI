// device_wgpu.h - the WebGPU (wgpu-native) VriDevice implementation + factory.
#pragma once

#include <webgpu/webgpu.h>

#include <vri/vri.h>

#include "core/device_base.h"
#include "objects_wgpu.h"

namespace vri::wgpu
{
    class DeviceWGPU final : public core::DeviceBase
    {
    public:
        ~DeviceWGPU() override;

        VriResult Init(const VriDeviceCreationDesc& desc);

        WGPUInstance Instance() const { return m_instance; }
        WGPUAdapter  Adapter() const { return m_adapter; }
        WGPUDevice   Device() const { return m_device; }
        WGPUQueue    Queue() const { return m_queue; }
        QueueWGPU*   GetQueue(VriQueueType /*type*/) { return &m_queueObj; }
        const VriDeviceDesc& Desc() const { return m_desc; }

        void ReportError(const char* message) const;
        // Route a backend diagnostic (native WebGPU error/warning) to the app's callback.
        void Diagnostic(VriMessageSeverity severity, const char* message) const;

    private:
        void FillDeviceDesc();
        void FillRegistry();

        WGPUInstance         m_instance = nullptr;
        WGPUAdapter          m_adapter = nullptr;
        WGPUDevice           m_device = nullptr;
        WGPUQueue            m_queue = nullptr;
        QueueWGPU            m_queueObj = {};
        VriDeviceDesc        m_desc = {};
        VriCallbackInterface m_callback = {};
    };

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult);
} // namespace vri::wgpu
