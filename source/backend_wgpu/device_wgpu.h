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

        using InitCallback = void (*)(void* userData, VriResult result);

        // Returns once the adapter and device requests have completed, pumping WebGPU's events while
        // it waits. In the browser that wait needs -sASYNCIFY; without it this reports Unsupported.
        VriResult Init(const VriDeviceCreationDesc& desc);
        // Requests the adapter and device and calls `done` once with the outcome: in the browser from
        // its event loop, natively from whoever pumps the instance's events (Init). `done` may delete
        // the device.
        void InitAsync(const VriDeviceCreationDesc& desc, InitCallback done, void* userData);

        WGPUInstance         Instance() const { return m_instance; }
        WGPUAdapter          Adapter() const { return m_adapter; }
        WGPUDevice           Device() const { return m_device; }
        WGPUQueue            Queue() const { return m_queue; }
        QueueWGPU*           GetQueue(VriQueueType /*type*/) { return &m_queueObj; }
        const VriDeviceDesc& Desc() const { return m_desc; }
        bool                 HasTimestamp() const { return m_hasTimestamp; }
        bool                 HasFloat32Filterable() const { return m_hasFloat32Filterable; }

        void ReportError(const char* message) const;
        void ReportWarning(const char* message) const;
        // Route a backend diagnostic (native WebGPU error/warning) to the app's callback.
        void Diagnostic(VriMessageSeverity severity, const char* message) const;

    private:
        static void OnAdapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView, void* ud1, void*);
        static void OnDevice(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView, void* ud1, void*);
        void        RequestDevice();
        void        FinishInit(VriResult result);
        void        FillDeviceDesc();
        void        FillRegistry();

        WGPUInstance         m_instance             = nullptr;
        WGPUAdapter          m_adapter              = nullptr;
        WGPUDevice           m_device               = nullptr;
        WGPUQueue            m_queue                = nullptr;
        QueueWGPU            m_queueObj             = {};
        VriDeviceDesc        m_desc                 = {};
        VriCallbackInterface m_callback             = {};
        bool                 m_hasTimestamp         = false; // WGPUFeatureName_TimestampQuery granted at creation
        bool                 m_hasFloat32Filterable = false; // WGPUFeatureName_Float32Filterable granted at creation

        // Held from InitAsync until the device callback, which the browser runs later.
        InitCallback             m_initDone     = nullptr;
        void*                    m_initUserData = nullptr;
        VriCoordinateConventions m_coordinates  = {};
    };

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult);

    // `done` receives the device (NULL unless VriResult_Success) once: in the browser from its event
    // loop, natively before CreateDeviceAsync returns.
    using DeviceCreatedCallback = void (*)(void* userData, VriResult result, core::DeviceBase* device);
    void CreateDeviceAsync(const VriDeviceCreationDesc& desc, DeviceCreatedCallback done, void* userData);
} // namespace vri::wgpu
