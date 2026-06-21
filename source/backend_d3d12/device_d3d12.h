// device_d3d12.h - the Direct3D 12 VriDevice implementation + factory.
#pragma once

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <vri/vri.h>

#include "core/device_base.h"
#include "objects_d3d12.h"

namespace vri::d3d12
{
    using Microsoft::WRL::ComPtr;

    class DeviceD3D12 final : public core::DeviceBase
    {
    public:
        ~DeviceD3D12() override;

        VriResult Init(const VriDeviceCreationDesc& desc);

        ID3D12Device*        Device() const { return m_device.Get(); }
        const VriDeviceDesc& Desc() const { return m_desc; }
        QueueD3D12*          GetQueue(VriQueueType /*type*/) { return &m_queue; }
        void                 ReportError(const char* message) const;

        // Bump-allocate one CPU render-target-view descriptor (non-shader-visible heap).
        D3D12_CPU_DESCRIPTOR_HANDLE AllocRtv();

    private:
        void      FillDeviceDesc(IDXGIAdapter1* adapter);
        void      FillRegistry();
        VriResult CreateDescriptorHeaps();

        ComPtr<IDXGIFactory4>        m_factory;
        ComPtr<ID3D12Device>         m_device;
        ComPtr<ID3D12DescriptorHeap> m_rtvHeap;       // CPU-only RTV heap
        uint32_t                     m_rtvSize = 0;    // descriptor increment
        uint32_t                     m_rtvNext = 0;    // bump cursor
        static constexpr uint32_t    kRtvHeapSize = 256;
        QueueD3D12            m_queue = {};
        VriDeviceDesc         m_desc = {};
        VriCallbackInterface  m_callback = {};
    };

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult);
} // namespace vri::d3d12
