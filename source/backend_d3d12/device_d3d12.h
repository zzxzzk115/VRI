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
        IDXGIFactory4*       Factory() const { return m_factory.Get(); }
        const VriDeviceDesc& Desc() const { return m_desc; }
        // Each VriQueueType has its own D3D12 engine: Graphics->DIRECT, Compute->COMPUTE,
        // Transfer->COPY. Compute/Transfer run independently of graphics; cross-queue ordering
        // is the caller's job via timeline fences (QueueSubmit wait/signal).
        QueueD3D12* GetQueue(VriQueueType type)
        {
            return &m_queues[type < VriQueueType_Count ? type : VriQueueType_Graphics];
        }
        uint64_t EnabledFeatures() const { return m_enabledFeatures; }
        void     ReportError(const char* message) const;
        // Route a backend diagnostic (D3D12 InfoQueue message) to the app's callback.
        void Diagnostic(VriMessageSeverity severity, const char* message) const;

        // Bump-allocate one CPU render-target-view / depth-stencil-view descriptor.
        D3D12_CPU_DESCRIPTOR_HANDLE AllocRtv();
        D3D12_CPU_DESCRIPTOR_HANDLE AllocDsv();

        // Lazily-created command signature for ExecuteIndirect(DispatchMesh) - must
        // outlive GPU execution, so it lives on the device (12-byte: 3x uint32 x,y,z).
        ID3D12CommandSignature* DispatchMeshSignature();
        // Lazily-created (and cached by stride) DRAW / DRAW_INDEXED signatures for indirect draws.
        ID3D12CommandSignature* DrawSignature(bool indexed, uint32_t stride);
        // The adapter, for live VRAM budget queries (QueryVideoMemoryInfo); null if unavailable.
        IDXGIAdapter3* Adapter() const { return m_adapter3.Get(); }

    private:
        VriResult NegotiateFeatures(const VriDeviceCreationDesc& desc);
        void      FillDeviceDesc(IDXGIAdapter1* adapter);
        void      FillRegistry();
        VriResult CreateDescriptorHeaps();

        ComPtr<IDXGIFactory4>          m_factory;
        ComPtr<IDXGIAdapter3>          m_adapter3; // chosen adapter, for QueryVideoMemoryInfo
        ComPtr<ID3D12Device>           m_device;
        ComPtr<ID3D12InfoQueue1>       m_infoQueue;       // debug-layer message sink (validation)
        DWORD                          m_msgCookie = 0;   // RegisterMessageCallback token
        ComPtr<ID3D12CommandSignature> m_dispatchMeshSig; // lazy, for indirect DispatchMesh
        std::unordered_map<uint64_t, ComPtr<ID3D12CommandSignature>>
                                     m_drawSigs;               // indirect draw sigs, keyed by (indexed,stride)
        ComPtr<ID3D12DescriptorHeap> m_rtvHeap;                // CPU-only RTV heap
        ComPtr<ID3D12DescriptorHeap> m_dsvHeap;                // CPU-only DSV heap
        uint32_t                     m_rtvSize            = 0; // descriptor increment
        uint32_t                     m_rtvNext            = 0; // bump cursor
        uint32_t                     m_dsvSize            = 0;
        uint32_t                     m_dsvNext            = 0;
        static constexpr uint32_t    kRtvHeapSize         = 256;
        static constexpr uint32_t    kDsvHeapSize         = 64;
        QueueD3D12           m_queues[VriQueueType_Count] = {}; // one per VriQueueType (DIRECT / COMPUTE / COPY)
        uint64_t             m_enabledFeatures            = 0;  // granted VriFeatureBits
        VriDeviceDesc        m_desc                       = {};
        VriCallbackInterface m_callback                   = {};
    };

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult);
} // namespace vri::d3d12
