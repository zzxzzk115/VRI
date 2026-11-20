// device_d3d12.h - the Direct3D 12 VriDevice implementation + factory.
#pragma once

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <vri/vri.h>

#include "core/device_base.h"
#include "objects_d3d12.h"

#include <mutex>
#include <vector>

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

        // The CPU-only render-target-view (or depth-stencil-view) descriptor of a texture view, made the first
        // time the view is an attachment and held until the view is destroyed. OMSetRenderTargets and the clears
        // read these descriptors when they are recorded, so a destroyed view's slot is free at once. False, with
        // an error reported, when the texture cannot be an attachment or every slot is taken.
        bool AttachmentView(const DescriptorD3D12& view, D3D12_CPU_DESCRIPTOR_HANDLE& out);
        void FreeAttachmentView(bool depth, D3D12_CPU_DESCRIPTOR_HANDLE handle);

        // Lazily-created command signature for ExecuteIndirect(DispatchMesh) - must
        // outlive GPU execution, so it lives on the device (12-byte: 3x uint32 x,y,z).
        ID3D12CommandSignature* DispatchMeshSignature();
        // Lazily-created command signature for ExecuteIndirect(Dispatch) - 12-byte: 3x uint32 x,y,z.
        ID3D12CommandSignature* DispatchSignature();
        // Lazily-created (and cached by stride) DRAW / DRAW_INDEXED signatures for indirect draws.
        ID3D12CommandSignature* DrawSignature(bool indexed, uint32_t stride);
        // The adapter, for live VRAM budget queries (QueryVideoMemoryInfo); null if unavailable.
        IDXGIAdapter3* Adapter() const { return m_adapter3.Get(); }
        // Stage a UAV for ClearUnorderedAccessView, which needs the view in BOTH a non-shader-visible
        // heap (-> outCpu) and the currently-bound shader-visible heap (-> outGpu, in *outGpuHeap).
        // Ring-allocated across the two dedicated clear heaps (lazily created).
        void ClearUavViews(ID3D12Resource*                         res,
                           const D3D12_UNORDERED_ACCESS_VIEW_DESC& desc,
                           D3D12_GPU_DESCRIPTOR_HANDLE&            outGpu,
                           D3D12_CPU_DESCRIPTOR_HANDLE&            outCpu,
                           ID3D12DescriptorHeap*&                  outGpuHeap);

        // The shader-visible CBV/SRV/UAV and sampler heaps every descriptor pool takes a block of: a command
        // list binds one heap of each type at a time, so tables of sets from different pools must share them.
        // Allocation and release are thread-safe (pools are created on any thread).
        ID3D12DescriptorHeap* ShaderVisibleHeap(D3D12_DESCRIPTOR_HEAP_TYPE type) const;
        bool AllocateShaderVisible(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t count, DescriptorBlockD3D12& out);
        void FreeShaderVisible(D3D12_DESCRIPTOR_HEAP_TYPE type, DescriptorBlockD3D12 block);

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
        ComPtr<ID3D12CommandSignature> m_dispatchSig;     // lazy, for indirect Dispatch
        std::unordered_map<uint64_t, ComPtr<ID3D12CommandSignature>>
                                     m_drawSigs;         // indirect draw sigs, keyed by (indexed,stride)
        ComPtr<ID3D12DescriptorHeap> m_clearUavCpuHeap;  // non-shader-visible UAVs for ClearUAV (lazy)
        ComPtr<ID3D12DescriptorHeap> m_clearUavGpuHeap;  // shader-visible UAVs for ClearUAV (lazy)
        uint32_t                     m_clearUavSize = 0; // CBV_SRV_UAV descriptor increment
        uint32_t                     m_clearUavNext = 0; // ring cursor across the clear heaps
        static constexpr uint32_t    kClearUavSlots = 1024;
        // CPU-only RTV and DSV heaps with their unused slots.
        struct AttachmentHeap
        {
            ComPtr<ID3D12DescriptorHeap> heap;
            uint32_t                     increment = 0;
            std::vector<uint32_t>        free;
        };
        AttachmentHeap            m_rtvHeap;
        AttachmentHeap            m_dsvHeap;
        mutable std::mutex        m_attachmentHeapMutex;
        static constexpr uint32_t kRtvHeapSize = 4096;
        static constexpr uint32_t kDsvHeapSize = 1024;
        // Shared shader-visible heaps with their free blocks, sorted by offset.
        struct SharedHeap
        {
            ComPtr<ID3D12DescriptorHeap>      heap;
            std::vector<DescriptorBlockD3D12> free;
        };
        SharedHeap                m_viewHeap;
        SharedHeap                m_samplerHeap;
        mutable std::mutex        m_sharedHeapMutex;
        static constexpr uint32_t kViewHeapSize = 1u << 18;
        // D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE: the most samplers a shader-visible heap can hold.
        static constexpr uint32_t kSamplerHeapSize             = 2048;
        QueueD3D12                m_queues[VriQueueType_Count] = {}; // one per VriQueueType (DIRECT / COMPUTE / COPY)
        uint64_t                  m_enabledFeatures            = 0;  // granted VriFeatureBits
        VriDeviceDesc             m_desc                       = {};
        VriCallbackInterface      m_callback                   = {};
    };

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult);
} // namespace vri::d3d12
