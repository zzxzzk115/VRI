// objects_d3d12.h - concrete D3D12 objects behind the opaque VRI handles.
//
// The D3D12 backend targets VRI's explicit model directly (command lists, descriptor
// heaps, resource-state barriers) - no emulation, unlike GL. Resources track their
// current D3D12_RESOURCE_STATES so CmdBarrier can emit transition barriers.
#pragma once

#include <d3d12.h>
#include <wrl/client.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <vri/vri.h>

namespace vri::d3d12
{
    using Microsoft::WRL::ComPtr;

    class DeviceD3D12;

    struct QueueD3D12
    {
        DeviceD3D12*               device = nullptr;
        ComPtr<ID3D12CommandQueue> queue;
    };

    struct BufferD3D12
    {
        DeviceD3D12*           device = nullptr;
        ComPtr<ID3D12Resource> resource;
        uint64_t               size = 0;
        D3D12_HEAP_TYPE        heapType = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_STATES  state = D3D12_RESOURCE_STATE_COMMON;
        void*                  mapped = nullptr; // persistent CPU pointer for upload/readback heaps
    };

    struct TextureD3D12
    {
        DeviceD3D12*           device = nullptr;
        ComPtr<ID3D12Resource> resource;
        DXGI_FORMAT            format = DXGI_FORMAT_UNKNOWN;
        uint32_t               width = 0, height = 0, depth = 1;
        uint32_t               mipNum = 1, layerNum = 1;
        uint32_t               texelSize = 0;
        bool                   isRenderTarget = false; // created with ALLOW_RENDER_TARGET (RTV-capable)
        bool                   isDepthStencil = false; // created with ALLOW_DEPTH_STENCIL (DSV-capable)
        D3D12_RESOURCE_STATES  state = D3D12_RESOURCE_STATE_COMMON;
    };

    // A view/sampler. RTV/DSV live in CPU-only heaps; SRV/CBV/UAV in a shader-visible
    // heap (later). Phase 1 needs RTV; phase 3a adds the constant-buffer view (bound as a
    // root CBV by GPU address, so no descriptor heap slot).
    struct DescriptorD3D12
    {
        enum class Kind { TextureRtv, TextureSrv, BufferView, Sampler } kind = Kind::TextureRtv;
        DeviceD3D12*                device = nullptr;
        const TextureD3D12*         texture = nullptr;
        const BufferD3D12*          buffer = nullptr;
        uint64_t                    bufferOffset = 0;
        uint64_t                    bufferSize = 0;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = {}; // for RTV/DSV
        uint32_t                    mip = 0;
        VriSamplerDesc              sampler = {}; // for Kind::Sampler (built into the sampler heap at update)
    };

    // Flattened pipeline-layout binding: a VRI (set, binding) -> its D3D12 binding slot.
    // Constant buffers -> root CBV (rootParam). SRV/sampler -> a per-set descriptor table
    // (the set's table root param) at heapOffset within that table (per-type from 0).
    struct LayoutBindingD3D12
    {
        uint32_t          set = 0;
        uint32_t          binding = 0;
        VriDescriptorType type = VriDescriptorType_ConstantBuffer;
        uint32_t          rootParam = 0;  // root CBV param (CBV) or table root param (SRV/sampler)
        uint32_t          heapOffset = 0; // SRV/sampler: slot within the set's heap block
    };

    // Per-set descriptor-table info: an SRV/UAV table and/or a sampler table.
    struct LayoutSetD3D12
    {
        uint32_t set = 0;
        int      srvTableParam = -1; uint32_t srvCount = 0;
        int      samplerTableParam = -1; uint32_t samplerCount = 0;
    };

    // A descriptor pool owns two shader-visible heaps (CBV/SRV/UAV + sampler) that
    // descriptor sets sub-allocate from via a bump cursor.
    struct DescriptorPoolD3D12
    {
        DeviceD3D12*                 device = nullptr;
        ComPtr<ID3D12DescriptorHeap> srvHeap;     // shader-visible CBV/SRV/UAV
        ComPtr<ID3D12DescriptorHeap> samplerHeap; // shader-visible SAMPLER
        uint32_t srvSize = 0, samplerSize = 0;    // descriptor increments
        uint32_t srvNext = 0, samplerNext = 0;    // bump cursors
    };

    struct CommandAllocatorD3D12
    {
        DeviceD3D12*                  device = nullptr;
        ComPtr<ID3D12CommandAllocator> allocator;
    };

    struct PipelineD3D12;

    struct CommandBufferD3D12
    {
        DeviceD3D12*                       device = nullptr;
        ComPtr<ID3D12GraphicsCommandList>  list;
        CommandAllocatorD3D12*             allocator = nullptr;
        D3D12_CPU_DESCRIPTOR_HANDLE        rtvs[8] = {}; // bound color RTVs for the active pass
        uint32_t                           rtvCount = 0;
        const PipelineD3D12*               boundPipeline = nullptr; // for vertex-buffer strides at draw
    };

    struct FenceD3D12
    {
        DeviceD3D12*        device = nullptr;
        ComPtr<ID3D12Fence> fence;
        void*               event = nullptr; // HANDLE for blocking waits
    };

    struct PipelineLayoutD3D12
    {
        DeviceD3D12*                    device = nullptr;
        ComPtr<ID3D12RootSignature>     rootSig;
        std::vector<LayoutBindingD3D12> bindings;
        std::vector<LayoutSetD3D12>     sets;

        const LayoutBindingD3D12* Find(uint32_t set, uint32_t binding) const
        {
            for (const LayoutBindingD3D12& b : bindings)
                if (b.set == set && b.binding == binding) return &b;
            return nullptr;
        }
        const LayoutSetD3D12* FindSet(uint32_t set) const
        {
            for (const LayoutSetD3D12& s : sets)
                if (s.set == set) return &s;
            return nullptr;
        }
    };

    // CPU-side descriptor set: records which view sits at each binding (like the GL
    // backend). At CmdSetDescriptorSet the recorded views are bound as root arguments.
    struct DescriptorSetD3D12
    {
        DeviceD3D12*                                          device = nullptr;
        DescriptorPoolD3D12*                                  pool = nullptr;
        const PipelineLayoutD3D12*                            layout = nullptr;
        uint32_t                                              setIndex = 0;
        std::unordered_map<uint32_t, const DescriptorD3D12*>  bound; // binding -> view
        // base of this set's blocks in the pool's shader-visible heaps (SRV + sampler).
        D3D12_CPU_DESCRIPTOR_HANDLE srvCpu = {}; D3D12_GPU_DESCRIPTOR_HANDLE srvGpu = {};
        D3D12_CPU_DESCRIPTOR_HANDLE samplerCpu = {}; D3D12_GPU_DESCRIPTOR_HANDLE samplerGpu = {};
    };

    struct PipelineGraphicsVB { uint32_t stride; uint32_t slot; }; // per-stream stride for IASetVertexBuffers

    struct PipelineD3D12
    {
        DeviceD3D12*                    device = nullptr;
        ComPtr<ID3D12PipelineState>     pso;
        ID3D12RootSignature*            rootSig = nullptr; // borrowed from the layout
        D3D12_PRIMITIVE_TOPOLOGY        topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        std::vector<PipelineGraphicsVB> vbStrides; // for the draw-time vertex-buffer binding
        bool                            isCompute = false;
    };

    inline VriPipelineLayout* ToHandle(PipelineLayoutD3D12* p) { return reinterpret_cast<VriPipelineLayout*>(p); }
    inline VriPipeline*       ToHandle(PipelineD3D12* p)       { return reinterpret_cast<VriPipeline*>(p); }
    inline VriDescriptorPool* ToHandle(DescriptorPoolD3D12* p) { return reinterpret_cast<VriDescriptorPool*>(p); }
    inline VriDescriptorSet*  ToHandle(DescriptorSetD3D12* s)  { return reinterpret_cast<VriDescriptorSet*>(s); }

    inline VriQueue*            ToHandle(QueueD3D12* q)            { return reinterpret_cast<VriQueue*>(q); }
    inline VriBuffer*           ToHandle(BufferD3D12* b)           { return reinterpret_cast<VriBuffer*>(b); }
    inline VriTexture*          ToHandle(TextureD3D12* t)          { return reinterpret_cast<VriTexture*>(t); }
    inline VriDescriptor*       ToHandle(DescriptorD3D12* d)       { return reinterpret_cast<VriDescriptor*>(d); }
    inline VriCommandAllocator* ToHandle(CommandAllocatorD3D12* a) { return reinterpret_cast<VriCommandAllocator*>(a); }
    inline VriCommandBuffer*    ToHandle(CommandBufferD3D12* c)    { return reinterpret_cast<VriCommandBuffer*>(c); }
    inline VriFence*            ToHandle(FenceD3D12* f)            { return reinterpret_cast<VriFence*>(f); }
} // namespace vri::d3d12
