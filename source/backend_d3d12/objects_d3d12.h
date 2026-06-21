// objects_d3d12.h - concrete D3D12 objects behind the opaque VRI handles.
//
// The D3D12 backend targets VRI's explicit model directly (command lists, descriptor
// heaps, resource-state barriers) - no emulation, unlike GL. Resources track their
// current D3D12_RESOURCE_STATES so CmdBarrier can emit transition barriers.
#pragma once

#include <d3d12.h>
#include <wrl/client.h>

#include <cstdint>
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
        D3D12_RESOURCE_STATES  state = D3D12_RESOURCE_STATE_COMMON;
    };

    // A view/sampler. RTV/DSV live in CPU-only heaps; SRV/CBV/UAV in a shader-visible
    // heap (later). Phase 1 only needs the render-target view (RTV).
    struct DescriptorD3D12
    {
        enum class Kind { TextureRtv, TextureSrv, BufferView, Sampler } kind = Kind::TextureRtv;
        DeviceD3D12*                device = nullptr;
        const TextureD3D12*         texture = nullptr;
        const BufferD3D12*          buffer = nullptr;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = {}; // for RTV/DSV
        uint32_t                    mip = 0;
    };

    struct CommandAllocatorD3D12
    {
        DeviceD3D12*                  device = nullptr;
        ComPtr<ID3D12CommandAllocator> allocator;
    };

    struct CommandBufferD3D12
    {
        DeviceD3D12*                       device = nullptr;
        ComPtr<ID3D12GraphicsCommandList>  list;
        CommandAllocatorD3D12*             allocator = nullptr;
        D3D12_CPU_DESCRIPTOR_HANDLE        rtvs[8] = {}; // bound color RTVs for the active pass
        uint32_t                           rtvCount = 0;
    };

    struct FenceD3D12
    {
        DeviceD3D12*        device = nullptr;
        ComPtr<ID3D12Fence> fence;
        void*               event = nullptr; // HANDLE for blocking waits
    };

    inline VriQueue*            ToHandle(QueueD3D12* q)            { return reinterpret_cast<VriQueue*>(q); }
    inline VriBuffer*           ToHandle(BufferD3D12* b)           { return reinterpret_cast<VriBuffer*>(b); }
    inline VriTexture*          ToHandle(TextureD3D12* t)          { return reinterpret_cast<VriTexture*>(t); }
    inline VriDescriptor*       ToHandle(DescriptorD3D12* d)       { return reinterpret_cast<VriDescriptor*>(d); }
    inline VriCommandAllocator* ToHandle(CommandAllocatorD3D12* a) { return reinterpret_cast<VriCommandAllocator*>(a); }
    inline VriCommandBuffer*    ToHandle(CommandBufferD3D12* c)    { return reinterpret_cast<VriCommandBuffer*>(c); }
    inline VriFence*            ToHandle(FenceD3D12* f)            { return reinterpret_cast<VriFence*>(f); }
} // namespace vri::d3d12
