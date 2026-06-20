// objects_d3d12.h - concrete D3D12 objects behind the opaque VRI handles.
//
// Phase 0 only defines the queue; resources/pipelines/command lists land as the
// backend is built up. The D3D12 backend targets VRI's explicit model directly
// (command lists, descriptor heaps, root signatures) - no emulation, unlike GL.
#pragma once

#include <d3d12.h>
#include <wrl/client.h>

#include <vri/vri.h>

namespace vri::d3d12
{
    class DeviceD3D12;

    struct QueueD3D12
    {
        DeviceD3D12*                               device = nullptr;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    };

    inline VriQueue* ToHandle(QueueD3D12* q) { return reinterpret_cast<VriQueue*>(q); }
} // namespace vri::d3d12
