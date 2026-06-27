// swapchain_d3d12.h - Direct3D 12 windowed presentation (DXGI flip-model swapchain).
//
// CreateSwapChainForHwnd on the present queue; the backbuffers are wrapped as VRI
// textures (RTV-capable) the app renders into. Win32 only; other window systems return
// VriResult_Unsupported. Like the GL windowed path, this has no CI display - verified
// locally via examples/triangle (VRI_API=d3d12).
#pragma once

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <vector>

#include <vri/ext/vri_ext_swapchain.h>
#include <vri/vri.h>

#include "objects_d3d12.h"

namespace vri::d3d12
{
    using Microsoft::WRL::ComPtr;

    struct SwapChainD3D12
    {
        DeviceD3D12*               device = nullptr;
        ComPtr<IDXGISwapChain3>    swapchain;
        std::vector<TextureD3D12*> textures; // backbuffers wrapped as VRI textures (owned)
        uint32_t                   width = 0, height = 0;
        VriFormat                  format       = VriFormat_Unknown;
        uint32_t                   bufferCount  = 0;
        UINT                       syncInterval = 1; // 1 = Fifo (vsync); 0 = Mailbox/Immediate (uncapped)
    };

    inline VriSwapChain* ToHandle(SwapChainD3D12* s) { return reinterpret_cast<VriSwapChain*>(s); }

    const VriSwapChainInterface* GetSwapChainInterfaceD3D12();
} // namespace vri::d3d12
