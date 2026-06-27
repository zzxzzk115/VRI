// swapchain_wgpu.h - WebGPU swapchain (surface) object + interface table.
#pragma once

#include <webgpu/webgpu.h>

#include <vri/ext/vri_ext_swapchain.h>
#include <vri/vri.h>

#include "objects_wgpu.h"

namespace vri::wgpu
{
    struct SwapChainWGPU
    {
        DeviceWGPU*       device;
        WGPUSurface       surface;
        WGPUTextureFormat format;
        uint32_t          width;
        uint32_t          height;
        bool              vsync;
        VriFormat         requestedFormat;
        // WebGPU hands back the current frame texture on acquire; we expose a
        // single stable VRI texture handle whose underlying WGPUTexture is
        // refreshed each AcquireNextTexture.
        TextureWGPU current;
        WGPUTexture acquired; // owned until Present
    };

    inline VriSwapChain* ToHandle(SwapChainWGPU* s) { return reinterpret_cast<VriSwapChain*>(s); }

    const VriSwapChainInterface* GetSwapChainInterfaceWGPU();
} // namespace vri::wgpu
