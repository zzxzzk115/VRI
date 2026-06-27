// swapchain_mtl.h - native Metal presentation (CAMetalLayer) interface.
#pragma once

#import <QuartzCore/CAMetalLayer.h>

#include <vri/vri.h>

#include "objects_mtl.h"

namespace vri::mtl
{
    struct SwapChainMTL
    {
        DeviceMTL*           device;
        QueueMTL*            queue;
        CAMetalLayer*        layer;
        MTLPixelFormat       format;
        uint32_t             width;
        uint32_t             height;
        VriPresentMode       presentMode;
        id<CAMetalDrawable>  drawable; // current acquired frame (retained until Present)
        TextureMTL           current;  // stable handle; its texture is refreshed each acquire
    };

    inline VriSwapChain* ToHandle(SwapChainMTL* s) { return reinterpret_cast<VriSwapChain*>(s); }

    const VriSwapChainInterface* GetSwapChainInterfaceMTL();
} // namespace vri::mtl
