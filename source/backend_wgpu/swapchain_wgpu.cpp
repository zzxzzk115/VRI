// swapchain_wgpu.cpp - WebGPU presentation. Surface is created from a native
// VriWindowHandle (Win32 for now). WebGPU returns the current frame texture on
// acquire (no fixed image array), so the VRI swapchain exposes one stable
// texture handle refreshed each AcquireNextTexture.

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    define NOMINMAX
#    include <windows.h>
#endif

#include "wgpu_native.h" // webgpu.h + native-only poll helpers

#include "swapchain_wgpu.h"
#include "conversions_wgpu.h"
#include "device_wgpu.h"

#include <vector>

namespace vri::wgpu
{
    namespace
    {
        inline SwapChainWGPU* SC(VriSwapChain* h) { return reinterpret_cast<SwapChainWGPU*>(h); }
        inline DeviceWGPU*    Dev(VriDevice* h)   { return reinterpret_cast<DeviceWGPU*>(h); }
        inline QueueWGPU*     Q(VriQueue* h)      { return reinterpret_cast<QueueWGPU*>(h); }

        WGPUSurface CreateSurface(DeviceWGPU* d, const VriWindowHandle& window)
        {
#if defined(__EMSCRIPTEN__)
            // Browser: surface from the HTML canvas selector (emdawnwebgpu).
            if (window.type != VriWindowSystem_Web || window.handle.web.selector == nullptr)
                return nullptr;
            WGPUEmscriptenSurfaceSourceCanvasHTMLSelector src = {};
            src.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
            src.selector = WGPUStringView{window.handle.web.selector, WGPU_STRLEN};
            WGPUSurfaceDescriptor sd = {};
            sd.nextInChain = &src.chain;
            return wgpuInstanceCreateSurface(d->Instance(), &sd);
#elif defined(_WIN32)
            if (window.type != VriWindowSystem_Win32)
                return nullptr;
            WGPUSurfaceSourceWindowsHWND src = {};
            src.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
            src.hinstance = window.handle.win32.hinstance ? window.handle.win32.hinstance : (void*)GetModuleHandleW(nullptr);
            src.hwnd = window.handle.win32.hwnd;
            WGPUSurfaceDescriptor sd = {};
            sd.nextInChain = &src.chain;
            return wgpuInstanceCreateSurface(d->Instance(), &sd);
#else
            (void)d; (void)window;
            return nullptr;
#endif
        }

        void Configure(SwapChainWGPU* sc)
        {
            WGPUSurfaceConfiguration cfg = {};
            cfg.device = sc->device->Device();
            cfg.format = sc->format;
            cfg.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc; // CopySrc: screenshot readback
            cfg.width = sc->width;
            cfg.height = sc->height;
            cfg.alphaMode = WGPUCompositeAlphaMode_Auto;
            cfg.presentMode = sc->vsync ? WGPUPresentMode_Fifo : WGPUPresentMode_Mailbox;
            wgpuSurfaceConfigure(sc->surface, &cfg);
        }

        VriResult VRI_CALL CreateSwapChain(VriDevice* device, const VriSwapChainDesc* desc, VriSwapChain** out)
        {
            DeviceWGPU* d = Dev(device);
            SwapChainWGPU* sc = new SwapChainWGPU{};
            sc->device = d;
            sc->surface = CreateSurface(d, desc->window);
            if (!sc->surface)
            {
                delete sc;
                return VriResult_Unsupported;
            }

            // choose a format: prefer the requested one if the surface supports it
            WGPUSurfaceCapabilities caps = {};
            wgpuSurfaceGetCapabilities(sc->surface, d->Adapter(), &caps);
            const WGPUTextureFormat wanted = ToWgpuFormat(desc->format);
            WGPUTextureFormat chosen = caps.formatCount ? caps.formats[0] : WGPUTextureFormat_BGRA8Unorm;
            for (size_t i = 0; i < caps.formatCount; ++i)
                if (caps.formats[i] == wanted) { chosen = wanted; break; }
            wgpuSurfaceCapabilitiesFreeMembers(caps);

            sc->format = chosen;
            sc->width = desc->width;
            sc->height = desc->height;
            sc->vsync = desc->vsync != VRI_FALSE;
            sc->requestedFormat = desc->format;
            Configure(sc);

            sc->current = {};
            sc->current.device = d;
            sc->current.format = chosen;
            sc->current.width = sc->width;
            sc->current.height = sc->height;
            sc->current.depth = 1;
            sc->current.mipNum = 1;
            sc->current.layerNum = 1;
            sc->current.texelSize = 4; // swapchain formats are 8-bit RGBA/BGRA (needed for readback bytesPerRow)
            sc->current.owned = false; // swapchain-managed
            *out = ToHandle(sc);
            return VriResult_Success;
        }

        void VRI_CALL DestroySwapChain(VriSwapChain* swapChain)
        {
            if (!swapChain) return;
            SwapChainWGPU* sc = SC(swapChain);
            PollDevice(sc->device->Device());
            if (sc->acquired) wgpuTextureRelease(sc->acquired);
            if (sc->surface) { wgpuSurfaceUnconfigure(sc->surface); wgpuSurfaceRelease(sc->surface); }
            delete sc;
        }

        VriResult VRI_CALL GetSwapChainTextures(VriSwapChain* swapChain, VriTexture** outTextures, uint32_t* ioCount)
        {
            SwapChainWGPU* sc = SC(swapChain);
            if (!outTextures) { *ioCount = 1; return VriResult_Success; }
            if (*ioCount >= 1) { outTextures[0] = ToHandle(&sc->current); *ioCount = 1; }
            return VriResult_Success;
        }

        VriResult VRI_CALL AcquireNextTexture(VriSwapChain* swapChain, VriFence* /*acquireFence*/, uint64_t /*signalValue*/, uint32_t* outIndex)
        {
            SwapChainWGPU* sc = SC(swapChain);
            if (sc->acquired) { wgpuTextureRelease(sc->acquired); sc->acquired = nullptr; }

            WGPUSurfaceTexture st = {};
            wgpuSurfaceGetCurrentTexture(sc->surface, &st);
            if (st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
                st.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)
            {
                if (st.texture) wgpuTextureRelease(st.texture);
                return VriResult_OutOfDate;
            }
            sc->acquired = st.texture;
            sc->current.texture = st.texture; // refresh the stable handle's underlying texture
            *outIndex = 0;
            return VriResult_Success;
        }

        VriResult VRI_CALL Present(VriSwapChain* swapChain, VriFence* /*waitFence*/, uint64_t /*waitValue*/)
        {
            SwapChainWGPU* sc = SC(swapChain);
#if defined(__EMSCRIPTEN__)
            // The browser presents the canvas automatically when the requestAnimationFrame
            // callback returns (the app drives it via emscripten_set_main_loop), and
            // wgpuSurfacePresent aborts on the web ("unsupported"). So just drop the frame's
            // acquired texture; presentation happens implicitly when control returns to JS.
            if (sc->acquired) { wgpuTextureRelease(sc->acquired); sc->acquired = nullptr; }
            sc->current.texture = nullptr;
            return VriResult_Success;
#else
            const WGPUStatus s = wgpuSurfacePresent(sc->surface);
            if (sc->acquired) { wgpuTextureRelease(sc->acquired); sc->acquired = nullptr; }
            sc->current.texture = nullptr;
            return s == WGPUStatus_Success ? VriResult_Success : VriResult_OutOfDate;
#endif
        }

        VriResult VRI_CALL Resize(VriSwapChain* swapChain, uint32_t width, uint32_t height)
        {
            SwapChainWGPU* sc = SC(swapChain);
            PollDevice(sc->device->Device());
            sc->width = width;
            sc->height = height;
            sc->current.width = width;
            sc->current.height = height;
            Configure(sc);
            return VriResult_Success;
        }

        const VriSwapChainInterface g_swapChainWGPU = {
            CreateSwapChain,
            DestroySwapChain,
            GetSwapChainTextures,
            AcquireNextTexture,
            Present,
            Resize,
        };
    } // namespace

    const VriSwapChainInterface* GetSwapChainInterfaceWGPU() { return &g_swapChainWGPU; }
} // namespace vri::wgpu
