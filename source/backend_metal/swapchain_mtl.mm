// swapchain_mtl.mm - native Metal presentation via CAMetalLayer.
//
// The window integration (SDL3) hands us a ready CAMetalLayer in cocoa.layer
// (same handle the MoltenVK backend wraps). nextDrawable returns the current
// backbuffer; we expose one stable VRI texture whose underlying drawable texture
// is refreshed on each AcquireNextTexture (mirrors the WebGPU backend's model).

#include "swapchain_mtl.h"
#include "conversions_mtl.h"
#include "device_mtl.h"

namespace vri::mtl
{
    namespace
    {
        inline SwapChainMTL* SC(VriSwapChain* h) { return reinterpret_cast<SwapChainMTL*>(h); }
        inline DeviceMTL*    Dev(VriDevice* h)   { return reinterpret_cast<DeviceMTL*>(h); }
        inline QueueMTL*     Q(VriQueue* h)      { return reinterpret_cast<QueueMTL*>(h); }
        inline FenceMTL*     Fen(VriFence* h)    { return reinterpret_cast<FenceMTL*>(h); }

        void ConfigureCurrent(SwapChainMTL* sc, VriFormat requested)
        {
            sc->current = {};
            sc->current.device = sc->device;
            sc->current.format = requested;
            sc->current.mtlFormat = sc->format;
            sc->current.type = VriTextureType_2D;
            sc->current.width = sc->width;
            sc->current.height = sc->height;
            sc->current.depth = 1;
            sc->current.mipNum = 1;
            sc->current.layerNum = 1;
            sc->current.sampleNum = 1;
            sc->current.texelSize = TexelSize(requested);
            sc->current.owned = false; // swapchain-managed (drawable owns the texture)
        }

        VriResult VRI_CALL CreateSwapChain(VriDevice* device, const VriSwapChainDesc* desc, VriSwapChain** out)
        {
            if (desc->window.type != VriWindowSystem_Cocoa || desc->window.handle.cocoa.layer == nullptr)
                return VriResult_Unsupported;

            DeviceMTL* d = Dev(device);
            SwapChainMTL* sc = new SwapChainMTL{};
            sc->device = d;
            sc->queue = desc->queue ? Q(desc->queue) : d->GetQueue(VriQueueType_Graphics);
            sc->layer = (CAMetalLayer*)desc->window.handle.cocoa.layer;
            [sc->layer retain];

            MTLPixelFormat fmt = ToMtlFormat(desc->format);
            if (fmt == MTLPixelFormatInvalid)
                fmt = MTLPixelFormatBGRA8Unorm;
            sc->format = fmt;
            sc->width = desc->width;
            sc->height = desc->height;
            sc->vsync = desc->vsync != VRI_FALSE;

            sc->layer.device = d->Device();
            sc->layer.pixelFormat = fmt;
            sc->layer.framebufferOnly = NO; // allow CopySrc / readback from the drawable
            sc->layer.drawableSize = CGSizeMake(desc->width, desc->height);
            sc->layer.displaySyncEnabled = sc->vsync ? YES : NO;
            if (desc->textureNum)
                sc->layer.maximumDrawableCount = desc->textureNum < 2 ? 2 : (desc->textureNum > 3 ? 3 : desc->textureNum);

            ConfigureCurrent(sc, desc->format);
            *out = ToHandle(sc);
            return VriResult_Success;
        }

        void VRI_CALL DestroySwapChain(VriSwapChain* swapChain)
        {
            if (!swapChain) return;
            SwapChainMTL* sc = SC(swapChain);
            if (sc->drawable) [sc->drawable release];
            if (sc->layer) [sc->layer release];
            delete sc;
        }

        VriResult VRI_CALL GetSwapChainTextures(VriSwapChain* swapChain, VriTexture** outTextures, uint32_t* ioCount)
        {
            SwapChainMTL* sc = SC(swapChain);
            if (!outTextures) { *ioCount = 1; return VriResult_Success; }
            if (*ioCount >= 1) { outTextures[0] = ToHandle(&sc->current); *ioCount = 1; }
            return VriResult_Success;
        }

        VriResult VRI_CALL AcquireNextTexture(VriSwapChain* swapChain, VriFence* acquireFence, uint64_t signalValue, uint32_t* outIndex)
        {
            SwapChainMTL* sc = SC(swapChain);
            if (sc->drawable) { [sc->drawable release]; sc->drawable = nil; }

            sc->drawable = [[sc->layer nextDrawable] retain]; // blocks until a backbuffer is free
            if (!sc->drawable)
                return VriResult_OutOfDate;
            sc->current.texture = [sc->drawable texture]; // owned by the drawable

            // nextDrawable is synchronous, so the acquire is already satisfied: signal the
            // fence on the CPU timeline immediately (the GPU never has to wait on it).
            if (acquireFence)
                Fen(acquireFence)->event.signaledValue = signalValue;
            *outIndex = 0;
            return VriResult_Success;
        }

        VriResult VRI_CALL Present(VriSwapChain* swapChain, VriFence* waitFence, uint64_t waitValue)
        {
            SwapChainMTL* sc = SC(swapChain);
            // Present on its own command buffer: the in-order queue runs it after the
            // already-committed render work, and an optional event wait gates it further.
            id<MTLCommandBuffer> pc = [sc->queue->queue commandBuffer];
            if (waitFence)
                [pc encodeWaitForEvent:Fen(waitFence)->event value:waitValue];
            if (sc->drawable)
                [pc presentDrawable:sc->drawable];
            [pc commit];
            if (sc->drawable) { [sc->drawable release]; sc->drawable = nil; }
            sc->current.texture = nil;
            return VriResult_Success;
        }

        VriResult VRI_CALL Resize(VriSwapChain* swapChain, uint32_t width, uint32_t height)
        {
            SwapChainMTL* sc = SC(swapChain);
            sc->width = width;
            sc->height = height;
            sc->layer.drawableSize = CGSizeMake(width, height);
            sc->current.width = width;
            sc->current.height = height;
            return VriResult_Success;
        }

        const VriSwapChainInterface g_swapChainMTL = {
            CreateSwapChain,
            DestroySwapChain,
            GetSwapChainTextures,
            AcquireNextTexture,
            Present,
            Resize,
        };
    } // namespace

    const VriSwapChainInterface* GetSwapChainInterfaceMTL() { return &g_swapChainMTL; }
} // namespace vri::mtl
