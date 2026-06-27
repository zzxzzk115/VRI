#include "swapchain_d3d12.h"

#include "conversions_d3d12.h"
#include "device_d3d12.h"

namespace vri::d3d12
{
    namespace
    {
        DeviceD3D12*    Dev(VriDevice* d) { return reinterpret_cast<DeviceD3D12*>(d); }
        SwapChainD3D12* SC(VriSwapChain* s) { return reinterpret_cast<SwapChainD3D12*>(s); }

        // Wrap each DXGI backbuffer as an RTV-capable VRI texture (state PRESENT).
        bool WrapBackbuffers(SwapChainD3D12* s)
        {
            const DxgiFormatInfo fi = ToDxgiFormat(s->format);
            for (uint32_t i = 0; i < s->bufferCount; ++i)
            {
                ComPtr<ID3D12Resource> res;
                if (FAILED(s->swapchain->GetBuffer(i, IID_PPV_ARGS(&res))))
                    return false;
                TextureD3D12* t   = new TextureD3D12 {};
                t->device         = s->device;
                t->resource       = res;
                t->format         = fi.format;
                t->texelSize      = fi.texelSize;
                t->width          = s->width;
                t->height         = s->height;
                t->depth          = 1;
                t->mipNum         = 1;
                t->layerNum       = 1;
                t->sampleCount    = 1;
                t->isRenderTarget = true;
                t->state          = D3D12_RESOURCE_STATE_PRESENT;
                s->textures.push_back(t);
            }
            return true;
        }
        void ReleaseBackbuffers(SwapChainD3D12* s)
        {
            for (TextureD3D12* t : s->textures)
                delete t;
            s->textures.clear();
        }

        VriResult VRI_CALL CreateSwapChain(VriDevice* device, const VriSwapChainDesc* desc, VriSwapChain** out)
        {
            DeviceD3D12* d = Dev(device);
            if (desc->window.type != VriWindowSystem_Win32)
            {
                d->ReportError("D3D12 swapchain: only the Win32 window system is supported");
                return VriResult_Unsupported;
            }
            HWND hwnd = static_cast<HWND>(desc->window.handle.win32.hwnd);
            if (!hwnd)
                return VriResult_InvalidArgument;
            ID3D12CommandQueue* queue = reinterpret_cast<QueueD3D12*>(desc->queue)->queue.Get();

            SwapChainD3D12* s = new SwapChainD3D12 {};
            s->device         = d;
            s->width          = desc->width ? desc->width : 1;
            s->height         = desc->height ? desc->height : 1;
            s->format         = desc->format;
            s->bufferCount    = desc->textureNum >= 2 ? desc->textureNum : 2;
            s->vsync          = desc->vsync != VRI_FALSE;

            DXGI_SWAP_CHAIN_DESC1 sd = {};
            sd.Width                 = s->width;
            sd.Height                = s->height;
            sd.Format                = ToDxgiFormat(desc->format).format;
            sd.SampleDesc.Count      = 1;
            sd.BufferUsage           = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            sd.BufferCount           = s->bufferCount;
            sd.SwapEffect            = DXGI_SWAP_EFFECT_FLIP_DISCARD;
            sd.Scaling               = DXGI_SCALING_STRETCH;
            sd.AlphaMode             = DXGI_ALPHA_MODE_IGNORE;

            ComPtr<IDXGISwapChain1> sc1;
            if (FAILED(d->Factory()->CreateSwapChainForHwnd(queue, hwnd, &sd, nullptr, nullptr, &sc1)))
            {
                delete s;
                d->ReportError("CreateSwapChainForHwnd failed");
                return VriResult_Failure;
            }
            if (FAILED(sc1.As(&s->swapchain)))
            {
                delete s;
                d->ReportError("IDXGISwapChain3 query failed");
                return VriResult_Failure;
            }
            if (!WrapBackbuffers(s))
            {
                ReleaseBackbuffers(s);
                delete s;
                d->ReportError("swapchain GetBuffer failed");
                return VriResult_Failure;
            }
            *out = ToHandle(s);
            return VriResult_Success;
        }

        void VRI_CALL DestroySwapChain(VriSwapChain* swapChain)
        {
            if (!swapChain)
                return;
            SwapChainD3D12* s = SC(swapChain);
            ReleaseBackbuffers(s);
            delete s;
        }

        VriResult VRI_CALL GetSwapChainTextures(VriSwapChain* swapChain, VriTexture** outTextures, uint32_t* ioCount)
        {
            SwapChainD3D12* s = SC(swapChain);
            const uint32_t  n = static_cast<uint32_t>(s->textures.size());
            if (!outTextures)
            {
                if (ioCount)
                    *ioCount = n;
                return VriResult_Success;
            }
            const uint32_t cap = ioCount ? *ioCount : 0u;
            const uint32_t c   = cap < n ? cap : n;
            for (uint32_t i = 0; i < c; ++i)
                outTextures[i] = ToHandle(s->textures[i]);
            if (ioCount)
                *ioCount = c;
            return VriResult_Success;
        }

        VriResult VRI_CALL AcquireNextTexture(VriSwapChain* swapChain,
                                              VriFence*     acquireFence,
                                              uint64_t      signalValue,
                                              uint32_t*     outIndex)
        {
            SwapChainD3D12* s     = SC(swapChain);
            const uint32_t  index = s->swapchain->GetCurrentBackBufferIndex();
            if (acquireFence)
                reinterpret_cast<FenceD3D12*>(acquireFence)
                    ->fence->Signal(signalValue); // CPU-side signal; image is ready
            if (outIndex)
                *outIndex = index;
            return VriResult_Success;
        }

        VriResult VRI_CALL Present(VriSwapChain* swapChain, VriFence* /*waitFence*/, uint64_t /*waitValue*/)
        {
            SwapChainD3D12* s = SC(swapChain);
            return SUCCEEDED(s->swapchain->Present(s->vsync ? 1 : 0, 0)) ? VriResult_Success : VriResult_Failure;
        }

        VriResult VRI_CALL Resize(VriSwapChain* swapChain, uint32_t width, uint32_t height)
        {
            SwapChainD3D12* s = SC(swapChain);
            ReleaseBackbuffers(s); // all backbuffer references must be released before ResizeBuffers
            s->width  = width ? width : 1;
            s->height = height ? height : 1;
            if (FAILED(s->swapchain->ResizeBuffers(
                    s->bufferCount, s->width, s->height, ToDxgiFormat(s->format).format, 0)))
            {
                s->device->ReportError("ResizeBuffers failed");
                return VriResult_Failure;
            }
            return WrapBackbuffers(s) ? VriResult_Success : VriResult_Failure;
        }
    } // namespace

    const VriSwapChainInterface* GetSwapChainInterfaceD3D12()
    {
        static const VriSwapChainInterface table = {
            CreateSwapChain,
            DestroySwapChain,
            GetSwapChainTextures,
            AcquireNextTexture,
            Present,
            Resize,
        };
        return &table;
    }
} // namespace vri::d3d12
