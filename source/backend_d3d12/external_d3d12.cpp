// external_d3d12.cpp - VriExternalInterface for D3D12: export OS handles for buffer/texture
// memory (shared committed resources) and timeline fences (shared ID3D12Fence), so an
// external consumer (CUDA / OptiX / another process or API) can import the same memory and
// timeline. VRI itself does NOT depend on CUDA -- it only produces the handles.
//
// D3D12 shares via NT HANDLEs: a committed resource on a D3D12_HEAP_FLAG_SHARED heap and a
// fence created with D3D12_FENCE_FLAG_SHARED, each exported with ID3D12Device::CreateSharedHandle.
// CUDA imports these as cudaExternalMemoryHandleTypeD3D12Resource (dedicated) and
// cudaExternalSemaphoreHandleTypeD3D12Fence respectively. The returned handle is owned by the
// caller (CloseHandle after import); the VriBuffer/Texture/Fence own the resource independently.

#include "external_d3d12.h"
#include "conversions_d3d12.h"
#include "device_d3d12.h"
#include "objects_d3d12.h"

namespace vri::d3d12
{
    namespace
    {
        inline DeviceD3D12*  Dev(VriDevice* h) { return reinterpret_cast<DeviceD3D12*>(h); }
        inline BufferD3D12*  Buf(VriBuffer* h) { return reinterpret_cast<BufferD3D12*>(h); }
        inline TextureD3D12* Tex(VriTexture* h) { return reinterpret_cast<TextureD3D12*>(h); }
        inline FenceD3D12*   Fen(VriFence* h) { return reinterpret_cast<FenceD3D12*>(h); }

        // D3D12 memory calls export a shared committed resource; fence calls a shared fence.
        bool MemTypeOk(VriExternalHandleType t) { return t == VriExternalHandleType_D3D12Resource; }
        bool FenceTypeOk(VriExternalHandleType t) { return t == VriExternalHandleType_D3D12Fence; }

        // Export an NT HANDLE for a shareable device child (resource or fence).
        VriResult ShareHandle(DeviceD3D12* d, ID3D12DeviceChild* obj, void** out)
        {
            HANDLE h = nullptr;
            if (FAILED(d->Device()->CreateSharedHandle(obj, nullptr, GENERIC_ALL, nullptr, &h)) || !h)
                return VriResult_Failure;
            *out = h;
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateExportableBuffer(VriDevice*            device,
                                                  const VriBufferDesc*  desc,
                                                  VriExternalHandleType handleType,
                                                  VriBuffer**           out)
        {
            if (!desc || !out || !MemTypeOk(handleType))
                return VriResult_InvalidArgument;
            DeviceD3D12* d = Dev(device);

            // Shareable committed resource: DEFAULT (device-local) heap, SHARED flag.
            const bool            uav = (desc->usage & VriBufferUsage_StorageBuffer) != 0;
            D3D12_HEAP_PROPERTIES hp  = {};
            hp.Type                   = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd    = {};
            rd.Dimension              = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width                  = desc->size ? desc->size : 1;
            rd.Height                 = 1;
            rd.DepthOrArraySize       = 1;
            rd.MipLevels              = 1;
            rd.Format                 = DXGI_FORMAT_UNKNOWN;
            rd.SampleDesc.Count       = 1;
            rd.Layout                 = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (uav)
                rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

            BufferD3D12* b = new BufferD3D12 {};
            if (FAILED(d->Device()->CreateCommittedResource(&hp,
                                                            D3D12_HEAP_FLAG_SHARED,
                                                            &rd,
                                                            D3D12_RESOURCE_STATE_COMMON,
                                                            nullptr,
                                                            IID_PPV_ARGS(&b->resource))))
            {
                delete b;
                d->ReportError("CreateCommittedResource (exportable buffer) failed");
                return VriResult_Failure;
            }
            b->device   = d;
            b->size     = desc->size;
            b->heapType = D3D12_HEAP_TYPE_DEFAULT;
            b->state    = uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COMMON;
            *out        = ToHandle(b);
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateExportableTexture(VriDevice*            device,
                                                   const VriTextureDesc* desc,
                                                   VriExternalHandleType handleType,
                                                   VriTexture**          out)
        {
            if (!desc || !out || !MemTypeOk(handleType))
                return VriResult_InvalidArgument;
            DeviceD3D12*         d  = Dev(device);
            const DxgiFormatInfo fi = ToDxgiFormat(desc->format);

            D3D12_HEAP_PROPERTIES hp = {};
            hp.Type                  = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd   = {};
            rd.Dimension             = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rd.Width                 = desc->width ? desc->width : 1;
            rd.Height                = desc->height ? desc->height : 1;
            rd.DepthOrArraySize      = static_cast<UINT16>(desc->layerNum ? desc->layerNum : 1);
            rd.MipLevels             = static_cast<UINT16>(desc->mipNum ? desc->mipNum : 1);
            rd.Format                = fi.format;
            rd.SampleDesc.Count      = desc->sampleNum ? desc->sampleNum : 1;
            rd.Layout                = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            if (desc->usage & VriTextureUsage_ColorAttachment)
                rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            if (desc->usage & VriTextureUsage_ShaderResourceStorage)
                rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

            TextureD3D12* t = new TextureD3D12 {};
            if (FAILED(d->Device()->CreateCommittedResource(&hp,
                                                            D3D12_HEAP_FLAG_SHARED,
                                                            &rd,
                                                            D3D12_RESOURCE_STATE_COMMON,
                                                            nullptr,
                                                            IID_PPV_ARGS(&t->resource))))
            {
                delete t;
                d->ReportError("CreateCommittedResource (exportable texture) failed");
                return VriResult_Failure;
            }
            t->device      = d;
            t->format      = rd.Format;
            t->dsvFormat   = fi.format;
            t->srvFormat   = fi.format;
            t->texelSize   = fi.texelSize;
            t->width       = desc->width;
            t->height      = desc->height ? desc->height : 1;
            t->depth       = 1;
            t->mipNum      = rd.MipLevels;
            t->layerNum    = rd.DepthOrArraySize;
            t->state       = D3D12_RESOURCE_STATE_COMMON;
            t->sampleCount = rd.SampleDesc.Count;
            *out           = ToHandle(t);
            return VriResult_Success;
        }

        VriResult FillResourceHandle(DeviceD3D12*           d,
                                     ID3D12Resource*        resource,
                                     VriExternalHandleType  handleType,
                                     VriExternalMemoryInfo* out)
        {
            if (!resource || !out || !MemTypeOk(handleType))
                return VriResult_InvalidArgument;
            const VriResult r = ShareHandle(d, resource, &out->handle);
            if (r != VriResult_Success)
                return r;
            // CUDA needs the total allocation size, not just the requested width.
            const D3D12_RESOURCE_DESC            rd = resource->GetDesc();
            const D3D12_RESOURCE_ALLOCATION_INFO ai = d->Device()->GetResourceAllocationInfo(0, 1, &rd);
            out->size                               = ai.SizeInBytes;
            out->dedicated                          = VRI_TRUE; // committed resource == a dedicated allocation
            return VriResult_Success;
        }

        VriResult VRI_CALL GetBufferMemoryHandle(VriDevice*             device,
                                                 VriBuffer*             buffer,
                                                 VriExternalHandleType  handleType,
                                                 VriExternalMemoryInfo* out)
        {
            if (!buffer)
                return VriResult_InvalidArgument;
            return FillResourceHandle(Dev(device), Buf(buffer)->resource.Get(), handleType, out);
        }

        VriResult VRI_CALL GetTextureMemoryHandle(VriDevice*             device,
                                                  VriTexture*            texture,
                                                  VriExternalHandleType  handleType,
                                                  VriExternalMemoryInfo* out)
        {
            if (!texture)
                return VriResult_InvalidArgument;
            return FillResourceHandle(Dev(device), Tex(texture)->resource.Get(), handleType, out);
        }

        VriResult VRI_CALL CreateExportableFence(VriDevice*            device,
                                                 uint64_t              initialValue,
                                                 VriExternalHandleType handleType,
                                                 VriFence**            out)
        {
            if (!out || !FenceTypeOk(handleType))
                return VriResult_InvalidArgument;
            DeviceD3D12* d = Dev(device);
            FenceD3D12*  f = new FenceD3D12 {};
            f->device      = d;
            if (FAILED(d->Device()->CreateFence(initialValue, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&f->fence))))
            {
                delete f;
                d->ReportError("CreateFence (shared) failed");
                return VriResult_Failure;
            }
            f->event = CreateEventA(nullptr, FALSE, FALSE, nullptr); // for host Wait, mirrors core CreateFence
            *out     = ToHandle(f);
            return VriResult_Success;
        }

        VriResult VRI_CALL GetFenceHandle(VriDevice*            device,
                                          VriFence*             fence,
                                          VriExternalHandleType handleType,
                                          void**                outHandle)
        {
            if (!fence || !outHandle || !FenceTypeOk(handleType))
                return VriResult_InvalidArgument;
            return ShareHandle(Dev(device), Fen(fence)->fence.Get(), outHandle);
        }

        const VriExternalInterface g_externalD3D12 = {
            CreateExportableBuffer,
            CreateExportableTexture,
            GetBufferMemoryHandle,
            GetTextureMemoryHandle,
            CreateExportableFence,
            GetFenceHandle,
        };
    } // namespace

    const VriExternalInterface* GetExternalInterfaceD3D12() { return &g_externalD3D12; }
} // namespace vri::d3d12
