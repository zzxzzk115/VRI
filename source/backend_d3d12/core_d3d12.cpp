// core_d3d12.cpp - the D3D12 core function table.
//
// Phase 1 implements the clear+readback path: resources (committed default/upload/
// readback heaps), RTV views, command allocator/list, resource-state barriers,
// render-pass clears, texture->buffer readback, fences and submission. Pipelines,
// descriptor sets and draws are stubbed (Unsupported/no-op) until Phase 2.
#include "core_d3d12.h"

#include "conversions_d3d12.h"
#include "device_d3d12.h"
#include "objects_d3d12.h"

#include <d3dcompiler.h> // D3DReflect (VS input-signature reflection for the input layout)

#include <cstring>
#include <string>
#include <vector>

namespace vri::d3d12
{
    namespace
    {
        DeviceD3D12*               Dev(VriDevice* d) { return reinterpret_cast<DeviceD3D12*>(d); }
        const DeviceD3D12*         Dev(const VriDevice* d) { return reinterpret_cast<const DeviceD3D12*>(d); }
        BufferD3D12*               Buf(VriBuffer* b) { return reinterpret_cast<BufferD3D12*>(b); }
        TextureD3D12*              Tex(VriTexture* t) { return reinterpret_cast<TextureD3D12*>(t); }
        DescriptorD3D12*           Desc(VriDescriptor* d) { return reinterpret_cast<DescriptorD3D12*>(d); }
        const DescriptorD3D12*     Desc(const VriDescriptor* d) { return reinterpret_cast<const DescriptorD3D12*>(d); }
        QueueD3D12*                Q(VriQueue* q) { return reinterpret_cast<QueueD3D12*>(q); }
        CommandAllocatorD3D12*     Alloc(VriCommandAllocator* a) { return reinterpret_cast<CommandAllocatorD3D12*>(a); }
        CommandBufferD3D12*        CB(VriCommandBuffer* c) { return reinterpret_cast<CommandBufferD3D12*>(c); }
        FenceD3D12*                Fen(VriFence* f) { return reinterpret_cast<FenceD3D12*>(f); }
        PipelineLayoutD3D12*       PL(VriPipelineLayout* p) { return reinterpret_cast<PipelineLayoutD3D12*>(p); }
        const PipelineLayoutD3D12* PL(const VriPipelineLayout* p)
        {
            return reinterpret_cast<const PipelineLayoutD3D12*>(p);
        }
        PipelineD3D12*       Pipe(VriPipeline* p) { return reinterpret_cast<PipelineD3D12*>(p); }
        DescriptorPoolD3D12* DPool(VriDescriptorPool* p) { return reinterpret_cast<DescriptorPoolD3D12*>(p); }
        DescriptorSetD3D12*  DSet(VriDescriptorSet* s) { return reinterpret_cast<DescriptorSetD3D12*>(s); }

        D3D12_RESOURCE_STATES ToState(VriLayout layout)
        {
            switch (layout)
            {
                case VriLayout_ColorAttachment:
                    return D3D12_RESOURCE_STATE_RENDER_TARGET;
                case VriLayout_DepthStencilAttachment:
                    return D3D12_RESOURCE_STATE_DEPTH_WRITE;
                case VriLayout_DepthStencilReadOnly:
                    return D3D12_RESOURCE_STATE_DEPTH_READ;
                case VriLayout_ShaderResource:
                    return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                case VriLayout_CopySource:
                    return D3D12_RESOURCE_STATE_COPY_SOURCE;
                case VriLayout_CopyDestination:
                    return D3D12_RESOURCE_STATE_COPY_DEST;
                case VriLayout_Present:
                    return D3D12_RESOURCE_STATE_PRESENT;
                case VriLayout_General:
                    return D3D12_RESOURCE_STATE_COMMON;
                default:
                    return D3D12_RESOURCE_STATE_COMMON;
            }
        }

        void Transition(CommandBufferD3D12* c, TextureD3D12* t, D3D12_RESOURCE_STATES after)
        {
            if (t->state == after)
                return;
            D3D12_RESOURCE_BARRIER b = {};
            b.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource   = t->resource.Get();
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            b.Transition.StateBefore = t->state;
            b.Transition.StateAfter  = after;
            c->list->ResourceBarrier(1, &b);
            t->state = after;
        }

        void TransitionBuffer(CommandBufferD3D12* c, BufferD3D12* b, D3D12_RESOURCE_STATES after)
        {
            if (b->heapType != D3D12_HEAP_TYPE_DEFAULT || b->state == after)
                return; // UPLOAD/READBACK heaps are fixed-state
            D3D12_RESOURCE_BARRIER rb = {};
            rb.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            rb.Transition.pResource   = b->resource.Get();
            rb.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            rb.Transition.StateBefore = b->state;
            rb.Transition.StateAfter  = after;
            c->list->ResourceBarrier(1, &rb);
            b->state = after;
        }

        D3D12_RESOURCE_STATES BufferStateForAccess(VriAccessFlags a)
        {
            if (a & VriAccess_CopySourceRead)
                return D3D12_RESOURCE_STATE_COPY_SOURCE;
            if (a & VriAccess_CopyDestinationWrite)
                return D3D12_RESOURCE_STATE_COPY_DEST;
            if (a & (VriAccess_ShaderResourceStorageWrite | VriAccess_ShaderResourceStorageRead))
                return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            if (a & VriAccess_IndexBufferRead)
                return D3D12_RESOURCE_STATE_INDEX_BUFFER;
            if (a & (VriAccess_VertexBufferRead | VriAccess_ConstantBufferRead))
                return D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
            return D3D12_RESOURCE_STATE_COMMON;
        }

        // ---- queries ---------------------------------------------------
        const VriDeviceDesc* VRI_CALL  GetDeviceDesc(const VriDevice* device) { return &Dev(device)->Desc(); }
        VriFormatSupportFlags VRI_CALL GetFormatSupport(const VriDevice*, VriFormat)
        {
            return VriFormatSupport_Texture | VriFormatSupport_ColorAttachment | VriFormatSupport_Blend |
                   VriFormatSupport_VertexBuffer;
        }
        VriResult VRI_CALL GetQueue(VriDevice* device, VriQueueType type, uint32_t, VriQueue** outQueue)
        {
            if (type >= VriQueueType_Count)
                return VriResult_InvalidArgument;
            *outQueue = ToHandle(Dev(device)->GetQueue(type));
            return VriResult_Success;
        }

        // ---- resources -------------------------------------------------
        VriResult VRI_CALL CreateBuffer(VriDevice* device, const VriBufferDesc* desc, VriBuffer** out)
        {
            DeviceD3D12*          d     = Dev(device);
            D3D12_HEAP_TYPE       heap  = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
            if (desc->memoryLocation == VriMemoryLocation_HostUpload)
            {
                heap  = D3D12_HEAP_TYPE_UPLOAD;
                state = D3D12_RESOURCE_STATE_GENERIC_READ;
            }
            else if (desc->memoryLocation == VriMemoryLocation_HostReadback)
            {
                heap  = D3D12_HEAP_TYPE_READBACK;
                state = D3D12_RESOURCE_STATE_COPY_DEST;
            }

            // Storage buffers are UAVs: they need the UAV resource flag. D3D12 *ignores* the
            // CreateCommittedResource InitialState for buffers (they are always created in COMMON,
            // and passing anything else trips a debug-layer warning), so the creation state stays
            // COMMON. We still TRACK storage buffers as UNORDERED_ACCESS: a buffer promotes
            // COMMON->UAV implicitly on first GPU access, so after the compute write the readback
            // barrier correctly emits UAV->COPY_SOURCE (StateBefore comes from the tracked state).
            const bool uav = (heap == D3D12_HEAP_TYPE_DEFAULT) && (desc->usage & VriBufferUsage_StorageBuffer);
            const D3D12_RESOURCE_STATES trackedState = uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : state;

            D3D12_HEAP_PROPERTIES hp = {};
            hp.Type                  = heap;
            D3D12_RESOURCE_DESC rd   = {};
            rd.Dimension             = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width                 = desc->size ? desc->size : 1;
            rd.Height                = 1;
            rd.DepthOrArraySize      = 1;
            rd.MipLevels             = 1;
            rd.Format                = DXGI_FORMAT_UNKNOWN;
            rd.SampleDesc.Count      = 1;
            rd.Layout                = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (uav)
                rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

            BufferD3D12* b = new BufferD3D12 {};
            if (FAILED(d->Device()->CreateCommittedResource(
                    &hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&b->resource))))
            {
                delete b;
                d->ReportError("CreateCommittedResource (buffer) failed");
                return VriResult_Failure;
            }
            b->device   = d;
            b->size     = desc->size;
            b->heapType = heap;
            b->state    = trackedState;
            if (heap != D3D12_HEAP_TYPE_DEFAULT) // persistently map upload/readback heaps
            {
                D3D12_RANGE none = {0, 0};
                b->resource->Map(0, heap == D3D12_HEAP_TYPE_READBACK ? nullptr : &none, &b->mapped);
            }
            *out = ToHandle(b);
            return VriResult_Success;
        }
        void VRI_CALL DestroyBuffer(VriBuffer* buffer)
        {
            if (buffer)
                delete Buf(buffer);
        }
        void* VRI_CALL MapBuffer(VriBuffer* buffer, uint64_t offset, uint64_t)
        {
            BufferD3D12* b = Buf(buffer);
            return b->mapped ? static_cast<char*>(b->mapped) + offset : nullptr;
        }
        void VRI_CALL     UnmapBuffer(VriBuffer*) {} // persistent mapping; nothing to do
        uint64_t VRI_CALL GetBufferDeviceAddress(const VriBuffer*) { return 0; }

        VriResult VRI_CALL CreateTexture(VriDevice* device, const VriTextureDesc* desc, VriTexture** out)
        {
            DeviceD3D12*          d  = Dev(device);
            const DxgiFormatInfo  fi = ToDxgiFormat(desc->format);
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
            // A depth target that is ALSO sampled (shadow map) must use a TYPELESS resource so it
            // can carry both a typed DSV and a typed SRV (D3D12 forbids a D*_ format SRV directly).
            DXGI_FORMAT dsvFormat = fi.format, srvFormat = fi.format;
            const bool  depthSampled = (desc->usage & VriTextureUsage_DepthStencilAttachment) &&
                                      (desc->usage & VriTextureUsage_ShaderResource);
            if (depthSampled)
            {
                switch (fi.format)
                {
                    case DXGI_FORMAT_D32_FLOAT:
                        rd.Format = DXGI_FORMAT_R32_TYPELESS;
                        srvFormat = DXGI_FORMAT_R32_FLOAT;
                        break;
                    case DXGI_FORMAT_D16_UNORM:
                        rd.Format = DXGI_FORMAT_R16_TYPELESS;
                        srvFormat = DXGI_FORMAT_R16_UNORM;
                        break;
                    case DXGI_FORMAT_D24_UNORM_S8_UINT:
                        rd.Format = DXGI_FORMAT_R24G8_TYPELESS;
                        srvFormat = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
                        break;
                    default:
                        break; // already a sampleable format
                }
            }
            if (desc->usage & VriTextureUsage_ColorAttachment)
                rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            if (desc->usage & VriTextureUsage_DepthStencilAttachment)
                rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            if (desc->usage & VriTextureUsage_ShaderResourceStorage)
                rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

            // Bake the app-declared clear value into the resource so D3D12 takes the fast-clear path.
            // It must match the value passed to CmdBeginRendering or the debug layer warns; the app
            // owns that (VriTextureDesc::clearValue mirrors VriAttachmentDesc::clearValue). A typeless
            // depth target needs the typed clear format.
            D3D12_CLEAR_VALUE        cv {};
            const D3D12_CLEAR_VALUE* pcv = nullptr;
            if (desc->usage & VriTextureUsage_DepthStencilAttachment)
            {
                cv.Format               = dsvFormat;
                cv.DepthStencil.Depth   = desc->clearValue.depthStencil.depth;
                cv.DepthStencil.Stencil = static_cast<UINT8>(desc->clearValue.depthStencil.stencil);
                pcv                     = &cv;
            }
            else if (desc->usage & VriTextureUsage_ColorAttachment)
            {
                cv.Format = fi.format;
                for (int i = 0; i < 4; ++i)
                    cv.Color[i] = desc->clearValue.color.f32[i];
                pcv = &cv;
            }

            TextureD3D12* t = new TextureD3D12 {};
            if (FAILED(d->Device()->CreateCommittedResource(
                    &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, pcv, IID_PPV_ARGS(&t->resource))))
            {
                delete t;
                d->ReportError("CreateCommittedResource (texture) failed");
                return VriResult_Failure;
            }
            t->device         = d;
            t->format         = rd.Format;
            t->dsvFormat      = dsvFormat;
            t->srvFormat      = srvFormat;
            t->texelSize      = fi.texelSize;
            t->width          = desc->width;
            t->height         = desc->height ? desc->height : 1;
            t->depth          = 1;
            t->mipNum         = rd.MipLevels;
            t->layerNum       = rd.DepthOrArraySize;
            t->state          = D3D12_RESOURCE_STATE_COMMON;
            t->sampleCount    = rd.SampleDesc.Count;
            t->isRenderTarget = (desc->usage & VriTextureUsage_ColorAttachment) != 0;
            t->isDepthStencil = (desc->usage & VriTextureUsage_DepthStencilAttachment) != 0;
            *out              = ToHandle(t);
            return VriResult_Success;
        }
        void VRI_CALL DestroyTexture(VriTexture* texture)
        {
            if (texture)
                delete Tex(texture);
        }

        // ---- views -----------------------------------------------------
        VriResult VRI_CALL CreateTextureView(VriDevice* device, const VriTextureViewDesc* desc, VriDescriptor** out)
        {
            DeviceD3D12*     d = Dev(device);
            DescriptorD3D12* v = new DescriptorD3D12 {};
            v->device          = d;
            v->texture         = reinterpret_cast<const TextureD3D12*>(desc->texture);
            v->mip             = desc->baseMip;
            v->viewType        = desc->viewType;
            // The view records the texture; it serves as an RTV (color attachment) and/or an
            // SRV source (sampling - the SRV is created into a descriptor set's heap slot at
            // UpdateDescriptorRanges). Pre-create the RTV only for RT-capable textures.
            v->kind = DescriptorD3D12::Kind::TextureRtv;
            if (v->texture->isDepthStencil) // depth-stencil view (used as the depth attachment)
            {
                v->cpu                            = d->AllocDsv();
                D3D12_DEPTH_STENCIL_VIEW_DESC dsv = {};
                dsv.Format             = v->texture->dsvFormat; // typed depth (resource may be TYPELESS for sampling)
                dsv.ViewDimension      = D3D12_DSV_DIMENSION_TEXTURE2D;
                dsv.Texture2D.MipSlice = desc->baseMip;
                d->Device()->CreateDepthStencilView(v->texture->resource.Get(), &dsv, v->cpu);
            }
            else if (v->texture->isRenderTarget)
            {
                v->cpu                            = d->AllocRtv();
                D3D12_RENDER_TARGET_VIEW_DESC rtv = {};
                rtv.Format                        = v->texture->format;
                if (v->texture->sampleCount > 1)
                    rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
                else
                {
                    rtv.ViewDimension      = D3D12_RTV_DIMENSION_TEXTURE2D;
                    rtv.Texture2D.MipSlice = desc->baseMip;
                }
                d->Device()->CreateRenderTargetView(v->texture->resource.Get(), &rtv, v->cpu);
            }
            *out = ToHandle(v);
            return VriResult_Success;
        }
        VriResult VRI_CALL CreateBufferView(VriDevice* device, const VriBufferViewDesc* desc, VriDescriptor** out)
        {
            // Records the buffer range; bound as a root CBV (by GPU address) at draw time.
            DescriptorD3D12* v = new DescriptorD3D12 {};
            v->kind            = DescriptorD3D12::Kind::BufferView;
            v->device          = Dev(device);
            v->buffer          = reinterpret_cast<const BufferD3D12*>(desc->buffer);
            v->bufferOffset    = desc->offset;
            v->bufferSize      = desc->size;
            *out               = ToHandle(v);
            return VriResult_Success;
        }
        VriResult VRI_CALL CreateSampler(VriDevice* device, const VriSamplerDesc* desc, VriDescriptor** out)
        {
            DescriptorD3D12* v = new DescriptorD3D12 {};
            v->kind            = DescriptorD3D12::Kind::Sampler;
            v->device          = Dev(device);
            v->sampler         = *desc; // realized into a sampler heap slot at UpdateDescriptorRanges
            *out               = ToHandle(v);
            return VriResult_Success;
        }
        void VRI_CALL DestroyDescriptor(VriDescriptor* descriptor)
        {
            if (descriptor)
                delete Desc(descriptor);
        }

        // ---- command allocation / recording ----------------------------
        VriResult VRI_CALL CreateCommandAllocator(VriDevice* device, VriQueueType type, VriCommandAllocator** out)
        {
            DeviceD3D12* d = Dev(device);
            // Allocator's engine must match the queue its lists submit to: Graphics->DIRECT,
            // Compute->COMPUTE, Transfer->COPY. Lists carry this type so submission stays valid.
            static constexpr D3D12_COMMAND_LIST_TYPE kListType[VriQueueType_Count] = {
                D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_COPY};
            const D3D12_COMMAND_LIST_TYPE listType =
                kListType[type < VriQueueType_Count ? type : VriQueueType_Graphics];
            CommandAllocatorD3D12* a = new CommandAllocatorD3D12 {};
            a->device                = d;
            a->listType              = listType;
            if (FAILED(d->Device()->CreateCommandAllocator(listType, IID_PPV_ARGS(&a->allocator))))
            {
                delete a;
                d->ReportError("CreateCommandAllocator failed");
                return VriResult_Failure;
            }
            *out = ToHandle(a);
            return VriResult_Success;
        }
        void VRI_CALL ResetCommandAllocator(VriCommandAllocator* a) { Alloc(a)->allocator->Reset(); }
        void VRI_CALL DestroyCommandAllocator(VriCommandAllocator* a)
        {
            if (a)
                delete Alloc(a);
        }
        VriResult VRI_CALL CreateCommandBuffer(VriCommandAllocator* allocator, VriCommandBuffer** out)
        {
            CommandAllocatorD3D12* a = Alloc(allocator);
            CommandBufferD3D12*    c = new CommandBufferD3D12 {};
            c->device                = a->device;
            c->allocator             = a;
            if (FAILED(a->device->Device()->CreateCommandList(
                    0, a->listType, a->allocator.Get(), nullptr, IID_PPV_ARGS(&c->list))))
            {
                delete c;
                a->device->ReportError("CreateCommandList failed");
                return VriResult_Failure;
            }
            c->list->Close(); // created open; BeginCommandBuffer resets it
            *out = ToHandle(c);
            return VriResult_Success;
        }
        VriResult VRI_CALL BeginCommandBuffer(VriCommandBuffer* cmd)
        {
            CommandBufferD3D12* c = CB(cmd);
            if (FAILED(c->list->Reset(c->allocator->allocator.Get(), nullptr)))
                return VriResult_Failure;
            c->rtvCount      = 0;
            c->boundPipeline = nullptr;
            c->boundLayout   = nullptr;
            for (auto& vb : c->pendingVBs)
                vb = {}; // stale bindings/strides must not leak across recordings
            c->vbDirty = false;
            c->tempUploads.clear(); // prior frame's bounce buffers are done (caller waited on the fence)
            return VriResult_Success;
        }
        VriResult VRI_CALL EndCommandBuffer(VriCommandBuffer* cmd)
        {
            return SUCCEEDED(CB(cmd)->list->Close()) ? VriResult_Success : VriResult_Failure;
        }

        // ---- barriers --------------------------------------------------
        void VRI_CALL CmdBarrier(VriCommandBuffer* cmd, const VriBarrierGroupDesc* g)
        {
            CommandBufferD3D12* c = CB(cmd);
            if (!g)
                return;
            for (uint32_t i = 0; i < g->textureNum; ++i)
            {
                const VriTextureBarrierDesc& tb = g->textures[i];
                if (tb.texture)
                    Transition(c, reinterpret_cast<TextureD3D12*>(tb.texture), ToState(tb.after.layout));
            }
            for (uint32_t i = 0; i < g->bufferNum; ++i)
            {
                const VriBufferBarrierDesc& bb = g->buffers[i];
                if (bb.buffer)
                    TransitionBuffer(
                        c, reinterpret_cast<BufferD3D12*>(bb.buffer), BufferStateForAccess(bb.after.access));
            }
        }

        // ---- render pass ----------------------------------------------
        void VRI_CALL CmdBeginRendering(VriCommandBuffer* cmd, const VriAttachmentsDesc* a)
        {
            CommandBufferD3D12* c = CB(cmd);
            c->rtvCount           = a->colorNum <= 8 ? a->colorNum : 8;
            c->resolveCount       = 0;
            for (uint32_t i = 0; i < c->rtvCount; ++i)
            {
                const DescriptorD3D12* v = Desc(a->colors[i].view);
                TextureD3D12*          t = const_cast<TextureD3D12*>(v->texture);
                Transition(c, t, D3D12_RESOURCE_STATE_RENDER_TARGET); // defensive (usually already RT)
                c->rtvs[i] = v->cpu;
                if (a->colors[i].resolveView) // MSAA -> single-sample resolve at EndRendering
                {
                    const DescriptorD3D12* rv      = Desc(a->colors[i].resolveView);
                    c->resolves[c->resolveCount++] = {t, const_cast<TextureD3D12*>(rv->texture)};
                }
            }
            D3D12_CPU_DESCRIPTOR_HANDLE dsv      = {};
            const bool                  hasDepth = a->depth && a->depth->view;
            if (hasDepth)
            {
                const DescriptorD3D12* dv = Desc(a->depth->view);
                Transition(c, const_cast<TextureD3D12*>(dv->texture), D3D12_RESOURCE_STATE_DEPTH_WRITE);
                dsv = dv->cpu;
            }
            c->list->OMSetRenderTargets(
                static_cast<UINT>(c->rtvCount), c->rtvCount ? c->rtvs : nullptr, FALSE, hasDepth ? &dsv : nullptr);
            for (uint32_t i = 0; i < c->rtvCount; ++i)
                if (a->colors[i].loadOp == VriAttachmentLoadOp_Clear)
                    c->list->ClearRenderTargetView(c->rtvs[i], a->colors[i].clearValue.color.f32, 0, nullptr);
            if (hasDepth && a->depth->loadOp == VriAttachmentLoadOp_Clear)
            {
                // Clear stencil too for combined depth+stencil targets (else the stencil plane keeps
                // stale values across frames - a stencil test against a clear value then misbehaves).
                const DXGI_FORMAT df    = Desc(a->depth->view)->texture->dsvFormat;
                D3D12_CLEAR_FLAGS flags = D3D12_CLEAR_FLAG_DEPTH;
                if (df == DXGI_FORMAT_D24_UNORM_S8_UINT || df == DXGI_FORMAT_D32_FLOAT_S8X24_UINT)
                    flags |= D3D12_CLEAR_FLAG_STENCIL;
                c->list->ClearDepthStencilView(dsv,
                                               flags,
                                               a->depth->clearValue.depthStencil.depth,
                                               static_cast<UINT8>(a->depth->clearValue.depthStencil.stencil),
                                               0,
                                               nullptr);
            }
        }
        void VRI_CALL CmdEndRendering(VriCommandBuffer* cmd)
        {
            CommandBufferD3D12* c = CB(cmd);
            for (uint32_t i = 0; i < c->resolveCount; ++i)
            {
                TextureD3D12* src = c->resolves[i].src;
                TextureD3D12* dst = c->resolves[i].dst;
                Transition(c, src, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
                Transition(c, dst, D3D12_RESOURCE_STATE_RESOLVE_DEST);
                c->list->ResolveSubresource(dst->resource.Get(), 0, src->resource.Get(), 0, dst->format);
            }
            c->resolveCount = 0;
        }
        void VRI_CALL CmdSetViewports(VriCommandBuffer* cmd, const VriViewport* vps, uint32_t num)
        {
            if (!num)
                return;
            std::vector<D3D12_VIEWPORT> v(num);
            for (uint32_t i = 0; i < num; ++i)
                v[i] = {vps[i].x, vps[i].y, vps[i].width, vps[i].height, vps[i].minDepth, vps[i].maxDepth};
            CB(cmd)->list->RSSetViewports(num, v.data());
        }
        void VRI_CALL CmdSetScissors(VriCommandBuffer* cmd, const VriRect* r, uint32_t num)
        {
            if (!num)
                return;
            std::vector<D3D12_RECT> rects(num);
            for (uint32_t i = 0; i < num; ++i)
                rects[i] = {
                    r[i].x, r[i].y, r[i].x + static_cast<LONG>(r[i].width), r[i].y + static_cast<LONG>(r[i].height)};
            CB(cmd)->list->RSSetScissorRects(num, rects.data());
        }

        // ---- readback (texture -> buffer) ------------------------------
        void VRI_CALL CmdReadbackTextureToBuffer(VriCommandBuffer*               cmd,
                                                 VriBuffer*                      dst,
                                                 VriTexture*                     src,
                                                 const VriBufferTextureCopyDesc* region)
        {
            CommandBufferD3D12*                c        = CB(cmd);
            TextureD3D12*                      t        = Tex(src);
            BufferD3D12*                       b        = Buf(dst);
            const UINT                         sub      = region ? region->texture.mip : 0;
            D3D12_RESOURCE_DESC                rd       = t->resource->GetDesc();
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp       = {};
            UINT                               rows     = 0;
            UINT64                             rowBytes = 0, total = 0;
            c->device->Device()->GetCopyableFootprints(
                &rd, sub, 1, region ? region->bufferOffset : 0, &fp, &rows, &rowBytes, &total);
            Transition(c, t, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
            dstLoc.pResource                   = b->resource.Get();
            dstLoc.Type                        = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dstLoc.PlacedFootprint             = fp;
            D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
            srcLoc.pResource                   = t->resource.Get();
            srcLoc.Type                        = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            srcLoc.SubresourceIndex            = sub;
            c->list->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        }

        // ---- submission / fences --------------------------------------
        void VRI_CALL QueueSubmit(VriQueue* queue, const VriQueueSubmitDesc* submit)
        {
            QueueD3D12* q = Q(queue);
            for (uint32_t i = 0; i < submit->waitFenceNum; ++i)
                q->queue->Wait(Fen(submit->waitFences[i].fence)->fence.Get(), submit->waitFences[i].value);
            std::vector<ID3D12CommandList*> lists(submit->commandBufferNum);
            for (uint32_t i = 0; i < submit->commandBufferNum; ++i)
                lists[i] = CB(submit->commandBuffers[i])->list.Get();
            if (!lists.empty())
                q->queue->ExecuteCommandLists(static_cast<UINT>(lists.size()), lists.data());
            for (uint32_t i = 0; i < submit->signalFenceNum; ++i)
                q->queue->Signal(Fen(submit->signalFences[i].fence)->fence.Get(), submit->signalFences[i].value);
        }
        void WaitQueueIdle(QueueD3D12* q)
        {
            if (!q->idleFence)
                return;
            const uint64_t v = ++q->idleValue;
            q->queue->Signal(q->idleFence.Get(), v);
            if (q->idleFence->GetCompletedValue() < v)
            {
                q->idleFence->SetEventOnCompletion(v, q->idleEvent);
                WaitForSingleObject(q->idleEvent, INFINITE);
            }
        }
        void VRI_CALL QueueWaitIdle(VriQueue* queue) { WaitQueueIdle(Q(queue)); }
        void VRI_CALL DeviceWaitIdle(VriDevice* device)
        {
            DeviceD3D12* d = Dev(device);
            for (int t = 0; t < VriQueueType_Count; ++t)
                WaitQueueIdle(d->GetQueue(static_cast<VriQueueType>(t)));
        }

        VriResult VRI_CALL CreateFence(VriDevice* device, uint64_t initialValue, VriFence** out)
        {
            DeviceD3D12* d = Dev(device);
            FenceD3D12*  f = new FenceD3D12 {};
            f->device      = d;
            if (FAILED(d->Device()->CreateFence(initialValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f->fence))))
            {
                delete f;
                d->ReportError("CreateFence failed");
                return VriResult_Failure;
            }
            f->event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
            *out     = ToHandle(f);
            return VriResult_Success;
        }
        void VRI_CALL DestroyFence(VriFence* fence)
        {
            if (!fence)
                return;
            FenceD3D12* f = Fen(fence);
            if (f->event)
                CloseHandle(f->event);
            delete f;
        }
        uint64_t VRI_CALL GetFenceValue(VriFence* fence) { return Fen(fence)->fence->GetCompletedValue(); }
        void VRI_CALL     Wait(VriFence* fence, uint64_t value)
        {
            FenceD3D12* f = Fen(fence);
            if (f->fence->GetCompletedValue() >= value)
                return;
            f->fence->SetEventOnCompletion(value, f->event);
            WaitForSingleObject(f->event, INFINITE);
        }

        // ---- not yet implemented (Phase 2) -----------------------------
        void VRI_CALL GetBufferMemoryDesc(const VriDevice*, const VriBufferDesc*, VriMemoryLocation, VriMemoryDesc* o)
        {
            if (o)
                *o = VriMemoryDesc {};
        }
        void VRI_CALL GetTextureMemoryDesc(const VriDevice*, const VriTextureDesc*, VriMemoryLocation, VriMemoryDesc* o)
        {
            if (o)
                *o = VriMemoryDesc {};
        }
        VriResult VRI_CALL AllocateMemory(VriDevice*, const VriMemoryDesc*, VriMemory**)
        {
            return VriResult_Unsupported;
        }
        void VRI_CALL      FreeMemory(VriMemory*) {}
        VriResult VRI_CALL BindBufferMemory(VriDevice*, VriBuffer*, VriMemory*, uint64_t)
        {
            return VriResult_Unsupported;
        }
        VriResult VRI_CALL BindTextureMemory(VriDevice*, VriTexture*, VriMemory*, uint64_t)
        {
            return VriResult_Unsupported;
        }
        VriResult VRI_CALL CreatePipelineLayout(VriDevice*                   device,
                                                const VriPipelineLayoutDesc* desc,
                                                VriPipelineLayout**          out)
        {
            DeviceD3D12*         d = Dev(device);
            PipelineLayoutD3D12* l = new PipelineLayoutD3D12 {};
            l->device              = d;
            // Constant buffers -> root CBV (by GPU address). Textures (SRV) and samplers ->
            // one descriptor table per set per heap-type. D3D12 registers are per-type, so we
            // flatten per-type from 0 within the set (matching Slang's HLSL assignment): the
            // first texture -> t0, first sampler -> s0, etc.
            std::vector<D3D12_ROOT_PARAMETER>                params;
            std::vector<std::vector<D3D12_DESCRIPTOR_RANGE>> rangeStore; // keep ranges alive for serialize
            rangeStore.reserve(desc->descriptorSetNum * 2);
            for (uint32_t s = 0; s < desc->descriptorSetNum; ++s)
            {
                const VriDescriptorSetDesc& sd = desc->descriptorSets[s];
                LayoutSetD3D12              setInfo {};
                setInfo.set                = sd.registerSpace;
                uint32_t            srvIdx = 0, uavIdx = 0, samplerIdx = 0;
                std::vector<size_t> srvBindings, uavBindings, samplerBindings; // indices into l->bindings to patch
                auto                isUav = [](VriDescriptorType t) { return t == VriDescriptorType_StorageTexture; };
                for (uint32_t r = 0; r < sd.rangeNum; ++r)
                {
                    const VriDescriptorRangeDesc& rd = sd.ranges[r];
                    const uint32_t                n  = rd.descriptorNum ? rd.descriptorNum : 1u;
                    if (rd.descriptorType == VriDescriptorType_ConstantBuffer ||
                        rd.descriptorType == VriDescriptorType_StorageBuffer)
                    {
                        // Constant buffer -> root CBV; storage (RW) buffer -> root UAV.
                        D3D12_ROOT_PARAMETER p      = {};
                        p.ParameterType             = rd.descriptorType == VriDescriptorType_ConstantBuffer ?
                                                          D3D12_ROOT_PARAMETER_TYPE_CBV :
                                                          D3D12_ROOT_PARAMETER_TYPE_UAV;
                        p.Descriptor.ShaderRegister = rd.baseRegister;
                        p.Descriptor.RegisterSpace  = sd.registerSpace;
                        p.ShaderVisibility          = D3D12_SHADER_VISIBILITY_ALL;
                        l->bindings.push_back({sd.registerSpace,
                                               rd.baseRegister,
                                               rd.descriptorType,
                                               static_cast<uint32_t>(params.size()),
                                               0});
                        params.push_back(p);
                    }
                    else if (rd.descriptorType == VriDescriptorType_Sampler)
                    {
                        l->bindings.push_back({sd.registerSpace, rd.baseRegister, rd.descriptorType, 0, samplerIdx});
                        samplerBindings.push_back(l->bindings.size() - 1);
                        samplerIdx += n;
                    }
                    else if (isUav(rd.descriptorType)) // StorageTexture -> UAV (u#) in the set's table
                    {
                        l->bindings.push_back({sd.registerSpace, rd.baseRegister, rd.descriptorType, 0, uavIdx});
                        uavBindings.push_back(l->bindings.size() - 1);
                        uavIdx += n;
                    }
                    else // Texture / StructuredBuffer / AccelerationStructure -> SRV (t#) in the set's table
                    {
                        l->bindings.push_back({sd.registerSpace, rd.baseRegister, rd.descriptorType, 0, srvIdx});
                        srvBindings.push_back(l->bindings.size() - 1);
                        srvIdx += n;
                    }
                }
                // SRV + UAV share one descriptor table (one root param). Heap layout (APPEND):
                // SRV slots [0, srvIdx) then UAV slots [srvIdx, srvIdx+uavIdx). UAV registers
                // (u#) restart at 0, so the UAV range's BaseShaderRegister is 0 too.
                if (srvIdx > 0 || uavIdx > 0)
                {
                    std::vector<D3D12_DESCRIPTOR_RANGE> ranges;
                    if (srvIdx > 0)
                    {
                        D3D12_DESCRIPTOR_RANGE rg            = {};
                        rg.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
                        rg.NumDescriptors                    = srvIdx;
                        rg.BaseShaderRegister                = 0;
                        rg.RegisterSpace                     = sd.registerSpace;
                        rg.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
                        ranges.push_back(rg);
                    }
                    if (uavIdx > 0)
                    {
                        D3D12_DESCRIPTOR_RANGE rg            = {};
                        rg.RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
                        rg.NumDescriptors                    = uavIdx;
                        rg.BaseShaderRegister                = 0;
                        rg.RegisterSpace                     = sd.registerSpace;
                        rg.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
                        ranges.push_back(rg);
                    }
                    rangeStore.push_back(std::move(ranges));
                    D3D12_ROOT_PARAMETER p                = {};
                    p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                    p.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(rangeStore.back().size());
                    p.DescriptorTable.pDescriptorRanges   = rangeStore.back().data();
                    p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
                    setInfo.srvTableParam                 = static_cast<int>(params.size());
                    setInfo.srvCount                      = srvIdx + uavIdx; // total heap slots for the table
                    for (size_t bi : srvBindings)
                        l->bindings[bi].rootParam = static_cast<uint32_t>(params.size());
                    for (size_t bi : uavBindings)
                    {
                        l->bindings[bi].rootParam = static_cast<uint32_t>(params.size());
                        l->bindings[bi].heapOffset += srvIdx;
                    }
                    params.push_back(p);
                }
                if (samplerIdx > 0)
                {
                    std::vector<D3D12_DESCRIPTOR_RANGE> ranges(1);
                    ranges[0].RangeType                         = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
                    ranges[0].NumDescriptors                    = samplerIdx;
                    ranges[0].BaseShaderRegister                = 0;
                    ranges[0].RegisterSpace                     = sd.registerSpace;
                    ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
                    rangeStore.push_back(std::move(ranges));
                    D3D12_ROOT_PARAMETER p                = {};
                    p.ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                    p.DescriptorTable.NumDescriptorRanges = 1;
                    p.DescriptorTable.pDescriptorRanges   = rangeStore.back().data();
                    p.ShaderVisibility                    = D3D12_SHADER_VISIBILITY_ALL;
                    setInfo.samplerTableParam             = static_cast<int>(params.size());
                    setInfo.samplerCount                  = samplerIdx;
                    for (size_t bi : samplerBindings)
                        l->bindings[bi].rootParam = static_cast<uint32_t>(params.size());
                    params.push_back(p);
                }
                l->sets.push_back(setInfo);
            }
            // Push constants -> a root 32-bit-constants parameter at the requested register.
            // Slang lowers [[vk::push_constant]] to an HLSL cbuffer at that b# register.
            if (desc->pushConstantNum > 0)
            {
                const VriPushConstantDesc& pc = desc->pushConstants[0];
                D3D12_ROOT_PARAMETER       p  = {};
                p.ParameterType               = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
                p.Constants.ShaderRegister    = pc.baseRegister;
                p.Constants.RegisterSpace     = 0;
                p.Constants.Num32BitValues    = (pc.size + 3u) / 4u;
                p.ShaderVisibility            = D3D12_SHADER_VISIBILITY_ALL;
                l->hasPush                    = true;
                l->pushRootParam              = static_cast<uint32_t>(params.size());
                l->push32Count                = (pc.size + 3u) / 4u;
                params.push_back(p);
            }
            D3D12_ROOT_SIGNATURE_DESC rs = {};
            rs.NumParameters             = static_cast<UINT>(params.size());
            rs.pParameters               = params.empty() ? nullptr : params.data();
            rs.Flags                     = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
            ComPtr<ID3DBlob> blob, err;
            if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)))
            {
                delete l;
                d->ReportError("D3D12SerializeRootSignature failed");
                return VriResult_Failure;
            }
            if (FAILED(d->Device()->CreateRootSignature(
                    0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&l->rootSig))))
            {
                delete l;
                d->ReportError("CreateRootSignature failed");
                return VriResult_Failure;
            }
            *out = ToHandle(l);
            return VriResult_Success;
        }
        void VRI_CALL DestroyPipelineLayout(VriPipelineLayout* layout)
        {
            if (layout)
                delete PL(layout);
        }

        // Mesh-shader pipeline (MS [+ AS] + PS) via the pipeline-state-stream API
        // (ID3D12Device2). No vertex input / input assembly. Returns Success/Failure.
        VriResult CreateMeshPipeline(DeviceD3D12* d, const VriGraphicsPipelineDesc* desc, VriPipeline** out)
        {
            PipelineLayoutD3D12*  layout = PL(desc->pipelineLayout);
            D3D12_SHADER_BYTECODE ms = {}, as = {}, ps = {};
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                const VriShaderDesc&        s  = desc->shaders[i];
                const D3D12_SHADER_BYTECODE bc = {s.bytecode, s.bytecodeSize};
                if (s.stage == VriShaderStage_Mesh)
                    ms = bc;
                else if (s.stage == VriShaderStage_Task)
                    as = bc;
                else if (s.stage == VriShaderStage_Fragment)
                    ps = bc;
            }

            D3D12_RASTERIZER_DESC raster = {};
            raster.FillMode              = D3D12_FILL_MODE_SOLID;
            raster.CullMode              = ToD3DCull(desc->rasterization.cullMode);
            raster.FrontCounterClockwise =
                desc->rasterization.frontFace == VriFrontFace_CounterClockwise ? TRUE : FALSE;
            raster.DepthClipEnable = TRUE;

            D3D12_BLEND_DESC blend               = {};
            blend.RenderTarget[0].SrcBlend       = D3D12_BLEND_ONE;
            blend.RenderTarget[0].DestBlend      = D3D12_BLEND_ZERO;
            blend.RenderTarget[0].BlendOp        = D3D12_BLEND_OP_ADD;
            blend.RenderTarget[0].SrcBlendAlpha  = D3D12_BLEND_ONE;
            blend.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
            blend.RenderTarget[0].BlendOpAlpha   = D3D12_BLEND_OP_ADD;
            const VriColorAttachmentDesc* c0 = desc->outputMerger.colorNum ? &desc->outputMerger.colors[0] : nullptr;
            const VriColorWriteFlags      wm = c0 ? c0->colorWriteMask : VriColorWrite_RGBA;
            blend.RenderTarget[0].RenderTargetWriteMask = static_cast<UINT8>((wm == 0 ? VriColorWrite_RGBA : wm) & 0xF);

            D3D12_DEPTH_STENCIL_DESC ds = {};
            ds.DepthEnable              = desc->depthStencil.depthTest ? TRUE : FALSE;
            ds.DepthWriteMask =
                desc->depthStencil.depthWrite ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
            ds.DepthFunc = ToD3DCompare(desc->depthStencil.depthCompareOp);
            // D3D12 has one read/write mask for both faces (VRI carries per-face); use the front's.
            ds.StencilEnable    = desc->depthStencil.stencilTest ? TRUE : FALSE;
            ds.StencilReadMask  = static_cast<UINT8>(desc->depthStencil.front.compareMask);
            ds.StencilWriteMask = static_cast<UINT8>(desc->depthStencil.front.writeMask);
            ds.FrontFace        = ToD3DStencilFace(desc->depthStencil.front);
            ds.BackFace         = ToD3DStencilFace(desc->depthStencil.back);

            D3D12_RT_FORMAT_ARRAY rtv = {};
            rtv.NumRenderTargets      = desc->outputMerger.colorNum <= 8 ? desc->outputMerger.colorNum : 8;
            for (uint32_t i = 0; i < rtv.NumRenderTargets; ++i)
                rtv.RTFormats[i] = ToDxgiFormat(desc->outputMerger.colors[i].format).format;
            DXGI_SAMPLE_DESC sample = {};
            sample.Count            = desc->multisample.sampleNum ? desc->multisample.sampleNum : 1;

            // Packed stream of subobjects (each aligned to void*); CreatePipelineState
            // reads them by their leading type tag.
            struct alignas(void*) SubRoot
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
                ID3D12RootSignature*                v;
            };
            struct alignas(void*) SubMS
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS;
                D3D12_SHADER_BYTECODE               v;
            };
            struct alignas(void*) SubAS
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS;
                D3D12_SHADER_BYTECODE               v;
            };
            struct alignas(void*) SubPS
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS;
                D3D12_SHADER_BYTECODE               v;
            };
            struct alignas(void*) SubRast
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER;
                D3D12_RASTERIZER_DESC               v;
            };
            struct alignas(void*) SubBlend
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND;
                D3D12_BLEND_DESC                    v;
            };
            struct alignas(void*) SubDS
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL;
                D3D12_DEPTH_STENCIL_DESC            v;
            };
            struct alignas(void*) SubRTV
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS;
                D3D12_RT_FORMAT_ARRAY               v;
            };
            struct alignas(void*) SubDSV
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT;
                DXGI_FORMAT                         v;
            };
            struct alignas(void*) SubSamp
            {
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC;
                DXGI_SAMPLE_DESC                    v;
            };
            struct Stream
            {
                SubRoot  rootSig;
                SubMS    msBc;
                SubAS    asBc;
                SubPS    psBc;
                SubRast  rast;
                SubBlend blend;
                SubDS    ds;
                SubRTV   rtv;
                SubDSV   dsv;
                SubSamp  sample;
            } stream;
            stream.rootSig.v = layout->rootSig.Get();
            stream.msBc.v    = ms;
            stream.asBc.v    = as;
            stream.psBc.v    = ps;
            stream.rast.v    = raster;
            stream.blend.v   = blend;
            stream.ds.v      = ds;
            stream.rtv.v     = rtv;
            stream.dsv.v     = desc->outputMerger.depthStencilFormat != VriFormat_Unknown ?
                                   ToDxgiFormat(desc->outputMerger.depthStencilFormat).format :
                                   DXGI_FORMAT_UNKNOWN;
            stream.sample.v  = sample;

            ComPtr<ID3D12Device2> dev2;
            if (FAILED(d->Device()->QueryInterface(IID_PPV_ARGS(&dev2))))
            {
                d->ReportError("mesh pipeline: ID3D12Device2 unavailable");
                return VriResult_Unsupported;
            }
            D3D12_PIPELINE_STATE_STREAM_DESC sdesc = {sizeof(stream), &stream};

            PipelineD3D12* p  = new PipelineD3D12 {};
            p->device         = d;
            p->rootSig        = layout->rootSig.Get();
            p->isMesh         = true;
            p->stencilEnabled = desc->depthStencil.stencilTest != VRI_FALSE;
            p->stencilRef     = desc->depthStencil.front.reference;
            if (FAILED(dev2->CreatePipelineState(&sdesc, IID_PPV_ARGS(&p->pso))))
            {
                delete p;
                d->ReportError("CreatePipelineState (mesh) failed");
                return VriResult_Failure;
            }
            *out = ToHandle(p);
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateGraphicsPipeline(VriDevice*                     device,
                                                  const VriGraphicsPipelineDesc* desc,
                                                  VriPipeline**                  out)
        {
            DeviceD3D12*         d      = Dev(device);
            PipelineLayoutD3D12* layout = PL(desc->pipelineLayout);

            for (uint32_t i = 0; i < desc->shaderNum; ++i)
                if (desc->shaders[i].stage == VriShaderStage_Mesh || desc->shaders[i].stage == VriShaderStage_Task)
                    return CreateMeshPipeline(d, desc, out);

            D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature                     = layout->rootSig.Get();
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                const VriShaderDesc&        s  = desc->shaders[i];
                const D3D12_SHADER_BYTECODE bc = {s.bytecode, s.bytecodeSize};
                if (s.stage == VriShaderStage_Vertex)
                    pd.VS = bc;
                else if (s.stage == VriShaderStage_Fragment)
                    pd.PS = bc;
                else if (s.stage == VriShaderStage_Geometry)
                    pd.GS = bc;
                else if (s.stage == VriShaderStage_TessControl)
                    pd.HS = bc;
                else if (s.stage == VriShaderStage_TessEval)
                    pd.DS = bc;
            }

            // Input layout (per-attribute). VRI attributes are location-indexed, but D3D
            // matches vertex inputs by semantic name+index. Reflect the VS input signature
            // (register N == VRI attribute/location N) so the layout uses the real
            // semantics (POSITION, COLOR0, ...) Slang emitted - no hardcoded convention.
            struct SemInfo
            {
                std::string name;
                UINT        index = 0;
            };
            std::vector<SemInfo> byRegister; // owns the semantic strings the layout points at
            if (pd.VS.pShaderBytecode && desc->vertexInput.attributeNum)
            {
                ComPtr<ID3D12ShaderReflection> refl;
                if (SUCCEEDED(D3DReflect(pd.VS.pShaderBytecode, pd.VS.BytecodeLength, IID_PPV_ARGS(&refl))))
                {
                    D3D12_SHADER_DESC sd = {};
                    refl->GetDesc(&sd);
                    byRegister.resize(sd.InputParameters);
                    for (UINT i = 0; i < sd.InputParameters; ++i)
                    {
                        D3D12_SIGNATURE_PARAMETER_DESC p = {};
                        refl->GetInputParameterDesc(i, &p);
                        if (p.Register < byRegister.size())
                            byRegister[p.Register] = {p.SemanticName ? p.SemanticName : "TEXCOORD", p.SemanticIndex};
                    }
                }
            }
            std::vector<D3D12_INPUT_ELEMENT_DESC> elems;
            std::vector<PipelineGraphicsVB>       vbStrides;
            for (uint32_t i = 0; i < desc->vertexInput.attributeNum; ++i)
            {
                const VriVertexAttributeDesc& a    = desc->vertexInput.attributes[i];
                uint32_t                      slot = a.streamIndex, stride = 0;
                bool                          perInstance = false;
                if (a.streamIndex < desc->vertexInput.streamNum)
                {
                    slot        = desc->vertexInput.streams[a.streamIndex].bindingSlot;
                    stride      = static_cast<uint32_t>(desc->vertexInput.streams[a.streamIndex].stride);
                    perInstance = desc->vertexInput.streams[a.streamIndex].stepRate == VriVertexStepRate_PerInstance;
                }
                D3D12_INPUT_ELEMENT_DESC e = {};
                if (i < byRegister.size() && !byRegister[i].name.empty())
                {
                    e.SemanticName  = byRegister[i].name.c_str();
                    e.SemanticIndex = byRegister[i].index;
                }
                else
                {
                    e.SemanticName  = "TEXCOORD";
                    e.SemanticIndex = i;
                }
                e.Format               = ToDxgiVertexFormat(a.format);
                e.InputSlot            = slot;
                e.AlignedByteOffset    = a.offset;
                e.InputSlotClass       = perInstance ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA :
                                                       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
                e.InstanceDataStepRate = perInstance ? 1u : 0u;
                elems.push_back(e);
                bool have = false;
                for (auto& vb : vbStrides)
                    if (vb.slot == slot)
                    {
                        have = true;
                        break;
                    }
                if (!have)
                    vbStrides.push_back({stride, slot});
            }
            if (!elems.empty())
            {
                pd.InputLayout.pInputElementDescs = elems.data();
                pd.InputLayout.NumElements        = static_cast<UINT>(elems.size());
            }

            pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            pd.RasterizerState.CullMode = ToD3DCull(desc->rasterization.cullMode);
            pd.RasterizerState.FrontCounterClockwise =
                desc->rasterization.frontFace == VriFrontFace_CounterClockwise ? TRUE : FALSE;
            pd.RasterizerState.DepthClipEnable    = TRUE;
            pd.RasterizerState.ConservativeRaster = desc->rasterization.conservativeRaster ?
                                                        D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON :
                                                        D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

            const VriColorAttachmentDesc* c0 = desc->outputMerger.colorNum ? &desc->outputMerger.colors[0] : nullptr;
            pd.BlendState.RenderTarget[0].SrcBlend       = D3D12_BLEND_ONE;
            pd.BlendState.RenderTarget[0].DestBlend      = D3D12_BLEND_ZERO;
            pd.BlendState.RenderTarget[0].BlendOp        = D3D12_BLEND_OP_ADD;
            pd.BlendState.RenderTarget[0].SrcBlendAlpha  = D3D12_BLEND_ONE;
            pd.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
            pd.BlendState.RenderTarget[0].BlendOpAlpha   = D3D12_BLEND_OP_ADD;
            if (c0 && c0->blend.enable)
            {
                pd.BlendState.RenderTarget[0].BlendEnable    = TRUE;
                pd.BlendState.RenderTarget[0].SrcBlend       = ToD3DBlend(c0->blend.srcColor);
                pd.BlendState.RenderTarget[0].DestBlend      = ToD3DBlend(c0->blend.dstColor);
                pd.BlendState.RenderTarget[0].BlendOp        = ToD3DBlendOp(c0->blend.colorOp);
                pd.BlendState.RenderTarget[0].SrcBlendAlpha  = ToD3DBlend(c0->blend.srcAlpha);
                pd.BlendState.RenderTarget[0].DestBlendAlpha = ToD3DBlend(c0->blend.dstAlpha);
                pd.BlendState.RenderTarget[0].BlendOpAlpha   = ToD3DBlendOp(c0->blend.alphaOp);
            }
            const VriColorWriteFlags wm = c0 ? c0->colorWriteMask : VriColorWrite_RGBA;
            pd.BlendState.RenderTarget[0].RenderTargetWriteMask =
                static_cast<UINT8>((wm == 0 ? VriColorWrite_RGBA : wm) & 0xF);

            pd.DepthStencilState.DepthEnable = desc->depthStencil.depthTest ? TRUE : FALSE;
            pd.DepthStencilState.DepthWriteMask =
                desc->depthStencil.depthWrite ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
            pd.DepthStencilState.DepthFunc = ToD3DCompare(desc->depthStencil.depthCompareOp);
            // D3D12 has one read/write mask for both faces (VRI carries per-face); use the front's.
            // The stencil reference is dynamic state in D3D12 (OMSetStencilRef), set when the pipeline binds.
            pd.DepthStencilState.StencilEnable    = desc->depthStencil.stencilTest ? TRUE : FALSE;
            pd.DepthStencilState.StencilReadMask  = static_cast<UINT8>(desc->depthStencil.front.compareMask);
            pd.DepthStencilState.StencilWriteMask = static_cast<UINT8>(desc->depthStencil.front.writeMask);
            pd.DepthStencilState.FrontFace        = ToD3DStencilFace(desc->depthStencil.front);
            pd.DepthStencilState.BackFace         = ToD3DStencilFace(desc->depthStencil.back);

            pd.SampleMask            = UINT_MAX;
            pd.PrimitiveTopologyType = ToD3DTopologyType(desc->inputAssembly.topology);
            pd.NumRenderTargets      = desc->outputMerger.colorNum <= 8 ? desc->outputMerger.colorNum : 8;
            for (uint32_t i = 0; i < pd.NumRenderTargets; ++i)
                pd.RTVFormats[i] = ToDxgiFormat(desc->outputMerger.colors[i].format).format;
            pd.DSVFormat        = desc->outputMerger.depthStencilFormat != VriFormat_Unknown ?
                                      ToDxgiFormat(desc->outputMerger.depthStencilFormat).format :
                                      DXGI_FORMAT_UNKNOWN;
            pd.SampleDesc.Count = desc->multisample.sampleNum ? desc->multisample.sampleNum : 1;

            PipelineD3D12* p  = new PipelineD3D12 {};
            p->device         = d;
            p->rootSig        = layout->rootSig.Get();
            p->vbStrides      = std::move(vbStrides);
            p->topology       = ToD3DTopology(desc->inputAssembly.topology, desc->tessellation.patchControlPoints);
            p->stencilEnabled = desc->depthStencil.stencilTest != VRI_FALSE;
            p->stencilRef     = desc->depthStencil.front.reference;
            if (FAILED(d->Device()->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&p->pso))))
            {
                delete p;
                d->ReportError("CreateGraphicsPipelineState failed");
                return VriResult_Failure;
            }
            *out = ToHandle(p);
            return VriResult_Success;
        }
        VriResult VRI_CALL CreateComputePipeline(VriDevice*                    device,
                                                 const VriComputePipelineDesc* desc,
                                                 VriPipeline**                 out)
        {
            DeviceD3D12*                      d      = Dev(device);
            PipelineLayoutD3D12*              layout = PL(desc->pipelineLayout);
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd     = {};
            pd.pRootSignature                        = layout->rootSig.Get();
            pd.CS                                    = {desc->shader.bytecode, desc->shader.bytecodeSize};
            PipelineD3D12* p                         = new PipelineD3D12 {};
            p->device                                = d;
            p->rootSig                               = layout->rootSig.Get();
            p->isCompute                             = true;
            if (FAILED(d->Device()->CreateComputePipelineState(&pd, IID_PPV_ARGS(&p->pso))))
            {
                delete p;
                d->ReportError("CreateComputePipelineState failed");
                return VriResult_Failure;
            }
            *out = ToHandle(p);
            return VriResult_Success;
        }
        void VRI_CALL DestroyPipeline(VriPipeline* pipeline)
        {
            if (pipeline)
                delete Pipe(pipeline);
        }
        VriResult VRI_CALL CreateDescriptorPool(VriDevice*                   device,
                                                const VriDescriptorPoolDesc* desc,
                                                VriDescriptorPool**          out)
        {
            DeviceD3D12*         d = Dev(device);
            DescriptorPoolD3D12* p = new DescriptorPoolD3D12 {};
            p->device              = d;
            // Shader-visible heaps for SRV/CBV/UAV + samplers (descriptor tables bind from
            // these). Sized from the pool request with sane minimums; root CBVs use neither.
            const uint32_t srvCap = (desc->textureMaxNum + desc->structuredBufferMaxNum + desc->storageBufferMaxNum +
                                     desc->storageTextureMaxNum + 16);
            const uint32_t samplerCap     = (desc->samplerMaxNum + 16);
            D3D12_DESCRIPTOR_HEAP_DESC hd = {};
            hd.Type                       = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            hd.NumDescriptors             = srvCap;
            hd.Flags                      = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            if (FAILED(d->Device()->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&p->srvHeap))))
            {
                delete p;
                d->ReportError("CreateDescriptorHeap (SRV) failed");
                return VriResult_Failure;
            }
            hd.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
            hd.NumDescriptors = samplerCap;
            if (FAILED(d->Device()->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&p->samplerHeap))))
            {
                delete p;
                d->ReportError("CreateDescriptorHeap (sampler) failed");
                return VriResult_Failure;
            }
            p->srvSize     = d->Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            p->samplerSize = d->Device()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
            *out           = ToHandle(p);
            return VriResult_Success;
        }
        void VRI_CALL ResetDescriptorPool(VriDescriptorPool* pool)
        {
            DescriptorPoolD3D12* p = DPool(pool);
            p->srvNext             = 0;
            p->samplerNext         = 0;
        }
        void VRI_CALL DestroyDescriptorPool(VriDescriptorPool* pool)
        {
            if (pool)
                delete DPool(pool);
        }
        VriResult VRI_CALL AllocateDescriptorSets(VriDescriptorPool*       pool,
                                                  const VriPipelineLayout* layout,
                                                  uint32_t                 setIndex,
                                                  VriDescriptorSet**       outSets,
                                                  uint32_t                 setNum)
        {
            DescriptorPoolD3D12*       p  = DPool(pool);
            const PipelineLayoutD3D12* l  = PL(layout);
            const LayoutSetD3D12*      si = l->FindSet(setIndex);
            for (uint32_t i = 0; i < setNum; ++i)
            {
                DescriptorSetD3D12* s = new DescriptorSetD3D12 {};
                s->device             = p->device;
                s->pool               = p;
                s->layout             = l;
                s->setIndex           = setIndex;
                if (si && si->srvCount > 0) // sub-allocate this set's SRV block
                {
                    s->srvCpu = p->srvHeap->GetCPUDescriptorHandleForHeapStart();
                    s->srvCpu.ptr += static_cast<SIZE_T>(p->srvNext) * p->srvSize;
                    s->srvGpu = p->srvHeap->GetGPUDescriptorHandleForHeapStart();
                    s->srvGpu.ptr += static_cast<UINT64>(p->srvNext) * p->srvSize;
                    p->srvNext += si->srvCount;
                }
                if (si && si->samplerCount > 0)
                {
                    s->samplerCpu = p->samplerHeap->GetCPUDescriptorHandleForHeapStart();
                    s->samplerCpu.ptr += static_cast<SIZE_T>(p->samplerNext) * p->samplerSize;
                    s->samplerGpu = p->samplerHeap->GetGPUDescriptorHandleForHeapStart();
                    s->samplerGpu.ptr += static_cast<UINT64>(p->samplerNext) * p->samplerSize;
                    p->samplerNext += si->samplerCount;
                }
                outSets[i] = ToHandle(s);
            }
            return VriResult_Success;
        }
        void VRI_CALL UpdateDescriptorRanges(VriDescriptorSet*                   set,
                                             uint32_t                            baseRange,
                                             uint32_t                            rangeNum,
                                             const VriDescriptorRangeUpdateDesc* updates)
        {
            DescriptorSetD3D12*                    s = DSet(set);
            DeviceD3D12*                           d = s->device;
            std::vector<const LayoutBindingD3D12*> setBindings; // this set's bindings in declaration order
            for (const LayoutBindingD3D12& b : s->layout->bindings)
                if (b.set == s->setIndex)
                    setBindings.push_back(&b);
            for (uint32_t r = 0; r < rangeNum; ++r)
            {
                const uint32_t idx = baseRange + r;
                if (idx >= setBindings.size() || updates[r].descriptorNum == 0)
                    continue;
                const LayoutBindingD3D12* lb = setBindings[idx];
                s->bound[lb->binding] =
                    reinterpret_cast<const DescriptorD3D12*>(updates[r].descriptors[0]); // root CBVs bind [0] at draw
                if (lb->type == VriDescriptorType_ConstantBuffer)
                    continue;
                // A range may carry an array of descriptors (descriptor indexing / bindless):
                // write each into consecutive heap slots from heapOffset + baseDescriptor.
                for (uint32_t k = 0; k < updates[r].descriptorNum; ++k)
                {
                    const DescriptorD3D12* v = reinterpret_cast<const DescriptorD3D12*>(updates[r].descriptors[k]);
                    if (!v)
                        continue;
                    const SIZE_T slot = static_cast<SIZE_T>(lb->heapOffset) + updates[r].baseDescriptor + k;
                    if (lb->type == VriDescriptorType_Sampler) // realize into the set's sampler heap slot
                    {
                        D3D12_SAMPLER_DESC          sd = ToD3DSampler(v->sampler);
                        D3D12_CPU_DESCRIPTOR_HANDLE h  = s->samplerCpu;
                        h.ptr += slot * s->pool->samplerSize;
                        d->Device()->CreateSampler(&sd, h);
                    }
                    else if (lb->type == VriDescriptorType_AccelerationStructure) // TLAS SRV (no resource, by GPU VA)
                    {
                        D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
                        srv.ViewDimension                   = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
                        srv.Shader4ComponentMapping         = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                        srv.RaytracingAccelerationStructure.Location = v->accelAddress;
                        D3D12_CPU_DESCRIPTOR_HANDLE h                = s->srvCpu;
                        h.ptr += slot * s->pool->srvSize;
                        d->Device()->CreateShaderResourceView(nullptr, &srv, h);
                    }
                    else if (lb->type == VriDescriptorType_StorageTexture && v->texture) // UAV
                    {
                        D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
                        uav.Format                           = v->texture->format;
                        uav.ViewDimension                    = D3D12_UAV_DIMENSION_TEXTURE2D;
                        D3D12_CPU_DESCRIPTOR_HANDLE h        = s->srvCpu;
                        h.ptr += slot * s->pool->srvSize;
                        d->Device()->CreateUnorderedAccessView(v->texture->resource.Get(), nullptr, &uav, h);
                    }
                    else if (v->texture) // SRV (sampled texture) into the set's CBV/SRV/UAV heap slot
                    {
                        D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
                        srv.Format =
                            v->texture->srvFormat; // typed sampleable format (R32_FLOAT for a sampled D32 depth)
                        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                        const bool isCube =
                            v->viewType == VriTextureViewType_Cube || v->viewType == VriTextureViewType_CubeArray;
                        if (isCube) // a TextureCube SRV samples a 6-layer resource by direction
                        {
                            srv.ViewDimension         = D3D12_SRV_DIMENSION_TEXTURECUBE;
                            srv.TextureCube.MipLevels = v->texture->mipNum;
                        }
                        else if (v->texture->layerNum > 1)
                        {
                            srv.ViewDimension            = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                            srv.Texture2DArray.MipLevels = v->texture->mipNum;
                            srv.Texture2DArray.ArraySize = v->texture->layerNum;
                        }
                        else
                        {
                            srv.ViewDimension       = D3D12_SRV_DIMENSION_TEXTURE2D;
                            srv.Texture2D.MipLevels = v->texture->mipNum;
                        }
                        D3D12_CPU_DESCRIPTOR_HANDLE h = s->srvCpu;
                        h.ptr += slot * s->pool->srvSize;
                        d->Device()->CreateShaderResourceView(v->texture->resource.Get(), &srv, h);
                    }
                }
            }
        }
        void VRI_CALL CmdSetPipelineLayout(VriCommandBuffer* cmd, VriPipelineLayout* layout)
        {
            CommandBufferD3D12* c   = CB(cmd);
            c->boundLayout          = PL(layout);
            ID3D12RootSignature* rs = PL(layout)->rootSig.Get();
            if (c->boundPipeline && c->boundPipeline->isCompute)
                c->list->SetComputeRootSignature(rs);
            else
                c->list->SetGraphicsRootSignature(rs);
        }
        void VRI_CALL CmdSetPipeline(VriCommandBuffer* cmd, VriPipeline* pipeline)
        {
            CommandBufferD3D12* c = CB(cmd);
            PipelineD3D12*      p = Pipe(pipeline);
            if (c->boundPipeline != p)
                c->vbDirty = true; // re-bind vertex buffers: the new pipeline's stride may differ
            c->boundPipeline = p;
            if (p->isRt) // DXR state object: bound via SetPipelineState1, compute root binding
            {
                Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> list4;
                if (SUCCEEDED(c->list.As(&list4)))
                    list4->SetPipelineState1(p->stateObject.Get());
                c->list->SetComputeRootSignature(p->rootSig);
                return;
            }
            c->list->SetPipelineState(p->pso.Get());
            // Set the root signature here too so a no-descriptor pipeline works without CmdSetPipelineLayout.
            if (p->isCompute)
            {
                c->list->SetComputeRootSignature(p->rootSig);
            }
            else
            {
                c->list->SetGraphicsRootSignature(p->rootSig);
                if (!p->isMesh)
                    c->list->IASetPrimitiveTopology(p->topology);
            }
            // The stencil reference is dynamic state in D3D12 (not in the PSO); apply the pipeline's.
            if (p->stencilEnabled)
                c->list->OMSetStencilRef(p->stencilRef);
        }
        void VRI_CALL CmdSetDescriptorSet(VriCommandBuffer* cmd, uint32_t setIndex, const VriDescriptorSet* set)
        {
            CommandBufferD3D12*       c = CB(cmd);
            const DescriptorSetD3D12* s = reinterpret_cast<const DescriptorSetD3D12*>(set);
            if (!s || !s->layout)
                return;
            const LayoutSetD3D12* si      = s->layout->FindSet(setIndex);
            const bool            compute = c->boundPipeline && c->boundPipeline->isCompute;
            // Bind the pool's shader-visible heaps (needed for the SRV/sampler tables).
            if (s->pool && (si && (si->srvCount || si->samplerCount)))
            {
                ID3D12DescriptorHeap* heaps[2] = {s->pool->srvHeap.Get(), s->pool->samplerHeap.Get()};
                c->list->SetDescriptorHeaps(2, heaps);
            }
            // Root CBV / UAV (by GPU address); graphics vs compute root binding point.
            for (const LayoutBindingD3D12& b : s->layout->bindings)
            {
                if (b.set != setIndex)
                    continue;
                auto it = s->bound.find(b.binding);
                if (it == s->bound.end() || !it->second || !it->second->buffer)
                    continue;
                const D3D12_GPU_VIRTUAL_ADDRESS va =
                    it->second->buffer->resource->GetGPUVirtualAddress() + it->second->bufferOffset;
                if (b.type == VriDescriptorType_ConstantBuffer)
                {
                    if (compute)
                        c->list->SetComputeRootConstantBufferView(b.rootParam, va);
                    else
                        c->list->SetGraphicsRootConstantBufferView(b.rootParam, va);
                }
                else if (b.type == VriDescriptorType_StorageBuffer)
                {
                    if (compute)
                        c->list->SetComputeRootUnorderedAccessView(b.rootParam, va);
                    else
                        c->list->SetGraphicsRootUnorderedAccessView(b.rootParam, va);
                }
            }
            // SRV / sampler descriptor tables (one each per set).
            if (si && si->srvTableParam >= 0)
            {
                if (compute)
                    c->list->SetComputeRootDescriptorTable(static_cast<UINT>(si->srvTableParam), s->srvGpu);
                else
                    c->list->SetGraphicsRootDescriptorTable(static_cast<UINT>(si->srvTableParam), s->srvGpu);
            }
            if (si && si->samplerTableParam >= 0)
            {
                if (compute)
                    c->list->SetComputeRootDescriptorTable(static_cast<UINT>(si->samplerTableParam), s->samplerGpu);
                else
                    c->list->SetGraphicsRootDescriptorTable(static_cast<UINT>(si->samplerTableParam), s->samplerGpu);
            }
        }
        void VRI_CALL CmdSetConstants(VriCommandBuffer* cmd, uint32_t, const void* data, uint32_t size)
        {
            CommandBufferD3D12* c = CB(cmd);
            if (!c->boundLayout || !c->boundLayout->hasPush || !data || !size)
                return;
            c->list->SetGraphicsRoot32BitConstants(c->boundLayout->pushRootParam, (size + 3u) / 4u, data, 0);
        }
        // Record the bindings only; the actual IASetVertexBuffers happens at draw time (FlushVertexBuffers),
        // because the per-stream stride comes from the pipeline and the app may bind buffers before it.
        void VRI_CALL CmdSetVertexBuffers(VriCommandBuffer*             cmd,
                                          uint32_t                      baseSlot,
                                          const VriVertexBufferBinding* bindings,
                                          uint32_t                      num)
        {
            CommandBufferD3D12* c = CB(cmd);
            if (!num)
                return;
            for (uint32_t i = 0; i < num; ++i)
            {
                const uint32_t slot = baseSlot + i;
                if (slot >= 8)
                    continue;
                BufferD3D12* b              = Buf(bindings[i].buffer);
                c->pendingVBs[slot].address = b->resource->GetGPUVirtualAddress() + bindings[i].offset;
                c->pendingVBs[slot].size    = static_cast<UINT>(b->size - bindings[i].offset);
                c->pendingVBs[slot].set     = true;
            }
            c->vbDirty = true;
        }
        static void FlushVertexBuffers(CommandBufferD3D12* c)
        {
            if (!c->vbDirty)
                return;
            for (uint32_t slot = 0; slot < 8; ++slot)
            {
                if (!c->pendingVBs[slot].set)
                    continue;
                uint32_t stride = 0;
                if (c->boundPipeline)
                    for (const PipelineGraphicsVB& vb : c->boundPipeline->vbStrides)
                        if (vb.slot == slot)
                        {
                            stride = vb.stride;
                            break;
                        }
                D3D12_VERTEX_BUFFER_VIEW v = {};
                v.BufferLocation           = c->pendingVBs[slot].address;
                v.SizeInBytes              = c->pendingVBs[slot].size;
                v.StrideInBytes            = stride;
                c->list->IASetVertexBuffers(slot, 1, &v);
            }
            c->vbDirty = false;
        }
        void VRI_CALL CmdSetIndexBuffer(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, VriIndexType type)
        {
            BufferD3D12*            b = Buf(buffer);
            D3D12_INDEX_BUFFER_VIEW v = {};
            v.BufferLocation          = b->resource->GetGPUVirtualAddress() + offset;
            v.SizeInBytes             = static_cast<UINT>(b->size - offset);
            v.Format                  = type == VriIndexType_UInt16 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;
            CB(cmd)->list->IASetIndexBuffer(&v);
        }
        void VRI_CALL CmdDraw(VriCommandBuffer* cmd, const VriDrawDesc* d)
        {
            FlushVertexBuffers(CB(cmd));
            CB(cmd)->list->DrawInstanced(
                d->vertexNum, d->instanceNum ? d->instanceNum : 1, d->baseVertex, d->baseInstance);
        }
        void VRI_CALL CmdDrawIndexed(VriCommandBuffer* cmd, const VriDrawIndexedDesc* d)
        {
            FlushVertexBuffers(CB(cmd));
            CB(cmd)->list->DrawIndexedInstanced(
                d->indexNum, d->instanceNum ? d->instanceNum : 1, d->baseIndex, d->vertexOffset, d->baseInstance);
        }
        void VRI_CALL CmdDrawIndirect(VriCommandBuffer*, VriBuffer*, uint64_t, uint32_t, uint32_t) {}
        void VRI_CALL CmdDispatch(VriCommandBuffer* cmd, const VriDispatchDesc* d)
        {
            CB(cmd)->list->Dispatch(d->x, d->y, d->z);
        }
        void VRI_CALL CmdDispatchIndirect(VriCommandBuffer*, VriBuffer*, uint64_t) {}
        void VRI_CALL CmdCopyBuffer(VriCommandBuffer* cmd, VriBuffer* dst, VriBuffer* src, const VriBufferCopyDesc* r)
        {
            // Buffers implicitly promote from COMMON to COPY_DEST on a direct queue. Record that
            // so a following read-state barrier (CmdBarrier -> TransitionBuffer) emits the correct
            // StateBefore (COPY_DEST), instead of a stale COMMON that the debug layer rejects.
            BufferD3D12* s = Buf(src);
            BufferD3D12* d = Buf(dst);
            CB(cmd)->list->CopyBufferRegion(d->resource.Get(), r->dstOffset, s->resource.Get(), r->srcOffset, r->size);
            if (d->heapType == D3D12_HEAP_TYPE_DEFAULT)
                d->state = D3D12_RESOURCE_STATE_COPY_DEST;
        }
        void VRI_CALL CmdCopyTexture(VriCommandBuffer*, VriTexture*, VriTexture*, const VriTextureCopyDesc*) {}
        void VRI_CALL CmdUploadBufferToTexture(VriCommandBuffer*               cmd,
                                               VriTexture*                     dst,
                                               VriBuffer*                      src,
                                               const VriBufferTextureCopyDesc* region)
        {
            CommandBufferD3D12* c = CB(cmd);
            TextureD3D12*       t = Tex(dst);
            BufferD3D12*        b = Buf(src);
            // Subresource = mip + arrayLayer * mipNum (baseLayer selects the array slice).
            const UINT sub = (region ? region->texture.mip : 0) + (region ? region->texture.baseLayer : 0) * t->mipNum;
            const UINT64                       srcOffset = region ? region->bufferOffset : 0;
            D3D12_RESOURCE_DESC                rd        = t->resource->GetDesc();
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp        = {};
            UINT                               rows      = 0;
            UINT64                             rowBytes = 0, total = 0;
            c->device->Device()->GetCopyableFootprints(
                &rd, sub, 1, 0, &fp, &rows, &rowBytes, &total); // footprint at offset 0
            Transition(c, t, D3D12_RESOURCE_STATE_COPY_DEST);

            D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
            dstLoc.pResource                   = t->resource.Get();
            dstLoc.Type                        = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dstLoc.SubresourceIndex            = sub;
            D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
            srcLoc.Type                        = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;

            // D3D12 buffer->texture copies need a 256-aligned row pitch and a 512-aligned source
            // offset. If the app's tightly-packed staging already meets that (typical wide
            // textures, offset 0) copy it directly; otherwise pack the rows into an aligned temp
            // upload buffer (CPU memcpy from the mapped staging) so any texture size/offset works.
            const bool aligned =
                rowBytes == fp.Footprint.RowPitch && (srcOffset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT) == 0;
            if (aligned || !b->mapped)
            {
                srcLoc.pResource              = b->resource.Get();
                srcLoc.PlacedFootprint        = fp;
                srcLoc.PlacedFootprint.Offset = srcOffset;
                c->list->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
            }
            else
            {
                ComPtr<ID3D12Resource> temp;
                D3D12_HEAP_PROPERTIES  hp = {};
                hp.Type                   = D3D12_HEAP_TYPE_UPLOAD;
                D3D12_RESOURCE_DESC bd    = {};
                bd.Dimension              = D3D12_RESOURCE_DIMENSION_BUFFER;
                bd.Width                  = total;
                bd.Height                 = 1;
                bd.DepthOrArraySize       = 1;
                bd.MipLevels              = 1;
                bd.Format                 = DXGI_FORMAT_UNKNOWN;
                bd.SampleDesc.Count       = 1;
                bd.Layout                 = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                if (FAILED(c->device->Device()->CreateCommittedResource(&hp,
                                                                        D3D12_HEAP_FLAG_NONE,
                                                                        &bd,
                                                                        D3D12_RESOURCE_STATE_GENERIC_READ,
                                                                        nullptr,
                                                                        IID_PPV_ARGS(&temp))))
                {
                    c->device->ReportError("texture-upload bounce buffer alloc failed");
                    return;
                }
                void*       tmp  = nullptr;
                D3D12_RANGE none = {0, 0};
                temp->Map(0, &none, &tmp);
                const uint8_t* srcBase = static_cast<const uint8_t*>(b->mapped) + srcOffset;
                for (UINT r = 0; r < rows; ++r)
                    std::memcpy(static_cast<uint8_t*>(tmp) + r * fp.Footprint.RowPitch,
                                srcBase + r * rowBytes,
                                static_cast<size_t>(rowBytes));
                temp->Unmap(0, nullptr);
                srcLoc.pResource       = temp.Get();
                srcLoc.PlacedFootprint = fp; // offset 0
                c->list->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
                c->tempUploads.push_back(std::move(temp));
            }
        }
        void VRI_CALL CmdBeginDebugGroup(VriCommandBuffer*, const char*) {}
        void VRI_CALL CmdEndDebugGroup(VriCommandBuffer*) {}
        void VRI_CALL SetDebugName(void*, const char*) {}
    } // namespace

    const VriCoreInterface* GetCoreInterfaceD3D12()
    {
        static const VriCoreInterface t = {
            GetDeviceDesc,
            GetFormatSupport,
            GetQueue,
            CreateCommandAllocator,
            ResetCommandAllocator,
            DestroyCommandAllocator,
            CreateCommandBuffer,
            BeginCommandBuffer,
            EndCommandBuffer,
            CreateBuffer,
            DestroyBuffer,
            MapBuffer,
            UnmapBuffer,
            GetBufferDeviceAddress,
            CreateTexture,
            DestroyTexture,
            GetBufferMemoryDesc,
            GetTextureMemoryDesc,
            AllocateMemory,
            FreeMemory,
            BindBufferMemory,
            BindTextureMemory,
            CreateBufferView,
            CreateTextureView,
            CreateSampler,
            DestroyDescriptor,
            CreatePipelineLayout,
            DestroyPipelineLayout,
            CreateGraphicsPipeline,
            CreateComputePipeline,
            DestroyPipeline,
            CreateDescriptorPool,
            ResetDescriptorPool,
            DestroyDescriptorPool,
            AllocateDescriptorSets,
            UpdateDescriptorRanges,
            CreateFence,
            DestroyFence,
            GetFenceValue,
            Wait,
            CmdBeginRendering,
            CmdEndRendering,
            CmdSetViewports,
            CmdSetScissors,
            CmdSetPipelineLayout,
            CmdSetPipeline,
            CmdSetDescriptorSet,
            CmdSetConstants,
            CmdSetVertexBuffers,
            CmdSetIndexBuffer,
            CmdDraw,
            CmdDrawIndexed,
            CmdDrawIndirect,
            CmdDispatch,
            CmdDispatchIndirect,
            CmdBarrier,
            CmdCopyBuffer,
            CmdCopyTexture,
            CmdUploadBufferToTexture,
            CmdReadbackTextureToBuffer,
            CmdBeginDebugGroup,
            CmdEndDebugGroup,
            QueueSubmit,
            QueueWaitIdle,
            DeviceWaitIdle,
            SetDebugName,
        };
        return &t;
    }
} // namespace vri::d3d12
