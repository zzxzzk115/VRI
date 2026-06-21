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
        DeviceD3D12*         Dev(VriDevice* d) { return reinterpret_cast<DeviceD3D12*>(d); }
        const DeviceD3D12*   Dev(const VriDevice* d) { return reinterpret_cast<const DeviceD3D12*>(d); }
        BufferD3D12*         Buf(VriBuffer* b) { return reinterpret_cast<BufferD3D12*>(b); }
        TextureD3D12*        Tex(VriTexture* t) { return reinterpret_cast<TextureD3D12*>(t); }
        DescriptorD3D12*     Desc(VriDescriptor* d) { return reinterpret_cast<DescriptorD3D12*>(d); }
        const DescriptorD3D12* Desc(const VriDescriptor* d) { return reinterpret_cast<const DescriptorD3D12*>(d); }
        QueueD3D12*          Q(VriQueue* q) { return reinterpret_cast<QueueD3D12*>(q); }
        CommandAllocatorD3D12* Alloc(VriCommandAllocator* a) { return reinterpret_cast<CommandAllocatorD3D12*>(a); }
        CommandBufferD3D12* CB(VriCommandBuffer* c) { return reinterpret_cast<CommandBufferD3D12*>(c); }
        FenceD3D12*         Fen(VriFence* f) { return reinterpret_cast<FenceD3D12*>(f); }
        PipelineLayoutD3D12* PL(VriPipelineLayout* p) { return reinterpret_cast<PipelineLayoutD3D12*>(p); }
        const PipelineLayoutD3D12* PL(const VriPipelineLayout* p) { return reinterpret_cast<const PipelineLayoutD3D12*>(p); }
        PipelineD3D12*      Pipe(VriPipeline* p) { return reinterpret_cast<PipelineD3D12*>(p); }
        DescriptorPoolD3D12* DPool(VriDescriptorPool* p) { return reinterpret_cast<DescriptorPoolD3D12*>(p); }
        DescriptorSetD3D12* DSet(VriDescriptorSet* s) { return reinterpret_cast<DescriptorSetD3D12*>(s); }

        D3D12_RESOURCE_STATES ToState(VriLayout layout)
        {
            switch (layout)
            {
                case VriLayout_ColorAttachment:    return D3D12_RESOURCE_STATE_RENDER_TARGET;
                case VriLayout_DepthStencilAttachment: return D3D12_RESOURCE_STATE_DEPTH_WRITE;
                case VriLayout_DepthStencilReadOnly: return D3D12_RESOURCE_STATE_DEPTH_READ;
                case VriLayout_ShaderResource:     return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                case VriLayout_CopySource:         return D3D12_RESOURCE_STATE_COPY_SOURCE;
                case VriLayout_CopyDestination:    return D3D12_RESOURCE_STATE_COPY_DEST;
                case VriLayout_Present:            return D3D12_RESOURCE_STATE_PRESENT;
                case VriLayout_General:            return D3D12_RESOURCE_STATE_COMMON;
                default:                           return D3D12_RESOURCE_STATE_COMMON;
            }
        }

        void Transition(CommandBufferD3D12* c, TextureD3D12* t, D3D12_RESOURCE_STATES after)
        {
            if (t->state == after) return;
            D3D12_RESOURCE_BARRIER b = {};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = t->resource.Get();
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            b.Transition.StateBefore = t->state;
            b.Transition.StateAfter = after;
            c->list->ResourceBarrier(1, &b);
            t->state = after;
        }

        // ---- queries ---------------------------------------------------
        const VriDeviceDesc* VRI_CALL GetDeviceDesc(const VriDevice* device) { return &Dev(device)->Desc(); }
        VriFormatSupportFlags VRI_CALL GetFormatSupport(const VriDevice*, VriFormat)
        {
            return VriFormatSupport_Texture | VriFormatSupport_ColorAttachment | VriFormatSupport_Blend | VriFormatSupport_VertexBuffer;
        }
        VriResult VRI_CALL GetQueue(VriDevice* device, VriQueueType type, uint32_t, VriQueue** outQueue)
        {
            if (type >= VriQueueType_Count) return VriResult_InvalidArgument;
            *outQueue = ToHandle(Dev(device)->GetQueue(type));
            return VriResult_Success;
        }

        // ---- resources -------------------------------------------------
        VriResult VRI_CALL CreateBuffer(VriDevice* device, const VriBufferDesc* desc, VriBuffer** out)
        {
            DeviceD3D12* d = Dev(device);
            D3D12_HEAP_TYPE heap = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
            if (desc->memoryLocation == VriMemoryLocation_HostUpload) { heap = D3D12_HEAP_TYPE_UPLOAD; state = D3D12_RESOURCE_STATE_GENERIC_READ; }
            else if (desc->memoryLocation == VriMemoryLocation_HostReadback) { heap = D3D12_HEAP_TYPE_READBACK; state = D3D12_RESOURCE_STATE_COPY_DEST; }

            D3D12_HEAP_PROPERTIES hp = {}; hp.Type = heap;
            D3D12_RESOURCE_DESC rd = {};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width = desc->size ? desc->size : 1; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
            rd.Format = DXGI_FORMAT_UNKNOWN; rd.SampleDesc.Count = 1;
            rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

            BufferD3D12* b = new BufferD3D12{};
            if (FAILED(d->Device()->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&b->resource))))
            { delete b; d->ReportError("CreateCommittedResource (buffer) failed"); return VriResult_Failure; }
            b->device = d; b->size = desc->size; b->heapType = heap; b->state = state;
            if (heap != D3D12_HEAP_TYPE_DEFAULT) // persistently map upload/readback heaps
            { D3D12_RANGE none = {0, 0}; b->resource->Map(0, heap == D3D12_HEAP_TYPE_READBACK ? nullptr : &none, &b->mapped); }
            *out = ToHandle(b);
            return VriResult_Success;
        }
        void VRI_CALL DestroyBuffer(VriBuffer* buffer) { if (buffer) delete Buf(buffer); }
        void* VRI_CALL MapBuffer(VriBuffer* buffer, uint64_t offset, uint64_t) { BufferD3D12* b = Buf(buffer); return b->mapped ? static_cast<char*>(b->mapped) + offset : nullptr; }
        void VRI_CALL UnmapBuffer(VriBuffer*) {} // persistent mapping; nothing to do
        uint64_t VRI_CALL GetBufferDeviceAddress(const VriBuffer*) { return 0; }

        VriResult VRI_CALL CreateTexture(VriDevice* device, const VriTextureDesc* desc, VriTexture** out)
        {
            DeviceD3D12* d = Dev(device);
            const DxgiFormatInfo fi = ToDxgiFormat(desc->format);
            D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd = {};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rd.Width = desc->width ? desc->width : 1; rd.Height = desc->height ? desc->height : 1;
            rd.DepthOrArraySize = static_cast<UINT16>(desc->layerNum ? desc->layerNum : 1);
            rd.MipLevels = static_cast<UINT16>(desc->mipNum ? desc->mipNum : 1);
            rd.Format = fi.format; rd.SampleDesc.Count = desc->sampleNum ? desc->sampleNum : 1;
            rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            if (desc->usage & VriTextureUsage_ColorAttachment) rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            if (desc->usage & VriTextureUsage_DepthStencilAttachment) rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            if (desc->usage & VriTextureUsage_ShaderResourceStorage) rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

            TextureD3D12* t = new TextureD3D12{};
            if (FAILED(d->Device()->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&t->resource))))
            { delete t; d->ReportError("CreateCommittedResource (texture) failed"); return VriResult_Failure; }
            t->device = d; t->format = fi.format; t->texelSize = fi.texelSize;
            t->width = desc->width; t->height = desc->height ? desc->height : 1; t->depth = 1;
            t->mipNum = rd.MipLevels; t->layerNum = rd.DepthOrArraySize; t->state = D3D12_RESOURCE_STATE_COMMON;
            *out = ToHandle(t);
            return VriResult_Success;
        }
        void VRI_CALL DestroyTexture(VriTexture* texture) { if (texture) delete Tex(texture); }

        // ---- views -----------------------------------------------------
        VriResult VRI_CALL CreateTextureView(VriDevice* device, const VriTextureViewDesc* desc, VriDescriptor** out)
        {
            DeviceD3D12* d = Dev(device);
            DescriptorD3D12* v = new DescriptorD3D12{};
            v->device = d; v->texture = reinterpret_cast<const TextureD3D12*>(desc->texture); v->mip = desc->baseMip;
            // Phase 1 needs the render-target view; sampling (SRV) lands with descriptor sets.
            v->kind = DescriptorD3D12::Kind::TextureRtv;
            v->cpu = d->AllocRtv();
            D3D12_RENDER_TARGET_VIEW_DESC rtv = {};
            rtv.Format = v->texture->format;
            rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            rtv.Texture2D.MipSlice = desc->baseMip;
            d->Device()->CreateRenderTargetView(v->texture->resource.Get(), &rtv, v->cpu);
            *out = ToHandle(v);
            return VriResult_Success;
        }
        VriResult VRI_CALL CreateBufferView(VriDevice* device, const VriBufferViewDesc* desc, VriDescriptor** out)
        {
            // Records the buffer range; bound as a root CBV (by GPU address) at draw time.
            DescriptorD3D12* v = new DescriptorD3D12{};
            v->kind = DescriptorD3D12::Kind::BufferView;
            v->device = Dev(device);
            v->buffer = reinterpret_cast<const BufferD3D12*>(desc->buffer);
            v->bufferOffset = desc->offset;
            v->bufferSize = desc->size;
            *out = ToHandle(v);
            return VriResult_Success;
        }
        VriResult VRI_CALL CreateSampler(VriDevice*, const VriSamplerDesc*, VriDescriptor**) { return VriResult_Unsupported; }
        void VRI_CALL DestroyDescriptor(VriDescriptor* descriptor) { if (descriptor) delete Desc(descriptor); }

        // ---- command allocation / recording ----------------------------
        VriResult VRI_CALL CreateCommandAllocator(VriDevice* device, VriQueueType, VriCommandAllocator** out)
        {
            DeviceD3D12* d = Dev(device);
            CommandAllocatorD3D12* a = new CommandAllocatorD3D12{};
            a->device = d;
            if (FAILED(d->Device()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a->allocator))))
            { delete a; d->ReportError("CreateCommandAllocator failed"); return VriResult_Failure; }
            *out = ToHandle(a);
            return VriResult_Success;
        }
        void VRI_CALL ResetCommandAllocator(VriCommandAllocator* a) { Alloc(a)->allocator->Reset(); }
        void VRI_CALL DestroyCommandAllocator(VriCommandAllocator* a) { if (a) delete Alloc(a); }
        VriResult VRI_CALL CreateCommandBuffer(VriCommandAllocator* allocator, VriCommandBuffer** out)
        {
            CommandAllocatorD3D12* a = Alloc(allocator);
            CommandBufferD3D12* c = new CommandBufferD3D12{};
            c->device = a->device; c->allocator = a;
            if (FAILED(a->device->Device()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, a->allocator.Get(), nullptr, IID_PPV_ARGS(&c->list))))
            { delete c; a->device->ReportError("CreateCommandList failed"); return VriResult_Failure; }
            c->list->Close(); // created open; BeginCommandBuffer resets it
            *out = ToHandle(c);
            return VriResult_Success;
        }
        VriResult VRI_CALL BeginCommandBuffer(VriCommandBuffer* cmd)
        {
            CommandBufferD3D12* c = CB(cmd);
            if (FAILED(c->list->Reset(c->allocator->allocator.Get(), nullptr))) return VriResult_Failure;
            c->rtvCount = 0;
            return VriResult_Success;
        }
        VriResult VRI_CALL EndCommandBuffer(VriCommandBuffer* cmd) { return SUCCEEDED(CB(cmd)->list->Close()) ? VriResult_Success : VriResult_Failure; }

        // ---- barriers --------------------------------------------------
        void VRI_CALL CmdBarrier(VriCommandBuffer* cmd, const VriBarrierGroupDesc* g)
        {
            CommandBufferD3D12* c = CB(cmd);
            if (!g) return;
            for (uint32_t i = 0; i < g->textureNum; ++i)
            {
                const VriTextureBarrierDesc& tb = g->textures[i];
                if (tb.texture) Transition(c, reinterpret_cast<TextureD3D12*>(tb.texture), ToState(tb.after.layout));
            }
            // Buffer barriers: D3D12 upload/readback heaps stay in a fixed state; default-heap
            // buffer transitions land with the copy/vertex paths in Phase 2.
        }

        // ---- render pass ----------------------------------------------
        void VRI_CALL CmdBeginRendering(VriCommandBuffer* cmd, const VriAttachmentsDesc* a)
        {
            CommandBufferD3D12* c = CB(cmd);
            c->rtvCount = a->colorNum <= 8 ? a->colorNum : 8;
            for (uint32_t i = 0; i < c->rtvCount; ++i)
            {
                const DescriptorD3D12* v = Desc(a->colors[i].view);
                TextureD3D12* t = const_cast<TextureD3D12*>(v->texture);
                Transition(c, t, D3D12_RESOURCE_STATE_RENDER_TARGET); // defensive (usually already RT)
                c->rtvs[i] = v->cpu;
            }
            c->list->OMSetRenderTargets(static_cast<UINT>(c->rtvCount), c->rtvCount ? c->rtvs : nullptr, FALSE, nullptr);
            for (uint32_t i = 0; i < c->rtvCount; ++i)
                if (a->colors[i].loadOp == VriAttachmentLoadOp_Clear)
                    c->list->ClearRenderTargetView(c->rtvs[i], a->colors[i].clearValue.color.f32, 0, nullptr);
        }
        void VRI_CALL CmdEndRendering(VriCommandBuffer*) {}
        void VRI_CALL CmdSetViewports(VriCommandBuffer* cmd, const VriViewport* vps, uint32_t num)
        {
            if (!num) return;
            std::vector<D3D12_VIEWPORT> v(num);
            for (uint32_t i = 0; i < num; ++i) v[i] = {vps[i].x, vps[i].y, vps[i].width, vps[i].height, vps[i].minDepth, vps[i].maxDepth};
            CB(cmd)->list->RSSetViewports(num, v.data());
        }
        void VRI_CALL CmdSetScissors(VriCommandBuffer* cmd, const VriRect* r, uint32_t num)
        {
            if (!num) return;
            std::vector<D3D12_RECT> rects(num);
            for (uint32_t i = 0; i < num; ++i) rects[i] = {r[i].x, r[i].y, r[i].x + static_cast<LONG>(r[i].width), r[i].y + static_cast<LONG>(r[i].height)};
            CB(cmd)->list->RSSetScissorRects(num, rects.data());
        }

        // ---- readback (texture -> buffer) ------------------------------
        void VRI_CALL CmdReadbackTextureToBuffer(VriCommandBuffer* cmd, VriBuffer* dst, VriTexture* src, const VriBufferTextureCopyDesc* region)
        {
            CommandBufferD3D12* c = CB(cmd);
            TextureD3D12* t = Tex(src);
            BufferD3D12* b = Buf(dst);
            const UINT sub = region ? region->texture.mip : 0;
            D3D12_RESOURCE_DESC rd = t->resource->GetDesc();
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
            UINT rows = 0; UINT64 rowBytes = 0, total = 0;
            c->device->Device()->GetCopyableFootprints(&rd, sub, 1, region ? region->bufferOffset : 0, &fp, &rows, &rowBytes, &total);
            Transition(c, t, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION dstLoc = {}; dstLoc.pResource = b->resource.Get(); dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dstLoc.PlacedFootprint = fp;
            D3D12_TEXTURE_COPY_LOCATION srcLoc = {}; srcLoc.pResource = t->resource.Get(); srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; srcLoc.SubresourceIndex = sub;
            c->list->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        }

        // ---- submission / fences --------------------------------------
        void VRI_CALL QueueSubmit(VriQueue* queue, const VriQueueSubmitDesc* submit)
        {
            QueueD3D12* q = Q(queue);
            for (uint32_t i = 0; i < submit->waitFenceNum; ++i)
                q->queue->Wait(Fen(submit->waitFences[i].fence)->fence.Get(), submit->waitFences[i].value);
            std::vector<ID3D12CommandList*> lists(submit->commandBufferNum);
            for (uint32_t i = 0; i < submit->commandBufferNum; ++i) lists[i] = CB(submit->commandBuffers[i])->list.Get();
            if (!lists.empty()) q->queue->ExecuteCommandLists(static_cast<UINT>(lists.size()), lists.data());
            for (uint32_t i = 0; i < submit->signalFenceNum; ++i)
                q->queue->Signal(Fen(submit->signalFences[i].fence)->fence.Get(), submit->signalFences[i].value);
        }
        void VRI_CALL QueueWaitIdle(VriQueue*) {}
        void VRI_CALL DeviceWaitIdle(VriDevice*) {}

        VriResult VRI_CALL CreateFence(VriDevice* device, uint64_t initialValue, VriFence** out)
        {
            DeviceD3D12* d = Dev(device);
            FenceD3D12* f = new FenceD3D12{};
            f->device = d;
            if (FAILED(d->Device()->CreateFence(initialValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f->fence))))
            { delete f; d->ReportError("CreateFence failed"); return VriResult_Failure; }
            f->event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
            *out = ToHandle(f);
            return VriResult_Success;
        }
        void VRI_CALL DestroyFence(VriFence* fence) { if (!fence) return; FenceD3D12* f = Fen(fence); if (f->event) CloseHandle(f->event); delete f; }
        uint64_t VRI_CALL GetFenceValue(VriFence* fence) { return Fen(fence)->fence->GetCompletedValue(); }
        void VRI_CALL Wait(VriFence* fence, uint64_t value)
        {
            FenceD3D12* f = Fen(fence);
            if (f->fence->GetCompletedValue() >= value) return;
            f->fence->SetEventOnCompletion(value, f->event);
            WaitForSingleObject(f->event, INFINITE);
        }

        // ---- not yet implemented (Phase 2) -----------------------------
        void      VRI_CALL GetBufferMemoryDesc(const VriDevice*, const VriBufferDesc*, VriMemoryLocation, VriMemoryDesc* o) { if (o) *o = VriMemoryDesc{}; }
        void      VRI_CALL GetTextureMemoryDesc(const VriDevice*, const VriTextureDesc*, VriMemoryLocation, VriMemoryDesc* o) { if (o) *o = VriMemoryDesc{}; }
        VriResult VRI_CALL AllocateMemory(VriDevice*, const VriMemoryDesc*, VriMemory**) { return VriResult_Unsupported; }
        void      VRI_CALL FreeMemory(VriMemory*) {}
        VriResult VRI_CALL BindBufferMemory(VriDevice*, VriBuffer*, VriMemory*, uint64_t) { return VriResult_Unsupported; }
        VriResult VRI_CALL BindTextureMemory(VriDevice*, VriTexture*, VriMemory*, uint64_t) { return VriResult_Unsupported; }
        VriResult VRI_CALL CreatePipelineLayout(VriDevice* device, const VriPipelineLayoutDesc* desc, VriPipelineLayout** out)
        {
            DeviceD3D12* d = Dev(device);
            PipelineLayoutD3D12* l = new PipelineLayoutD3D12{};
            l->device = d;
            // Phase 3a: each constant buffer becomes a ROOT CBV (bound by GPU address - no
            // descriptor heap). register = VRI binding, space = VRI set. Textures/samplers/
            // storage need descriptor tables (Phase 3b); ignored here so the IA flag + any
            // CBVs still produce a valid root signature.
            std::vector<D3D12_ROOT_PARAMETER> params;
            for (uint32_t s = 0; s < desc->descriptorSetNum; ++s)
            {
                const VriDescriptorSetDesc& sd = desc->descriptorSets[s];
                for (uint32_t r = 0; r < sd.rangeNum; ++r)
                {
                    const VriDescriptorRangeDesc& rd = sd.ranges[r];
                    if (rd.descriptorType != VriDescriptorType_ConstantBuffer) continue;
                    D3D12_ROOT_PARAMETER p = {};
                    p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
                    p.Descriptor.ShaderRegister = rd.baseRegister;
                    p.Descriptor.RegisterSpace = sd.registerSpace;
                    p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                    l->bindings.push_back({sd.registerSpace, rd.baseRegister, rd.descriptorType, static_cast<uint32_t>(params.size())});
                    params.push_back(p);
                }
            }
            D3D12_ROOT_SIGNATURE_DESC rs = {};
            rs.NumParameters = static_cast<UINT>(params.size());
            rs.pParameters = params.empty() ? nullptr : params.data();
            rs.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
            ComPtr<ID3DBlob> blob, err;
            if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)))
            { delete l; d->ReportError("D3D12SerializeRootSignature failed"); return VriResult_Failure; }
            if (FAILED(d->Device()->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&l->rootSig))))
            { delete l; d->ReportError("CreateRootSignature failed"); return VriResult_Failure; }
            *out = ToHandle(l);
            return VriResult_Success;
        }
        void VRI_CALL DestroyPipelineLayout(VriPipelineLayout* layout) { if (layout) delete PL(layout); }

        VriResult VRI_CALL CreateGraphicsPipeline(VriDevice* device, const VriGraphicsPipelineDesc* desc, VriPipeline** out)
        {
            DeviceD3D12* d = Dev(device);
            PipelineLayoutD3D12* layout = PL(desc->pipelineLayout);

            D3D12_GRAPHICS_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = layout->rootSig.Get();
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                const VriShaderDesc& s = desc->shaders[i];
                const D3D12_SHADER_BYTECODE bc = {s.bytecode, s.bytecodeSize};
                if (s.stage == VriShaderStage_Vertex) pd.VS = bc;
                else if (s.stage == VriShaderStage_Fragment) pd.PS = bc;
                else if (s.stage == VriShaderStage_Geometry) pd.GS = bc;
                else if (s.stage == VriShaderStage_TessControl) pd.HS = bc;
                else if (s.stage == VriShaderStage_TessEval) pd.DS = bc;
            }

            // Input layout (per-attribute). VRI attributes are location-indexed, but D3D
            // matches vertex inputs by semantic name+index. Reflect the VS input signature
            // (register N == VRI attribute/location N) so the layout uses the real
            // semantics (POSITION, COLOR0, ...) Slang emitted - no hardcoded convention.
            struct SemInfo { std::string name; UINT index = 0; };
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
                        if (p.Register < byRegister.size()) byRegister[p.Register] = {p.SemanticName ? p.SemanticName : "TEXCOORD", p.SemanticIndex};
                    }
                }
            }
            std::vector<D3D12_INPUT_ELEMENT_DESC> elems;
            std::vector<PipelineGraphicsVB> vbStrides;
            for (uint32_t i = 0; i < desc->vertexInput.attributeNum; ++i)
            {
                const VriVertexAttributeDesc& a = desc->vertexInput.attributes[i];
                uint32_t slot = a.streamIndex, stride = 0;
                if (a.streamIndex < desc->vertexInput.streamNum)
                { slot = desc->vertexInput.streams[a.streamIndex].bindingSlot; stride = static_cast<uint32_t>(desc->vertexInput.streams[a.streamIndex].stride); }
                D3D12_INPUT_ELEMENT_DESC e = {};
                if (i < byRegister.size() && !byRegister[i].name.empty()) { e.SemanticName = byRegister[i].name.c_str(); e.SemanticIndex = byRegister[i].index; }
                else { e.SemanticName = "TEXCOORD"; e.SemanticIndex = i; }
                e.Format = ToDxgiVertexFormat(a.format);
                e.InputSlot = slot; e.AlignedByteOffset = a.offset;
                e.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
                elems.push_back(e);
                bool have = false; for (auto& vb : vbStrides) if (vb.slot == slot) { have = true; break; }
                if (!have) vbStrides.push_back({stride, slot});
            }
            if (!elems.empty()) { pd.InputLayout.pInputElementDescs = elems.data(); pd.InputLayout.NumElements = static_cast<UINT>(elems.size()); }

            pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            pd.RasterizerState.CullMode = ToD3DCull(desc->rasterization.cullMode);
            pd.RasterizerState.FrontCounterClockwise = desc->rasterization.frontFace == VriFrontFace_CounterClockwise ? TRUE : FALSE;
            pd.RasterizerState.DepthClipEnable = TRUE;

            const VriColorAttachmentDesc* c0 = desc->outputMerger.colorNum ? &desc->outputMerger.colors[0] : nullptr;
            pd.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_ONE; pd.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_ZERO; pd.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
            pd.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE; pd.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO; pd.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
            if (c0 && c0->blend.enable)
            {
                pd.BlendState.RenderTarget[0].BlendEnable = TRUE;
                pd.BlendState.RenderTarget[0].SrcBlend = ToD3DBlend(c0->blend.srcColor); pd.BlendState.RenderTarget[0].DestBlend = ToD3DBlend(c0->blend.dstColor); pd.BlendState.RenderTarget[0].BlendOp = ToD3DBlendOp(c0->blend.colorOp);
                pd.BlendState.RenderTarget[0].SrcBlendAlpha = ToD3DBlend(c0->blend.srcAlpha); pd.BlendState.RenderTarget[0].DestBlendAlpha = ToD3DBlend(c0->blend.dstAlpha); pd.BlendState.RenderTarget[0].BlendOpAlpha = ToD3DBlendOp(c0->blend.alphaOp);
            }
            const VriColorWriteFlags wm = c0 ? c0->colorWriteMask : VriColorWrite_RGBA;
            pd.BlendState.RenderTarget[0].RenderTargetWriteMask = static_cast<UINT8>((wm == 0 ? VriColorWrite_RGBA : wm) & 0xF);

            pd.DepthStencilState.DepthEnable = desc->depthStencil.depthTest ? TRUE : FALSE;
            pd.DepthStencilState.DepthWriteMask = desc->depthStencil.depthWrite ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
            pd.DepthStencilState.DepthFunc = ToD3DCompare(desc->depthStencil.depthCompareOp);

            pd.SampleMask = UINT_MAX;
            pd.PrimitiveTopologyType = ToD3DTopologyType(desc->inputAssembly.topology);
            pd.NumRenderTargets = desc->outputMerger.colorNum <= 8 ? desc->outputMerger.colorNum : 8;
            for (uint32_t i = 0; i < pd.NumRenderTargets; ++i) pd.RTVFormats[i] = ToDxgiFormat(desc->outputMerger.colors[i].format).format;
            pd.DSVFormat = desc->outputMerger.depthStencilFormat != VriFormat_Unknown ? ToDxgiFormat(desc->outputMerger.depthStencilFormat).format : DXGI_FORMAT_UNKNOWN;
            pd.SampleDesc.Count = desc->multisample.sampleNum ? desc->multisample.sampleNum : 1;

            PipelineD3D12* p = new PipelineD3D12{};
            p->device = d; p->rootSig = layout->rootSig.Get(); p->vbStrides = std::move(vbStrides);
            p->topology = ToD3DTopology(desc->inputAssembly.topology, desc->tessellation.patchControlPoints);
            if (FAILED(d->Device()->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&p->pso))))
            { delete p; d->ReportError("CreateGraphicsPipelineState failed"); return VriResult_Failure; }
            *out = ToHandle(p);
            return VriResult_Success;
        }
        VriResult VRI_CALL CreateComputePipeline(VriDevice*, const VriComputePipelineDesc*, VriPipeline**) { return VriResult_Unsupported; }
        void VRI_CALL DestroyPipeline(VriPipeline* pipeline) { if (pipeline) delete Pipe(pipeline); }
        VriResult VRI_CALL CreateDescriptorPool(VriDevice* device, const VriDescriptorPoolDesc*, VriDescriptorPool** out)
        {
            *out = ToHandle(new DescriptorPoolD3D12{Dev(device)}); // CPU-side bookkeeping (root CBVs need no heap)
            return VriResult_Success;
        }
        void VRI_CALL ResetDescriptorPool(VriDescriptorPool*) {}
        void VRI_CALL DestroyDescriptorPool(VriDescriptorPool* pool) { if (pool) delete DPool(pool); }
        VriResult VRI_CALL AllocateDescriptorSets(VriDescriptorPool* pool, const VriPipelineLayout* layout, uint32_t setIndex, VriDescriptorSet** outSets, uint32_t setNum)
        {
            DescriptorPoolD3D12* p = DPool(pool);
            const PipelineLayoutD3D12* l = PL(layout);
            for (uint32_t i = 0; i < setNum; ++i)
            {
                DescriptorSetD3D12* s = new DescriptorSetD3D12{};
                s->device = p->device; s->layout = l; s->setIndex = setIndex;
                outSets[i] = ToHandle(s);
            }
            return VriResult_Success;
        }
        void VRI_CALL UpdateDescriptorRanges(VriDescriptorSet* set, uint32_t baseRange, uint32_t rangeNum, const VriDescriptorRangeUpdateDesc* updates)
        {
            DescriptorSetD3D12* s = DSet(set);
            std::vector<const LayoutBindingD3D12*> setBindings; // this set's bindings in declaration order
            for (const LayoutBindingD3D12& b : s->layout->bindings)
                if (b.set == s->setIndex) setBindings.push_back(&b);
            for (uint32_t r = 0; r < rangeNum; ++r)
            {
                const uint32_t idx = baseRange + r;
                if (idx >= setBindings.size() || updates[r].descriptorNum == 0) continue;
                s->bound[setBindings[idx]->binding] = reinterpret_cast<const DescriptorD3D12*>(updates[r].descriptors[0]);
            }
        }
        void VRI_CALL CmdSetPipelineLayout(VriCommandBuffer* cmd, VriPipelineLayout* layout) { CB(cmd)->list->SetGraphicsRootSignature(PL(layout)->rootSig.Get()); }
        void VRI_CALL CmdSetPipeline(VriCommandBuffer* cmd, VriPipeline* pipeline)
        {
            CommandBufferD3D12* c = CB(cmd);
            PipelineD3D12* p = Pipe(pipeline);
            c->boundPipeline = p;
            c->list->SetGraphicsRootSignature(p->rootSig); // also set here so a no-descriptor pipeline works without CmdSetPipelineLayout
            c->list->SetPipelineState(p->pso.Get());
            c->list->IASetPrimitiveTopology(p->topology);
        }
        void VRI_CALL CmdSetDescriptorSet(VriCommandBuffer* cmd, uint32_t setIndex, const VriDescriptorSet* set)
        {
            CommandBufferD3D12* c = CB(cmd);
            const DescriptorSetD3D12* s = reinterpret_cast<const DescriptorSetD3D12*>(set);
            if (!s || !s->layout) return;
            // Bind each recorded constant buffer as a root CBV at its mapped root slot.
            for (const LayoutBindingD3D12& b : s->layout->bindings)
            {
                if (b.set != setIndex || b.type != VriDescriptorType_ConstantBuffer) continue;
                auto it = s->bound.find(b.binding);
                if (it == s->bound.end() || !it->second || !it->second->buffer) continue;
                const DescriptorD3D12* v = it->second;
                c->list->SetGraphicsRootConstantBufferView(b.rootParam, v->buffer->resource->GetGPUVirtualAddress() + v->bufferOffset);
            }
        }
        void VRI_CALL CmdSetConstants(VriCommandBuffer*, uint32_t, const void*, uint32_t) {}
        void VRI_CALL CmdSetVertexBuffers(VriCommandBuffer* cmd, uint32_t baseSlot, const VriVertexBufferBinding* bindings, uint32_t num)
        {
            CommandBufferD3D12* c = CB(cmd);
            if (!num) return;
            std::vector<D3D12_VERTEX_BUFFER_VIEW> views(num);
            for (uint32_t i = 0; i < num; ++i)
            {
                BufferD3D12* b = Buf(bindings[i].buffer);
                uint32_t stride = 0;
                if (c->boundPipeline)
                    for (const PipelineGraphicsVB& vb : c->boundPipeline->vbStrides)
                        if (vb.slot == baseSlot + i) { stride = vb.stride; break; }
                views[i].BufferLocation = b->resource->GetGPUVirtualAddress() + bindings[i].offset;
                views[i].SizeInBytes = static_cast<UINT>(b->size - bindings[i].offset);
                views[i].StrideInBytes = stride;
            }
            c->list->IASetVertexBuffers(baseSlot, num, views.data());
        }
        void VRI_CALL CmdSetIndexBuffer(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, VriIndexType type)
        {
            BufferD3D12* b = Buf(buffer);
            D3D12_INDEX_BUFFER_VIEW v = {};
            v.BufferLocation = b->resource->GetGPUVirtualAddress() + offset;
            v.SizeInBytes = static_cast<UINT>(b->size - offset);
            v.Format = type == VriIndexType_UInt16 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;
            CB(cmd)->list->IASetIndexBuffer(&v);
        }
        void VRI_CALL CmdDraw(VriCommandBuffer* cmd, const VriDrawDesc* d)
        {
            CB(cmd)->list->DrawInstanced(d->vertexNum, d->instanceNum ? d->instanceNum : 1, d->baseVertex, d->baseInstance);
        }
        void VRI_CALL CmdDrawIndexed(VriCommandBuffer* cmd, const VriDrawIndexedDesc* d)
        {
            CB(cmd)->list->DrawIndexedInstanced(d->indexNum, d->instanceNum ? d->instanceNum : 1, d->baseIndex, d->vertexOffset, d->baseInstance);
        }
        void VRI_CALL CmdDrawIndirect(VriCommandBuffer*, VriBuffer*, uint64_t, uint32_t, uint32_t) {}
        void VRI_CALL CmdDispatch(VriCommandBuffer*, const VriDispatchDesc*) {}
        void VRI_CALL CmdDispatchIndirect(VriCommandBuffer*, VriBuffer*, uint64_t) {}
        void VRI_CALL CmdCopyBuffer(VriCommandBuffer* cmd, VriBuffer* dst, VriBuffer* src, const VriBufferCopyDesc* r)
        {
            // Buffers implicitly promote from COMMON to COPY_DEST/COPY_SOURCE on a direct
            // queue and decay back, so no explicit transition barriers are needed here.
            BufferD3D12* s = Buf(src);
            BufferD3D12* d = Buf(dst);
            CB(cmd)->list->CopyBufferRegion(d->resource.Get(), r->dstOffset, s->resource.Get(), r->srcOffset, r->size);
        }
        void VRI_CALL CmdCopyTexture(VriCommandBuffer*, VriTexture*, VriTexture*, const VriTextureCopyDesc*) {}
        void VRI_CALL CmdUploadBufferToTexture(VriCommandBuffer*, VriTexture*, VriBuffer*, const VriBufferTextureCopyDesc*) {}
        void VRI_CALL CmdBeginDebugGroup(VriCommandBuffer*, const char*) {}
        void VRI_CALL CmdEndDebugGroup(VriCommandBuffer*) {}
        void VRI_CALL SetDebugName(void*, const char*) {}
    } // namespace

    const VriCoreInterface* GetCoreInterfaceD3D12()
    {
        static const VriCoreInterface t = {
            GetDeviceDesc, GetFormatSupport, GetQueue,
            CreateCommandAllocator, ResetCommandAllocator, DestroyCommandAllocator, CreateCommandBuffer, BeginCommandBuffer, EndCommandBuffer,
            CreateBuffer, DestroyBuffer, MapBuffer, UnmapBuffer, GetBufferDeviceAddress, CreateTexture, DestroyTexture,
            GetBufferMemoryDesc, GetTextureMemoryDesc, AllocateMemory, FreeMemory, BindBufferMemory, BindTextureMemory,
            CreateBufferView, CreateTextureView, CreateSampler, DestroyDescriptor,
            CreatePipelineLayout, DestroyPipelineLayout, CreateGraphicsPipeline, CreateComputePipeline, DestroyPipeline,
            CreateDescriptorPool, ResetDescriptorPool, DestroyDescriptorPool, AllocateDescriptorSets, UpdateDescriptorRanges,
            CreateFence, DestroyFence, GetFenceValue, Wait,
            CmdBeginRendering, CmdEndRendering, CmdSetViewports, CmdSetScissors, CmdSetPipelineLayout, CmdSetPipeline,
            CmdSetDescriptorSet, CmdSetConstants, CmdSetVertexBuffers, CmdSetIndexBuffer,
            CmdDraw, CmdDrawIndexed, CmdDrawIndirect, CmdDispatch, CmdDispatchIndirect, CmdBarrier,
            CmdCopyBuffer, CmdCopyTexture, CmdUploadBufferToTexture, CmdReadbackTextureToBuffer, CmdBeginDebugGroup, CmdEndDebugGroup,
            QueueSubmit, QueueWaitIdle, DeviceWaitIdle, SetDebugName,
        };
        return &t;
    }
} // namespace vri::d3d12
