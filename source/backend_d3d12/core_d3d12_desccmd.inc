// core_d3d12_desccmd.inc - descriptor pools/sets and the draw/dispatch/copy/debug command
// recording. #included into core_d3d12.cpp (one TU); see core_d3d12.cpp.
VriResult VRI_CALL CreateDescriptorPool(VriDevice* device, const VriDescriptorPoolDesc* desc, VriDescriptorPool** out)
{
    DeviceD3D12*         d = Dev(device);
    DescriptorPoolD3D12* p = new DescriptorPoolD3D12 {};
    p->device              = d;
    // Shader-visible heaps for SRV/CBV/UAV + samplers (descriptor tables bind from
    // these). Sized from the pool request with sane minimums; root CBVs use neither.
    const uint32_t srvCap         = (desc->textureMaxNum + desc->structuredBufferMaxNum + desc->storageBufferMaxNum +
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
                D3D12_SHADER_RESOURCE_VIEW_DESC srv          = {};
                srv.ViewDimension                            = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
                srv.Shader4ComponentMapping                  = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
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
                srv.Format = v->texture->srvFormat; // typed sampleable format (R32_FLOAT for a sampled D32 depth)
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
        c->boundSrvHeap     = heaps[0];
        c->boundSamplerHeap = heaps[1];
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
    CB(cmd)->list->DrawInstanced(d->vertexNum, d->instanceNum ? d->instanceNum : 1, d->baseVertex, d->baseInstance);
}
void VRI_CALL CmdDrawIndexed(VriCommandBuffer* cmd, const VriDrawIndexedDesc* d)
{
    FlushVertexBuffers(CB(cmd));
    CB(cmd)->list->DrawIndexedInstanced(
        d->indexNum, d->instanceNum ? d->instanceNum : 1, d->baseIndex, d->vertexOffset, d->baseInstance);
}
// Indirect draws via ExecuteIndirect + a DRAW/DRAW_INDEXED command signature (cached by stride).
// The *Count variants pass the count buffer; the others pass null and a fixed drawNum.
void VRI_CALL
CmdDrawIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint32_t drawNum, uint32_t stride)
{
    CommandBufferD3D12* c = CB(cmd);
    c->list->ExecuteIndirect(
        c->device->DrawSignature(false, stride), drawNum, Buf(buffer)->resource.Get(), offset, nullptr, 0);
}
void VRI_CALL
CmdDrawIndexedIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint32_t drawNum, uint32_t stride)
{
    CommandBufferD3D12* c = CB(cmd);
    c->list->ExecuteIndirect(
        c->device->DrawSignature(true, stride), drawNum, Buf(buffer)->resource.Get(), offset, nullptr, 0);
}
void VRI_CALL CmdDrawIndirectCount(VriCommandBuffer* cmd,
                                   VriBuffer*        buffer,
                                   uint64_t          offset,
                                   VriBuffer*        countBuffer,
                                   uint64_t          countOffset,
                                   uint32_t          maxDrawNum,
                                   uint32_t          stride)
{
    CommandBufferD3D12* c = CB(cmd);
    c->list->ExecuteIndirect(c->device->DrawSignature(false, stride),
                             maxDrawNum,
                             Buf(buffer)->resource.Get(),
                             offset,
                             Buf(countBuffer)->resource.Get(),
                             countOffset);
}
void VRI_CALL CmdDrawIndexedIndirectCount(VriCommandBuffer* cmd,
                                          VriBuffer*        buffer,
                                          uint64_t          offset,
                                          VriBuffer*        countBuffer,
                                          uint64_t          countOffset,
                                          uint32_t          maxDrawNum,
                                          uint32_t          stride)
{
    CommandBufferD3D12* c = CB(cmd);
    c->list->ExecuteIndirect(c->device->DrawSignature(true, stride),
                             maxDrawNum,
                             Buf(buffer)->resource.Get(),
                             offset,
                             Buf(countBuffer)->resource.Get(),
                             countOffset);
}
void VRI_CALL CmdDispatch(VriCommandBuffer* cmd, const VriDispatchDesc* d)
{
    CB(cmd)->list->Dispatch(d->x, d->y, d->z);
}
// ExecuteIndirect with a single DISPATCH record: the {x,y,z} group count is read from
// `buffer` at `offset` (the app barriers it to IndirectBufferRead first). Uses the compute
// root signature / PSO already bound by CmdSetPipeline{,Layout} + CmdSetDescriptorSet.
void VRI_CALL CmdDispatchIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset)
{
    CommandBufferD3D12* c = CB(cmd);
    c->list->ExecuteIndirect(c->device->DispatchSignature(), 1, Buf(buffer)->resource.Get(), offset, nullptr, 0);
}
// D3D12 has no direct buffer fill: ClearUnorderedAccessViewUint via a transient R32_UINT UAV
// staged in both clear heaps (see DeviceD3D12::ClearUavViews). The buffer is bounced into
// UNORDERED_ACCESS for the clear and the shader-visible heap is swapped in then restored.
void VRI_CALL
CmdClearStorageBuffer(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint64_t size, uint32_t value)
{
    CommandBufferD3D12*              c   = CB(cmd);
    BufferD3D12*                     b   = Buf(buffer);
    const uint64_t                   sz  = size ? size : (b->size - offset);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format                           = DXGI_FORMAT_R32_UINT;
    uav.ViewDimension                    = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.FirstElement              = offset / 4;
    uav.Buffer.NumElements               = static_cast<UINT>(sz / 4);
    D3D12_GPU_DESCRIPTOR_HANDLE gpu {};
    D3D12_CPU_DESCRIPTOR_HANDLE cpuView {};
    ID3D12DescriptorHeap*       gpuHeap = nullptr;
    c->device->ClearUavViews(b->resource.Get(), uav, gpu, cpuView, gpuHeap);
    TransitionBuffer(c, b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    c->list->SetDescriptorHeaps(1, &gpuHeap);
    const UINT values[4] = {value, value, value, value};
    c->list->ClearUnorderedAccessViewUint(gpu, cpuView, b->resource.Get(), values, 0, nullptr);
    if (c->boundSrvHeap) // restore the app's shader-visible heaps
    {
        ID3D12DescriptorHeap* h[2] = {c->boundSrvHeap, c->boundSamplerHeap};
        c->list->SetDescriptorHeaps(2, h);
    }
}
void VRI_CALL CmdClearStorageTexture(VriCommandBuffer* cmd, VriTexture* texture, const VriClearColor* value)
{
    CommandBufferD3D12*              c   = CB(cmd);
    TextureD3D12*                    t   = Tex(texture);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format                           = t->format;
    if (t->layerNum > 1)
    {
        uav.ViewDimension            = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        uav.Texture2DArray.ArraySize = t->layerNum;
    }
    else
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu {};
    D3D12_CPU_DESCRIPTOR_HANDLE cpuView {};
    ID3D12DescriptorHeap*       gpuHeap = nullptr;
    c->device->ClearUavViews(t->resource.Get(), uav, gpu, cpuView, gpuHeap);
    Transition(c, t, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    c->list->SetDescriptorHeaps(1, &gpuHeap);
    if (IsIntegerDxgi(t->format))
        c->list->ClearUnorderedAccessViewUint(gpu, cpuView, t->resource.Get(), value->u32, 0, nullptr);
    else
        c->list->ClearUnorderedAccessViewFloat(gpu, cpuView, t->resource.Get(), value->f32, 0, nullptr);
    if (c->boundSrvHeap)
    {
        ID3D12DescriptorHeap* h[2] = {c->boundSrvHeap, c->boundSamplerHeap};
        c->list->SetDescriptorHeaps(2, h);
    }
}
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
// Texture->texture copy. The app barriers src to CopySource and dst to CopyDestination, so
// this just issues CopyTextureRegion between the selected subresources (subresource = mip +
// baseLayer * mipNum, matching CmdUploadBufferToTexture). A zero-width src region copies the
// whole subresource (pSrcBox = nullptr); a non-zero width copies that sub-box.
void VRI_CALL CmdCopyTexture(VriCommandBuffer* cmd, VriTexture* dst, VriTexture* src, const VriTextureCopyDesc* region)
{
    CommandBufferD3D12* c = CB(cmd);
    TextureD3D12*       d = Tex(dst);
    TextureD3D12*       s = Tex(src);

    D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
    dstLoc.pResource                   = d->resource.Get();
    dstLoc.Type                        = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex            = region ? region->dst.mip + region->dst.baseLayer * d->mipNum : 0;
    D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
    srcLoc.pResource                   = s->resource.Get();
    srcLoc.Type                        = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLoc.SubresourceIndex            = region ? region->src.mip + region->src.baseLayer * s->mipNum : 0;

    const UINT dx = region ? static_cast<UINT>(region->dst.x) : 0;
    const UINT dy = region ? static_cast<UINT>(region->dst.y) : 0;
    const UINT dz = region ? static_cast<UINT>(region->dst.z) : 0;
    if (region && region->src.width)
    {
        D3D12_BOX box = {};
        box.left      = static_cast<UINT>(region->src.x);
        box.top       = static_cast<UINT>(region->src.y);
        box.front     = static_cast<UINT>(region->src.z);
        box.right     = box.left + region->src.width;
        box.bottom    = box.top + (region->src.height ? region->src.height : 1);
        box.back      = box.front + (region->src.depth ? region->src.depth : 1);
        c->list->CopyTextureRegion(&dstLoc, dx, dy, dz, &srcLoc, &box);
    }
    else
        c->list->CopyTextureRegion(&dstLoc, dx, dy, dz, &srcLoc, nullptr);
}
void VRI_CALL CmdUploadBufferToTexture(VriCommandBuffer*               cmd,
                                       VriTexture*                     dst,
                                       VriBuffer*                      src,
                                       const VriBufferTextureCopyDesc* region)
{
    CommandBufferD3D12* c = CB(cmd);
    TextureD3D12*       t = Tex(dst);
    BufferD3D12*        b = Buf(src);
    // Subresource = mip + arrayLayer * mipNum (baseLayer selects the array slice).
    const UINT          sub = (region ? region->texture.mip : 0) + (region ? region->texture.baseLayer : 0) * t->mipNum;
    const UINT64        srcOffset               = region ? region->bufferOffset : 0;
    D3D12_RESOURCE_DESC rd                      = t->resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp       = {};
    UINT                               rows     = 0;
    UINT64                             rowBytes = 0, total = 0;
    c->device->Device()->GetCopyableFootprints(&rd, sub, 1, 0, &fp, &rows, &rowBytes, &total); // footprint at offset 0
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
    const bool aligned = rowBytes == fp.Footprint.RowPitch && (srcOffset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT) == 0;
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
        if (FAILED(c->device->Device()->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&temp))))
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
// PIX/RenderDoc debug-group markers. We emit them through the command list's
// BeginEvent/EndEvent using the legacy PIX string-marker encoding: metadata ==
// PIX_EVENT_ANSI_VERSION means pData is a null-terminated ANSI string and Size is its
// byte length including the terminator. PIX (backward-compat) and RenderDoc both decode
// this, so the groups are visible in captures without a WinPixEventRuntime dependency.
// (The modern PIX3-blob metadata value 2 requires a fully encoded blob; a raw string
// there is rejected and unbalances the runtime's begin/end accounting, so we use the
// legacy string path instead.)
constexpr UINT kPixEventAnsiVersion = 1; // PIX_EVENT_ANSI_VERSION (pix3.h)
void VRI_CALL  CmdBeginDebugGroup(VriCommandBuffer* cmd, const char* name)
{
    CommandBufferD3D12* c = CB(cmd);
    const char*         n = name ? name : "";
    c->list->BeginEvent(kPixEventAnsiVersion, n, static_cast<UINT>(std::strlen(n) + 1));
}
void VRI_CALL CmdEndDebugGroup(VriCommandBuffer* cmd) { CB(cmd)->list->EndEvent(); }
// Was a no-op. It records the label against the tracked object, which is what makes an
// EnumerateObjects listing readable - without it every row is a pointer and a size. Every
// backend object struct begins with its owning device, which is what lets one entry point
// name any of them without a per-type dispatch table.
void VRI_CALL SetDebugName(void* object, const char* name)
{
    if (object == nullptr)
        return;
    DeviceD3D12* d = *reinterpret_cast<DeviceD3D12**>(object);
    if (d != nullptr)
        d->Objects().SetName(object, name);
}

VriResult VRI_CALL EnumerateObjects(const VriDevice* device, uint32_t* count, VriObjectInfo* out)
{
    if (device == nullptr)
        return VriResult_InvalidArgument;
    return Dev(const_cast<VriDevice*>(device))->Objects().Snapshot(count, out);
}
