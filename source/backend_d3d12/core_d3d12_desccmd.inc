// core_d3d12_desccmd.inc - descriptor pools/sets and the draw/dispatch/copy/debug command
// recording. #included into core_d3d12.cpp (one TU); see core_d3d12.cpp.
VriResult VRI_CALL CreateDescriptorPool(VriDevice* device, const VriDescriptorPoolDesc* desc, VriDescriptorPool** out)
{
    DeviceD3D12*         d = Dev(device);
    DescriptorPoolD3D12* p = new DescriptorPoolD3D12 {};
    p->device              = d;
    // Blocks of the device's shared shader-visible heaps for SRV/CBV/UAV + samplers (descriptor tables
    // bind from these). Sized from the pool request with sane minimums; root CBVs use neither.
    const uint32_t srvCap     = (desc->textureMaxNum + desc->structuredBufferMaxNum + desc->storageBufferMaxNum +
                             desc->storageTextureMaxNum + 16);
    const uint32_t samplerCap = (desc->samplerMaxNum + 16);
    if (!d->AllocateShaderVisible(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, srvCap, p->srvBlock))
    {
        delete p;
        d->ReportError("CreateDescriptorPool: the shared CBV/SRV/UAV heap is full");
        return VriResult_OutOfMemory;
    }
    if (!d->AllocateShaderVisible(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, samplerCap, p->samplerBlock))
    {
        d->FreeShaderVisible(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, p->srvBlock);
        delete p;
        d->ReportError("CreateDescriptorPool: the shared sampler heap is full (2048 samplers per device)");
        return VriResult_OutOfMemory;
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
    if (!pool)
        return;
    DescriptorPoolD3D12* p = DPool(pool);
    p->device->FreeShaderVisible(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, p->srvBlock);
    p->device->FreeShaderVisible(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, p->samplerBlock);
    delete p;
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
    const uint32_t srvNeed     = si ? si->srvCount * setNum : 0;
    const uint32_t samplerNeed = si ? si->samplerCount * setNum : 0;
    if (p->srvNext + srvNeed > p->srvBlock.count || p->samplerNext + samplerNeed > p->samplerBlock.count)
    {
        p->device->ReportError("AllocateDescriptorSets: the descriptor pool is full");
        return VriResult_OutOfMemory;
    }
    ID3D12DescriptorHeap* srvHeap     = p->device->ShaderVisibleHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    ID3D12DescriptorHeap* samplerHeap = p->device->ShaderVisibleHeap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    for (uint32_t i = 0; i < setNum; ++i)
    {
        DescriptorSetD3D12* s = new DescriptorSetD3D12 {};
        s->device             = p->device;
        s->pool               = p;
        s->layout             = l;
        s->setIndex           = setIndex;
        if (si && si->srvCount > 0) // sub-allocate this set's SRV block
        {
            const uint32_t slot = p->srvBlock.offset + p->srvNext;
            s->srvCpu           = srvHeap->GetCPUDescriptorHandleForHeapStart();
            s->srvCpu.ptr += static_cast<SIZE_T>(slot) * p->srvSize;
            s->srvGpu = srvHeap->GetGPUDescriptorHandleForHeapStart();
            s->srvGpu.ptr += static_cast<UINT64>(slot) * p->srvSize;
            p->srvNext += si->srvCount;
        }
        if (si && si->samplerCount > 0)
        {
            const uint32_t slot = p->samplerBlock.offset + p->samplerNext;
            s->samplerCpu       = samplerHeap->GetCPUDescriptorHandleForHeapStart();
            s->samplerCpu.ptr += static_cast<SIZE_T>(slot) * p->samplerSize;
            s->samplerGpu = samplerHeap->GetGPUDescriptorHandleForHeapStart();
            s->samplerGpu.ptr += static_cast<UINT64>(slot) * p->samplerSize;
            p->samplerNext += si->samplerCount;
        }
        outSets[i] = ToHandle(s);
    }
    return VriResult_Success;
}
// A texture view covers its own mips and layers. A view of one slice of an array resource needs
// an array descriptor: D3D12's non-array descriptors always address slice 0.
bool ArrayView(const DescriptorD3D12* v)
{
    return v->viewType == VriTextureViewType_1DArray || v->viewType == VriTextureViewType_2DArray ||
           v->texture->layerNum > 1;
}

D3D12_UNORDERED_ACCESS_VIEW_DESC StorageViewDesc(const DescriptorD3D12* v)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format                           = v->texture->format;
    if (v->texture->dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
    {
        uav.ViewDimension         = D3D12_UAV_DIMENSION_TEXTURE3D;
        uav.Texture3D.MipSlice    = v->mip;
        uav.Texture3D.FirstWSlice = 0;
        uav.Texture3D.WSize       = UINT(-1); // every depth slice of the mip
    }
    else if (v->texture->dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && !ArrayView(v))
    {
        uav.ViewDimension      = D3D12_UAV_DIMENSION_TEXTURE1D;
        uav.Texture1D.MipSlice = v->mip;
    }
    else if (v->texture->dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D)
    {
        uav.ViewDimension                  = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
        uav.Texture1DArray.MipSlice        = v->mip;
        uav.Texture1DArray.FirstArraySlice = v->baseLayer;
        uav.Texture1DArray.ArraySize       = v->layerNum;
    }
    else if (ArrayView(v))
    {
        uav.ViewDimension                  = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        uav.Texture2DArray.MipSlice        = v->mip;
        uav.Texture2DArray.FirstArraySlice = v->baseLayer;
        uav.Texture2DArray.ArraySize       = v->layerNum;
    }
    else
    {
        uav.ViewDimension      = D3D12_UAV_DIMENSION_TEXTURE2D;
        uav.Texture2D.MipSlice = v->mip;
    }
    return uav;
}

D3D12_SHADER_RESOURCE_VIEW_DESC SampledViewDesc(const DescriptorD3D12* v)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
    srv.Format                  = v->texture->srvFormat; // typed sampleable format (R32_FLOAT for a sampled D32)
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (v->texture->dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
    {
        srv.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE3D;
        srv.Texture3D.MostDetailedMip = v->mip;
        srv.Texture3D.MipLevels       = v->mipNum;
    }
    else if (v->texture->dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && !ArrayView(v))
    {
        srv.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE1D;
        srv.Texture1D.MostDetailedMip = v->mip;
        srv.Texture1D.MipLevels       = v->mipNum;
    }
    else if (v->texture->dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D)
    {
        srv.ViewDimension                  = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
        srv.Texture1DArray.MostDetailedMip = v->mip;
        srv.Texture1DArray.MipLevels       = v->mipNum;
        srv.Texture1DArray.FirstArraySlice = v->baseLayer;
        srv.Texture1DArray.ArraySize       = v->layerNum;
    }
    else if (v->viewType == VriTextureViewType_CubeArray) // samples cubes by direction and index
    {
        srv.ViewDimension                     = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
        srv.TextureCubeArray.MostDetailedMip  = v->mip;
        srv.TextureCubeArray.MipLevels        = v->mipNum;
        srv.TextureCubeArray.First2DArrayFace = v->baseLayer;
        srv.TextureCubeArray.NumCubes         = v->layerNum / 6;
    }
    else if (v->viewType == VriTextureViewType_Cube) // samples a 6-layer resource by direction
    {
        srv.ViewDimension               = D3D12_SRV_DIMENSION_TEXTURECUBE;
        srv.TextureCube.MostDetailedMip = v->mip;
        srv.TextureCube.MipLevels       = v->mipNum;
    }
    else if (ArrayView(v))
    {
        srv.ViewDimension                  = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        srv.Texture2DArray.MostDetailedMip = v->mip;
        srv.Texture2DArray.MipLevels       = v->mipNum;
        srv.Texture2DArray.FirstArraySlice = v->baseLayer;
        srv.Texture2DArray.ArraySize       = v->layerNum;
    }
    else
    {
        srv.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Texture2D.MostDetailedMip = v->mip;
        srv.Texture2D.MipLevels       = v->mipNum;
    }
    return srv;
}
// A read-only buffer view: structured with the buffer's structureStride (an HLSL StructuredBuffer),
// or raw 32-bit words when the buffer has none (a ByteAddressBuffer).
D3D12_SHADER_RESOURCE_VIEW_DESC BufferViewDesc(const DescriptorD3D12* v)
{
    const uint64_t size = v->bufferSize ? v->bufferSize : v->buffer->size - v->bufferOffset;
    const uint32_t stride = v->buffer->structureStride;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
    srv.ViewDimension                   = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping         = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (stride)
    {
        srv.Format                     = DXGI_FORMAT_UNKNOWN;
        srv.Buffer.FirstElement        = v->bufferOffset / stride;
        srv.Buffer.NumElements         = static_cast<UINT>(size / stride);
        srv.Buffer.StructureByteStride = stride;
    }
    else
    {
        srv.Format              = DXGI_FORMAT_R32_TYPELESS;
        srv.Buffer.FirstElement = v->bufferOffset / 4;
        srv.Buffer.NumElements  = static_cast<UINT>(size / 4);
        srv.Buffer.Flags        = D3D12_BUFFER_SRV_FLAG_RAW;
    }
    return srv;
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
        if (lb->type == VriDescriptorType_ConstantBuffer || lb->type == VriDescriptorType_StorageBuffer)
        {
            // Root descriptors bind descriptor [0] by GPU address at CmdSetDescriptorSet.
            const auto* v = reinterpret_cast<const DescriptorD3D12*>(updates[r].descriptors[0]);
            if (v && v->buffer)
                s->rootAddresses[lb->binding] = v->buffer->resource->GetGPUVirtualAddress() + v->bufferOffset;
            continue;
        }
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
            else if (lb->type == VriDescriptorType_StructuredBuffer && v->buffer) // buffer SRV in the table
            {
                const D3D12_SHADER_RESOURCE_VIEW_DESC srv = BufferViewDesc(v);
                D3D12_CPU_DESCRIPTOR_HANDLE           h   = s->srvCpu;
                h.ptr += slot * s->pool->srvSize;
                d->Device()->CreateShaderResourceView(v->buffer->resource.Get(), &srv, h);
            }
            else if (lb->type == VriDescriptorType_StorageTexture && v->texture) // UAV
            {
                const D3D12_UNORDERED_ACCESS_VIEW_DESC uav = StorageViewDesc(v);
                D3D12_CPU_DESCRIPTOR_HANDLE            h   = s->srvCpu;
                h.ptr += slot * s->pool->srvSize;
                d->Device()->CreateUnorderedAccessView(v->texture->resource.Get(), nullptr, &uav, h);
            }
            else if (v->texture) // SRV (sampled texture) into the set's CBV/SRV/UAV heap slot
            {
                const D3D12_SHADER_RESOURCE_VIEW_DESC srv = SampledViewDesc(v);
                D3D12_CPU_DESCRIPTOR_HANDLE           h   = s->srvCpu;
                h.ptr += slot * s->pool->srvSize;
                d->Device()->CreateShaderResourceView(v->texture->resource.Get(), &srv, h);
            }
        }
    }
}
// Whether layouts, descriptor sets and constants bind on the compute root: always on a compute
// list, otherwise by the bound pipeline, or by the layout while no pipeline is bound yet.
bool ComputeBinding(const CommandBufferD3D12* c)
{
    if (c->allocator->listType == D3D12_COMMAND_LIST_TYPE_COMPUTE)
        return true;
    if (c->boundPipeline)
        return c->boundPipeline->isCompute;
    return c->boundLayout && c->boundLayout->computeOnly;
}
void VRI_CALL CmdSetPipelineLayout(VriCommandBuffer* cmd, VriPipelineLayout* layout)
{
    CommandBufferD3D12* c   = CB(cmd);
    c->boundLayout          = PL(layout);
    ID3D12RootSignature* rs = PL(layout)->rootSig.Get();
    if (ComputeBinding(c))
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
    const bool            compute = ComputeBinding(c);
    // Bind the pool's shader-visible heaps (needed for the SRV/sampler tables).
    if (s->pool && (si && (si->srvCount || si->samplerCount)))
    {
        ID3D12DescriptorHeap* heaps[2] = {c->device->ShaderVisibleHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
                                          c->device->ShaderVisibleHeap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)};
        c->list->SetDescriptorHeaps(2, heaps);
        c->boundSrvHeap     = heaps[0];
        c->boundSamplerHeap = heaps[1];
    }
    // Root CBV / UAV (by GPU address); graphics vs compute root binding point.
    for (const LayoutBindingD3D12& b : s->layout->bindings)
    {
        if (b.set != setIndex)
            continue;
        const auto it = s->rootAddresses.find(b.binding);
        if (it == s->rootAddresses.end())
            continue;
        const D3D12_GPU_VIRTUAL_ADDRESS va = it->second;
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
    if (ComputeBinding(c))
        c->list->SetComputeRoot32BitConstants(c->boundLayout->pushRootParam, (size + 3u) / 4u, data, 0);
    else
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
// The *Count variants pass the count buffer; the others pass null and a fixed drawNum. Like direct draws
// they first bind the pending vertex buffers.
void VRI_CALL
CmdDrawIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint32_t drawNum, uint32_t stride)
{
    CommandBufferD3D12* c = CB(cmd);
    FlushVertexBuffers(c);
    c->list->ExecuteIndirect(
        c->device->DrawSignature(false, stride), drawNum, Buf(buffer)->resource.Get(), offset, nullptr, 0);
}
void VRI_CALL
CmdDrawIndexedIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint32_t drawNum, uint32_t stride)
{
    CommandBufferD3D12* c = CB(cmd);
    FlushVertexBuffers(c);
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
    FlushVertexBuffers(c);
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
    FlushVertexBuffers(c);
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
    CommandBufferD3D12* c = CB(cmd);
    TextureD3D12*       t = Tex(texture);
    // Mip 0 of every layer (every depth slice of a 3D texture).
    DescriptorD3D12 whole {};
    whole.texture                              = t;
    whole.viewType                             = VriTextureViewType_2D;
    whole.layerNum                             = t->layerNum;
    const D3D12_UNORDERED_ACCESS_VIEW_DESC uav = StorageViewDesc(&whole);
    D3D12_GPU_DESCRIPTOR_HANDLE            gpu {};
    D3D12_CPU_DESCRIPTOR_HANDLE cpuView {};
    ID3D12DescriptorHeap*       gpuHeap = nullptr;
    c->device->ClearUavViews(t->resource.Get(), uav, gpu, cpuView, gpuHeap);
    // D3D12 clears through a UAV, but the caller barriered the texture for a transfer-domain clear
    // (CopyDestination): move only mip 0 to UNORDERED_ACCESS and give it back its states afterwards,
    // so the caller's view of every subresource stays true.
    std::vector<D3D12_RESOURCE_STATES> before(t->layerNum);
    for (uint32_t layer = 0; layer < t->layerNum; ++layer)
        before[layer] = t->states[layer * t->mipNum];
    Transition(c, t, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, {0, 1, 0, t->layerNum});
    c->list->SetDescriptorHeaps(1, &gpuHeap);
    if (IsIntegerDxgi(t->format))
        c->list->ClearUnorderedAccessViewUint(gpu, cpuView, t->resource.Get(), value->u32, 0, nullptr);
    else
        c->list->ClearUnorderedAccessViewFloat(gpu, cpuView, t->resource.Get(), value->f32, 0, nullptr);
    for (uint32_t layer = 0; layer < t->layerNum; ++layer)
        Transition(c, t, before[layer], {0, 1, layer, 1});
    if (c->boundSrvHeap)
    {
        ID3D12DescriptorHeap* h[2] = {c->boundSrvHeap, c->boundSamplerHeap};
        c->list->SetDescriptorHeaps(2, h);
    }
}
void VRI_CALL CmdCopyBuffer(VriCommandBuffer* cmd, VriBuffer* dst, VriBuffer* src, const VriBufferCopyDesc* r)
{
    // Implicit promotion to COPY_DEST covers only a buffer in COMMON; a buffer refreshed by a copy
    // after being read is in a read state, so both sides are transitioned explicitly (upload and
    // readback heaps keep their fixed states).
    CommandBufferD3D12* c = CB(cmd);
    BufferD3D12*        s = Buf(src);
    BufferD3D12*        d = Buf(dst);
    TransitionBuffer(c, s, D3D12_RESOURCE_STATE_COPY_SOURCE);
    TransitionBuffer(c, d, D3D12_RESOURCE_STATE_COPY_DEST);
    c->list->CopyBufferRegion(d->resource.Get(), r->dstOffset, s->resource.Get(), r->srcOffset, r->size);
}
// Texture->texture copy. The app barriers src to CopySource and dst to CopyDestination, so
// this just issues CopyTextureRegion between the selected subresources (subresource = mip +
// layer * mipNum, matching CmdUploadBufferToTexture), once per layer of src.layerNum (0 or 1 =
// one layer). A zero-width src region copies the whole subresource (pSrcBox = nullptr); a
// non-zero width copies that sub-box.
void VRI_CALL CmdCopyTexture(VriCommandBuffer* cmd, VriTexture* dst, VriTexture* src, const VriTextureCopyDesc* region)
{
    CommandBufferD3D12* c = CB(cmd);
    TextureD3D12*       d = Tex(dst);
    TextureD3D12*       s = Tex(src);

    const UINT dx = region ? static_cast<UINT>(region->dst.x) : 0;
    const UINT dy = region ? static_cast<UINT>(region->dst.y) : 0;
    const UINT dz = region ? static_cast<UINT>(region->dst.z) : 0;
    D3D12_BOX  box = {};
    if (region && region->src.width)
    {
        box.left   = static_cast<UINT>(region->src.x);
        box.top    = static_cast<UINT>(region->src.y);
        box.front  = static_cast<UINT>(region->src.z);
        box.right  = box.left + region->src.width;
        box.bottom = box.top + (region->src.height ? region->src.height : 1);
        box.back   = box.front + (region->src.depth ? region->src.depth : 1);
    }
    const UINT layers = region && region->src.layerNum ? region->src.layerNum : 1;
    for (UINT layer = 0; layer < layers; ++layer)
    {
        D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
        dstLoc.pResource                   = d->resource.Get();
        dstLoc.Type                        = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dstLoc.SubresourceIndex = region ? region->dst.mip + (region->dst.baseLayer + layer) * d->mipNum : 0;
        D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
        srcLoc.pResource                   = s->resource.Get();
        srcLoc.Type                        = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        srcLoc.SubresourceIndex = region ? region->src.mip + (region->src.baseLayer + layer) * s->mipNum : 0;
        c->list->CopyTextureRegion(&dstLoc, dx, dy, dz, &srcLoc, region && region->src.width ? &box : nullptr);
    }
}
void VRI_CALL CmdUploadBufferToTexture(VriCommandBuffer*               cmd,
                                       VriTexture*                     dst,
                                       VriBuffer*                      src,
                                       const VriBufferTextureCopyDesc* region)
{
    CommandBufferD3D12* c         = CB(cmd);
    TextureD3D12*       t         = Tex(dst);
    BufferD3D12*        b         = Buf(src);
    const UINT          mip       = region ? region->texture.mip : 0;
    const UINT          baseLayer = region ? region->texture.baseLayer : 0;
    const UINT          layers    = region && region->texture.layerNum ? region->texture.layerNum : 1;
    if (mip >= t->mipNum || baseLayer >= t->layerNum || layers > t->layerNum - baseLayer)
    {
        c->device->ReportError("texture-upload subresource out of bounds");
        return;
    }
    const UINT x         = region ? region->texture.x : 0;
    const UINT y         = region ? region->texture.y : 0;
    const UINT z         = region ? region->texture.z : 0;
    const UINT mipWidth  = (std::max)(1u, t->width >> mip);
    const UINT mipHeight = (std::max)(1u, t->height >> mip);
    const UINT mipDepth  = (std::max)(1u, t->depth >> mip);
    const UINT width     = region && region->texture.width ? region->texture.width : mipWidth;
    const UINT height    = region && region->texture.height ? region->texture.height : mipHeight;
    const UINT depth     = region && region->texture.depth ? region->texture.depth : mipDepth;
    if (x >= mipWidth || width > mipWidth - x || y >= mipHeight || height > mipHeight - y || z >= mipDepth ||
        depth > mipDepth - z)
    {
        c->device->ReportError("texture-upload region out of bounds");
        return;
    }
    D3D12_RESOURCE_DESC rd = t->resource->GetDesc();
    // Describe the requested region, not the entire destination mip. Footprints account
    // for compressed blocks even when the logical mip is smaller than one block.
    rd.Width                                    = width;
    rd.Height                                   = height;
    rd.DepthOrArraySize                         = static_cast<UINT16>(depth);
    rd.MipLevels                                = 1;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp       = {};
    UINT                               rows     = 0;
    UINT64                             rowBytes = 0, total = 0;
    c->device->Device()->GetCopyableFootprints(&rd, 0, 1, 0, &fp, &rows, &rowBytes, &total);
    if (total == UINT64_MAX || rows == 0 || rowBytes == 0)
    {
        c->device->ReportError("texture-upload invalid footprint");
        return;
    }
    const UINT block        = t->texelSize == 0 ? 4 : 1;
    const UINT sourceWidth  = region && region->bufferRowLength ? region->bufferRowLength : width;
    const UINT sourceHeight = region && region->bufferImageHeight ? region->bufferImageHeight : height;
    if (sourceWidth < width || sourceHeight < height || x % block || y % block ||
        (width % block && x + width != mipWidth) || (height % block && y + height != mipHeight) ||
        (region && region->bufferRowLength && sourceWidth % block) ||
        (region && region->bufferImageHeight && sourceHeight % block))
    {
        c->device->ReportError("texture-upload invalid block region or source stride");
        return;
    }
    const UINT64 blockBytes  = rowBytes / ((width + block - 1) / block);
    const UINT64 sourcePitch = ((UINT64(sourceWidth) + block - 1) / block) * blockBytes;
    const UINT64 sourceRows  = (UINT64(sourceHeight) + block - 1) / block;
    const UINT64 srcOffset   = region ? region->bufferOffset : 0;
    // Check divisions before multiplying application-controlled strides.
    if (srcOffset > b->size || sourcePitch == 0 || sourceRows > UINT64_MAX / sourcePitch)
    {
        c->device->ReportError("texture-upload source buffer too small");
        return;
    }
    const UINT64 sourceSlice = sourcePitch * sourceRows;
    const UINT64 slices      = UINT64(depth) * layers;
    const UINT64 tail        = (UINT64(rows) - 1) * sourcePitch + rowBytes;
    if (tail > b->size - srcOffset || slices - 1 > (b->size - srcOffset - tail) / sourceSlice)
    {
        c->device->ReportError("texture-upload source buffer too small");
        return;
    }
    Transition(c, t, D3D12_RESOURCE_STATE_COPY_DEST, {mip, 1, baseLayer, layers});

    D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
    dstLoc.pResource                   = t->resource.Get();
    dstLoc.Type                        = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
    srcLoc.Type                        = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;

    // D3D12 buffer->texture copies need a 256-aligned row pitch and a 512-aligned source
    // offset. If the app's tightly-packed staging already meets that (typical wide
    // textures, offset 0) copy it directly; otherwise pack the rows into an aligned temp
    // upload buffer (CPU memcpy from the mapped staging) so any texture size/offset works.
    for (UINT layer = 0; layer < layers; ++layer)
    {
        dstLoc.SubresourceIndex  = mip + (baseLayer + layer) * t->mipNum;
        const UINT64 layerOffset = srcOffset + layer * depth * sourceSlice;
        const bool   aligned     = sourcePitch % D3D12_TEXTURE_DATA_PITCH_ALIGNMENT == 0 && sourcePitch <= UINT32_MAX &&
                             sourceRows == rows && layerOffset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT == 0;
        if (aligned)
        {
            if (b->heapType == D3D12_HEAP_TYPE_DEFAULT)
                TransitionBuffer(c, b, D3D12_RESOURCE_STATE_COPY_SOURCE);
            srcLoc.pResource                          = b->resource.Get();
            srcLoc.PlacedFootprint                    = fp;
            srcLoc.PlacedFootprint.Footprint.RowPitch = static_cast<UINT>(sourcePitch);
            srcLoc.PlacedFootprint.Offset             = layerOffset;
            c->list->CopyTextureRegion(&dstLoc, x, y, z, &srcLoc, nullptr);
        }
        else
        {
            if (!b->mapped)
            {
                c->device->ReportError("unaligned texture-upload requires a mapped upload buffer");
                return;
            }
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
            if (FAILED(temp->Map(0, &none, &tmp)))
            {
                c->device->ReportError("texture-upload bounce buffer map failed");
                return;
            }
            const uint8_t* srcBase = static_cast<const uint8_t*>(b->mapped) + layerOffset;
            for (UINT slice = 0; slice < depth; ++slice)
                for (UINT r = 0; r < rows; ++r)
                    std::memcpy(static_cast<uint8_t*>(tmp) + (UINT64(slice) * rows + r) * fp.Footprint.RowPitch,
                                srcBase + slice * sourceSlice + r * sourcePitch,
                                static_cast<size_t>(rowBytes));
            temp->Unmap(0, nullptr);
            srcLoc.pResource       = temp.Get();
            srcLoc.PlacedFootprint = fp; // offset 0
            c->list->CopyTextureRegion(&dstLoc, x, y, z, &srcLoc, nullptr);
            c->tempUploads.push_back(std::move(temp));
        }
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
