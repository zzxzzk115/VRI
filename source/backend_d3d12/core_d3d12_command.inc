        // core_d3d12_command.inc - command allocators/buffers, command recording, draws/dispatch,
        // copies, barriers, submission and fences for the D3D12 backend. #included into
        // core_d3d12.cpp (one translation unit / anonymous namespace); not compiled on its own.
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
            c->rtvCount         = 0;
            c->boundPipeline    = nullptr;
            c->boundLayout      = nullptr;
            c->boundSrvHeap     = nullptr; // no heap bound until the first CmdSetDescriptorSet
            c->boundSamplerHeap = nullptr;
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
            CommandBufferD3D12* c = CB(cmd);
            TextureD3D12*       t = Tex(src);
            BufferD3D12*        b = Buf(dst);
            // Subresource = mip + arrayLayer * mipNum (baseLayer selects the array slice), matching
            // the upload path - so a per-layer readback of an array target reads the right slice.
            const UINT sub = (region ? region->texture.mip : 0) + (region ? region->texture.baseLayer : 0) * t->mipNum;
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
            CB(cmd)->list->DrawInstanced(
                d->vertexNum, d->instanceNum ? d->instanceNum : 1, d->baseVertex, d->baseInstance);
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
        void VRI_CALL CmdDrawIndexedIndirect(VriCommandBuffer* cmd,
                                             VriBuffer*        buffer,
                                             uint64_t          offset,
                                             uint32_t          drawNum,
                                             uint32_t          stride)
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
            c->list->ExecuteIndirect(
                c->device->DispatchSignature(), 1, Buf(buffer)->resource.Get(), offset, nullptr, 0);
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
        void VRI_CALL CmdCopyTexture(VriCommandBuffer*         cmd,
                                     VriTexture*               dst,
                                     VriTexture*               src,
                                     const VriTextureCopyDesc* region)
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
        void VRI_CALL SetDebugName(void*, const char*) {}
