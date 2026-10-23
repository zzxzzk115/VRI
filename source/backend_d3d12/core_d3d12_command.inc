        // core_d3d12_command.inc - command allocation/recording, barriers, render pass, readback,
        // submission and fences. #included into core_d3d12.cpp (one TU); see core_d3d12.cpp.
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
        VriResult VRI_CALL QueueSubmit(VriQueue* queue, const VriQueueSubmitDesc* submit)
        {
            if (!queue || !core::ValidSubmitPointers(submit))
                return VriResult_InvalidArgument;
            QueueD3D12* q = Q(queue);
            for (uint32_t i = 0; i < submit->waitFenceNum; ++i)
                if (Fen(submit->waitFences[i].fence)->device != q->device)
                    return VriResult_InvalidArgument;
            for (uint32_t i = 0; i < submit->signalFenceNum; ++i)
                if (Fen(submit->signalFences[i].fence)->device != q->device)
                    return VriResult_InvalidArgument;
            for (uint32_t i = 0; i < submit->commandBufferNum; ++i)
                if (CB(submit->commandBuffers[i])->device != q->device)
                    return VriResult_InvalidArgument;
            for (uint32_t i = 0; i < submit->waitFenceNum; ++i)
            {
                const auto& wait = submit->waitFences[i];
                if (FAILED(q->queue->Wait(Fen(wait.fence)->fence.Get(), wait.value)))
                    return VriResult_DeviceLost;
            }
            std::vector<ID3D12CommandList*> lists(submit->commandBufferNum);
            for (uint32_t i = 0; i < submit->commandBufferNum; ++i)
                lists[i] = CB(submit->commandBuffers[i])->list.Get();
            if (!lists.empty())
                q->queue->ExecuteCommandLists(static_cast<UINT>(lists.size()), lists.data());
            for (uint32_t i = 0; i < submit->signalFenceNum; ++i)
            {
                const auto& signal = submit->signalFences[i];
                if (FAILED(q->queue->Signal(Fen(signal.fence)->fence.Get(), signal.value)))
                    return VriResult_DeviceLost;
            }
            return FAILED(q->device->Device()->GetDeviceRemovedReason()) ? VriResult_DeviceLost : VriResult_Success;
        }
        VriResult WaitQueueIdle(QueueD3D12* q)
        {
            if (!q->idleFence)
                return VriResult_Failure;
            const uint64_t value = ++q->idleValue;
            if (FAILED(q->queue->Signal(q->idleFence.Get(), value)))
                return VriResult_DeviceLost;
            if (q->idleFence->GetCompletedValue() == UINT64_MAX)
                return VriResult_DeviceLost;
            if (q->idleFence->GetCompletedValue() < value)
            {
                if (FAILED(q->idleFence->SetEventOnCompletion(value, q->idleEvent)))
                    return VriResult_Failure;
                if (WaitForSingleObject(q->idleEvent, INFINITE) != WAIT_OBJECT_0)
                    return VriResult_Failure;
            }
            return q->idleFence->GetCompletedValue() == UINT64_MAX ? VriResult_DeviceLost : VriResult_Success;
        }
        VriResult VRI_CALL QueueWaitIdle(VriQueue* queue)
        {
            return queue ? WaitQueueIdle(Q(queue)) : VriResult_InvalidArgument;
        }
        VriResult VRI_CALL DeviceWaitIdle(VriDevice* device)
        {
            if (!device)
                return VriResult_InvalidArgument;
            for (uint32_t type = 0; type < VriQueueType_Count; ++type)
            {
                const auto status = WaitQueueIdle(Dev(device)->GetQueue(static_cast<VriQueueType>(type)));
                if (status != VriResult_Success)
                    return status;
            }
            return VriResult_Success;
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
            *out     = ToHandle(f);
            return VriResult_Success;
        }
        void VRI_CALL DestroyFence(VriFence* fence)
        {
            if (!fence)
                return;
            FenceD3D12* f = Fen(fence);
            delete f;
        }
        uint64_t VRI_CALL GetFenceValue(VriFence* fence) { return Fen(fence)->fence->GetCompletedValue(); }
        VriResult VRI_CALL WaitForFence(VriFence* fence, uint64_t value, uint64_t timeoutNs)
        {
            if (!fence)
                return VriResult_InvalidArgument;
            FenceD3D12* f = Fen(fence);
            uint64_t    completed = f->fence->GetCompletedValue();
            if (completed == UINT64_MAX)
                return VriResult_DeviceLost;
            if (completed >= value)
                return VriResult_Success;
            if (!timeoutNs)
                return VriResult_Timeout;
            struct WaitEvent
            {
                HANDLE handle = CreateEventA(nullptr, FALSE, FALSE, nullptr);
                ~WaitEvent()
                {
                    if (handle)
                        CloseHandle(handle);
                }
            } event;
            if (!event.handle || FAILED(f->fence->SetEventOnCompletion(value, event.handle)))
                return VriResult_Failure;
            core::WaitDeadline deadline(timeoutNs);
            while (true)
            {
                const uint64_t remaining = deadline.Remaining();
                if (!remaining)
                    return VriResult_Timeout;
                const uint64_t milliseconds = remaining / 1000000 + (remaining % 1000000 != 0 ? 1 : 0);
                const DWORD    waitMs       = remaining == UINT64_MAX ?
                                                  INFINITE :
                                                  static_cast<DWORD>(std::min<uint64_t>(milliseconds, INFINITE - 1));
                const DWORD    result       = WaitForSingleObject(event.handle, waitMs);
                if (result != WAIT_OBJECT_0 && result != WAIT_TIMEOUT)
                    return VriResult_Failure;
                completed = f->fence->GetCompletedValue();
                if (completed == UINT64_MAX)
                    return VriResult_DeviceLost;
                if (completed >= value)
                    return VriResult_Success;
            }
        }

        void VRI_CALL Wait(VriFence* fence, uint64_t value)
        {
            if (WaitForFence(fence, value, UINT64_MAX) != VriResult_Success && fence)
                Fen(fence)->device->ReportError("Wait: D3D12 timeline synchronization failed");
        }
