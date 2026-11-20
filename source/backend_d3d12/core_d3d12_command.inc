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
    const D3D12_COMMAND_LIST_TYPE listType = kListType[type < VriQueueType_Count ? type : VriQueueType_Graphics];
    CommandAllocatorD3D12*        a        = new CommandAllocatorD3D12 {};
    a->device                              = d;
    a->listType                            = listType;
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
    if (!a)
        return;
    // Each command list holds a reference to the device, so a list left behind keeps the device alive.
    for (CommandBufferD3D12* c : Alloc(a)->buffers)
        delete c;
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
    a->buffers.push_back(c);
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
    c->concurrentTextures.clear();
    return VriResult_Success;
}
VriResult VRI_CALL EndCommandBuffer(VriCommandBuffer* cmd)
{
    CommandBufferD3D12* c = CB(cmd);
    for (TextureD3D12* t : c->concurrentTextures)
        Transition(c, t, D3D12_RESOURCE_STATE_COMMON);
    c->concurrentTextures.clear();
    return SUCCEEDED(c->list->Close()) ? VriResult_Success : VriResult_Failure;
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
        if (!tb.texture)
            continue;
        TextureD3D12*               t     = reinterpret_cast<TextureD3D12*>(tb.texture);
        const D3D12_RESOURCE_STATES after = ForList(c, ToState(tb.after.layout, tb.after.access));
        // 0 mips or layers select every remaining one; the range is clamped to the texture.
        SubresourceRange r;
        r.baseMip   = (std::min)(tb.baseMip, t->mipNum);
        r.mipNum    = (std::min)(tb.mipNum ? tb.mipNum : t->mipNum, t->mipNum - r.baseMip);
        r.baseLayer = (std::min)(tb.baseLayer, t->layerNum);
        r.layerNum  = (std::min)(tb.layerNum ? tb.layerNum : t->layerNum, t->layerNum - r.baseLayer);
        if (after == D3D12_RESOURCE_STATE_UNORDERED_ACCESS && InState(t, r, after))
            UavBarrier(c, t->resource.Get());
        else
            Transition(c, t, after, r);
    }
    for (uint32_t i = 0; i < g->bufferNum; ++i)
    {
        const VriBufferBarrierDesc& bb = g->buffers[i];
        if (!bb.buffer)
            continue;
        BufferD3D12*                b     = reinterpret_cast<BufferD3D12*>(bb.buffer);
        const D3D12_RESOURCE_STATES after = BufferStateForAccess(bb.after.access);
        if (after == D3D12_RESOURCE_STATE_UNORDERED_ACCESS && b->state == after)
            UavBarrier(c, b->resource.Get());
        else
            TransitionBuffer(c, b, after);
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
        Transition(c, t, D3D12_RESOURCE_STATE_RENDER_TARGET, ViewRange(v)); // usually already RT
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
        Transition(c, const_cast<TextureD3D12*>(dv->texture), D3D12_RESOURCE_STATE_DEPTH_WRITE, ViewRange(dv));
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
        rects[i] = {r[i].x, r[i].y, r[i].x + static_cast<LONG>(r[i].width), r[i].y + static_cast<LONG>(r[i].height)};
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
    const UINT mip = region ? region->texture.mip : 0;
    const UINT sub = mip + (region ? region->texture.baseLayer : 0) * t->mipNum;
    // The region's box within the mip; zero extents mean the rest of the mip, as on Vulkan.
    const UINT mipWidth  = (std::max)(1u, t->width >> mip);
    const UINT mipHeight = (std::max)(1u, t->height >> mip);
    const UINT mipDepth  = (std::max)(1u, t->depth >> mip);
    D3D12_BOX  box       = {};
    box.left             = region ? static_cast<UINT>(region->texture.x) : 0;
    box.top              = region ? static_cast<UINT>(region->texture.y) : 0;
    box.front            = region ? static_cast<UINT>(region->texture.z) : 0;
    box.right            = box.left + (region && region->texture.width ? region->texture.width : mipWidth - box.left);
    box.bottom           = box.top + (region && region->texture.height ? region->texture.height : mipHeight - box.top);
    box.back             = box.front + (region && region->texture.depth ? region->texture.depth : mipDepth - box.front);
    if (box.right > mipWidth || box.bottom > mipHeight || box.back > mipDepth)
    {
        c->device->ReportError("texture-readback region out of bounds");
        return;
    }
    // The footprint describes the box alone: the buffer holds only what is copied.
    D3D12_RESOURCE_DESC rd                      = t->resource->GetDesc();
    rd.Width                                    = box.right - box.left;
    rd.Height                                   = box.bottom - box.top;
    rd.DepthOrArraySize                         = static_cast<UINT16>(box.back - box.front);
    rd.MipLevels                                = 1;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp       = {};
    UINT                               rows     = 0;
    UINT64                             rowBytes = 0, total = 0;
    c->device->Device()->GetCopyableFootprints(
        &rd, 0, 1, region ? region->bufferOffset : 0, &fp, &rows, &rowBytes, &total);
    // The buffer layout the caller asked for: rows bufferRowLength texels apart, slices
    // bufferImageHeight rows apart. Zero keeps the tight layout at D3D12's 256-byte row pitch.
    const UINT block = t->texelSize == 0 ? 4 : 1; // compressed formats copy 4x4 blocks
    if (region && region->bufferRowLength)
    {
        const UINT64 blockBytes = rowBytes / ((fp.Footprint.Width + block - 1) / block);
        const UINT64 pitch      = (UINT64(region->bufferRowLength) + block - 1) / block * blockBytes;
        if (region->bufferRowLength < fp.Footprint.Width || region->bufferRowLength % block ||
            pitch % D3D12_TEXTURE_DATA_PITCH_ALIGNMENT || pitch > UINT32_MAX)
        {
            c->device->ReportError("texture-readback bufferRowLength must cover the region and give a row pitch "
                                   "that is a multiple of 256 bytes");
            return;
        }
        fp.Footprint.RowPitch = static_cast<UINT>(pitch);
    }
    if (region && region->bufferImageHeight)
    {
        if (region->bufferImageHeight < fp.Footprint.Height || region->bufferImageHeight % block)
        {
            c->device->ReportError("texture-readback bufferImageHeight must cover the region");
            return;
        }
        fp.Footprint.Height = region->bufferImageHeight;
    }
    const UINT64 sliceBytes = UINT64(fp.Footprint.RowPitch) * ((fp.Footprint.Height + block - 1) / block);
    const UINT64 needed = sliceBytes * (fp.Footprint.Depth - 1) + UINT64(fp.Footprint.RowPitch) * (rows - 1) + rowBytes;
    if (fp.Offset > b->size || needed > b->size - fp.Offset)
    {
        c->device->ReportError("texture-readback destination buffer too small");
        return;
    }
    const bool whole = box.left == 0 && box.top == 0 && box.front == 0 && box.right == mipWidth &&
                       box.bottom == mipHeight && box.back == mipDepth;
    Transition(c,
               t,
               D3D12_RESOURCE_STATE_COPY_SOURCE,
               {region ? region->texture.mip : 0, 1, region ? region->texture.baseLayer : 0, 1});
    D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
    dstLoc.pResource                   = b->resource.Get();
    dstLoc.Type                        = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLoc.PlacedFootprint             = fp;
    D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
    srcLoc.pResource                   = t->resource.Get();
    srcLoc.Type                        = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLoc.SubresourceIndex            = sub;
    // Depth-stencil copies must cover the whole subresource, so only a partial box is passed.
    c->list->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, whole ? nullptr : &box);
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
    *out = ToHandle(f);
    return VriResult_Success;
}
void VRI_CALL DestroyFence(VriFence* fence)
{
    if (!fence)
        return;
    FenceD3D12* f = Fen(fence);
    delete f;
}
uint64_t VRI_CALL  GetFenceValue(VriFence* fence) { return Fen(fence)->fence->GetCompletedValue(); }
VriResult VRI_CALL WaitForFence(VriFence* fence, uint64_t value, uint64_t timeoutNs)
{
    if (!fence)
        return VriResult_InvalidArgument;
    FenceD3D12* f         = Fen(fence);
    uint64_t    completed = f->fence->GetCompletedValue();
    if (completed == UINT64_MAX)
        return VriResult_DeviceLost;
    if (completed >= value)
        return VriResult_Success;
    if (!timeoutNs)
        return VriResult_Timeout;
    if (timeoutNs != UINT64_MAX)
    {
        // D3D12 cannot cancel a timed-out event registration. Finite waits
        // poll instead of retaining a kernel registration until a future signal.
        core::WaitDeadline deadline(timeoutNs);
        while (deadline.Remaining())
        {
            deadline.Pause();
            completed = f->fence->GetCompletedValue();
            if (completed == UINT64_MAX)
                return VriResult_DeviceLost;
            if (completed >= value)
                return VriResult_Success;
        }
        return VriResult_Timeout;
    }
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
    if (WaitForSingleObject(event.handle, INFINITE) != WAIT_OBJECT_0)
        return VriResult_Failure;
    completed = f->fence->GetCompletedValue();
    if (completed == UINT64_MAX)
        return VriResult_DeviceLost;
    return completed >= value ? VriResult_Success : VriResult_Failure;
}

void VRI_CALL Wait(VriFence* fence, uint64_t value)
{
    if (WaitForFence(fence, value, UINT64_MAX) != VriResult_Success && fence)
        Fen(fence)->device->ReportError("Wait: D3D12 timeline synchronization failed");
}
