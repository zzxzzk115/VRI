// core_d3d12_resource.inc - device queries, buffers, textures, views and samplers.
// #included into core_d3d12.cpp (one TU); see core_d3d12.cpp for the split rationale.
// ---- queries ---------------------------------------------------
const VriDeviceDesc* VRI_CALL  GetDeviceDesc(const VriDevice* device) { return &Dev(device)->Desc(); }
VriFormatSupportFlags VRI_CALL GetFormatSupport(const VriDevice* device, VriFormat format)
{
    VriFormatSupportFlags             r  = VriFormatSupport_None;
    D3D12_FEATURE_DATA_FORMAT_SUPPORT fs = {};
    fs.Format                            = ToDxgiFormat(format).format;
    if (fs.Format == DXGI_FORMAT_UNKNOWN ||
        FAILED(Dev(device)->Device()->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs))))
        return r;
    const D3D12_FORMAT_SUPPORT1 s = fs.Support1;
    if (s & (D3D12_FORMAT_SUPPORT1_TEXTURE2D | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE))
        r |= VriFormatSupport_Texture;
    if (s & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)
        r |= VriFormatSupport_StorageTexture;
    if (s & D3D12_FORMAT_SUPPORT1_RENDER_TARGET)
        r |= VriFormatSupport_ColorAttachment;
    if (s & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL)
        r |= VriFormatSupport_DepthStencil;
    if (s & D3D12_FORMAT_SUPPORT1_BLENDABLE)
        r |= VriFormatSupport_Blend;
    if (s & D3D12_FORMAT_SUPPORT1_IA_VERTEX_BUFFER)
        r |= VriFormatSupport_VertexBuffer;
    return r;
}
VriResult VRI_CALL GetQueue(VriDevice* device, VriQueueType type, uint32_t, VriQueue** outQueue)
{
    if (type >= VriQueueType_Count)
        return VriResult_InvalidArgument;
    *outQueue = ToHandle(Dev(device)->GetQueue(type));
    return VriResult_Success;
}
VriResult VRI_CALL GetVideoMemoryInfo(const VriDevice* device, VriMemoryLocation location, VriVideoMemoryInfo* out)
{
    if (!out)
        return VriResult_InvalidArgument;
    IDXGIAdapter3* a = Dev(device)->Adapter();
    if (!a)
        return VriResult_Unsupported;
    const DXGI_MEMORY_SEGMENT_GROUP seg =
        (location == VriMemoryLocation_Device || location == VriMemoryLocation_DeviceUpload) ?
            DXGI_MEMORY_SEGMENT_GROUP_LOCAL :
            DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL;
    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    if (FAILED(a->QueryVideoMemoryInfo(0, seg, &info)))
        return VriResult_Failure;
    out->budget = info.Budget;
    out->usage  = info.CurrentUsage;
    return VriResult_Success;
}

// ---- resources -------------------------------------------------
// D3D12's own accounting for what a resource costs, so the figure includes alignment and
// tiling exactly as the Vulkan backend's VMA query does. GetResourceAllocationInfo takes
// the desc, which is available on both the committed and the deferred/placed paths.
[[nodiscard]] uint64_t AllocatedBytes(DeviceD3D12* d, const D3D12_RESOURCE_DESC& rd)
{
    const D3D12_RESOURCE_ALLOCATION_INFO info = d->Device()->GetResourceAllocationInfo(0, 1, &rd);
    return info.SizeInBytes == UINT64_MAX ? 0 : info.SizeInBytes;
}

VriResult VRI_CALL CreateBuffer(VriDevice* device, const VriBufferDesc* desc, VriBuffer** out)
{
    DeviceD3D12* d = Dev(device);
    if (core::RejectDesc(*d, core::CheckBufferDesc("CreateBuffer", *desc)))
    {
        *out = nullptr;
        return VriResult_InvalidArgument;
    }
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
    const bool                  uav = (heap == D3D12_HEAP_TYPE_DEFAULT) && (desc->usage & VriBufferUsage_StorageBuffer);
    const D3D12_RESOURCE_STATES trackedState = uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : state;

    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type                  = heap;
    D3D12_RESOURCE_DESC rd   = {};
    rd.Dimension             = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width                 = desc->size;
    rd.Height                = 1;
    rd.DepthOrArraySize      = 1;
    rd.MipLevels             = 1;
    rd.Format                = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count      = 1;
    rd.Layout                = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (uav)
        rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    BufferD3D12* b     = new BufferD3D12 {};
    b->device          = d;
    b->size            = desc->size;
    b->structureStride = desc->structureStride;
    b->wantsUav        = uav;
    // Explicit-memory path: create the buffer UNBOUND and defer the placed-resource creation
    // to BindBufferMemory (D3D12 has no memory-less resource, so we stash the desc instead).
    if (desc->memoryLocation == VriMemoryLocation_Undefined)
    {
        b->deferred   = true;
        b->placedDesc = rd;
        *out          = ToHandle(b);
        return VriResult_Success;
    }
    if (FAILED(d->Device()->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&b->resource))))
    {
        delete b;
        d->ReportError("CreateCommittedResource (buffer) failed");
        return VriResult_Failure;
    }
    b->heapType = heap;
    b->state    = trackedState;
    if (heap != D3D12_HEAP_TYPE_DEFAULT) // persistently map upload/readback heaps
    {
        D3D12_RANGE none = {0, 0};
        b->resource->Map(0, heap == D3D12_HEAP_TYPE_READBACK ? nullptr : &none, &b->mapped);
    }
    {
        VriObjectInfo info = MakeObjectInfo(VriObjectType_Buffer);
        info.memoryBytes   = AllocatedBytes(d, rd);
        info.width         = static_cast<uint32_t>(desc->size);
        d->Objects().Track(ToHandle(b), info);
    }
    *out = ToHandle(b);
    return VriResult_Success;
}
void VRI_CALL DestroyBuffer(VriBuffer* buffer)
{
    if (buffer)
    {
        Buf(buffer)->device->Objects().Untrack(buffer);
        delete Buf(buffer);
    }
}
void* VRI_CALL MapBuffer(VriBuffer* buffer, uint64_t offset, uint64_t)
{
    BufferD3D12* b = Buf(buffer);
    return b->mapped ? static_cast<char*>(b->mapped) + offset : nullptr;
}
void VRI_CALL     UnmapBuffer(VriBuffer*) {} // persistent mapping; nothing to do
VriResult VRI_CALL MapBufferAsync(VriBuffer*              buffer,
                                  uint64_t                offset,
                                  uint64_t                size,
                                  VriBufferMappedCallback callback,
                                  void*                   userData)
{
    return core::MapBufferNow(MapBuffer, buffer, offset, size, callback, userData);
}
uint64_t VRI_CALL GetBufferDeviceAddress(const VriBuffer*) { return 0; }

VriResult VRI_CALL CreateTexture(VriDevice* device, const VriTextureDesc* desc, VriTexture** out)
{
    DeviceD3D12* d = Dev(device);
    if (core::RejectDesc(*d, core::CheckTextureDesc("CreateTexture", *desc)))
    {
        *out = nullptr;
        return VriResult_InvalidArgument;
    }
    const DxgiFormatInfo fi = ToDxgiFormat(desc->format);
    if (fi.format == DXGI_FORMAT_UNKNOWN)
    {
        *out = nullptr;
        d->ReportError("CreateTexture: unmapped DXGI format");
        return VriResult_Unsupported;
    }
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type                  = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd   = {};
    SetTextureExtent(rd, desc);
    rd.MipLevels        = static_cast<UINT16>(desc->mipNum);
    rd.Format           = fi.format;
    rd.SampleDesc.Count = desc->sampleNum;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    // A depth target that is ALSO sampled (shadow map) must use a TYPELESS resource so it
    // can carry both a typed DSV and a typed SRV (D3D12 forbids a D*_ format SRV directly).
    DXGI_FORMAT dsvFormat = fi.format, srvFormat = fi.format;
    const bool  depthSampled =
        (desc->usage & VriTextureUsage_DepthStencilAttachment) && (desc->usage & VriTextureUsage_ShaderResource);
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
            case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
                rd.Format = DXGI_FORMAT_R32G8X24_TYPELESS;
                srvFormat = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
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
    t->device       = d;
    t->format       = rd.Format;
    t->dsvFormat    = dsvFormat;
    t->srvFormat    = srvFormat;
    t->texelSize    = fi.texelSize;
    t->dimension    = rd.Dimension;
    t->width        = static_cast<uint32_t>(rd.Width);
    t->height       = rd.Height;
    t->depth        = rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? rd.DepthOrArraySize : 1u;
    t->mipNum       = rd.MipLevels;
    t->layerNum     = rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1u : rd.DepthOrArraySize;
    t->ResetStates(D3D12_RESOURCE_STATE_COMMON);
    t->sampleCount    = rd.SampleDesc.Count;
    t->isRenderTarget = (desc->usage & VriTextureUsage_ColorAttachment) != 0;
    t->isDepthStencil = (desc->usage & VriTextureUsage_DepthStencilAttachment) != 0;
    t->concurrent     = (desc->usage & VriTextureUsage_ConcurrentQueues) != 0;
    // Explicit-memory path: create the texture UNBOUND and defer the placed-resource creation
    // to BindTextureMemory (stash the desc + optional fast-clear value).
    if (desc->memoryLocation == VriMemoryLocation_Undefined)
    {
        t->deferred   = true;
        t->placedDesc = rd;
        t->hasClear   = pcv != nullptr;
        if (pcv)
            t->placedClear = cv;
        *out = ToHandle(t);
        return VriResult_Success;
    }
    if (FAILED(d->Device()->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, pcv, IID_PPV_ARGS(&t->resource))))
    {
        delete t;
        d->ReportError("CreateCommittedResource (texture) failed");
        return VriResult_Failure;
    }
    {
        VriObjectInfo info = MakeObjectInfo(VriObjectType_Texture);
        info.memoryBytes   = AllocatedBytes(d, rd);
        info.width         = t->width;
        info.height        = t->height;
        info.depth         = t->depth;
        info.mipNum        = t->mipNum;
        info.layerNum      = t->layerNum;
        info.format        = desc->format;
        d->Objects().Track(ToHandle(t), info);
    }
    *out = ToHandle(t);
    return VriResult_Success;
}
void VRI_CALL DestroyTexture(VriTexture* texture)
{
    if (texture)
    {
        Tex(texture)->device->Objects().Untrack(texture);
        delete Tex(texture);
    }
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
    // 0 selects every remaining mip or layer.
    v->mipNum    = desc->mipNum ? desc->mipNum : v->texture->mipNum - desc->baseMip;
    v->baseLayer = desc->baseLayer;
    v->layerNum  = desc->layerNum ? desc->layerNum : v->texture->layerNum - desc->baseLayer;
    // The view records the texture; it serves as an attachment (its RTV/DSV is made the first time it
    // is one, DeviceD3D12::AttachmentView) and/or an SRV source (the SRV is created into a descriptor
    // set's heap slot at UpdateDescriptorRanges).
    v->kind       = DescriptorD3D12::Kind::TextureRtv;
    v->cpuIsDepth = v->texture->isDepthStencil;

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
    if (!descriptor)
        return;
    DescriptorD3D12* v = Desc(descriptor);
    if (v->cpu.ptr)
        v->device->FreeAttachmentView(v->cpuIsDepth, v->cpu);
    delete v;
}
