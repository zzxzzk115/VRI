        // core_d3d12_memory.inc - explicit placed-resource memory (AllocateMemory + Bind).
        // #included into core_d3d12.cpp (one TU); see core_d3d12.cpp for the split rationale.
        // ---- explicit memory: placed resources (AllocateMemory + Bind) -----------
        // GetBufferMemoryDesc / GetTextureMemoryDesc report the size, alignment and heap category a
        // resource needs; AllocateMemory creates an ID3D12Heap of that category; BindBufferMemory /
        // BindTextureMemory create the resource as a placed resource inside the heap. This is the
        // Vulkan-parity path (aliasing / pooling). First pass covers buffers + standard (non-MSAA)
        // textures; heap-tier mismatches and MSAA are diagnosed (never silently mis-handled).
        void VRI_CALL GetBufferMemoryDesc(const VriDevice*     device,
                                          const VriBufferDesc* desc,
                                          VriMemoryLocation    location,
                                          VriMemoryDesc*       o)
        {
            if (!o)
                return;
            *o                     = VriMemoryDesc {};
            D3D12_RESOURCE_DESC rd = {};
            rd.Dimension           = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width               = desc->size ? desc->size : 1;
            rd.Height              = 1;
            rd.DepthOrArraySize    = 1;
            rd.MipLevels           = 1;
            rd.Format              = DXGI_FORMAT_UNKNOWN;
            rd.SampleDesc.Count    = 1;
            rd.Layout              = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (desc->usage & VriBufferUsage_StorageBuffer)
                rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            const D3D12_RESOURCE_ALLOCATION_INFO ai = Dev(device)->Device()->GetResourceAllocationInfo(0, 1, &rd);
            o->size                                 = ai.SizeInBytes;
            o->alignment                            = ai.Alignment;
            o->location                             = location;
            o->memoryTypeMask                       = HeapCat_Buffers;
        }
        void VRI_CALL GetTextureMemoryDesc(const VriDevice*      device,
                                           const VriTextureDesc* desc,
                                           VriMemoryLocation     location,
                                           VriMemoryDesc*        o)
        {
            if (!o)
                return;
            *o                      = VriMemoryDesc {};
            const DxgiFormatInfo fi = ToDxgiFormat(desc->format);
            D3D12_RESOURCE_DESC  rd = {};
            rd.Dimension            = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rd.Width                = desc->width ? desc->width : 1;
            rd.Height               = desc->height ? desc->height : 1;
            rd.DepthOrArraySize     = static_cast<UINT16>(desc->layerNum ? desc->layerNum : 1);
            rd.MipLevels            = static_cast<UINT16>(desc->mipNum ? desc->mipNum : 1);
            rd.Format               = fi.format;
            rd.SampleDesc.Count     = desc->sampleNum ? desc->sampleNum : 1;
            rd.Layout               = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            // Mirror CreateTexture's typeless substitution + flags so the allocation info matches the
            // resource that BindTextureMemory will place (the clear value doesn't affect the size).
            const bool depthSampled = (desc->usage & VriTextureUsage_DepthStencilAttachment) &&
                                      (desc->usage & VriTextureUsage_ShaderResource);
            if (depthSampled)
            {
                switch (fi.format)
                {
                    case DXGI_FORMAT_D32_FLOAT:
                        rd.Format = DXGI_FORMAT_R32_TYPELESS;
                        break;
                    case DXGI_FORMAT_D16_UNORM:
                        rd.Format = DXGI_FORMAT_R16_TYPELESS;
                        break;
                    case DXGI_FORMAT_D24_UNORM_S8_UINT:
                        rd.Format = DXGI_FORMAT_R24G8_TYPELESS;
                        break;
                    default:
                        break;
                }
            }
            const bool rtds =
                (desc->usage & (VriTextureUsage_ColorAttachment | VriTextureUsage_DepthStencilAttachment)) != 0;
            if (desc->usage & VriTextureUsage_ColorAttachment)
                rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            if (desc->usage & VriTextureUsage_DepthStencilAttachment)
                rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            if (desc->usage & VriTextureUsage_ShaderResourceStorage)
                rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            const D3D12_RESOURCE_ALLOCATION_INFO ai = Dev(device)->Device()->GetResourceAllocationInfo(0, 1, &rd);
            o->size                                 = ai.SizeInBytes;
            o->alignment                            = ai.Alignment;
            o->location                             = location;
            o->memoryTypeMask                       = rtds ? HeapCat_RtDsTex : HeapCat_NonRtDsTex;
        }
        VriResult VRI_CALL AllocateMemory(VriDevice* device, const VriMemoryDesc* desc, VriMemory** out)
        {
            DeviceD3D12*     d     = Dev(device);
            D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE;
            switch (desc->memoryTypeMask)
            {
                case HeapCat_Buffers:
                    flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
                    break;
                case HeapCat_NonRtDsTex:
                    flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
                    break;
                case HeapCat_RtDsTex:
                    flags = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;
                    break;
                default:
                    d->ReportError("AllocateMemory: unknown heap category - call GetBufferMemoryDesc / "
                                   "GetTextureMemoryDesc to fill VriMemoryDesc first");
                    return VriResult_InvalidArgument;
            }
            D3D12_HEAP_DESC hd = {};
            hd.SizeInBytes     = desc->size;
            hd.Properties.Type = HeapTypeForLocation(desc->location);
            hd.Alignment       = 0; // 0 => 64KB default; MSAA (4MB) heaps are guarded out at bind
            hd.Flags           = flags;
            MemoryD3D12* m     = new MemoryD3D12 {};
            if (FAILED(d->Device()->CreateHeap(&hd, IID_PPV_ARGS(&m->heap))))
            {
                delete m;
                d->ReportError("AllocateMemory: CreateHeap failed");
                return VriResult_OutOfMemory;
            }
            m->device   = d;
            m->heapType = hd.Properties.Type;
            *out        = ToHandle(m);
            return VriResult_Success;
        }
        void VRI_CALL FreeMemory(VriMemory* memory)
        {
            if (memory)
                delete Mem(memory); // ComPtr releases the heap
        }
        VriResult VRI_CALL BindBufferMemory(VriDevice* device, VriBuffer* buffer, VriMemory* memory, uint64_t offset)
        {
            DeviceD3D12* d = Dev(device);
            BufferD3D12* b = Buf(buffer);
            MemoryD3D12* m = Mem(memory);
            if (!b->deferred)
            {
                d->ReportError("BindBufferMemory: buffer was not created with VriMemoryLocation_Undefined");
                return VriResult_InvalidArgument;
            }
            D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
            if (m->heapType == D3D12_HEAP_TYPE_UPLOAD)
                state = D3D12_RESOURCE_STATE_GENERIC_READ;
            else if (m->heapType == D3D12_HEAP_TYPE_READBACK)
                state = D3D12_RESOURCE_STATE_COPY_DEST;
            if (FAILED(d->Device()->CreatePlacedResource(
                    m->heap.Get(), offset, &b->placedDesc, state, nullptr, IID_PPV_ARGS(&b->resource))))
            {
                d->ReportError("BindBufferMemory: CreatePlacedResource failed - check heap category/tier "
                               "and that offset is 64KB-aligned");
                return VriResult_Failure;
            }
            b->heapType    = m->heapType;
            const bool uav = b->wantsUav && m->heapType == D3D12_HEAP_TYPE_DEFAULT;
            b->state       = uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : state;
            if (m->heapType != D3D12_HEAP_TYPE_DEFAULT) // persistently map upload/readback heaps
            {
                D3D12_RANGE none = {0, 0};
                b->resource->Map(0, m->heapType == D3D12_HEAP_TYPE_READBACK ? nullptr : &none, &b->mapped);
            }
            return VriResult_Success;
        }
        VriResult VRI_CALL BindTextureMemory(VriDevice* device, VriTexture* texture, VriMemory* memory, uint64_t offset)
        {
            DeviceD3D12*  d = Dev(device);
            TextureD3D12* t = Tex(texture);
            MemoryD3D12*  m = Mem(memory);
            if (!t->deferred)
            {
                d->ReportError("BindTextureMemory: texture was not created with VriMemoryLocation_Undefined");
                return VriResult_InvalidArgument;
            }
            if (t->sampleCount > 1)
            {
                // MSAA placed resources need a 4MB-aligned heap; not wired in this first pass.
                d->ReportError("BindTextureMemory: explicit placed memory for MSAA textures is not yet supported");
                return VriResult_Unsupported;
            }
            const D3D12_CLEAR_VALUE* pcv = t->hasClear ? &t->placedClear : nullptr;
            if (FAILED(d->Device()->CreatePlacedResource(m->heap.Get(),
                                                         offset,
                                                         &t->placedDesc,
                                                         D3D12_RESOURCE_STATE_COMMON,
                                                         pcv,
                                                         IID_PPV_ARGS(&t->resource))))
            {
                d->ReportError("BindTextureMemory: CreatePlacedResource failed - check heap category/tier "
                               "(RT/DS vs non-RT/DS) and that offset is 64KB-aligned");
                return VriResult_Failure;
            }
            t->state = D3D12_RESOURCE_STATE_COMMON;
            return VriResult_Success;
        }
