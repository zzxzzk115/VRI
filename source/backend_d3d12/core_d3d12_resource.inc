        // core_d3d12_resource.inc - device queries, buffers, textures, views and samplers.
        // #included into core_d3d12.cpp (one TU); see core_d3d12.cpp for the split rationale.
        // ---- queries ---------------------------------------------------
        const VriDeviceDesc* VRI_CALL  GetDeviceDesc(const VriDevice* device) { return &Dev(device)->Desc(); }
        VriFormatSupportFlags VRI_CALL GetFormatSupport(const VriDevice* device, VriFormat format)
        {
            VriFormatSupportFlags             r  = VriFormatSupport_None;
            D3D12_FEATURE_DATA_FORMAT_SUPPORT fs = {};
            // OrNone, not ToDxgiFormat: the latter's RGBA8 fallback made every unmapped
            // format report RGBA8's capabilities instead of "unsupported", which also
            // kept the DXGI_FORMAT_UNKNOWN guard below from ever firing.
            fs.Format                            = ToDxgiFormatOrNone(format).format;
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
        VriResult VRI_CALL GetVideoMemoryInfo(const VriDevice*    device,
                                              VriMemoryLocation   location,
                                              VriVideoMemoryInfo* out)
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
            b->device      = d;
            b->size        = desc->size;
            b->wantsUav    = uav;
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

            TextureD3D12* t   = new TextureD3D12 {};
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
            *out = ToHandle(t);
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
                // Array RTV for a layered target (viewType 2DArray, or > 1 layer): multiview
                // (ViewInstancing) targets one array slice per view, so the RTV must span them.
                uint32_t layers = desc->layerNum;
                if (layers == 0)
                    layers = v->texture->layerNum > desc->baseLayer ? v->texture->layerNum - desc->baseLayer : 1;
                const bool array = desc->viewType == VriTextureViewType_2DArray || layers > 1;
                if (v->texture->sampleCount > 1)
                {
                    if (array)
                    {
                        rtv.ViewDimension                    = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
                        rtv.Texture2DMSArray.FirstArraySlice = desc->baseLayer;
                        rtv.Texture2DMSArray.ArraySize       = layers;
                    }
                    else
                        rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
                }
                else if (array)
                {
                    rtv.ViewDimension                  = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
                    rtv.Texture2DArray.MipSlice        = desc->baseMip;
                    rtv.Texture2DArray.FirstArraySlice = desc->baseLayer;
                    rtv.Texture2DArray.ArraySize       = layers;
                }
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

