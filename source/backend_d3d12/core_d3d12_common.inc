        // core_d3d12_common.inc - handle accessors, DXGI/state conversions, transition helpers
        // and explicit-memory heap-category helpers. #included into core_d3d12.cpp (one TU).
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
        MemoryD3D12*         Mem(VriMemory* m) { return reinterpret_cast<MemoryD3D12*>(m); }

        // Explicit-memory heap category, encoded into VriMemoryDesc::memoryTypeMask by
        // GetBufferMemoryDesc/GetTextureMemoryDesc and decoded by AllocateMemory. A heap-tier-1
        // device restricts a heap to ONE category, so it must be fixed at allocation time.
        enum HeapCategory : uint32_t
        {
            HeapCat_Buffers    = 1,
            HeapCat_NonRtDsTex = 2,
            HeapCat_RtDsTex    = 3,
        };
        D3D12_HEAP_TYPE HeapTypeForLocation(VriMemoryLocation loc)
        {
            switch (loc)
            {
                case VriMemoryLocation_HostUpload:
                case VriMemoryLocation_DeviceUpload: // no distinct ReBAR heap in D3D12; use UPLOAD (host-visible)
                    return D3D12_HEAP_TYPE_UPLOAD;
                case VriMemoryLocation_HostReadback:
                    return D3D12_HEAP_TYPE_READBACK;
                default:
                    return D3D12_HEAP_TYPE_DEFAULT;
            }
        }
        DescriptorSetD3D12* DSet(VriDescriptorSet* s) { return reinterpret_cast<DescriptorSetD3D12*>(s); }

        // The resource dimension and extent of `desc`: 1D and 3D textures keep their dimension
        // (a 3D texture's depth is its third extent); 2D textures, arrays and cubes are 2D array
        // resources with one slice per layer.
        void SetTextureExtent(D3D12_RESOURCE_DESC& rd, const VriTextureDesc* desc)
        {
            const UINT16 layers = static_cast<UINT16>(desc->layerNum);
            rd.Width            = desc->width;
            switch (desc->type)
            {
                case VriTextureType_1D:
                case VriTextureType_1DArray:
                    rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
                    rd.Height           = 1;
                    rd.DepthOrArraySize = layers;
                    break;
                case VriTextureType_3D:
                    rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
                    rd.Height           = desc->height;
                    rd.DepthOrArraySize = static_cast<UINT16>(desc->depth);
                    break;
                default:
                    rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                    rd.Height           = desc->height;
                    rd.DepthOrArraySize = layers;
                    break;
            }
        }

        // `access` decides the General layout: storage access needs UNORDERED_ACCESS, which textures
        // are never promoted to implicitly.
        D3D12_RESOURCE_STATES ToState(VriLayout layout, VriAccessFlags access)
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
                case VriLayout_ShaderResourceStorage:
                    return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                case VriLayout_CopySource:
                    return D3D12_RESOURCE_STATE_COPY_SOURCE;
                case VriLayout_CopyDestination:
                    return D3D12_RESOURCE_STATE_COPY_DEST;
                case VriLayout_Present:
                    return D3D12_RESOURCE_STATE_PRESENT;
                case VriLayout_General:
                    return (access & (VriAccess_ShaderResourceStorageRead | VriAccess_ShaderResourceStorageWrite)) ?
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS :
                               D3D12_RESOURCE_STATE_COMMON;
                default:
                    return D3D12_RESOURCE_STATE_COMMON;
            }
        }

        // Integer (UINT/SINT) color formats need ClearUnorderedAccessViewUint; everything else uses
        // the Float clear (which converts to the format). Storage textures are always typed/color.
        bool IsIntegerDxgi(DXGI_FORMAT f)
        {
            switch (f)
            {
                case DXGI_FORMAT_R8_UINT:
                case DXGI_FORMAT_R8_SINT:
                case DXGI_FORMAT_R8G8_UINT:
                case DXGI_FORMAT_R8G8_SINT:
                case DXGI_FORMAT_R8G8B8A8_UINT:
                case DXGI_FORMAT_R8G8B8A8_SINT:
                case DXGI_FORMAT_R16_UINT:
                case DXGI_FORMAT_R16_SINT:
                case DXGI_FORMAT_R16G16_UINT:
                case DXGI_FORMAT_R16G16_SINT:
                case DXGI_FORMAT_R16G16B16A16_UINT:
                case DXGI_FORMAT_R16G16B16A16_SINT:
                case DXGI_FORMAT_R32_UINT:
                case DXGI_FORMAT_R32_SINT:
                case DXGI_FORMAT_R32G32_UINT:
                case DXGI_FORMAT_R32G32_SINT:
                case DXGI_FORMAT_R32G32B32_UINT:
                case DXGI_FORMAT_R32G32B32_SINT:
                case DXGI_FORMAT_R32G32B32A32_UINT:
                case DXGI_FORMAT_R32G32B32A32_SINT:
                    return true;
                default:
                    return false;
            }
        }

        // Mips [baseMip, baseMip + mipNum) of layers [baseLayer, baseLayer + layerNum).
        struct SubresourceRange
        {
            uint32_t baseMip   = 0;
            uint32_t mipNum    = 0;
            uint32_t baseLayer = 0;
            uint32_t layerNum  = 0;
        };

        SubresourceRange WholeTexture(const TextureD3D12* t) { return {0, t->mipNum, 0, t->layerNum}; }

        // An attachment view renders to one mip of its layers.
        SubresourceRange ViewRange(const DescriptorD3D12* v) { return {v->mip, 1, v->baseLayer, v->layerNum}; }

        bool InState(const TextureD3D12* t, const SubresourceRange& r, D3D12_RESOURCE_STATES state)
        {
            for (uint32_t layer = r.baseLayer; layer < r.baseLayer + r.layerNum; ++layer)
                for (uint32_t mip = r.baseMip; mip < r.baseMip + r.mipNum; ++mip)
                    if (t->states[mip + layer * t->mipNum] != state)
                        return false;
            return true;
        }

        // Moves the subresources of `r` to `after`. A whole texture in one state takes a single
        // all-subresources barrier; otherwise each subresource that changes gets its own, for
        // both planes of a depth-stencil format.
        void Transition(CommandBufferD3D12* c, TextureD3D12* t, D3D12_RESOURCE_STATES after, const SubresourceRange& r)
        {
            const bool whole = r.baseMip == 0 && r.mipNum == t->mipNum && r.baseLayer == 0 && r.layerNum == t->layerNum;
            D3D12_RESOURCE_BARRIER b = {};
            b.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource   = t->resource.Get();
            b.Transition.StateAfter  = after;
            if (whole && InState(t, r, t->states[0]))
            {
                if (t->states[0] == after)
                    return;
                b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                b.Transition.StateBefore = t->states[0];
                c->list->ResourceBarrier(1, &b);
                t->ResetStates(after);
            }
            else
            {
                const bool planes = t->dsvFormat == DXGI_FORMAT_D24_UNORM_S8_UINT ||
                                    t->dsvFormat == DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
                const UINT subresources = t->mipNum * t->layerNum;
                std::vector<D3D12_RESOURCE_BARRIER> barriers;
                for (uint32_t layer = r.baseLayer; layer < r.baseLayer + r.layerNum; ++layer)
                    for (uint32_t mip = r.baseMip; mip < r.baseMip + r.mipNum; ++mip)
                    {
                        D3D12_RESOURCE_STATES& state = t->states[mip + layer * t->mipNum];
                        if (state == after)
                            continue;
                        b.Transition.StateBefore = state;
                        b.Transition.Subresource = mip + layer * t->mipNum;
                        barriers.push_back(b);
                        if (planes)
                        {
                            b.Transition.Subresource += subresources;
                            barriers.push_back(b);
                        }
                        state = after;
                    }
                if (barriers.empty())
                    return;
                c->list->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
            }
            if (t->concurrent && after != D3D12_RESOURCE_STATE_COMMON &&
                std::find(c->concurrentTextures.begin(), c->concurrentTextures.end(), t) == c->concurrentTextures.end())
                c->concurrentTextures.push_back(t);
        }

        void Transition(CommandBufferD3D12* c, TextureD3D12* t, D3D12_RESOURCE_STATES after)
        {
            Transition(c, t, after, WholeTexture(t));
        }

        // Compute lists accept no graphics-only states: a read-only texture there is a
        // non-pixel shader resource.
        D3D12_RESOURCE_STATES ForList(const CommandBufferD3D12* c, D3D12_RESOURCE_STATES state)
        {
            if (c->allocator->listType != D3D12_COMMAND_LIST_TYPE_COMPUTE)
                return state;
            const D3D12_RESOURCE_STATES stripped = state & ~D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            return stripped != state && stripped == D3D12_RESOURCE_STATE_COMMON ?
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE :
                       stripped;
        }

        // Orders unordered-access writes before later unordered-access reads or writes of `resource`;
        // a transition cannot, because both sides are in the same state.
        void UavBarrier(CommandBufferD3D12* c, ID3D12Resource* resource)
        {
            D3D12_RESOURCE_BARRIER b = {};
            b.Type                   = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            b.UAV.pResource          = resource;
            c->list->ResourceBarrier(1, &b);
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
            if (a & VriAccess_IndirectBufferRead)
                return D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT; // ExecuteIndirect arg + count buffers
            return D3D12_RESOURCE_STATE_COMMON;
        }

