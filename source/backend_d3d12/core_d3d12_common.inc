        // core_d3d12_common.inc - handle accessors, heap/state conversion helpers and shared
        // utilities for the D3D12 backend. #included into core_d3d12.cpp (one translation unit /
        // anonymous namespace); not compiled on its own. See core_d3d12.cpp for the split rationale.
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

        D3D12_RESOURCE_STATES ToState(VriLayout layout)
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
                case VriLayout_CopySource:
                    return D3D12_RESOURCE_STATE_COPY_SOURCE;
                case VriLayout_CopyDestination:
                    return D3D12_RESOURCE_STATE_COPY_DEST;
                case VriLayout_Present:
                    return D3D12_RESOURCE_STATE_PRESENT;
                case VriLayout_General:
                    return D3D12_RESOURCE_STATE_COMMON;
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

        void Transition(CommandBufferD3D12* c, TextureD3D12* t, D3D12_RESOURCE_STATES after)
        {
            if (t->state == after)
                return;
            D3D12_RESOURCE_BARRIER b = {};
            b.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource   = t->resource.Get();
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            b.Transition.StateBefore = t->state;
            b.Transition.StateAfter  = after;
            c->list->ResourceBarrier(1, &b);
            t->state = after;
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
