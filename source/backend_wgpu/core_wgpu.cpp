// core_wgpu.cpp - WebGPU (wgpu-native) implementation of VriCoreInterface.
//
// Triangle vertical slice: resources, texture views, render pipeline from WGSL,
// command recording (render pass), copy/readback, queue submit, and emulated
// timeline fences. WebGPU auto-synchronizes, so CmdBarrier is a no-op and there
// are no resource layouts. Explicit memory and descriptor pools/sets are stubbed
// (WebGPU has no memory objects; descriptor sets = bind groups, landing later).

#include "core_wgpu.h"
#include "conversions_wgpu.h"
#include "device_wgpu.h"
#include "objects_wgpu.h"

#include "wgpu_native.h" // webgpu.h + native-only poll helpers / callback mode

#include <cstring>
#include <string>
#include <vector>

namespace vri::wgpu
{
    namespace
    {
        inline DeviceWGPU*           Dev(VriDevice* h) { return reinterpret_cast<DeviceWGPU*>(h); }
        inline const DeviceWGPU*     Dev(const VriDevice* h) { return reinterpret_cast<const DeviceWGPU*>(h); }
        inline QueueWGPU*            Q(VriQueue* h) { return reinterpret_cast<QueueWGPU*>(h); }
        inline CommandAllocatorWGPU* CA(VriCommandAllocator* h) { return reinterpret_cast<CommandAllocatorWGPU*>(h); }
        inline CommandBufferWGPU*    CB(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferWGPU*>(h); }
        inline BufferWGPU*           Buf(VriBuffer* h) { return reinterpret_cast<BufferWGPU*>(h); }
        inline TextureWGPU*          Tex(VriTexture* h) { return reinterpret_cast<TextureWGPU*>(h); }
        inline DescriptorWGPU*       Desc(VriDescriptor* h) { return reinterpret_cast<DescriptorWGPU*>(h); }
        inline PipelineLayoutWGPU*   PL(VriPipelineLayout* h) { return reinterpret_cast<PipelineLayoutWGPU*>(h); }
        inline const PipelineLayoutWGPU* PLc(const VriPipelineLayout* h)
        {
            return reinterpret_cast<const PipelineLayoutWGPU*>(h);
        }
        inline PipelineWGPU*       Pipe(VriPipeline* h) { return reinterpret_cast<PipelineWGPU*>(h); }
        inline FenceWGPU*          Fen(VriFence* h) { return reinterpret_cast<FenceWGPU*>(h); }
        inline DescriptorPoolWGPU* DPool(VriDescriptorPool* h) { return reinterpret_cast<DescriptorPoolWGPU*>(h); }
        inline DescriptorSetWGPU*  DSet(VriDescriptorSet* h) { return reinterpret_cast<DescriptorSetWGPU*>(h); }

        inline WGPUStringView SV(const char* s) { return WGPUStringView {s, WGPU_STRLEN}; }

        WGPUShaderStage ToWgpuShaderStage(VriShaderStageFlags s)
        {
            WGPUShaderStage r = WGPUShaderStage_None;
            if (s & VriShaderStage_Vertex)
                r |= WGPUShaderStage_Vertex;
            if (s & VriShaderStage_Fragment)
                r |= WGPUShaderStage_Fragment;
            if (s & VriShaderStage_Compute)
                r |= WGPUShaderStage_Compute;
            return r;
        }

        uint32_t TexelSize(VriFormat f)
        {
            switch (f)
            {
                case VriFormat_R8_UNORM:
                case VriFormat_R8_UINT:
                case VriFormat_R8_SNORM:
                case VriFormat_R8_SINT:
                    return 1;
                case VriFormat_RG8_UNORM:
                case VriFormat_R16_SFLOAT:
                case VriFormat_R16_UNORM:
                case VriFormat_D16_UNORM:
                    return 2;
                case VriFormat_RGBA8_UNORM:
                case VriFormat_RGBA8_SRGB:
                case VriFormat_BGRA8_UNORM:
                case VriFormat_BGRA8_SRGB:
                case VriFormat_RG16_SFLOAT:
                case VriFormat_R32_SFLOAT:
                case VriFormat_R32_UINT:
                case VriFormat_RGB10A2_UNORM:
                case VriFormat_D32_SFLOAT:
                    return 4;
                case VriFormat_RGBA16_SFLOAT:
                case VriFormat_RG32_SFLOAT:
                    return 8;
                case VriFormat_RGBA32_SFLOAT:
                    return 16;
                default:
                    return 4;
            }
        }

        // Slang emits a push-constant block as an *undecorated* `var<uniform>` (no @group/
        // @binding), which is invalid WGSL. WebGPU has no push constants, so VRI binds the
        // data as a uniform in a reserved bind group; inject that group/binding here onto the
        // one undecorated uniform var.
        std::string InjectPushBinding(const char* wgsl, uint32_t group)
        {
            std::string       s(wgsl);
            const std::string needle = "var<uniform>";
            for (size_t pos = s.find(needle); pos != std::string::npos; pos = s.find(needle, pos))
            {
                const size_t nl        = s.rfind('\n', pos);
                const size_t lineStart = (nl == std::string::npos) ? 0 : nl + 1;
                if (s.find("@binding", lineStart) >= pos) // no @binding before it on this line -> push constant
                {
                    const std::string dec = "@group(" + std::to_string(group) + ") @binding(0) ";
                    s.insert(pos, dec);
                    pos += dec.size() + needle.size();
                }
                else
                    pos += needle.size();
            }
            return s;
        }

        WGPUShaderModule MakeWgslModule(WGPUDevice device, const void* bytecode, int pushGroup = -1)
        {
            std::string patched;
            const char* code = static_cast<const char*>(bytecode);
            if (pushGroup >= 0)
            {
                patched = InjectPushBinding(code, static_cast<uint32_t>(pushGroup));
                code    = patched.c_str();
            }
            WGPUShaderSourceWGSL src     = {};
            src.chain.sType              = WGPUSType_ShaderSourceWGSL;
            src.code                     = SV(code);
            WGPUShaderModuleDescriptor d = {};
            d.nextInChain                = &src.chain;
            return wgpuDeviceCreateShaderModule(device, &d);
        }

        // ---- queries -------------------------------------------------------
        const VriDeviceDesc* VRI_CALL GetDeviceDesc(const VriDevice* device) { return &Dev(device)->Desc(); }

        // Answered from the same tables that decide what this backend actually does.
        // The previous reply claimed Texture|VertexBuffer for every format including
        // ones ToWgpuFormat maps to Undefined, gave depth formats ColorAttachment|Blend,
        // and never set DepthStencil at all - so the usual "probe a depth format, else
        // fall back" pattern got no usable answer here. Report per format instead.
        VriFormatSupportFlags VRI_CALL GetFormatSupport(const VriDevice*, VriFormat format)
        {
            VriFormatSupportFlags r = VriFormatSupport_None;

            // Depth32FloatStencil8 is behind the optional depth32float-stencil8
            // feature, and DeviceWGPU::Init requests only the timestamp ones - so
            // this device cannot create the format for ANY usage, however capable
            // the adapter is. It has to drop out here rather than lower down: a
            // caller probing for a sampled texture would act on a lone Texture bit
            // just as readily as on DepthStencil, and hit the same validation
            // failure at CreateTexture. Unusable is unusable; report nothing.
            const bool usable =
                ToWgpuFormat(format) != WGPUTextureFormat_Undefined && format != VriFormat_D32_SFLOAT_S8_UINT;

            if (usable)
            {
                r |= VriFormatSupport_Texture; // sampleable, depth included (shadow maps)
                if (vri::FormatIsDepthStencil(format))
                {
                    r |= VriFormatSupport_DepthStencil;
                }
                else
                {
                    // Every color format this backend maps is renderable in core
                    // WebGPU, but blending is narrower: integer targets are never
                    // blendable, and the 32-bit float ones only become blendable
                    // with the optional float32-blendable feature, which
                    // DeviceWGPU::Init does not request.
                    r |= VriFormatSupport_ColorAttachment;
                    switch (format)
                    {
                        case VriFormat_R8_UINT:
                        case VriFormat_R32_UINT:
                        case VriFormat_R32_SFLOAT:
                        case VriFormat_RG32_SFLOAT:
                        case VriFormat_RGBA32_SFLOAT:
                            break;
                        default:
                            r |= VriFormatSupport_Blend;
                            break;
                    }
                    // No StorageTexture: CreatePipelineLayout pins every
                    // storage-texture binding to RGBA8Unorm, so a view of any other
                    // format fails bind-group validation. Advertising the bit needs
                    // that layout to carry the range's real format first.
                }
            }

            if (HasWgpuVertexFormat(format))
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
        // WebGPU exposes no memory-budget query; the implementation manages residency internally.
        VriResult VRI_CALL GetVideoMemoryInfo(const VriDevice*, VriMemoryLocation, VriVideoMemoryInfo*)
        {
            return VriResult_Unsupported;
        }

        // ---- command allocation / lifecycle --------------------------------
        VriResult VRI_CALL CreateCommandAllocator(VriDevice* device, VriQueueType, VriCommandAllocator** out)
        {
            *out = ToHandle(new CommandAllocatorWGPU {Dev(device)});
            return VriResult_Success;
        }
        void VRI_CALL ResetCommandAllocator(VriCommandAllocator*) {}
        void VRI_CALL DestroyCommandAllocator(VriCommandAllocator* a) { delete CA(a); }

        VriResult VRI_CALL CreateCommandBuffer(VriCommandAllocator* allocator, VriCommandBuffer** out)
        {
            *out = ToHandle(
                new CommandBufferWGPU {CA(allocator)->device, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});
            return VriResult_Success;
        }

        VriResult VRI_CALL BeginCommandBuffer(VriCommandBuffer* cmd)
        {
            CommandBufferWGPU*           c = CB(cmd);
            WGPUCommandEncoderDescriptor d = {};
            for (WGPUBuffer t : c->tempUploads)
                wgpuBufferRelease(t); // prior frame's bounce buffers are done
            c->tempUploads.clear();
            c->encoder     = wgpuDeviceCreateCommandEncoder(c->device->Device(), &d);
            c->pass        = nullptr;
            c->computePass = nullptr;
            c->finished    = nullptr;
            return c->encoder ? VriResult_Success : VriResult_Failure;
        }

        VriResult VRI_CALL EndCommandBuffer(VriCommandBuffer* cmd)
        {
            CommandBufferWGPU* c = CB(cmd);
            if (c->computePass) // a trailing dispatch with no following encoder-level op
            {
                wgpuComputePassEncoderEnd(c->computePass);
                wgpuComputePassEncoderRelease(c->computePass);
                c->computePass = nullptr;
            }
            WGPUCommandBufferDescriptor d = {};
            c->finished                   = wgpuCommandEncoderFinish(c->encoder, &d);
            wgpuCommandEncoderRelease(c->encoder);
            c->encoder = nullptr;
            return c->finished ? VriResult_Success : VriResult_Failure;
        }

        // ---- resources -----------------------------------------------------
        VriResult VRI_CALL CreateBuffer(VriDevice* device, const VriBufferDesc* desc, VriBuffer** out)
        {
            DeviceWGPU*     d           = Dev(device);
            WGPUBufferUsage usage       = ToWgpuBufferUsage(desc->usage);
            WGPUMapMode     mapMode     = WGPUMapMode_None;
            bool            shadowWrite = false;
            if (desc->memoryLocation == VriMemoryLocation_HostReadback)
            {
                usage |= WGPUBufferUsage_MapRead; // readback must map to read GPU results
                mapMode = WGPUMapMode_Read;
            }
            else if (desc->memoryLocation == VriMemoryLocation_HostUpload)
            {
                // Write via a CPU shadow + queueWriteBuffer (synchronous) rather than MapWrite (async).
                usage |= WGPUBufferUsage_CopyDst; // queueWriteBuffer's destination
                shadowWrite = true;
            }

            WGPUBufferDescriptor bd = {};
            bd.usage                = usage;
            bd.size           = (desc->size + 3u) & ~uint64_t(3u); // queueWriteBuffer needs a 4-byte-multiple size
            WGPUBuffer buffer = wgpuDeviceCreateBuffer(d->Device(), &bd);
            if (!buffer)
                return VriResult_OutOfMemory;
            BufferWGPU* b = new BufferWGPU {d, buffer, desc->size, mapMode};
            if (shadowWrite)
                b->shadow = new uint8_t[bd.size]();
            *out = ToHandle(b);
            return VriResult_Success;
        }

        void VRI_CALL DestroyBuffer(VriBuffer* buffer)
        {
            if (!buffer)
                return;
            BufferWGPU* b = Buf(buffer);
            delete[] b->shadow;
            wgpuBufferRelease(b->buffer);
            delete b;
        }

        struct MapState
        {
            bool               done;
            WGPUMapAsyncStatus status;
        };
        void OnMap(WGPUMapAsyncStatus status, WGPUStringView, void* ud1, void*)
        {
            auto* s   = static_cast<MapState*>(ud1);
            s->done   = true;
            s->status = status;
        }

        void* VRI_CALL MapBuffer(VriBuffer* buffer, uint64_t offset, uint64_t size)
        {
            BufferWGPU* b = Buf(buffer);
            if (b->shadow) // host-upload: hand back the CPU shadow; Unmap pushes it via queueWriteBuffer
            {
                b->mapOffset = offset;
                b->mapLen    = size ? size : (b->size - offset);
                return b->shadow + offset;
            }
            const WGPUMapMode         mode = b->mapMode != WGPUMapMode_None ? b->mapMode : WGPUMapMode_Read;
            MapState                  st   = {};
            WGPUBufferMapCallbackInfo cb   = {};
            cb.mode                        = kCallbackMode;
            cb.callback                    = OnMap;
            cb.userdata1                   = &st;
            wgpuBufferMapAsync(b->buffer, mode, offset, size ? size : WGPU_WHOLE_MAP_SIZE, cb);
            for (int i = 0; i < 100000 && !st.done; ++i)
                PollDevice(b->device->Device());
            if (st.status != WGPUMapAsyncStatus_Success)
                return nullptr;
            const uint64_t range = size ? size : WGPU_WHOLE_MAP_SIZE;
            // Dawn (emdawnwebgpu) requires the const accessor for read-mapped buffers;
            // GetMappedRange asserts there. wgpu-native accepts either.
            if (mode == WGPUMapMode_Read)
                return const_cast<void*>(wgpuBufferGetConstMappedRange(b->buffer, offset, range));
            return wgpuBufferGetMappedRange(b->buffer, offset, range);
        }

        void VRI_CALL UnmapBuffer(VriBuffer* buffer)
        {
            BufferWGPU* b = Buf(buffer);
            if (b->shadow) // flush the written range to the GPU buffer (no map, no ASYNCIFY yield)
            {
                uint64_t len = (b->mapLen + 3u) & ~uint64_t(3u); // queueWriteBuffer size must be a 4-byte multiple
                const uint64_t cap = ((b->size + 3u) & ~uint64_t(3u)) - b->mapOffset; // shadow/GPU are 4-aligned
                if (len > cap)
                    len = cap;
                wgpuQueueWriteBuffer(b->device->Queue(), b->buffer, b->mapOffset, b->shadow + b->mapOffset, len);
                return;
            }
            wgpuBufferUnmap(b->buffer);
        }

        uint64_t VRI_CALL GetBufferDeviceAddress(const VriBuffer*) { return 0; } // not exposed by WebGPU

        VriResult VRI_CALL CreateTexture(VriDevice* device, const VriTextureDesc* desc, VriTexture** out)
        {
            DeviceWGPU*           d    = Dev(device);
            WGPUTextureDescriptor td   = {};
            td.usage                   = ToWgpuTextureUsage(desc->usage);
            td.dimension               = desc->type == VriTextureType_3D ? WGPUTextureDimension_3D :
                                         desc->type == VriTextureType_1D || desc->type == VriTextureType_1DArray ?
                                                                           WGPUTextureDimension_1D :
                                                                           WGPUTextureDimension_2D;
            td.size.width              = desc->width;
            td.size.height             = desc->height ? desc->height : 1u;
            td.size.depthOrArrayLayers = desc->type == VriTextureType_3D ? (desc->depth ? desc->depth : 1u) :
                                                                           (desc->layerNum ? desc->layerNum : 1u);
            td.format                  = ToWgpuFormat(desc->format);
            td.mipLevelCount           = desc->mipNum ? desc->mipNum : 1u;
            td.sampleCount             = desc->sampleNum ? desc->sampleNum : 1u;

            WGPUTexture texture = wgpuDeviceCreateTexture(d->Device(), &td);
            if (!texture)
                return VriResult_OutOfMemory;

            TextureWGPU* t = new TextureWGPU {};
            t->device      = d;
            t->texture     = texture;
            t->format      = td.format;
            t->width       = td.size.width;
            t->height      = td.size.height;
            t->depth       = td.size.depthOrArrayLayers;
            t->mipNum      = td.mipLevelCount;
            t->layerNum    = desc->type == VriTextureType_3D ? 1u : td.size.depthOrArrayLayers;
            t->texelSize   = TexelSize(desc->format);
            t->owned       = true;
            *out           = ToHandle(t);
            return VriResult_Success;
        }

        void VRI_CALL DestroyTexture(VriTexture* texture)
        {
            if (!texture)
                return;
            TextureWGPU* t = Tex(texture);
            if (t->owned && t->texture)
                wgpuTextureRelease(t->texture);
            delete t;
        }

        // ---- explicit memory: unsupported on WebGPU (no memory objects) ----
        void VRI_CALL GetBufferMemoryDesc(const VriDevice*, const VriBufferDesc*, VriMemoryLocation, VriMemoryDesc* o)
        {
            if (o)
                *o = VriMemoryDesc {};
        }
        void VRI_CALL GetTextureMemoryDesc(const VriDevice*, const VriTextureDesc*, VriMemoryLocation, VriMemoryDesc* o)
        {
            if (o)
                *o = VriMemoryDesc {};
        }
        VriResult VRI_CALL AllocateMemory(VriDevice*, const VriMemoryDesc*, VriMemory**)
        {
            return VriResult_Unsupported;
        }
        void VRI_CALL      FreeMemory(VriMemory*) {}
        VriResult VRI_CALL BindBufferMemory(VriDevice*, VriBuffer*, VriMemory*, uint64_t)
        {
            return VriResult_Unsupported;
        }
        VriResult VRI_CALL BindTextureMemory(VriDevice*, VriTexture*, VriMemory*, uint64_t)
        {
            return VriResult_Unsupported;
        }

        // ---- views & samplers ----------------------------------------------
        VriResult VRI_CALL CreateBufferView(VriDevice* device, const VriBufferViewDesc* desc, VriDescriptor** out)
        {
            DescriptorWGPU* v = new DescriptorWGPU {};
            v->kind           = DescriptorWGPU::Kind::BufferView;
            v->device         = Dev(device);
            v->buffer         = Buf(desc->buffer);
            v->bufferOffset   = desc->offset;
            v->bufferRange    = desc->size;
            *out              = ToHandle(v);
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateTextureView(VriDevice* device, const VriTextureViewDesc* desc, VriDescriptor** out)
        {
            const TextureWGPU*        t  = reinterpret_cast<const TextureWGPU*>(desc->texture);
            WGPUTextureViewDescriptor vd = {};
            vd.format                    = desc->format == VriFormat_Unknown ? t->format : ToWgpuFormat(desc->format);
            switch (desc->viewType)
            {
                case VriTextureViewType_1D:
                    vd.dimension = WGPUTextureViewDimension_1D;
                    break;
                case VriTextureViewType_2DArray:
                    vd.dimension = WGPUTextureViewDimension_2DArray;
                    break;
                case VriTextureViewType_3D:
                    vd.dimension = WGPUTextureViewDimension_3D;
                    break;
                case VriTextureViewType_Cube:
                    vd.dimension = WGPUTextureViewDimension_Cube;
                    break;
                case VriTextureViewType_CubeArray:
                    vd.dimension = WGPUTextureViewDimension_CubeArray;
                    break;
                default:
                    vd.dimension = WGPUTextureViewDimension_2D;
                    break;
            }
            vd.baseMipLevel    = desc->baseMip;
            vd.mipLevelCount   = desc->mipNum ? desc->mipNum : (t->mipNum - desc->baseMip);
            vd.baseArrayLayer  = desc->baseLayer;
            vd.arrayLayerCount = desc->layerNum ? desc->layerNum : (t->layerNum - desc->baseLayer);
            vd.aspect          = WGPUTextureAspect_All;

            WGPUTextureView view = wgpuTextureCreateView(t->texture, &vd);
            if (!view)
                return VriResult_Failure;

            DescriptorWGPU* v = new DescriptorWGPU {};
            v->kind           = DescriptorWGPU::Kind::TextureView;
            v->device         = Dev(device);
            v->view           = view;
            v->format         = vd.format;
            *out              = ToHandle(v);
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateSampler(VriDevice* device, const VriSamplerDesc* desc, VriDescriptor** out)
        {
            auto toAddr = [](VriAddressMode m) {
                switch (m)
                {
                    case VriAddressMode_MirroredRepeat:
                        return WGPUAddressMode_MirrorRepeat;
                    case VriAddressMode_ClampToEdge:
                        return WGPUAddressMode_ClampToEdge;
                    case VriAddressMode_ClampToBorder:
                        return WGPUAddressMode_ClampToEdge; // WebGPU has no border
                    default:
                        return WGPUAddressMode_Repeat;
                }
            };
            WGPUSamplerDescriptor sd = {};
            sd.magFilter = desc->magFilter == VriFilter_Linear ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
            sd.minFilter = desc->minFilter == VriFilter_Linear ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
            sd.mipmapFilter =
                desc->mipmapMode == VriMipmapMode_Linear ? WGPUMipmapFilterMode_Linear : WGPUMipmapFilterMode_Nearest;
            sd.addressModeU = toAddr(desc->addressModeU);
            sd.addressModeV = toAddr(desc->addressModeV);
            sd.addressModeW = toAddr(desc->addressModeW);
            sd.lodMinClamp  = desc->minLod;
            sd.lodMaxClamp  = desc->maxLod;
            sd.maxAnisotropy =
                desc->anisotropyEnable ? (desc->maxAnisotropy > 1.0f ? (uint16_t)desc->maxAnisotropy : 1) : 1;
            // A comparison sampler (shadow PCF): set sd.compare so it becomes a "comparison" sampler;
            // the matching bind-group-layout entry must declare WGPUSamplerBindingType_Comparison.
            if (desc->compareEnable)
                sd.compare = ToWgpuCompareOp(desc->compareOp);

            WGPUSampler sampler = wgpuDeviceCreateSampler(Dev(device)->Device(), &sd);
            if (!sampler)
                return VriResult_Failure;
            DescriptorWGPU* v = new DescriptorWGPU {};
            v->kind           = DescriptorWGPU::Kind::Sampler;
            v->device         = Dev(device);
            v->sampler        = sampler;
            *out              = ToHandle(v);
            return VriResult_Success;
        }

        void VRI_CALL DestroyDescriptor(VriDescriptor* descriptor)
        {
            if (!descriptor)
                return;
            DescriptorWGPU* v = Desc(descriptor);
            if (v->view)
                wgpuTextureViewRelease(v->view);
            if (v->sampler)
                wgpuSamplerRelease(v->sampler);
            delete v;
        }

        // ---- pipeline layout & pipelines -----------------------------------
        VriResult VRI_CALL CreatePipelineLayout(VriDevice*                   device,
                                                const VriPipelineLayoutDesc* desc,
                                                VriPipelineLayout**          out)
        {
            DeviceWGPU*         d      = Dev(device);
            PipelineLayoutWGPU* layout = new PipelineLayoutWGPU {};
            layout->device             = d;

            for (uint32_t s = 0; s < desc->descriptorSetNum; ++s)
            {
                const VriDescriptorSetDesc&           set = desc->descriptorSets[s];
                std::vector<WGPUBindGroupLayoutEntry> entries;
                std::vector<RangeInfoWGPU>            rangeInfos;
                entries.reserve(set.rangeNum);
                rangeInfos.reserve(set.rangeNum);
                for (uint32_t r = 0; r < set.rangeNum; ++r)
                {
                    const VriDescriptorRangeDesc& range = set.ranges[r];
                    WGPUBindGroupLayoutEntry      e     = {};
                    e.binding                           = range.baseRegister;
                    e.visibility                        = ToWgpuShaderStage(range.shaderStages);
                    switch (range.descriptorType)
                    {
                        case VriDescriptorType_ConstantBuffer:
                            e.buffer.type = WGPUBufferBindingType_Uniform;
                            break;
                        case VriDescriptorType_StorageBuffer:
                            e.buffer.type = WGPUBufferBindingType_Storage;
                            break;
                        case VriDescriptorType_StructuredBuffer:
                            e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
                            break;
                        case VriDescriptorType_Texture:
                            // Depth texture sampled for comparison -> sampleType must be Depth (shadow map).
                            e.texture.sampleType    = (range.flags & VriDescriptorRange_Comparison) ?
                                                          WGPUTextureSampleType_Depth :
                                                          WGPUTextureSampleType_Float;
                            e.texture.viewDimension = range.viewType == VriTextureViewType_2DArray ?
                                                          WGPUTextureViewDimension_2DArray :
                                                      range.viewType == VriTextureViewType_Cube ?
                                                          WGPUTextureViewDimension_Cube :
                                                          WGPUTextureViewDimension_2D; // 0/1D/2D -> 2D
                            break;
                        case VriDescriptorType_StorageTexture:
                            e.storageTexture.access        = WGPUStorageTextureAccess_WriteOnly;
                            e.storageTexture.format        = WGPUTextureFormat_RGBA8Unorm;
                            e.storageTexture.viewDimension = WGPUTextureViewDimension_2D;
                            break;
                        case VriDescriptorType_Sampler:
                            e.sampler.type = (range.flags & VriDescriptorRange_Comparison) ?
                                                 WGPUSamplerBindingType_Comparison :
                                                 WGPUSamplerBindingType_Filtering;
                            break;
                        default:
                            break; // acceleration structure: not WebGPU core
                    }
                    entries.push_back(e);
                    rangeInfos.push_back({range.baseRegister, range.descriptorType, range.descriptorNum});
                }

                WGPUBindGroupLayoutDescriptor ld = {};
                ld.entryCount                    = static_cast<uint32_t>(entries.size());
                ld.entries                       = entries.data();
                WGPUBindGroupLayout bgl          = wgpuDeviceCreateBindGroupLayout(d->Device(), &ld);
                layout->bindGroupLayouts.push_back(bgl);
                layout->setRanges.push_back(std::move(rangeInfos));
            }

            // Push constants -> a reserved bind group (one uniform buffer at binding 0) right
            // after the descriptor-set groups. The WGSL injection points at this group index.
            if (desc->pushConstantNum > 0)
            {
                WGPUBindGroupLayoutEntry e       = {};
                e.binding                        = 0;
                e.visibility                     = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
                e.buffer.type                    = WGPUBufferBindingType_Uniform;
                e.buffer.hasDynamicOffset        = true; // per-draw push values live at different ring offsets
                WGPUBindGroupLayoutDescriptor ld = {};
                ld.entryCount                    = 1;
                ld.entries                       = &e;
                layout->hasPush                  = true;
                layout->pushGroup                = static_cast<uint32_t>(layout->bindGroupLayouts.size());
                layout->pushSize                 = desc->pushConstants[0].size;
                layout->bindGroupLayouts.push_back(wgpuDeviceCreateBindGroupLayout(d->Device(), &ld));
                layout->setRanges.push_back({});
            }

            WGPUPipelineLayoutDescriptor pld = {};
            pld.bindGroupLayoutCount         = static_cast<uint32_t>(layout->bindGroupLayouts.size());
            pld.bindGroupLayouts = layout->bindGroupLayouts.empty() ? nullptr : layout->bindGroupLayouts.data();
            layout->layout       = wgpuDeviceCreatePipelineLayout(d->Device(), &pld);
            *out                 = ToHandle(layout);
            return VriResult_Success;
        }

        void VRI_CALL DestroyPipelineLayout(VriPipelineLayout* layout)
        {
            if (!layout)
                return;
            PipelineLayoutWGPU* l = PL(layout);
            if (l->pushBindGroup)
                wgpuBindGroupRelease(l->pushBindGroup);
            if (l->pushBuffer)
                wgpuBufferRelease(l->pushBuffer);
            for (WGPUBindGroupLayout bgl : l->bindGroupLayouts)
                wgpuBindGroupLayoutRelease(bgl);
            if (l->layout)
                wgpuPipelineLayoutRelease(l->layout);
            delete l;
        }

        VriResult VRI_CALL CreateGraphicsPipeline(VriDevice*                     device,
                                                  const VriGraphicsPipelineDesc* desc,
                                                  VriPipeline**                  out)
        {
            DeviceWGPU* d = Dev(device);
            // WebGPU has no geometry/tessellation stages - reject rather than ignore.
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                const VriShaderStageBits st = desc->shaders[i].stage;
                if (st == VriShaderStage_Geometry || st == VriShaderStage_TessControl || st == VriShaderStage_TessEval)
                    return VriResult_Unsupported;
            }
            const PipelineLayoutWGPU* pl = desc->pipelineLayout ? PL(desc->pipelineLayout) : nullptr;
            const int pushGroup    = (pl && pl->hasPush) ? static_cast<int>(pl->pushGroup) : -1; // inject WGSL binding
            WGPUShaderModule vsMod = nullptr, fsMod = nullptr;
            const char*      vsEntry = "main";
            const char*      fsEntry = "main";
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                const VriShaderDesc& s = desc->shaders[i];
                if (s.stage == VriShaderStage_Vertex)
                {
                    vsMod   = MakeWgslModule(d->Device(), s.bytecode, pushGroup);
                    vsEntry = s.entryPointName ? s.entryPointName : "main";
                }
                else if (s.stage == VriShaderStage_Fragment)
                {
                    fsMod   = MakeWgslModule(d->Device(), s.bytecode, pushGroup);
                    fsEntry = s.entryPointName ? s.entryPointName : "main";
                }
            }

            WGPUVertexState vertex = {};
            vertex.module          = vsMod;
            vertex.entryPoint      = SV(vsEntry);
            // Vertex buffer layout: one WGPUVertexBufferLayout per stream, with its
            // attributes (shaderLocation = attribute index, matching VK/GL).
            std::vector<WGPUVertexBufferLayout>           vbLayouts;
            std::vector<std::vector<WGPUVertexAttribute>> vbAttrs(desc->vertexInput.streamNum);
            vbLayouts.reserve(desc->vertexInput.streamNum);
            for (uint32_t s = 0; s < desc->vertexInput.streamNum; ++s)
            {
                const VriVertexStreamDesc& stream = desc->vertexInput.streams[s];
                for (uint32_t a = 0; a < desc->vertexInput.attributeNum; ++a)
                {
                    const VriVertexAttributeDesc& attr = desc->vertexInput.attributes[a];
                    if (attr.streamIndex != s)
                        continue;
                    WGPUVertexAttribute va = {};
                    va.format              = ToWgpuVertexFormat(attr.format);
                    va.offset              = attr.offset;
                    va.shaderLocation      = a; // global attribute index == GLSL/SPIR-V location
                    vbAttrs[s].push_back(va);
                }
                WGPUVertexBufferLayout vl = {};
                vl.arrayStride            = stream.stride;
                vl.stepMode       = stream.stepRate == VriVertexStepRate_PerInstance ? WGPUVertexStepMode_Instance :
                                                                                       WGPUVertexStepMode_Vertex;
                vl.attributeCount = vbAttrs[s].size();
                vl.attributes     = vbAttrs[s].data();
                vbLayouts.push_back(vl);
            }
            vertex.bufferCount = vbLayouts.size();
            vertex.buffers     = vbLayouts.empty() ? nullptr : vbLayouts.data();

            std::vector<WGPUColorTargetState> targets;
            std::vector<WGPUBlendState>       blends;
            targets.reserve(desc->outputMerger.colorNum);
            blends.reserve(desc->outputMerger.colorNum);
            for (uint32_t i = 0; i < desc->outputMerger.colorNum; ++i)
            {
                const VriColorAttachmentDesc& c  = desc->outputMerger.colors[i];
                WGPUColorTargetState          ct = {};
                ct.format                        = ToWgpuFormat(c.format);
                WGPUColorWriteMask mask          = WGPUColorWriteMask_None;
                if (c.colorWriteMask == 0 || c.colorWriteMask == VriColorWrite_RGBA)
                    mask = WGPUColorWriteMask_All;
                else
                {
                    if (c.colorWriteMask & VriColorWrite_R)
                        mask |= WGPUColorWriteMask_Red;
                    if (c.colorWriteMask & VriColorWrite_G)
                        mask |= WGPUColorWriteMask_Green;
                    if (c.colorWriteMask & VriColorWrite_B)
                        mask |= WGPUColorWriteMask_Blue;
                    if (c.colorWriteMask & VriColorWrite_A)
                        mask |= WGPUColorWriteMask_Alpha;
                }
                ct.writeMask = mask;
                if (c.blend.enable)
                {
                    WGPUBlendState bs  = {};
                    bs.color.srcFactor = ToWgpuBlendFactor(c.blend.srcColor);
                    bs.color.dstFactor = ToWgpuBlendFactor(c.blend.dstColor);
                    bs.color.operation = ToWgpuBlendOp(c.blend.colorOp);
                    bs.alpha.srcFactor = ToWgpuBlendFactor(c.blend.srcAlpha);
                    bs.alpha.dstFactor = ToWgpuBlendFactor(c.blend.dstAlpha);
                    bs.alpha.operation = ToWgpuBlendOp(c.blend.alphaOp);
                    blends.push_back(bs);
                    ct.blend = &blends.back();
                }
                targets.push_back(ct);
            }

            WGPUFragmentState fragment = {};
            fragment.module            = fsMod;
            fragment.entryPoint        = SV(fsEntry);
            fragment.targetCount       = static_cast<uint32_t>(targets.size());
            fragment.targets           = targets.data();

            WGPURenderPipelineDescriptor pd = {};
            pd.layout                       = desc->pipelineLayout ? PL(desc->pipelineLayout)->layout : nullptr;
            pd.vertex                       = vertex;
            pd.primitive.topology           = ToWgpuTopology(desc->inputAssembly.topology);
            pd.primitive.frontFace          = ToWgpuFrontFace(desc->rasterization.frontFace);
            pd.primitive.cullMode           = ToWgpuCullMode(desc->rasterization.cullMode);
            pd.multisample.count            = desc->multisample.sampleNum ? desc->multisample.sampleNum : 1u;
            pd.multisample.mask             = 0xFFFFFFFFu;
            pd.fragment                     = &fragment;

            WGPUDepthStencilState depthStencil = {};
            if (desc->outputMerger.depthStencilFormat != VriFormat_Unknown)
            {
                depthStencil.format = ToWgpuFormat(desc->outputMerger.depthStencilFormat);
                depthStencil.depthWriteEnabled =
                    desc->depthStencil.depthWrite ? WGPUOptionalBool_True : WGPUOptionalBool_False;
                depthStencil.depthCompare      = desc->depthStencil.depthTest ?
                                                     ToWgpuCompareOp(desc->depthStencil.depthCompareOp) :
                                                     WGPUCompareFunction_Always;
                const VriDepthStencilDesc& dss = desc->depthStencil;
                if (dss.stencilTest)
                {
                    auto toFace = [](const VriStencilOpDesc& s) {
                        WGPUStencilFaceState f = {};
                        f.compare              = ToWgpuCompareOp(s.compareOp);
                        f.failOp               = ToWgpuStencilOp(s.failOp);
                        f.depthFailOp          = ToWgpuStencilOp(s.depthFailOp);
                        f.passOp               = ToWgpuStencilOp(s.passOp);
                        return f;
                    };
                    depthStencil.stencilFront     = toFace(dss.front);
                    depthStencil.stencilBack      = toFace(dss.back);
                    depthStencil.stencilReadMask  = dss.front.compareMask; // WGPU masks are not per-face
                    depthStencil.stencilWriteMask = dss.front.writeMask;
                }
                else
                {
                    // No stencil test: both faces must be Always/Keep.
                    depthStencil.stencilFront.compare = WGPUCompareFunction_Always;
                    depthStencil.stencilBack.compare  = WGPUCompareFunction_Always;
                }
                pd.depthStencil = &depthStencil;
            }

            WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(d->Device(), &pd);

            if (vsMod)
                wgpuShaderModuleRelease(vsMod);
            if (fsMod)
                wgpuShaderModuleRelease(fsMod);
            if (!pipeline)
                return VriResult_Failure;

            PipelineWGPU* pw     = new PipelineWGPU {d, pipeline, nullptr, false};
            pw->stencilTest      = desc->depthStencil.stencilTest != VRI_FALSE;
            pw->stencilReference = desc->depthStencil.front.reference;
            *out                 = ToHandle(pw);
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateComputePipeline(VriDevice*                    device,
                                                 const VriComputePipelineDesc* desc,
                                                 VriPipeline**                 out)
        {
            DeviceWGPU*                   d   = Dev(device);
            WGPUShaderModule              mod = MakeWgslModule(d->Device(), desc->shader.bytecode);
            WGPUComputePipelineDescriptor pd  = {};
            pd.layout                         = desc->pipelineLayout ? PL(desc->pipelineLayout)->layout : nullptr;
            pd.compute.module                 = mod;
            pd.compute.entryPoint             = SV(desc->shader.entryPointName ? desc->shader.entryPointName : "main");
            WGPUComputePipeline pipe          = wgpuDeviceCreateComputePipeline(d->Device(), &pd);
            if (mod)
                wgpuShaderModuleRelease(mod);
            if (!pipe)
                return VriResult_Failure;
            *out = ToHandle(new PipelineWGPU {d, nullptr, pipe, true});
            return VriResult_Success;
        }

        void VRI_CALL DestroyPipeline(VriPipeline* pipeline)
        {
            if (!pipeline)
                return;
            PipelineWGPU* p = Pipe(pipeline);
            if (p->render)
                wgpuRenderPipelineRelease(p->render);
            if (p->compute)
                wgpuComputePipelineRelease(p->compute);
            delete p;
        }

        // ---- descriptor pools / sets (= WebGPU bind groups) ----------------
        VriResult VRI_CALL CreateDescriptorPool(VriDevice* device,
                                                const VriDescriptorPoolDesc*,
                                                VriDescriptorPool** out)
        {
            *out = ToHandle(new DescriptorPoolWGPU {Dev(device), {}});
            return VriResult_Success;
        }

        void VRI_CALL ResetDescriptorPool(VriDescriptorPool* pool)
        {
            DescriptorPoolWGPU* p = DPool(pool);
            for (DescriptorSetWGPU* s : p->sets)
            {
                if (s->bindGroup)
                    wgpuBindGroupRelease(s->bindGroup);
                delete s;
            }
            p->sets.clear();
        }

        void VRI_CALL DestroyDescriptorPool(VriDescriptorPool* pool)
        {
            if (!pool)
                return;
            ResetDescriptorPool(pool);
            delete DPool(pool);
        }

        VriResult VRI_CALL AllocateDescriptorSets(VriDescriptorPool*       pool,
                                                  const VriPipelineLayout* layout,
                                                  uint32_t                 setIndex,
                                                  VriDescriptorSet**       outSets,
                                                  uint32_t                 setNum)
        {
            DescriptorPoolWGPU*       p = DPool(pool);
            const PipelineLayoutWGPU* l = PLc(layout);
            if (setIndex >= l->bindGroupLayouts.size())
                return VriResult_InvalidArgument;
            for (uint32_t i = 0; i < setNum; ++i)
            {
                DescriptorSetWGPU* s = new DescriptorSetWGPU {p->device, l, setIndex, nullptr};
                p->sets.push_back(s);
                outSets[i] = ToHandle(s);
            }
            return VriResult_Success;
        }

        void VRI_CALL UpdateDescriptorRanges(VriDescriptorSet*                   set,
                                             uint32_t                            baseRange,
                                             uint32_t                            rangeNum,
                                             const VriDescriptorRangeUpdateDesc* updates)
        {
            DescriptorSetWGPU*                s      = DSet(set);
            const std::vector<RangeInfoWGPU>& ranges = s->layout->setRanges[s->setIndex];

            std::vector<WGPUBindGroupEntry> entries;
            for (uint32_t r = 0; r < rangeNum; ++r)
            {
                const VriDescriptorRangeUpdateDesc& u        = updates[r];
                const RangeInfoWGPU&                info     = ranges[baseRange + r];
                const bool                          isBuffer = info.type == VriDescriptorType_ConstantBuffer ||
                                      info.type == VriDescriptorType_StorageBuffer ||
                                      info.type == VriDescriptorType_StructuredBuffer;
                for (uint32_t k = 0; k < u.descriptorNum; ++k)
                {
                    const DescriptorWGPU* d = Desc(const_cast<VriDescriptor*>(u.descriptors[k]));
                    WGPUBindGroupEntry    e = {};
                    e.binding               = info.binding + k;
                    if (isBuffer)
                    {
                        e.buffer = d->buffer ? d->buffer->buffer : nullptr;
                        e.offset = d->bufferOffset;
                        e.size   = d->bufferRange ? d->bufferRange : WGPU_WHOLE_SIZE;
                    }
                    else if (info.type == VriDescriptorType_Sampler)
                    {
                        e.sampler = d->sampler;
                    }
                    else
                    {
                        e.textureView = d->view;
                    }
                    entries.push_back(e);
                }
            }

            WGPUBindGroupDescriptor bd = {};
            bd.layout                  = s->layout->bindGroupLayouts[s->setIndex];
            bd.entryCount              = static_cast<uint32_t>(entries.size());
            bd.entries                 = entries.data();
            if (s->bindGroup)
                wgpuBindGroupRelease(s->bindGroup);
            s->bindGroup = wgpuDeviceCreateBindGroup(s->device->Device(), &bd);
        }

        // ---- synchronization (emulated timeline) ---------------------------
        VriResult VRI_CALL CreateFence(VriDevice* device, uint64_t initialValue, VriFence** out)
        {
            *out = ToHandle(new FenceWGPU {Dev(device), initialValue});
            return VriResult_Success;
        }
        void VRI_CALL     DestroyFence(VriFence* fence) { delete Fen(fence); }
        uint64_t VRI_CALL GetFenceValue(VriFence* fence) { return Fen(fence)->value; }
        void VRI_CALL     Wait(VriFence* fence, uint64_t value)
        {
            FenceWGPU* f = Fen(fence);
#if !defined(__EMSCRIPTEN__)
            // Native has no rAF pacing: poll the device so the CPU can't race far ahead of the GPU
            // (backpressure). It's a cheap blocking poll, not a stack-unwinding yield.
            wgpuDevicePoll(f->device->Device(), /*wait*/ true, nullptr);
#endif
            // Browser: do NOT emscripten_sleep here. requestAnimationFrame already paces frames, and
            // WebGPU's serial queue orders this frame's work before the next frame's writes/submits,
            // so the per-frame fence wait protects nothing the queue doesn't already guarantee.
            // Skipping it removes the last per-frame ASYNCIFY yield. Real readback still syncs at the
            // capture buffer's mapAsync, and resources used by in-flight submits stay alive until done.
            if (value > f->value)
                f->value = value;
        }

        // ---- command recording ---------------------------------------------
        // WebGPU has no implicit compute scope; VRI dispatches outside a render pass.
        // Lazily open a compute pass on first compute command and close it before any
        // encoder-level op (render pass, copy, finish) needs the encoder back.
        void EnsureComputePass(CommandBufferWGPU* c)
        {
            if (!c->computePass)
            {
                WGPUComputePassDescriptor cpd = {};
                c->computePass                = wgpuCommandEncoderBeginComputePass(c->encoder, &cpd);
            }
        }
        void EndComputePass(CommandBufferWGPU* c)
        {
            if (c->computePass)
            {
                wgpuComputePassEncoderEnd(c->computePass);
                wgpuComputePassEncoderRelease(c->computePass);
                c->computePass = nullptr;
            }
        }

        void VRI_CALL CmdBeginRendering(VriCommandBuffer* cmd, const VriAttachmentsDesc* a)
        {
            CommandBufferWGPU* c = CB(cmd);
            EndComputePass(c);
            std::vector<WGPURenderPassColorAttachment> colors(a->colorNum);
            for (uint32_t i = 0; i < a->colorNum; ++i)
            {
                const VriAttachmentDesc&       src = a->colors[i];
                WGPURenderPassColorAttachment& ca  = colors[i];
                ca                                 = {};
                ca.view                            = Desc(src.view)->view;
                ca.depthSlice                      = WGPU_DEPTH_SLICE_UNDEFINED;
                ca.loadOp                          = src.loadOp == VriAttachmentLoadOp_Clear    ? WGPULoadOp_Clear :
                                                     src.loadOp == VriAttachmentLoadOp_DontCare ? WGPULoadOp_Clear :
                                                                                                  WGPULoadOp_Load;
                ca.storeOp    = src.storeOp == VriAttachmentStoreOp_DontCare ? WGPUStoreOp_Discard : WGPUStoreOp_Store;
                ca.clearValue = WGPUColor {src.clearValue.color.f32[0],
                                           src.clearValue.color.f32[1],
                                           src.clearValue.color.f32[2],
                                           src.clearValue.color.f32[3]};
                if (src.resolveView) // MSAA resolve target
                    ca.resolveTarget = Desc(src.resolveView)->view;
            }

            WGPURenderPassDescriptor rp = {};
            rp.colorAttachmentCount     = static_cast<uint32_t>(colors.size());
            rp.colorAttachments         = colors.data();

            WGPURenderPassDepthStencilAttachment ds = {};
            if (a->depth)
            {
                ds.view        = Desc(a->depth->view)->view;
                ds.depthLoadOp = a->depth->loadOp == VriAttachmentLoadOp_Load ? WGPULoadOp_Load : WGPULoadOp_Clear;
                ds.depthStoreOp =
                    a->depth->storeOp == VriAttachmentStoreOp_DontCare ? WGPUStoreOp_Discard : WGPUStoreOp_Store;
                ds.depthClearValue = a->depth->clearValue.depthStencil.depth;
                // Stencil aspect: WGPU requires stencil load/store ops iff the format has stencil.
                const WGPUTextureFormat dsFmt = Desc(a->depth->view)->format;
                if (dsFmt == WGPUTextureFormat_Depth24PlusStencil8 || dsFmt == WGPUTextureFormat_Depth32FloatStencil8 ||
                    dsFmt == WGPUTextureFormat_Stencil8)
                {
                    ds.stencilLoadOp =
                        a->depth->loadOp == VriAttachmentLoadOp_Load ? WGPULoadOp_Load : WGPULoadOp_Clear;
                    ds.stencilStoreOp =
                        a->depth->storeOp == VriAttachmentStoreOp_DontCare ? WGPUStoreOp_Discard : WGPUStoreOp_Store;
                    ds.stencilClearValue = a->depth->clearValue.depthStencil.stencil;
                }
                rp.depthStencilAttachment = &ds;
            }

            c->pass = wgpuCommandEncoderBeginRenderPass(c->encoder, &rp);
        }

        void VRI_CALL CmdEndRendering(VriCommandBuffer* cmd)
        {
            CommandBufferWGPU* c = CB(cmd);
            wgpuRenderPassEncoderEnd(c->pass);
            wgpuRenderPassEncoderRelease(c->pass);
            c->pass = nullptr;
        }

        void VRI_CALL CmdSetViewports(VriCommandBuffer* cmd, const VriViewport* vps, uint32_t num)
        {
            if (num == 0)
                return;
            wgpuRenderPassEncoderSetViewport(
                CB(cmd)->pass, vps[0].x, vps[0].y, vps[0].width, vps[0].height, vps[0].minDepth, vps[0].maxDepth);
        }

        void VRI_CALL CmdSetScissors(VriCommandBuffer* cmd, const VriRect* rects, uint32_t num)
        {
            if (num == 0)
                return;
            wgpuRenderPassEncoderSetScissorRect(CB(cmd)->pass, rects[0].x, rects[0].y, rects[0].width, rects[0].height);
        }

        void VRI_CALL CmdSetPipelineLayout(VriCommandBuffer* cmd, VriPipelineLayout* layout)
        {
            CB(cmd)->boundLayout = layout;
        }

        void VRI_CALL CmdSetPipeline(VriCommandBuffer* cmd, VriPipeline* pipeline)
        {
            CommandBufferWGPU* c = CB(cmd);
            PipelineWGPU*      p = Pipe(pipeline);
            if (p->isCompute)
            {
                EnsureComputePass(c);
                wgpuComputePassEncoderSetPipeline(c->computePass, p->compute);
                return;
            }
            c->boundPipeline = p->render;
            if (p->render && c->pass)
            {
                wgpuRenderPassEncoderSetPipeline(c->pass, p->render);
                if (p->stencilTest)
                    wgpuRenderPassEncoderSetStencilReference(c->pass, p->stencilReference);
            }
        }

        void VRI_CALL CmdSetDescriptorSet(VriCommandBuffer* cmd, uint32_t setIndex, const VriDescriptorSet* set)
        {
            CommandBufferWGPU* c = CB(cmd);
            if (!set)
                return;
            const DescriptorSetWGPU* s = reinterpret_cast<const DescriptorSetWGPU*>(set);
            if (!s->bindGroup)
                return;
            if (c->computePass)
                wgpuComputePassEncoderSetBindGroup(c->computePass, setIndex, s->bindGroup, 0, nullptr);
            else if (c->pass)
                wgpuRenderPassEncoderSetBindGroup(c->pass, setIndex, s->bindGroup, 0, nullptr);
        }
        // WebGPU has no push constants; emulate with a uniform buffer in the reserved bind
        // group (see CreatePipelineLayout + the WGSL injection). The buffer + bind group are
        // created lazily on the layout and reused; the data is uploaded via the queue.
        void VRI_CALL CmdSetConstants(VriCommandBuffer* cmd, uint32_t, const void* data, uint32_t size)
        {
            CommandBufferWGPU* c = CB(cmd);
            if (!c->boundLayout || !data || !size)
                return;
            PipelineLayoutWGPU* l = PL(c->boundLayout);
            if (!l->hasPush)
                return;
            DeviceWGPU* d = c->device;
            // Ring of dynamic-offset slots so per-draw push values coexist within one pass. Each
            // slot is 256-byte aligned (the dynamic-offset granularity); the cursor wraps mod the
            // slot count, which is far more than any frame's draw count, so an in-flight slot is
            // never overwritten (queue writes + submits stay ordered on the queue timeline).
            constexpr uint32_t kPushSlots = 256u;
            constexpr uint32_t kAlign     = 256u;
            if (!l->pushBuffer)
            {
                l->pushStride               = ((l->pushSize ? l->pushSize : 1u) + kAlign - 1u) & ~(kAlign - 1u);
                WGPUBufferDescriptor bd     = {};
                bd.usage                    = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
                bd.size                     = static_cast<uint64_t>(l->pushStride) * kPushSlots;
                l->pushBuffer               = wgpuDeviceCreateBuffer(d->Device(), &bd);
                WGPUBindGroupEntry e        = {};
                e.binding                   = 0;
                e.buffer                    = l->pushBuffer;
                e.offset                    = 0;
                e.size                      = l->pushStride; // dynamic offset slides this window
                WGPUBindGroupDescriptor bgd = {};
                bgd.layout                  = l->bindGroupLayouts[l->pushGroup];
                bgd.entryCount              = 1;
                bgd.entries                 = &e;
                l->pushBindGroup            = wgpuDeviceCreateBindGroup(d->Device(), &bgd);
            }
            const uint32_t offset = (l->pushCursor++ % kPushSlots) * l->pushStride;
            wgpuQueueWriteBuffer(d->Queue(), l->pushBuffer, offset, data, (size + 3u) & ~3u);
            if (c->pass)
                wgpuRenderPassEncoderSetBindGroup(c->pass, l->pushGroup, l->pushBindGroup, 1, &offset);
            else if (c->computePass)
                wgpuComputePassEncoderSetBindGroup(c->computePass, l->pushGroup, l->pushBindGroup, 1, &offset);
        }

        void VRI_CALL CmdSetVertexBuffers(VriCommandBuffer*             cmd,
                                          uint32_t                      baseSlot,
                                          const VriVertexBufferBinding* bindings,
                                          uint32_t                      num)
        {
            for (uint32_t i = 0; i < num; ++i)
                wgpuRenderPassEncoderSetVertexBuffer(
                    CB(cmd)->pass, baseSlot + i, Buf(bindings[i].buffer)->buffer, bindings[i].offset, WGPU_WHOLE_SIZE);
        }

        void VRI_CALL CmdSetIndexBuffer(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, VriIndexType type)
        {
            wgpuRenderPassEncoderSetIndexBuffer(CB(cmd)->pass,
                                                Buf(buffer)->buffer,
                                                type == VriIndexType_UInt16 ? WGPUIndexFormat_Uint16 :
                                                                              WGPUIndexFormat_Uint32,
                                                offset,
                                                WGPU_WHOLE_SIZE);
        }

        void VRI_CALL CmdDraw(VriCommandBuffer* cmd, const VriDrawDesc* d)
        {
            wgpuRenderPassEncoderDraw(CB(cmd)->pass, d->vertexNum, d->instanceNum, d->baseVertex, d->baseInstance);
        }
        void VRI_CALL CmdDrawIndexed(VriCommandBuffer* cmd, const VriDrawIndexedDesc* d)
        {
            wgpuRenderPassEncoderDrawIndexed(
                CB(cmd)->pass, d->indexNum, d->instanceNum, d->baseIndex, d->vertexOffset, d->baseInstance);
        }
        void VRI_CALL CmdDrawIndirect(VriCommandBuffer* cmd,
                                      VriBuffer*        buffer,
                                      uint64_t          offset,
                                      uint32_t          drawNum,
                                      uint32_t /*stride*/)
        {
            for (uint32_t i = 0; i < drawNum; ++i)
                wgpuRenderPassEncoderDrawIndirect(CB(cmd)->pass, Buf(buffer)->buffer, offset);
        }
        // One indexed-indirect draw per record (WebGPU draws a single record per call). The indirect
        // *count* variants stay unsupported - WebGPU core has no multi-draw-indirect-count.
        void VRI_CALL CmdDrawIndexedIndirect(VriCommandBuffer* cmd,
                                             VriBuffer*        buffer,
                                             uint64_t          offset,
                                             uint32_t          drawNum,
                                             uint32_t          stride)
        {
            for (uint32_t i = 0; i < drawNum; ++i)
                wgpuRenderPassEncoderDrawIndexedIndirect(
                    CB(cmd)->pass, Buf(buffer)->buffer, offset + static_cast<uint64_t>(i) * stride);
        }
        void VRI_CALL
        CmdDrawIndirectCount(VriCommandBuffer*, VriBuffer*, uint64_t, VriBuffer*, uint64_t, uint32_t, uint32_t)
        {}
        void VRI_CALL
        CmdDrawIndexedIndirectCount(VriCommandBuffer*, VriBuffer*, uint64_t, VriBuffer*, uint64_t, uint32_t, uint32_t)
        {}
        void VRI_CALL CmdDispatch(VriCommandBuffer* cmd, const VriDispatchDesc* d)
        {
            CommandBufferWGPU* c = CB(cmd);
            EnsureComputePass(c);
            wgpuComputePassEncoderDispatchWorkgroups(c->computePass, d->x, d->y, d->z);
        }
        void VRI_CALL CmdDispatchIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset)
        {
            CommandBufferWGPU* c = CB(cmd);
            EnsureComputePass(c);
            wgpuComputePassEncoderDispatchWorkgroupsIndirect(c->computePass, Buf(buffer)->buffer, offset);
        }

        void VRI_CALL CmdBarrier(VriCommandBuffer*, const VriBarrierGroupDesc*) {} // WebGPU auto-synchronizes

        // WebGPU's clearBuffer only zeroes; an arbitrary-value storage fill is unsupported here.
        void VRI_CALL CmdClearStorageBuffer(VriCommandBuffer* cmd, VriBuffer*, uint64_t, uint64_t, uint32_t)
        {
            CB(cmd)->device->ReportError("CmdClearStorageBuffer: unsupported on WebGPU");
        }
        // WebGPU can only clear a texture via a render pass loadOp, not an out-of-pass command.
        void VRI_CALL CmdClearStorageTexture(VriCommandBuffer* cmd, VriTexture*, const VriClearColor*)
        {
            CB(cmd)->device->ReportError("CmdClearStorageTexture: unsupported on WebGPU");
        }
        void VRI_CALL CmdCopyBuffer(VriCommandBuffer* cmd, VriBuffer* dst, VriBuffer* src, const VriBufferCopyDesc* r)
        {
            CommandBufferWGPU* c = CB(cmd);
            EndComputePass(c);
            wgpuCommandEncoderCopyBufferToBuffer(
                c->encoder, Buf(src)->buffer, r->srcOffset, Buf(dst)->buffer, r->dstOffset, r->size);
        }

        void VRI_CALL CmdReadbackTextureToBuffer(VriCommandBuffer*               cmd,
                                                 VriBuffer*                      dst,
                                                 VriTexture*                     src,
                                                 const VriBufferTextureCopyDesc* region)
        {
            EndComputePass(CB(cmd));
            const TextureWGPU* t = reinterpret_cast<const TextureWGPU*>(src);
            const uint32_t     w = region->texture.width ? region->texture.width : t->width;
            const uint32_t     h = region->texture.height ? region->texture.height : t->height;

            WGPUTexelCopyTextureInfo srcInfo = {};
            srcInfo.texture                  = t->texture;
            srcInfo.mipLevel                 = region->texture.mip;
            srcInfo.origin                   = {static_cast<uint32_t>(region->texture.x),
                                                static_cast<uint32_t>(region->texture.y),
                                                static_cast<uint32_t>(region->texture.z)};
            srcInfo.aspect                   = WGPUTextureAspect_All;

            WGPUTexelCopyBufferInfo dstInfo = {};
            dstInfo.buffer                  = Buf(dst)->buffer;
            dstInfo.layout.offset           = region->bufferOffset;
            dstInfo.layout.bytesPerRow =
                region->bufferRowLength ? region->bufferRowLength * t->texelSize : w * t->texelSize;
            dstInfo.layout.rowsPerImage = region->bufferImageHeight ? region->bufferImageHeight : h;

            WGPUExtent3D ext = {w, h, region->texture.depth ? region->texture.depth : 1u};
            wgpuCommandEncoderCopyTextureToBuffer(CB(cmd)->encoder, &srcInfo, &dstInfo, &ext);
        }

        void VRI_CALL CmdUploadBufferToTexture(VriCommandBuffer*               cmd,
                                               VriTexture*                     dst,
                                               VriBuffer*                      src,
                                               const VriBufferTextureCopyDesc* region)
        {
            CommandBufferWGPU* c = CB(cmd);
            const TextureWGPU* t = reinterpret_cast<const TextureWGPU*>(dst);
            const uint32_t     w = region->texture.width ? region->texture.width : t->width;
            const uint32_t     h = region->texture.height ? region->texture.height : t->height;

            // 2D array textures address the destination slice via baseLayer/layerNum; 3D
            // textures via z/depth. WebGPU uses origin.z + extent.depthOrArrayLayers for both.
            const bool     isArray  = t->layerNum > 1;
            const uint32_t originZ  = isArray ? region->texture.baseLayer : static_cast<uint32_t>(region->texture.z);
            const uint32_t extDepth = isArray ? (region->texture.layerNum ? region->texture.layerNum : 1u) :
                                                (region->texture.depth ? region->texture.depth : 1u);
            WGPUTexelCopyTextureInfo dstInfo = {};
            dstInfo.texture                  = t->texture;
            dstInfo.mipLevel                 = region->texture.mip;
            dstInfo.origin                   = {
                static_cast<uint32_t>(region->texture.x), static_cast<uint32_t>(region->texture.y), originZ};
            dstInfo.aspect   = WGPUTextureAspect_All;
            WGPUExtent3D ext = {w, h, extDepth};

            // copyBufferToTexture requires bytesPerRow to be a multiple of 256. If the app's
            // staging is tightly packed and narrower than that, pack the rows into a temp buffer
            // at the aligned pitch first (per-row buffer copy), so any texture size just works.
            const uint32_t          tightPitch = (region->bufferRowLength ? region->bufferRowLength : w) * t->texelSize;
            const uint32_t          alignedPitch = (tightPitch + 255u) & ~255u;
            WGPUTexelCopyBufferInfo srcInfo      = {};
            srcInfo.layout.rowsPerImage          = region->bufferImageHeight ? region->bufferImageHeight : h;
            if (tightPitch == alignedPitch)
            {
                srcInfo.buffer             = Buf(src)->buffer;
                srcInfo.layout.offset      = region->bufferOffset;
                srcInfo.layout.bytesPerRow = tightPitch;
                wgpuCommandEncoderCopyBufferToTexture(c->encoder, &srcInfo, &dstInfo, &ext);
            }
            else
            {
                WGPUBufferDescriptor bd = {};
                bd.size                 = uint64_t(alignedPitch) * h;
                bd.usage                = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
                WGPUBuffer temp         = wgpuDeviceCreateBuffer(c->device->Device(), &bd);
                for (uint32_t r = 0; r < h; ++r)
                    wgpuCommandEncoderCopyBufferToBuffer(c->encoder,
                                                         Buf(src)->buffer,
                                                         region->bufferOffset + uint64_t(r) * tightPitch,
                                                         temp,
                                                         uint64_t(r) * alignedPitch,
                                                         tightPitch);
                srcInfo.buffer              = temp;
                srcInfo.layout.offset       = 0;
                srcInfo.layout.bytesPerRow  = alignedPitch;
                srcInfo.layout.rowsPerImage = h;
                wgpuCommandEncoderCopyBufferToTexture(c->encoder, &srcInfo, &dstInfo, &ext);
                c->tempUploads.push_back(temp); // keep alive until the GPU consumes it
            }
        }

        void VRI_CALL CmdCopyTexture(VriCommandBuffer*         cmd,
                                     VriTexture*               dst,
                                     VriTexture*               src,
                                     const VriTextureCopyDesc* region)
        {
            const TextureWGPU*       s  = reinterpret_cast<const TextureWGPU*>(src);
            const TextureWGPU*       d  = reinterpret_cast<const TextureWGPU*>(dst);
            WGPUTexelCopyTextureInfo si = {};
            si.texture                  = s->texture;
            si.aspect                   = WGPUTextureAspect_All;
            WGPUTexelCopyTextureInfo di = {};
            di.texture                  = d->texture;
            di.aspect                   = WGPUTextureAspect_All;
            if (region)
            {
                si.mipLevel = region->src.mip;
                di.mipLevel = region->dst.mip;
            }
            WGPUExtent3D ext = {s->width, s->height, s->depth};
            wgpuCommandEncoderCopyTextureToTexture(CB(cmd)->encoder, &si, &di, &ext);
        }

        void VRI_CALL CmdBeginDebugGroup(VriCommandBuffer*, const char*) {}
        void VRI_CALL CmdEndDebugGroup(VriCommandBuffer*) {}

        // ---- submission ----------------------------------------------------
        void VRI_CALL QueueSubmit(VriQueue* queue, const VriQueueSubmitDesc* submit)
        {
            QueueWGPU*                     q = Q(queue);
            std::vector<WGPUCommandBuffer> cbs;
            cbs.reserve(submit->commandBufferNum);
            for (uint32_t i = 0; i < submit->commandBufferNum; ++i)
            {
                CommandBufferWGPU* c = CB(submit->commandBuffers[i]);
                if (c->finished)
                    cbs.push_back(c->finished);
            }
            wgpuQueueSubmit(q->queue, cbs.size(), cbs.data());
            for (uint32_t i = 0; i < submit->commandBufferNum; ++i)
            {
                CommandBufferWGPU* c = CB(submit->commandBuffers[i]);
                if (c->finished)
                {
                    wgpuCommandBufferRelease(c->finished);
                    c->finished = nullptr;
                }
            }
            // record intended signal values (completion guaranteed by Wait's poll)
            for (uint32_t i = 0; i < submit->signalFenceNum; ++i)
                Fen(submit->signalFences[i].fence)->value = submit->signalFences[i].value;
        }

        void VRI_CALL QueueWaitIdle(VriQueue* queue) { PollDevice(Q(queue)->device->Device()); }
        void VRI_CALL DeviceWaitIdle(VriDevice* device) { PollDevice(Dev(device)->Device()); }
        void VRI_CALL SetDebugName(void*, const char*) {}

        VriCoreInterface MakeTable()
        {
            VriCoreInterface t            = {};
            t.GetDeviceDesc               = GetDeviceDesc;
            t.GetFormatSupport            = GetFormatSupport;
            t.GetQueue                    = GetQueue;
            t.CreateCommandAllocator      = CreateCommandAllocator;
            t.ResetCommandAllocator       = ResetCommandAllocator;
            t.DestroyCommandAllocator     = DestroyCommandAllocator;
            t.CreateCommandBuffer         = CreateCommandBuffer;
            t.BeginCommandBuffer          = BeginCommandBuffer;
            t.EndCommandBuffer            = EndCommandBuffer;
            t.CreateBuffer                = CreateBuffer;
            t.DestroyBuffer               = DestroyBuffer;
            t.MapBuffer                   = MapBuffer;
            t.UnmapBuffer                 = UnmapBuffer;
            t.GetBufferDeviceAddress      = GetBufferDeviceAddress;
            t.CreateTexture               = CreateTexture;
            t.DestroyTexture              = DestroyTexture;
            t.GetBufferMemoryDesc         = GetBufferMemoryDesc;
            t.GetTextureMemoryDesc        = GetTextureMemoryDesc;
            t.AllocateMemory              = AllocateMemory;
            t.FreeMemory                  = FreeMemory;
            t.BindBufferMemory            = BindBufferMemory;
            t.BindTextureMemory           = BindTextureMemory;
            t.CreateBufferView            = CreateBufferView;
            t.CreateTextureView           = CreateTextureView;
            t.CreateSampler               = CreateSampler;
            t.DestroyDescriptor           = DestroyDescriptor;
            t.CreatePipelineLayout        = CreatePipelineLayout;
            t.DestroyPipelineLayout       = DestroyPipelineLayout;
            t.CreateGraphicsPipeline      = CreateGraphicsPipeline;
            t.CreateComputePipeline       = CreateComputePipeline;
            t.DestroyPipeline             = DestroyPipeline;
            t.CreateDescriptorPool        = CreateDescriptorPool;
            t.ResetDescriptorPool         = ResetDescriptorPool;
            t.DestroyDescriptorPool       = DestroyDescriptorPool;
            t.AllocateDescriptorSets      = AllocateDescriptorSets;
            t.UpdateDescriptorRanges      = UpdateDescriptorRanges;
            t.CreateFence                 = CreateFence;
            t.DestroyFence                = DestroyFence;
            t.GetFenceValue               = GetFenceValue;
            t.Wait                        = Wait;
            t.CmdBeginRendering           = CmdBeginRendering;
            t.CmdEndRendering             = CmdEndRendering;
            t.CmdSetViewports             = CmdSetViewports;
            t.CmdSetScissors              = CmdSetScissors;
            t.CmdSetPipelineLayout        = CmdSetPipelineLayout;
            t.CmdSetPipeline              = CmdSetPipeline;
            t.CmdSetDescriptorSet         = CmdSetDescriptorSet;
            t.CmdSetConstants             = CmdSetConstants;
            t.CmdSetVertexBuffers         = CmdSetVertexBuffers;
            t.CmdSetIndexBuffer           = CmdSetIndexBuffer;
            t.CmdDraw                     = CmdDraw;
            t.CmdDrawIndexed              = CmdDrawIndexed;
            t.CmdDrawIndirect             = CmdDrawIndirect;
            t.CmdDrawIndexedIndirect      = CmdDrawIndexedIndirect;
            t.CmdDrawIndirectCount        = CmdDrawIndirectCount;
            t.CmdDrawIndexedIndirectCount = CmdDrawIndexedIndirectCount;
            t.CmdDispatch                 = CmdDispatch;
            t.CmdDispatchIndirect         = CmdDispatchIndirect;
            t.CmdBarrier                  = CmdBarrier;
            t.CmdCopyBuffer               = CmdCopyBuffer;
            t.CmdCopyTexture              = CmdCopyTexture;
            t.CmdUploadBufferToTexture    = CmdUploadBufferToTexture;
            t.CmdReadbackTextureToBuffer  = CmdReadbackTextureToBuffer;
            t.CmdBeginDebugGroup          = CmdBeginDebugGroup;
            t.CmdEndDebugGroup            = CmdEndDebugGroup;
            t.QueueSubmit                 = QueueSubmit;
            t.QueueWaitIdle               = QueueWaitIdle;
            t.DeviceWaitIdle              = DeviceWaitIdle;
            t.SetDebugName                = SetDebugName;
            t.GetVideoMemoryInfo          = GetVideoMemoryInfo;
            t.CmdClearStorageBuffer       = CmdClearStorageBuffer;
            t.CmdClearStorageTexture      = CmdClearStorageTexture;
            return t;
        }

        const VriCoreInterface g_coreWGPU = MakeTable();
    } // namespace

    const VriCoreInterface* GetCoreInterfaceWGPU() { return &g_coreWGPU; }
} // namespace vri::wgpu
