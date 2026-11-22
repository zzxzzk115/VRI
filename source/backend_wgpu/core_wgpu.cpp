// WebGPU core implementation and native completion tracking.

#include "core_wgpu.h"
#include "conversions_wgpu.h"
#include "device_wgpu.h"
#include "objects_wgpu.h"

#include "wgpu_native.h" // native-only polling and browser callback mode

#include "core/resource_desc_checks.h"
#include "core/synchronization.h"

#include <algorithm>
#include <cstring>
#include <memory>
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
                case VriFormat_BC1_UNORM:
                case VriFormat_BC1_SRGB:
                case VriFormat_BC4_UNORM:
                case VriFormat_BC4_SNORM:
                    return 8; // per 4x4 block
                case VriFormat_BC2_UNORM:
                case VriFormat_BC2_SRGB:
                case VriFormat_BC3_UNORM:
                case VriFormat_BC3_SRGB:
                case VriFormat_BC5_UNORM:
                case VriFormat_BC5_SNORM:
                case VriFormat_BC6H_UFLOAT:
                case VriFormat_BC6H_SFLOAT:
                case VriFormat_BC7_UNORM:
                case VriFormat_BC7_SRGB:
                case VriFormat_ETC2_RGBA8_UNORM:
                case VriFormat_ETC2_RGBA8_SRGB:
                case VriFormat_ASTC_4x4_UNORM:
                case VriFormat_ASTC_4x4_SRGB:
                    return 16; // per 4x4 block
                default:
                    return 4;
            }
        }

        // The feature that enables a compressed format, or WGPUFeatureName_Force32 for an uncompressed one.
        WGPUFeatureName CompressionFeature(VriFormat f)
        {
            switch (f)
            {
                case VriFormat_BC1_UNORM:
                case VriFormat_BC1_SRGB:
                case VriFormat_BC2_UNORM:
                case VriFormat_BC2_SRGB:
                case VriFormat_BC3_UNORM:
                case VriFormat_BC3_SRGB:
                case VriFormat_BC4_UNORM:
                case VriFormat_BC4_SNORM:
                case VriFormat_BC5_UNORM:
                case VriFormat_BC5_SNORM:
                case VriFormat_BC6H_UFLOAT:
                case VriFormat_BC6H_SFLOAT:
                case VriFormat_BC7_UNORM:
                case VriFormat_BC7_SRGB:
                    return WGPUFeatureName_TextureCompressionBC;
                case VriFormat_ETC2_RGBA8_UNORM:
                case VriFormat_ETC2_RGBA8_SRGB:
                    return WGPUFeatureName_TextureCompressionETC2;
                case VriFormat_ASTC_4x4_UNORM:
                case VriFormat_ASTC_4x4_SRGB:
                    return WGPUFeatureName_TextureCompressionASTC;
                default:
                    return WGPUFeatureName_Force32;
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
        VriFormatSupportFlags VRI_CALL GetFormatSupport(const VriDevice* device, VriFormat format)
        {
            VriFormatSupportFlags r = VriFormatSupport_None;

            // Compressed formats are only sampled, and only with the feature of their family.
            const WGPUFeatureName compression = CompressionFeature(format);
            if (compression != WGPUFeatureName_Force32)
            {
                return wgpuDeviceHasFeature(Dev(device)->Device(), compression) != 0 ? VriFormatSupport_Texture :
                                                                                       VriFormatSupport_None;
            }

            // Depth32FloatStencil8 is behind the optional depth32float-stencil8
            // feature, which DeviceWGPU::Init does not request - so
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
                    // The mapped formats core WebGPU can store; a layout names the one it binds
                    // through VriDescriptorRangeDesc::format. Read-write access is core for
                    // the r32 formats only; the others need VriDescriptorRange_WriteOnly.
                    switch (format)
                    {
                        case VriFormat_RGBA8_UNORM:
                        case VriFormat_RGBA16_SFLOAT:
                        case VriFormat_R32_UINT:
                        case VriFormat_R32_SFLOAT:
                        case VriFormat_RG32_SFLOAT:
                        case VriFormat_RGBA32_SFLOAT:
                            r |= VriFormatSupport_StorageTexture;
                            break;
                        default:
                            break;
                    }
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
            DeviceWGPU* d = Dev(device);
            if (core::RejectDesc(*d, core::CheckBufferDesc("CreateBuffer", *desc)))
            {
                *out = nullptr;
                return VriResult_InvalidArgument;
            }
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
            {
                // WebGPU exposes no allocation query, so this is the padded request - the size the
                // implementation was asked for, which is the honest figure available here.
                VriObjectInfo info = MakeObjectInfo(VriObjectType_Buffer);
                info.memoryBytes   = bd.size;
                info.width         = static_cast<uint32_t>(desc->size);
                d->Objects().Track(ToHandle(b), info);
            }
            *out = ToHandle(b);
            return VriResult_Success;
        }

        void VRI_CALL DestroyBuffer(VriBuffer* buffer)
        {
            if (!buffer)
                return;
            BufferWGPU* b = Buf(buffer);
#if defined(__EMSCRIPTEN__)
            // Destroying the buffer cancels its MapBufferAsync: VRI's callback runs here, once, with failure. The
            // browser's answer (an abort, from wgpuBufferRelease below) finds the request cancelled and only frees it.
            // A buffer has at most one: UnmapBuffer's abort calls back inside wgpuBufferUnmap and clears pendingMap.
            if (AsyncMapWGPU* request = std::exchange(b->pendingMap, nullptr))
            {
                request->cancelled = true;
                if (request->device)
                    std::erase(request->device->PendingMaps(), request);
                request->callback(request->userData, VriResult_Failure, nullptr);
            }
#endif
            b->device->Objects().Untrack(buffer);
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

        void* MappedRange(WGPUBuffer buffer, WGPUMapMode mode, uint64_t offset, uint64_t range)
        {
            // Dawn (emdawnwebgpu) requires the const accessor for read-mapped buffers;
            // GetMappedRange asserts there. wgpu-native accepts either.
            if (mode == WGPUMapMode_Read)
                return const_cast<void*>(wgpuBufferGetConstMappedRange(buffer, offset, range));
            return wgpuBufferGetMappedRange(buffer, offset, range);
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
            if (!CanWait())
            {
                b->device->ReportError("MapBuffer: mapping a GPU buffer in the browser waits for the GPU, which "
                                       "needs -sASYNCIFY");
                return nullptr;
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
            return MappedRange(b->buffer, mode, offset, size ? size : WGPU_WHOLE_MAP_SIZE);
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

#if defined(__EMSCRIPTEN__)
        void OnAsyncMap(WGPUMapAsyncStatus status, WGPUStringView message, void* ud1, void*)
        {
            const std::unique_ptr<AsyncMapWGPU> request(static_cast<AsyncMapWGPU*>(ud1));
            if (request->cancelled) // DestroyBuffer already called back; the buffer is gone
                return;
            request->buffer->pendingMap = nullptr;
            if (request->device)
                std::erase(request->device->PendingMaps(), request.get());
            if (status != WGPUMapAsyncStatus_Success)
            {
                if (request->device)
                {
                    std::string text = "MapBufferAsync: wgpuBufferMapAsync failed";
                    if (message.data && message.length)
                        text.append(": ").append(
                            message.data, message.length == WGPU_STRLEN ? std::strlen(message.data) : message.length);
                    request->device->ReportError(text.c_str());
                }
                request->callback(request->userData, VriResult_Failure, nullptr);
                return;
            }
            void* data = MappedRange(request->buffer->buffer, request->mode, request->offset, request->range);
            request->callback(request->userData, data ? VriResult_Success : VriResult_Failure, data);
        }
#endif

        VriResult VRI_CALL MapBufferAsync(VriBuffer*              buffer,
                                          uint64_t                offset,
                                          uint64_t                size,
                                          VriBufferMappedCallback callback,
                                          void*                   userData)
        {
#if defined(__EMSCRIPTEN__)
            // A GPU buffer in the browser maps from the event loop; a host-upload buffer hands back its
            // CPU shadow at once, as everywhere else.
            BufferWGPU* b = buffer ? Buf(buffer) : nullptr;
            if (b && callback && !b->shadow)
            {
                const WGPUMapMode mode    = b->mapMode != WGPUMapMode_None ? b->mapMode : WGPUMapMode_Read;
                const uint64_t    range   = size ? size : WGPU_WHOLE_MAP_SIZE;
                auto*             request = new AsyncMapWGPU {b->device, b, mode, offset, range, callback, userData};
                b->pendingMap             = request;
                b->device->PendingMaps().push_back(request);
                WGPUBufferMapCallbackInfo cb = {};
                cb.mode                      = kCallbackMode;
                cb.callback                  = OnAsyncMap;
                cb.userdata1                 = request;
                wgpuBufferMapAsync(b->buffer, mode, offset, range, cb);
                return VriResult_Success;
            }
#endif
            return core::MapBufferNow(MapBuffer, buffer, offset, size, callback, userData);
        }

        uint64_t VRI_CALL GetBufferDeviceAddress(const VriBuffer*) { return 0; } // not exposed by WebGPU

        VriResult VRI_CALL CreateTexture(VriDevice* device, const VriTextureDesc* desc, VriTexture** out)
        {
            DeviceWGPU* d = Dev(device);
            if (core::RejectDesc(*d, core::CheckTextureDesc("CreateTexture", *desc)))
            {
                *out = nullptr;
                return VriResult_InvalidArgument;
            }
            WGPUTextureDescriptor td   = {};
            td.usage                   = ToWgpuTextureUsage(desc->usage);
            td.dimension               = desc->type == VriTextureType_3D ? WGPUTextureDimension_3D :
                                         desc->type == VriTextureType_1D || desc->type == VriTextureType_1DArray ?
                                                                           WGPUTextureDimension_1D :
                                                                           WGPUTextureDimension_2D;
            td.size.width              = desc->width;
            td.size.height             = desc->height;
            td.size.depthOrArrayLayers = desc->type == VriTextureType_3D ? desc->depth : desc->layerNum;
            td.format                  = ToWgpuFormat(desc->format);
            td.mipLevelCount           = desc->mipNum;
            td.sampleCount             = desc->sampleNum;

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
            t->blockDim    = CompressionFeature(desc->format) != WGPUFeatureName_Force32 ? 4u : 1u;
            t->owned       = true;
            {
                VriObjectInfo  info = MakeObjectInfo(VriObjectType_Texture);
                const uint32_t blocks =
                    ((t->width + t->blockDim - 1) / t->blockDim) * ((t->height + t->blockDim - 1) / t->blockDim);
                uint64_t size = static_cast<uint64_t>(blocks) * t->depth * t->texelSize;
                if (t->mipNum > 1)
                    size += size / 3;
                info.memoryBytes = size;
                info.width       = t->width;
                info.height      = t->height;
                info.depth       = t->depth;
                info.mipNum      = t->mipNum;
                info.layerNum    = t->layerNum;
                info.format      = desc->format;
                t->device->Objects().Track(ToHandle(t), info);
            }
            *out = ToHandle(t);
            return VriResult_Success;
        }

        void VRI_CALL DestroyTexture(VriTexture* texture)
        {
            if (!texture)
                return;
            TextureWGPU* t = Tex(texture);
            t->device->Objects().Untrack(texture);
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
            if (!device || !desc || !out || (desc->descriptorSetNum && !desc->descriptorSets))
                return VriResult_InvalidArgument;
            *out = nullptr;
            for (uint32_t s = 0; s < desc->descriptorSetNum; ++s)
            {
                const auto& set = desc->descriptorSets[s];
                if (set.rangeNum && !set.ranges)
                    return VriResult_InvalidArgument;
                for (uint32_t r = 0; r < set.rangeNum; ++r)
                {
                    if (set.ranges[r].descriptorNum != 1 ||
                        (set.ranges[r].flags & (VriDescriptorRange_PartiallyBound | VriDescriptorRange_VariableSized)))
                        return VriResult_Unsupported;
                }
            }
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
                            // Depth textures read without comparison are unfilterable floats (a depth view is
                            // accepted there), and so are 32-bit float ones unless the device has
                            // float32-filterable; the mapped integer ones are uint.
                            e.texture.sampleType = WGPUTextureSampleType_Float;
                            if (range.flags & VriDescriptorRange_Comparison)
                                e.texture.sampleType = WGPUTextureSampleType_Depth;
                            else if (vri::FormatHasDepth(range.format) ||
                                     (!d->HasFloat32Filterable() &&
                                      (range.format == VriFormat_R32_SFLOAT || range.format == VriFormat_RG32_SFLOAT ||
                                       range.format == VriFormat_RGBA32_SFLOAT)))
                                e.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
                            else if (range.format == VriFormat_R8_UINT || range.format == VriFormat_R32_UINT)
                                e.texture.sampleType = WGPUTextureSampleType_Uint;
                            if (range.viewType == VriTextureViewType_2DArray)
                                e.texture.viewDimension = WGPUTextureViewDimension_2DArray;
                            else if (range.viewType == VriTextureViewType_Cube)
                                e.texture.viewDimension = WGPUTextureViewDimension_Cube;
                            else if (range.viewType == VriTextureViewType_3D)
                                e.texture.viewDimension = WGPUTextureViewDimension_3D;
                            else
                                e.texture.viewDimension = WGPUTextureViewDimension_2D; // 0/1D/2D -> 2D
                            break;
                        case VriDescriptorType_StorageTexture:
                            // Access, format and dimension must match the WGSL declaration exactly.
                            e.storageTexture.access = (range.flags & VriDescriptorRange_WriteOnly) ?
                                                          WGPUStorageTextureAccess_WriteOnly :
                                                          WGPUStorageTextureAccess_ReadWrite;
                            e.storageTexture.format = range.format == VriFormat_Unknown ? WGPUTextureFormat_RGBA8Unorm :
                                                                                          ToWgpuFormat(range.format);
                            if (range.viewType == VriTextureViewType_2DArray)
                                e.storageTexture.viewDimension = WGPUTextureViewDimension_2DArray;
                            else if (range.viewType == VriTextureViewType_3D)
                                e.storageTexture.viewDimension = WGPUTextureViewDimension_3D;
                            else
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
                    // WebGPU forbids writable storage in the vertex stage. Layouts written for the other
                    // backends often name every stage, so drop the vertex bit where it can never apply.
                    if (range.descriptorType == VriDescriptorType_StorageBuffer ||
                        range.descriptorType == VriDescriptorType_StorageTexture)
                        e.visibility &= ~WGPUShaderStage_Vertex;
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
                WGPUBindGroupLayoutEntry e = {};
                e.binding                  = 0;
                e.visibility              = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment | WGPUShaderStage_Compute;
                e.buffer.type             = WGPUBufferBindingType_Uniform;
                e.buffer.hasDynamicOffset = true; // per-draw push values live at different ring offsets
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
            DeviceWGPU* d = Dev(device);
            // A compute layout with push constants needs the same WGSL binding injection as a render one.
            const PipelineLayoutWGPU*     pl        = desc->pipelineLayout ? PL(desc->pipelineLayout) : nullptr;
            const int                     pushGroup = (pl && pl->hasPush) ? static_cast<int>(pl->pushGroup) : -1;
            WGPUShaderModule              mod       = MakeWgslModule(d->Device(), desc->shader.bytecode, pushGroup);
            WGPUComputePipelineDescriptor pd        = {};
            pd.layout                               = desc->pipelineLayout ? PL(desc->pipelineLayout)->layout : nullptr;
            pd.compute.module                       = mod;
            pd.compute.entryPoint    = SV(desc->shader.entryPointName ? desc->shader.entryPointName : "main");
            WGPUComputePipeline pipe = wgpuDeviceCreateComputePipeline(d->Device(), &pd);
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
                s->bound.resize(l->setRanges[setIndex].size(), nullptr);
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
            if (!set)
                return;
            DescriptorSetWGPU* s      = DSet(set);
            const auto&        ranges = s->layout->setRanges[s->setIndex];
            if (baseRange > ranges.size() || rangeNum > ranges.size() - baseRange || (rangeNum && !updates))
            {
                s->device->ReportError("UpdateDescriptorRanges: range index is outside the set layout");
                return;
            }
            for (uint32_t i = 0; i < rangeNum; ++i)
            {
                const auto& update = updates[i];
                if (update.descriptorNum == 0)
                    continue;
                if (update.baseDescriptor != 0 || update.descriptorNum != 1 || !update.descriptors ||
                    !update.descriptors[0])
                {
                    s->device->ReportError("UpdateDescriptorRanges: invalid descriptor array range");
                    return;
                }
                const auto* descriptor = Desc(const_cast<VriDescriptor*>(update.descriptors[0]));
                if (descriptor->device != s->device)
                {
                    s->device->ReportError("UpdateDescriptorRanges: descriptor belongs to another device");
                    return;
                }
            }
            for (uint32_t i = 0; i < rangeNum; ++i)
            {
                if (updates[i].descriptorNum)
                {
                    s->bound[baseRange + i] = Desc(const_cast<VriDescriptor*>(updates[i].descriptors[0]));
                    s->dirty                = true;
                }
            }
        }

        bool UpdateBindGroup(DescriptorSetWGPU* set)
        {
            if (!set->dirty)
                return set->bindGroup != nullptr;
            const auto&                     ranges = set->layout->setRanges[set->setIndex];
            std::vector<WGPUBindGroupEntry> entries;
            entries.reserve(ranges.size());
            for (size_t i = 0; i < ranges.size(); ++i)
            {
                const DescriptorWGPU* descriptor = set->bound[i];
                if (!descriptor)
                {
                    set->device->ReportError("CmdSetDescriptorSet: a required descriptor range is not initialized");
                    return false;
                }
                const auto&        range = ranges[i];
                WGPUBindGroupEntry entry {};
                entry.binding = range.binding;
                if (range.type == VriDescriptorType_ConstantBuffer || range.type == VriDescriptorType_StorageBuffer ||
                    range.type == VriDescriptorType_StructuredBuffer)
                {
                    entry.buffer = descriptor->buffer ? descriptor->buffer->buffer : nullptr;
                    entry.offset = descriptor->bufferOffset;
                    entry.size   = descriptor->bufferRange ? descriptor->bufferRange : WGPU_WHOLE_SIZE;
                }
                else if (range.type == VriDescriptorType_Sampler)
                    entry.sampler = descriptor->sampler;
                else
                    entry.textureView = descriptor->view;
                entries.push_back(entry);
            }
            WGPUBindGroupDescriptor desc {};
            desc.layout         = set->layout->bindGroupLayouts[set->setIndex];
            desc.entryCount     = entries.size();
            desc.entries        = entries.data();
            WGPUBindGroup group = wgpuDeviceCreateBindGroup(set->device->Device(), &desc);
            if (!group)
                return false;
            if (set->bindGroup)
                wgpuBindGroupRelease(set->bindGroup);
            set->bindGroup = group;
            set->dirty     = false;
            return true;
        }

        // ---- synchronization (emulated timeline) ---------------------------
        VriResult VRI_CALL CreateFence(VriDevice* device, uint64_t initialValue, VriFence** out)
        {
            if (!device || !out)
                return VriResult_InvalidArgument;
            *out = ToHandle(new FenceWGPU {Dev(device), initialValue, initialValue});
            return VriResult_Success;
        }
        void VRI_CALL DestroyFence(VriFence* fence)
        {
            if (!fence)
                return;
            FenceWGPU* f = Fen(fence);
            // A submission that signals this fence may still be in flight (the browser cannot wait for it). Its
            // completion record stays until its callback has run; drop this fence from it so draining it later never
            // writes through the destroyed fence.
            QueueWGPU* queue = f->device->GetQueue(VriQueueType_Graphics);
            {
                std::lock_guard lock(queue->completionMutex);
                for (auto& completion : queue->completions)
                    std::erase_if(completion->signals, [f](const auto& signal) { return signal.first == f; });
            }
            delete f;
        }
        bool DrainCompletions(QueueWGPU* queue)
        {
            std::lock_guard lock(queue->completionMutex);
            // Native callbacks can arrive out of order. Publish only the completed
            // submission prefix, so a later value cannot hide an earlier callback.
            while (!queue->completions.empty() && queue->completions.front()->done.load(std::memory_order_acquire))
            {
                const auto& completion = *queue->completions.front();
                for (const auto& [fence, value] : completion.signals)
                {
                    if (completion.status == WGPUQueueWorkDoneStatus_Success)
                        fence->value.store(value, std::memory_order_release);
                    else
                        fence->failed.store(true, std::memory_order_release);
                }
                queue->completions.pop_front();
            }
            return queue->completions.empty();
        }

        uint64_t VRI_CALL GetFenceValue(VriFence* fence)
        {
            FenceWGPU* f = Fen(fence);
#if !defined(__EMSCRIPTEN__)
            wgpuDevicePoll(f->device->Device(), false, nullptr);
#endif
            DrainCompletions(f->device->GetQueue(VriQueueType_Graphics));
            return f->value.load(std::memory_order_acquire);
        }
        VriResult VRI_CALL WaitForFence(VriFence* fence, uint64_t value, uint64_t timeoutNs)
        {
            if (!fence)
                return VriResult_InvalidArgument;
            FenceWGPU*         f = Fen(fence);
            core::WaitDeadline deadline(timeoutNs);
            while (GetFenceValue(fence) < value)
            {
                if (f->failed.load(std::memory_order_acquire))
                    return VriResult_DeviceLost;
                if (!deadline.Remaining())
                    return VriResult_Timeout;
#if defined(__EMSCRIPTEN__)
                if (!CanWait())
                {
                    // The fence can only advance once the page returns to the browser event loop.
                    if (timeoutNs != UINT64_MAX)
                        return VriResult_Timeout;
                    f->device->ReportError("WaitForFence: an unbounded wait in the browser needs -sASYNCIFY; poll "
                                           "GetFenceValue on a later frame instead");
                    return VriResult_Unsupported;
                }
                emscripten_sleep(1);
#else
                if (timeoutNs == UINT64_MAX)
                    wgpuDevicePoll(f->device->Device(), true, nullptr);
                else
                    deadline.Pause();
#endif
            }
            return VriResult_Success;
        }

        void VRI_CALL Wait(VriFence* fence, uint64_t value)
        {
            if (WaitForFence(fence, value, UINT64_MAX) != VriResult_Success && fence)
                Fen(fence)->device->ReportError("Wait: WebGPU timeline synchronization failed");
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
            DescriptorSetWGPU* s = const_cast<DescriptorSetWGPU*>(reinterpret_cast<const DescriptorSetWGPU*>(set));
            if (!UpdateBindGroup(s))
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
        // IEEE half nearest to `value`, rounding ties up; enough for clear colors.
        uint16_t ToHalf(float value)
        {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            const uint32_t sign     = (bits >> 16) & 0x8000u;
            const uint32_t biased   = (bits >> 23) & 0xffu;
            const int32_t  exponent = int32_t(biased) - 127 + 15;
            uint32_t       mantissa = bits & 0x7fffffu;
            if (biased == 0xffu)
                return uint16_t(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
            if (exponent >= 31)
                return uint16_t(sign | 0x7c00u);
            if (exponent <= 0)
            {
                if (exponent < -10)
                    return uint16_t(sign);
                mantissa |= 0x800000u;
                const uint32_t shift = uint32_t(14 - exponent);
                return uint16_t(sign | ((mantissa >> shift) + ((mantissa >> (shift - 1)) & 1u)));
            }
            return uint16_t((sign | (uint32_t(exponent) << 10) | (mantissa >> 13)) + ((mantissa >> 12) & 1u));
        }

        // One texel of `format` holding `value`, for the formats WebGPU stores; empty for every other format.
        std::vector<uint8_t> ClearTexel(WGPUTextureFormat format, const VriClearColor& value)
        {
            std::vector<uint8_t> texel;
            const auto           raw = [&](uint32_t components) {
                texel.resize(components * 4);
                std::memcpy(texel.data(), value.u32, texel.size());
            };
            const auto half = [&](uint32_t components) {
                for (uint32_t i = 0; i < components; ++i)
                {
                    const uint16_t h = ToHalf(value.f32[i]);
                    texel.push_back(uint8_t(h & 0xffu));
                    texel.push_back(uint8_t(h >> 8));
                }
            };
            switch (format)
            {
                case WGPUTextureFormat_R32Float:
                case WGPUTextureFormat_R32Uint:
                case WGPUTextureFormat_R32Sint:
                    raw(1);
                    break;
                case WGPUTextureFormat_RG32Float:
                case WGPUTextureFormat_RG32Uint:
                case WGPUTextureFormat_RG32Sint:
                    raw(2);
                    break;
                case WGPUTextureFormat_RGBA32Float:
                case WGPUTextureFormat_RGBA32Uint:
                case WGPUTextureFormat_RGBA32Sint:
                    raw(4);
                    break;
                case WGPUTextureFormat_R16Float:
                    half(1);
                    break;
                case WGPUTextureFormat_RG16Float:
                    half(2);
                    break;
                case WGPUTextureFormat_RGBA16Float:
                    half(4);
                    break;
                case WGPUTextureFormat_RGBA8Unorm:
                    for (uint32_t i = 0; i < 4; ++i)
                    {
                        float unit = value.f32[i];
                        if (unit < 0.0f)
                            unit = 0.0f;
                        if (unit > 1.0f)
                            unit = 1.0f;
                        texel.push_back(uint8_t(unit * 255.0f + 0.5f));
                    }
                    break;
                default:
                    break;
            }
            return texel;
        }

        // WebGPU clears a texture only through a render pass loadOp, so this copies mip 0 from a buffer of
        // the value instead, which needs CopyDst usage (as vkCmdClearColorImage needs TransferDst).
        void VRI_CALL CmdClearStorageTexture(VriCommandBuffer* cmd, VriTexture* texture, const VriClearColor* value)
        {
            CommandBufferWGPU*         c     = CB(cmd);
            const TextureWGPU*         t     = reinterpret_cast<const TextureWGPU*>(texture);
            const std::vector<uint8_t> texel = ClearTexel(t->format, *value);
            if (texel.empty())
            {
                c->device->ReportError("CmdClearStorageTexture: unsupported texture format on WebGPU");
                return;
            }
            EndComputePass(c);
            const uint32_t       slices = t->layerNum > 1 ? t->layerNum : t->depth;
            const uint32_t       pitch  = (uint32_t(texel.size()) * t->width + 255u) & ~255u;
            WGPUBufferDescriptor bd     = {};
            bd.size                     = uint64_t(pitch) * t->height * slices;
            bd.usage                    = WGPUBufferUsage_CopySrc;
            bd.mappedAtCreation         = true;
            WGPUBuffer temp             = wgpuDeviceCreateBuffer(c->device->Device(), &bd);
            auto*      data = temp ? static_cast<uint8_t*>(wgpuBufferGetMappedRange(temp, 0, bd.size)) : nullptr;
            if (!data)
            {
                if (temp)
                    wgpuBufferRelease(temp);
                c->device->ReportError("CmdClearStorageTexture: could not create the clear buffer");
                return;
            }
            for (uint64_t row = 0; row < uint64_t(t->height) * slices; ++row)
                for (uint32_t x = 0; x < t->width; ++x)
                    std::memcpy(data + row * pitch + x * texel.size(), texel.data(), texel.size());
            wgpuBufferUnmap(temp);

            WGPUTexelCopyBufferInfo srcInfo  = {};
            srcInfo.buffer                   = temp;
            srcInfo.layout.bytesPerRow       = pitch;
            srcInfo.layout.rowsPerImage      = t->height;
            WGPUTexelCopyTextureInfo dstInfo = {};
            dstInfo.texture                  = t->texture;
            dstInfo.aspect                   = WGPUTextureAspect_All;
            WGPUExtent3D ext                 = {t->width, t->height, slices};
            wgpuCommandEncoderCopyBufferToTexture(c->encoder, &srcInfo, &dstInfo, &ext);
            c->tempUploads.push_back(temp); // keep alive until the GPU consumes it
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

            // As in CmdUploadBufferToTexture: layers of an array are origin.z and depthOrArrayLayers, too.
            const bool     isArray  = t->layerNum > 1;
            const uint32_t originZ  = isArray ? region->texture.baseLayer : static_cast<uint32_t>(region->texture.z);
            const uint32_t extDepth = isArray ? (region->texture.layerNum ? region->texture.layerNum : 1u) :
                                                (region->texture.depth ? region->texture.depth : 1u);
            WGPUTexelCopyTextureInfo srcInfo = {};
            srcInfo.texture                  = t->texture;
            srcInfo.mipLevel                 = region->texture.mip;
            srcInfo.origin                   = {
                static_cast<uint32_t>(region->texture.x), static_cast<uint32_t>(region->texture.y), originZ};
            srcInfo.aspect = WGPUTextureAspect_All;

            // As in CmdUploadBufferToTexture, compressed formats count rows and pitches in blocks.
            const uint32_t          block     = t->blockDim;
            const uint32_t          rowTexels = region->bufferRowLength ? region->bufferRowLength : w;
            const uint32_t          imageRows = region->bufferImageHeight ? region->bufferImageHeight : h;
            WGPUTexelCopyBufferInfo dstInfo   = {};
            dstInfo.buffer                    = Buf(dst)->buffer;
            dstInfo.layout.offset             = region->bufferOffset;
            dstInfo.layout.bytesPerRow        = (rowTexels + block - 1) / block * t->texelSize;
            dstInfo.layout.rowsPerImage       = (imageRows + block - 1) / block;

            const uint32_t     blockRows = (h + block - 1) / block;
            const WGPUExtent3D ext       = {(w + block - 1) / block * block, blockRows * block, extDepth};
            wgpuCommandEncoderCopyTextureToBuffer(CB(cmd)->encoder, &srcInfo, &dstInfo, &ext);
        }

        void VRI_CALL CmdUploadBufferToTexture(VriCommandBuffer*               cmd,
                                               VriTexture*                     dst,
                                               VriBuffer*                      src,
                                               const VriBufferTextureCopyDesc* region)
        {
            CommandBufferWGPU* c = CB(cmd);
            EndComputePass(c);
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
            dstInfo.aspect = WGPUTextureAspect_All;

            // Compressed formats copy whole blocks: rows and pitches count blocks, and the extent of a mip
            // smaller than a block, or not a multiple of it, is its physical size in blocks.
            const uint32_t block     = t->blockDim;
            const uint32_t blockRows = (h + block - 1) / block;
            const uint32_t rowTexels = region->bufferRowLength ? region->bufferRowLength : w;
            const uint32_t imageRows = region->bufferImageHeight ? region->bufferImageHeight : h;
            WGPUExtent3D   ext       = {(w + block - 1) / block * block, blockRows * block, extDepth};

            // copyBufferToTexture requires bytesPerRow to be a multiple of 256. If the app's
            // staging is tightly packed and narrower than that, pack the rows into a temp buffer
            // at the aligned pitch first (per-row buffer copy), so any texture size just works.
            const uint32_t          tightPitch   = (rowTexels + block - 1) / block * t->texelSize;
            const uint32_t          alignedPitch = (tightPitch + 255u) & ~255u;
            const uint32_t          sourceRows   = (imageRows + block - 1) / block; // block rows per source slice
            WGPUTexelCopyBufferInfo srcInfo      = {};
            srcInfo.layout.rowsPerImage          = sourceRows;
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
                bd.size                 = uint64_t(alignedPitch) * blockRows * extDepth;
                bd.usage                = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
                WGPUBuffer temp         = wgpuDeviceCreateBuffer(c->device->Device(), &bd);
                for (uint32_t slice = 0; slice < extDepth; ++slice)
                    for (uint32_t r = 0; r < blockRows; ++r)
                        wgpuCommandEncoderCopyBufferToBuffer(c->encoder,
                                                             Buf(src)->buffer,
                                                             region->bufferOffset +
                                                                 (uint64_t(slice) * sourceRows + r) * tightPitch,
                                                             temp,
                                                             (uint64_t(slice) * blockRows + r) * alignedPitch,
                                                             tightPitch);
                srcInfo.buffer              = temp;
                srcInfo.layout.offset       = 0;
                srcInfo.layout.bytesPerRow  = alignedPitch;
                srcInfo.layout.rowsPerImage = blockRows;
                wgpuCommandEncoderCopyBufferToTexture(c->encoder, &srcInfo, &dstInfo, &ext);
                c->tempUploads.push_back(temp); // keep alive until the GPU consumes it
            }
        }

        void VRI_CALL CmdCopyTexture(VriCommandBuffer*         cmd,
                                     VriTexture*               dst,
                                     VriTexture*               src,
                                     const VriTextureCopyDesc* region)
        {
            CommandBufferWGPU* c = CB(cmd);
            EndComputePass(c);
            const TextureWGPU*         s           = reinterpret_cast<const TextureWGPU*>(src);
            const TextureWGPU*         d           = reinterpret_cast<const TextureWGPU*>(dst);
            const VriTextureRegionDesc source      = region ? region->src : VriTextureRegionDesc {};
            const VriTextureRegionDesc destination = region ? region->dst : VriTextureRegionDesc {};
            // As in CmdUploadBufferToTexture: array layers are origin.z and depthOrArrayLayers, as volume slices are.
            // A zero size is the whole source mip, as in the Vulkan backend.
            const bool               sourceArray      = s->layerNum > 1;
            const bool               destinationArray = d->layerNum > 1;
            const uint32_t           layers           = source.layerNum ? source.layerNum : 1u;
            WGPUTexelCopyTextureInfo si               = {};
            si.texture                                = s->texture;
            si.aspect                                 = WGPUTextureAspect_All;
            si.mipLevel                               = source.mip;
            si.origin                                 = {static_cast<uint32_t>(source.x),
                                                         static_cast<uint32_t>(source.y),
                         sourceArray ? source.baseLayer : static_cast<uint32_t>(source.z)};
            WGPUTexelCopyTextureInfo di               = {};
            di.texture                                = d->texture;
            di.aspect                                 = WGPUTextureAspect_All;
            di.mipLevel                               = destination.mip;
            di.origin                                 = {static_cast<uint32_t>(destination.x),
                                                         static_cast<uint32_t>(destination.y),
                         destinationArray ? destination.baseLayer : static_cast<uint32_t>(destination.z)};
            // Compressed copies cover whole blocks, as in CmdUploadBufferToTexture.
            const uint32_t     block  = s->blockDim;
            const uint32_t     width  = source.width ? source.width : std::max(s->width >> source.mip, 1u);
            const uint32_t     height = source.height ? source.height : std::max(s->height >> source.mip, 1u);
            const WGPUExtent3D ext    = {
                (width + block - 1) / block * block,
                (height + block - 1) / block * block,
                sourceArray ? layers : (source.depth ? source.depth : std::max(s->depth >> source.mip, 1u))};
            wgpuCommandEncoderCopyTextureToTexture(c->encoder, &si, &di, &ext);
        }

        void VRI_CALL CmdBeginDebugGroup(VriCommandBuffer*, const char*) {}
        void VRI_CALL CmdEndDebugGroup(VriCommandBuffer*) {}

        // ---- submission ----------------------------------------------------
        VriResult VRI_CALL QueueSubmit(VriQueue* queue, const VriQueueSubmitDesc* submit)
        {
            if (!queue || !core::ValidSubmitPointers(submit))
                return VriResult_InvalidArgument;
            QueueWGPU* q = Q(queue);
            for (uint32_t i = 0; i < submit->waitFenceNum; ++i)
                if (Fen(submit->waitFences[i].fence)->device != q->device)
                    return VriResult_InvalidArgument;
            for (uint32_t i = 0; i < submit->commandBufferNum; ++i)
            {
                const CommandBufferWGPU* command = CB(submit->commandBuffers[i]);
                if (command->device != q->device || !command->finished)
                    return VriResult_InvalidArgument;
            }
            for (uint32_t i = 0; i < submit->signalFenceNum; ++i)
            {
                const auto& signal = submit->signalFences[i];
                if (Fen(signal.fence)->device != q->device || signal.value <= Fen(signal.fence)->submittedValue)
                    return VriResult_InvalidArgument;
            }
            // Submit-only loops must retire completed signals without introducing a GPU wait.
#if !defined(__EMSCRIPTEN__)
            wgpuDevicePoll(q->device->Device(), false, nullptr);
#endif
            DrainCompletions(q);
            // Every logical queue is the one native queue, which runs submissions in submission order: a wait on a
            // value an earlier submission signals is already ordered on the GPU and needs no host wait. A value no
            // submission signals can never arrive before this one.
            for (uint32_t i = 0; i < submit->waitFenceNum; ++i)
            {
                const auto& wait = submit->waitFences[i];
                FenceWGPU*  f    = Fen(wait.fence);
                if (f->failed.load(std::memory_order_acquire))
                    return VriResult_DeviceLost;
                if (wait.value > f->submittedValue)
                {
                    q->device->ReportError("QueueSubmit: a wait fence value that no earlier submission signals");
                    return VriResult_InvalidArgument;
                }
            }

            std::vector<WGPUCommandBuffer> commands;
            commands.reserve(submit->commandBufferNum);
            for (uint32_t i = 0; i < submit->commandBufferNum; ++i)
                commands.push_back(CB(submit->commandBuffers[i])->finished);
            SubmissionCompletionWGPU* completion = nullptr;
            if (submit->signalFenceNum)
            {
                auto pending = std::make_unique<SubmissionCompletionWGPU>();
                pending->signals.reserve(submit->signalFenceNum);
                for (uint32_t i = 0; i < submit->signalFenceNum; ++i)
                {
                    const auto& signal = submit->signalFences[i];
                    pending->signals.emplace_back(Fen(signal.fence), signal.value);
                }
                completion = pending.get();
                std::lock_guard lock(q->completionMutex);
                q->completions.push_back(std::move(pending));
            }
            wgpuQueueSubmit(q->queue, commands.size(), commands.data());
            for (uint32_t i = 0; i < submit->commandBufferNum; ++i)
            {
                auto* command = CB(submit->commandBuffers[i]);
                wgpuCommandBufferRelease(command->finished);
                command->finished = nullptr;
            }
            if (completion)
            {
                for (const auto& [fence, value] : completion->signals)
                    fence->submittedValue = value;
                WGPUQueueWorkDoneCallbackInfo callback {};
                callback.mode      = kCallbackMode;
                callback.userdata1 = completion;
                callback.callback  = [](WGPUQueueWorkDoneStatus status,
#if defined(__EMSCRIPTEN__)
                                       WGPUStringView, // Dawn includes a diagnostic message; wgpu-native does not.
#endif
                                       void* user,
                                       void*) {
                    // The queue owns this record until the host observes done. The
                    // callback never dereferences borrowed fence handles.
                    auto* result   = static_cast<SubmissionCompletionWGPU*>(user);
                    result->status = status;
                    result->done.store(true, std::memory_order_release);
                };
                wgpuQueueOnSubmittedWorkDone(q->queue, callback);
            }
            return VriResult_Success;
        }

        VriResult VRI_CALL QueueWaitIdle(VriQueue* queue)
        {
            if (!queue)
                return VriResult_InvalidArgument;
            if (!CanWait())
            {
                Q(queue)->device->ReportError("QueueWaitIdle: waiting for the GPU in the browser needs -sASYNCIFY");
                return VriResult_Unsupported;
            }
            struct Completion
            {
                std::atomic<bool>       done {false};
                WGPUQueueWorkDoneStatus status = WGPUQueueWorkDoneStatus_Error;
            } completion;
            WGPUQueueWorkDoneCallbackInfo callback {};
            callback.mode      = kCallbackMode;
            callback.userdata1 = &completion;
            callback.callback  = [](WGPUQueueWorkDoneStatus status,
#if defined(__EMSCRIPTEN__)
                                   WGPUStringView, // Dawn includes a diagnostic message; wgpu-native does not.
#endif
                                   void* user,
                                   void*) {
                auto* result   = static_cast<Completion*>(user);
                result->status = status;
                result->done.store(true, std::memory_order_release);
            };
            QueueWGPU* q = Q(queue);
            wgpuQueueOnSubmittedWorkDone(q->queue, callback);
            while (!completion.done.load(std::memory_order_acquire) || !DrainCompletions(q))
            {
#if defined(__EMSCRIPTEN__)
                emscripten_sleep(1);
#else
                wgpuDevicePoll(q->device->Device(), true, nullptr);
#endif
            }
            return completion.status == WGPUQueueWorkDoneStatus_Success ? VriResult_Success : VriResult_DeviceLost;
        }
        VriResult VRI_CALL DeviceWaitIdle(VriDevice* device)
        {
            if (!device)
                return VriResult_InvalidArgument;
            return QueueWaitIdle(ToHandle(Dev(device)->GetQueue(VriQueueType_Graphics)));
        }
        // Was a no-op. It records the label against the tracked object, which is what makes an
        // EnumerateObjects listing readable - without it every row is a pointer and a size. Every
        // backend object struct begins with its owning device, which is what lets one entry point
        // name any of them without a per-type dispatch table.
        void VRI_CALL SetDebugName(void* object, const char* name)
        {
            if (object == nullptr)
                return;
            DeviceWGPU* d = *reinterpret_cast<DeviceWGPU**>(object);
            if (d != nullptr)
                d->Objects().SetName(object, name);
        }

        VriResult VRI_CALL EnumerateObjects(const VriDevice* device, uint32_t* count, VriObjectInfo* out)
        {
            if (device == nullptr)
                return VriResult_InvalidArgument;
            return Dev(const_cast<VriDevice*>(device))->Objects().Snapshot(count, out);
        }

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
            t.WaitForFence                = WaitForFence;
            t.MapBufferAsync              = MapBufferAsync;
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
            t.EnumerateObjects            = EnumerateObjects;
            t.GetVideoMemoryInfo          = GetVideoMemoryInfo;
            t.CmdClearStorageBuffer       = CmdClearStorageBuffer;
            t.CmdClearStorageTexture      = CmdClearStorageTexture;
            return t;
        }

        const VriCoreInterface g_coreWGPU = MakeTable();
    } // namespace

    const VriCoreInterface* GetCoreInterfaceWGPU() { return &g_coreWGPU; }
} // namespace vri::wgpu
