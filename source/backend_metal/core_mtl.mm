// core_mtl.mm - native Metal implementation of VriCoreInterface.
//
// Shaders arrive as SPIR-V (VRI's portable shader IR) and are transpiled to MSL
// at pipeline-creation time via SPIRV-Cross (mirrors the GL backend's SPIRV->GLSL
// path), then compiled with newLibraryWithSource. The descriptor model maps each
// VRI (set, binding) to a fixed Metal argument-table slot recorded on the pipeline
// layout, and bound directly (no Metal argument buffers in the MVP).
//
// Metal's clip space is Y-Up, matching VRI's convention, so viewports need no flip
// (unlike the Vulkan backend). Hazards are auto-tracked within a command buffer, so
// CmdBarrier is a no-op. Manual retain/release (no ARC).

#include "core_mtl.h"
#include "conversions_mtl.h"
#include "device_mtl.h"
#include "objects_mtl.h"

#import <Metal/Metal.h>

#include <spirv_cross/spirv_msl.hpp>

#include <string>
#include <vector>

namespace vri::mtl
{
    namespace
    {
        inline DeviceMTL*           Dev(VriDevice* h)            { return reinterpret_cast<DeviceMTL*>(h); }
        inline const DeviceMTL*     Dev(const VriDevice* h)      { return reinterpret_cast<const DeviceMTL*>(h); }
        inline QueueMTL*            Q(VriQueue* h)               { return reinterpret_cast<QueueMTL*>(h); }
        inline CommandAllocatorMTL* CA(VriCommandAllocator* h)   { return reinterpret_cast<CommandAllocatorMTL*>(h); }
        inline CommandBufferMTL*    CB(VriCommandBuffer* h)      { return reinterpret_cast<CommandBufferMTL*>(h); }
        inline BufferMTL*           Buf(VriBuffer* h)            { return reinterpret_cast<BufferMTL*>(h); }
        inline TextureMTL*          Tex(VriTexture* h)           { return reinterpret_cast<TextureMTL*>(h); }
        inline DescriptorMTL*       Desc(VriDescriptor* h)       { return reinterpret_cast<DescriptorMTL*>(h); }
        inline const DescriptorMTL* Desc(const VriDescriptor* h) { return reinterpret_cast<const DescriptorMTL*>(h); }
        inline PipelineLayoutMTL*   PL(VriPipelineLayout* h)     { return reinterpret_cast<PipelineLayoutMTL*>(h); }
        inline const PipelineLayoutMTL* PLc(const VriPipelineLayout* h) { return reinterpret_cast<const PipelineLayoutMTL*>(h); }
        inline PipelineMTL*         Pipe(VriPipeline* h)         { return reinterpret_cast<PipelineMTL*>(h); }
        inline FenceMTL*            Fen(VriFence* h)             { return reinterpret_cast<FenceMTL*>(h); }
        inline DescriptorPoolMTL*   DPool(VriDescriptorPool* h)  { return reinterpret_cast<DescriptorPoolMTL*>(h); }
        inline DescriptorSetMTL*    DSet(VriDescriptorSet* h)    { return reinterpret_cast<DescriptorSetMTL*>(h); }
        inline MemoryMTL*           Mem(VriMemory* h)            { return reinterpret_cast<MemoryMTL*>(h); }

        inline bool IsBufferType(VriDescriptorType t)
        {
            return t == VriDescriptorType_ConstantBuffer || t == VriDescriptorType_StorageBuffer ||
                   t == VriDescriptorType_StructuredBuffer;
        }
        inline bool IsTextureType(VriDescriptorType t)
        {
            return t == VriDescriptorType_Texture || t == VriDescriptorType_StorageTexture;
        }

        // Shared listener for CPU-side timeline waits (MTLSharedEvent has no direct
        // blocking wait); created once, lives for the process.
        MTLSharedEventListener* Listener()
        {
            static MTLSharedEventListener* l = [[MTLSharedEventListener alloc] init];
            return l;
        }

        spv::ExecutionModel ToSpvModel(VriShaderStageBits stage)
        {
            switch (stage)
            {
                case VriShaderStage_Vertex:   return spv::ExecutionModelVertex;
                case VriShaderStage_Fragment: return spv::ExecutionModelFragment;
                case VriShaderStage_Compute:  return spv::ExecutionModelGLCompute;
                default:                      return spv::ExecutionModelVertex;
            }
        }

        MTLResourceOptions StorageOptions(VriMemoryLocation loc)
        {
            // Apple Silicon is unified memory: Shared is host-visible AND GPU-local, so it
            // covers every host-visible location. Device-only resources use Private.
            return loc == VriMemoryLocation_Device ? MTLResourceStorageModePrivate
                                                   : MTLResourceStorageModeShared;
        }

        // ---- SPIR-V -> MSL -------------------------------------------------
        struct MslResult
        {
            std::string source;
            std::string entry;     // cleansed MSL function name (e.g. "main0")
            MTLSize     localSize = {1, 1, 1};
            bool        ok = false;
        };

        MslResult SpirvToMsl(const DeviceMTL* d, const PipelineLayoutMTL* layout, const VriShaderDesc& sh,
                             spv::ExecutionModel model)
        {
            MslResult r;
            const uint32_t* words = static_cast<const uint32_t*>(sh.bytecode);
            const size_t wordCount = sh.bytecodeSize / 4;
            const char* entry = sh.entryPointName ? sh.entryPointName : "main";
            try
            {
                spirv_cross::CompilerMSL comp(words, wordCount);
                spirv_cross::CompilerMSL::Options o = comp.get_msl_options();
                o.platform = spirv_cross::CompilerMSL::Options::macOS;
                o.set_msl_version(2, 1);
                o.argument_buffers = false; // MVP binds resources directly to the argument tables
                comp.set_msl_options(o);
                comp.set_entry_point(entry, model);

                // Pin every (set, binding) used by this stage to the Metal slot the
                // pipeline layout assigned, so binding at draw time matches the shader.
                if (layout)
                {
                    const VriShaderStageFlags stageBit =
                        model == spv::ExecutionModelVertex     ? VriShaderStage_Vertex :
                        model == spv::ExecutionModelFragment   ? VriShaderStage_Fragment :
                                                                 VriShaderStage_Compute;
                    for (uint32_t set = 0; set < layout->setBindings.size(); ++set)
                    {
                        for (const BindingMTL& b : layout->setBindings[set])
                        {
                            if (!(b.stages & stageBit))
                                continue;
                            spirv_cross::MSLResourceBinding rb = {};
                            rb.stage = model;
                            rb.desc_set = set;
                            rb.binding = b.baseRegister;
                            rb.count = b.count;
                            if (IsBufferType(b.type))   rb.msl_buffer = b.mslIndex;
                            else if (IsTextureType(b.type)) rb.msl_texture = b.mslIndex;
                            else                        rb.msl_sampler = b.mslIndex;
                            comp.add_msl_resource_binding(rb);
                        }
                    }
                    if (layout->hasPush && (layout->pushStages &
                        (model == spv::ExecutionModelVertex ? VriShaderStage_Vertex :
                         model == spv::ExecutionModelFragment ? VriShaderStage_Fragment : VriShaderStage_Compute)))
                    {
                        spirv_cross::MSLResourceBinding rb = {};
                        rb.stage = model;
                        rb.desc_set = spirv_cross::ResourceBindingPushConstantDescriptorSet;
                        rb.binding = spirv_cross::ResourceBindingPushConstantBinding;
                        rb.msl_buffer = layout->pushBufferIndex;
                        comp.add_msl_resource_binding(rb);
                    }
                }

                r.source = comp.compile();
                r.entry = comp.get_cleansed_entry_point_name(entry, model);
                if (model == spv::ExecutionModelGLCompute)
                {
                    r.localSize.width  = comp.get_execution_mode_argument(spv::ExecutionModeLocalSize, 0);
                    r.localSize.height = comp.get_execution_mode_argument(spv::ExecutionModeLocalSize, 1);
                    r.localSize.depth  = comp.get_execution_mode_argument(spv::ExecutionModeLocalSize, 2);
                    if (r.localSize.width == 0)  r.localSize.width = 1;
                    if (r.localSize.height == 0) r.localSize.height = 1;
                    if (r.localSize.depth == 0)  r.localSize.depth = 1;
                }
                if (std::getenv("VRI_DUMP_MSL"))
                    std::fprintf(stderr, "=== VRI MSL model=%d entry=%s ===\n%s\n",
                                 static_cast<int>(model), r.entry.c_str(), r.source.c_str());
                r.ok = true;
            }
            catch (const std::exception& e)
            {
                std::string msg = "SPIRV-Cross MSL transpile failed: ";
                msg += e.what();
                d->ReportError(msg.c_str());
            }
            return r;
        }

        // Compile MSL source and fetch a function. Returns nil on failure (diagnosed).
        id<MTLFunction> MakeFunction(const DeviceMTL* d, const MslResult& msl)
        {
            if (!msl.ok)
                return nil;
            NSError* err = nil;
            NSString* src = [NSString stringWithUTF8String:msl.source.c_str()];
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            id<MTLLibrary> lib = [d->Device() newLibraryWithSource:src options:opts error:&err];
            [opts release];
            if (!lib)
            {
                std::string m = "newLibraryWithSource failed: ";
                m += err ? [[err localizedDescription] UTF8String] : "unknown";
                d->ReportError(m.c_str());
                return nil;
            }
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:msl.entry.c_str()]];
            [lib release];
            if (!fn)
                d->ReportError("newFunctionWithName: entry not found in compiled MSL");
            return fn; // +1 owned by caller
        }

        // ---- encoder lifecycle (one Metal encoder open at a time) -----------
        void EndAux(CommandBufferMTL* c)
        {
            if (c->blitEnc)    { [c->blitEnc endEncoding];    [c->blitEnc release];    c->blitEnc = nil; }
            if (c->computeEnc) { [c->computeEnc endEncoding]; [c->computeEnc release]; c->computeEnc = nil; }
        }
        void EnsureBlit(CommandBufferMTL* c)
        {
            if (c->computeEnc) { [c->computeEnc endEncoding]; [c->computeEnc release]; c->computeEnc = nil; }
            if (!c->blitEnc) c->blitEnc = [[c->cmd blitCommandEncoder] retain];
        }
        void EnsureCompute(CommandBufferMTL* c)
        {
            if (c->blitEnc) { [c->blitEnc endEncoding]; [c->blitEnc release]; c->blitEnc = nil; }
            if (!c->computeEnc) c->computeEnc = [[c->cmd computeCommandEncoder] retain];
        }

        // ---- queries -------------------------------------------------------
        const VriDeviceDesc* VRI_CALL GetDeviceDesc(const VriDevice* device) { return &Dev(device)->Desc(); }

        VriFormatSupportFlags VRI_CALL GetFormatSupport(const VriDevice*, VriFormat format)
        {
            if (ToMtlFormat(format) == MTLPixelFormatInvalid && ToMtlVertexFormat(format) == MTLVertexFormatInvalid)
                return VriFormatSupport_None;
            VriFormatSupportFlags r = VriFormatSupport_VertexBuffer;
            if (ToMtlFormat(format) != MTLPixelFormatInvalid)
            {
                r |= VriFormatSupport_Texture;
                if (FormatHasDepth(format) || FormatHasStencil(format))
                    r |= VriFormatSupport_DepthStencil;
                else
                    r |= VriFormatSupport_ColorAttachment | VriFormatSupport_Blend | VriFormatSupport_StorageTexture;
            }
            return r;
        }

        VriResult VRI_CALL GetQueue(VriDevice* device, VriQueueType type, uint32_t, VriQueue** outQueue)
        {
            if (type >= VriQueueType_Count)
                return VriResult_InvalidArgument;
            *outQueue = ToHandle(Dev(device)->GetQueue(type));
            return VriResult_Success;
        }

        // ---- command allocation / lifecycle --------------------------------
        VriResult VRI_CALL CreateCommandAllocator(VriDevice* device, VriQueueType type, VriCommandAllocator** out)
        {
            *out = ToHandle(new CommandAllocatorMTL{Dev(device), type});
            return VriResult_Success;
        }
        void VRI_CALL ResetCommandAllocator(VriCommandAllocator*) {}
        void VRI_CALL DestroyCommandAllocator(VriCommandAllocator* a) { delete CA(a); }

        VriResult VRI_CALL CreateCommandBuffer(VriCommandAllocator* allocator, VriCommandBuffer** out)
        {
            CommandBufferMTL* c = new CommandBufferMTL{};
            c->device = CA(allocator)->device;
            *out = ToHandle(c);
            return VriResult_Success;
        }

        VriResult VRI_CALL BeginCommandBuffer(VriCommandBuffer* cmd)
        {
            CommandBufferMTL* c = CB(cmd);
            c->cmd = [[c->device->Queue() commandBuffer] retain];
            c->renderEnc = nil; c->computeEnc = nil; c->blitEnc = nil;
            c->boundLayout = nullptr; c->boundPipeline = nullptr;
            c->indexBuffer = nil; c->indexOffset = 0; c->indexType = MTLIndexTypeUInt16;
            return c->cmd ? VriResult_Success : VriResult_Failure;
        }

        VriResult VRI_CALL EndCommandBuffer(VriCommandBuffer* cmd)
        {
            EndAux(CB(cmd)); // close any trailing blit/compute encoder
            return VriResult_Success;
        }

        // ---- resources -----------------------------------------------------
        VriResult VRI_CALL CreateBuffer(VriDevice* device, const VriBufferDesc* desc, VriBuffer** out)
        {
            DeviceMTL* d = Dev(device);
            BufferMTL* b = new BufferMTL{};
            b->device = d;
            b->size = desc->size;
            b->options = StorageOptions(desc->memoryLocation);
            if (desc->memoryLocation == VriMemoryLocation_Undefined)
            {
                b->buffer = nil; // bound later via BindBufferMemory
                b->owned = false;
            }
            else
            {
                b->buffer = [d->Device() newBufferWithLength:(desc->size ? desc->size : 1) options:b->options];
                if (!b->buffer) { delete b; return VriResult_OutOfMemory; }
                b->owned = true;
            }
            *out = ToHandle(b);
            return VriResult_Success;
        }

        void VRI_CALL DestroyBuffer(VriBuffer* buffer)
        {
            if (!buffer) return;
            BufferMTL* b = Buf(buffer);
            if (b->owned && b->buffer) [b->buffer release];
            delete b;
        }

        void* VRI_CALL MapBuffer(VriBuffer* buffer, uint64_t offset, uint64_t)
        {
            BufferMTL* b = Buf(buffer);
            if (!b->buffer || [b->buffer storageMode] == MTLStorageModePrivate)
                return nullptr; // GPU-only memory is not host-mappable
            return static_cast<uint8_t*>([b->buffer contents]) + offset;
        }

        void VRI_CALL UnmapBuffer(VriBuffer*) {} // Shared storage is coherent on Apple Silicon

        uint64_t VRI_CALL GetBufferDeviceAddress(const VriBuffer* buffer)
        {
            const BufferMTL* b = reinterpret_cast<const BufferMTL*>(buffer);
            return (b && b->buffer) ? [b->buffer gpuAddress] : 0;
        }

        MTLTextureDescriptor* BuildTextureDescriptor(const VriTextureDesc* desc)
        {
            MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
            const uint32_t sampleNum = desc->sampleNum ? desc->sampleNum : 1;
            const uint32_t layerNum = desc->layerNum ? desc->layerNum : 1;
            const bool is3D = desc->type == VriTextureType_3D;
            const bool isCube = desc->type == VriTextureType_Cube || desc->type == VriTextureType_CubeArray;
            const bool is1D = desc->type == VriTextureType_1D || desc->type == VriTextureType_1DArray;
            td.textureType = ToMtlTextureType(desc->type, sampleNum);
            // A texture with >1 array layer must use an *array* texture type, even when the caller
            // labelled it plain 2D/1D (the WebGPU/VK backends key arrays off layerNum, not type).
            if (!is3D && !isCube && layerNum > 1)
                td.textureType = is1D ? MTLTextureType1DArray
                               : (sampleNum > 1 ? MTLTextureType2DMultisampleArray : MTLTextureType2DArray);
            td.pixelFormat = ToMtlFormat(desc->format);
            td.width = desc->width ? desc->width : 1;
            td.height = desc->height ? desc->height : 1;
            td.depth = is3D ? (desc->depth ? desc->depth : 1) : 1;
            td.mipmapLevelCount = desc->mipNum ? desc->mipNum : 1;
            td.sampleCount = sampleNum;
            if (desc->type == VriTextureType_CubeArray)
                td.arrayLength = (layerNum ? layerNum : 6) / 6;
            else if (!is3D && !isCube && layerNum > 1)
                td.arrayLength = layerNum;

            MTLTextureUsage usage = MTLTextureUsageUnknown;
            if (desc->usage & VriTextureUsage_ShaderResource)         usage |= MTLTextureUsageShaderRead;
            if (desc->usage & VriTextureUsage_ShaderResourceStorage)  usage |= (MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
            if (desc->usage & (VriTextureUsage_ColorAttachment | VriTextureUsage_DepthStencilAttachment))
                usage |= MTLTextureUsageRenderTarget;
            // PixelFormatView lets CreateTextureView reinterpret format/type (e.g. cube faces).
            usage |= MTLTextureUsagePixelFormatView;
            td.usage = usage;
            return td; // +1 owned by caller
        }

        VriResult VRI_CALL CreateTexture(VriDevice* device, const VriTextureDesc* desc, VriTexture** out)
        {
            DeviceMTL* d = Dev(device);
            MTLTextureDescriptor* td = BuildTextureDescriptor(desc);
            td.storageMode = MTLStorageModePrivate;

            TextureMTL* t = new TextureMTL{};
            t->device = d;
            t->format = desc->format;
            t->mtlFormat = td.pixelFormat;
            t->type = desc->type;
            t->width = (uint32_t)td.width;
            t->height = (uint32_t)td.height;
            t->depth = (uint32_t)td.depth;
            t->mipNum = (uint32_t)td.mipmapLevelCount;
            t->layerNum = desc->type == VriTextureType_3D ? 1u : (desc->layerNum ? desc->layerNum : 1u);
            t->sampleNum = (uint32_t)td.sampleCount;
            t->texelSize = TexelSize(desc->format);

            if (desc->memoryLocation == VriMemoryLocation_Undefined)
            {
                t->descriptor = td; // retained; texture created at BindTextureMemory
                t->texture = nil;
                t->owned = false;
            }
            else
            {
                t->texture = [d->Device() newTextureWithDescriptor:td];
                [td release];
                if (!t->texture) { delete t; return VriResult_OutOfMemory; }
                t->owned = true;
            }
            *out = ToHandle(t);
            return VriResult_Success;
        }

        void VRI_CALL DestroyTexture(VriTexture* texture)
        {
            if (!texture) return;
            TextureMTL* t = Tex(texture);
            if (t->owned && t->texture) [t->texture release];
            if (t->descriptor) [t->descriptor release];
            delete t;
        }

        // ---- explicit memory (MTLHeap) -------------------------------------
        void VRI_CALL GetBufferMemoryDesc(const VriDevice* device, const VriBufferDesc* desc, VriMemoryLocation location, VriMemoryDesc* o)
        {
            if (!o) return;
            *o = {};
            MTLSizeAndAlign sa = [Dev(device)->Device() heapBufferSizeAndAlignWithLength:(desc->size ? desc->size : 1)
                                                                                 options:StorageOptions(location)];
            o->size = sa.size; o->alignment = sa.align; o->location = location;
        }
        void VRI_CALL GetTextureMemoryDesc(const VriDevice* device, const VriTextureDesc* desc, VriMemoryLocation location, VriMemoryDesc* o)
        {
            if (!o) return;
            *o = {};
            MTLTextureDescriptor* td = BuildTextureDescriptor(desc);
            td.storageMode = location == VriMemoryLocation_Device ? MTLStorageModePrivate : MTLStorageModeShared;
            MTLSizeAndAlign sa = [Dev(device)->Device() heapTextureSizeAndAlignWithDescriptor:td];
            [td release];
            o->size = sa.size; o->alignment = sa.align; o->location = location;
        }
        VriResult VRI_CALL AllocateMemory(VriDevice* device, const VriMemoryDesc* desc, VriMemory** out)
        {
            MTLHeapDescriptor* hd = [[MTLHeapDescriptor alloc] init];
            hd.size = desc->size;
            hd.storageMode = desc->location == VriMemoryLocation_Device ? MTLStorageModePrivate : MTLStorageModeShared;
            hd.type = MTLHeapTypeAutomatic;
            id<MTLHeap> heap = [Dev(device)->Device() newHeapWithDescriptor:hd];
            [hd release];
            if (!heap) return VriResult_OutOfMemory;
            *out = ToHandle(new MemoryMTL{Dev(device), heap, desc->size, desc->location});
            return VriResult_Success;
        }
        void VRI_CALL FreeMemory(VriMemory* memory)
        {
            if (!memory) return;
            MemoryMTL* m = Mem(memory);
            if (m->heap) [m->heap release];
            delete m;
        }
        VriResult VRI_CALL BindBufferMemory(VriDevice*, VriBuffer* buffer, VriMemory* memory, uint64_t)
        {
            BufferMTL* b = Buf(buffer);
            MemoryMTL* m = Mem(memory);
            const MTLResourceOptions opt = m->location == VriMemoryLocation_Device ? MTLResourceStorageModePrivate
                                                                                   : MTLResourceStorageModeShared;
            b->buffer = [m->heap newBufferWithLength:(b->size ? b->size : 1) options:opt];
            if (!b->buffer) return VriResult_OutOfMemory;
            b->owned = true;
            return VriResult_Success;
        }
        VriResult VRI_CALL BindTextureMemory(VriDevice*, VriTexture* texture, VriMemory* memory, uint64_t)
        {
            TextureMTL* t = Tex(texture);
            MemoryMTL* m = Mem(memory);
            if (!t->descriptor) return VriResult_InvalidArgument;
            t->descriptor.storageMode = m->location == VriMemoryLocation_Device ? MTLStorageModePrivate : MTLStorageModeShared;
            t->texture = [m->heap newTextureWithDescriptor:t->descriptor];
            if (!t->texture) return VriResult_OutOfMemory;
            t->owned = true;
            return VriResult_Success;
        }

        // ---- views & samplers ----------------------------------------------
        VriResult VRI_CALL CreateBufferView(VriDevice* device, const VriBufferViewDesc* desc, VriDescriptor** out)
        {
            DescriptorMTL* v = new DescriptorMTL{};
            v->kind = DescriptorMTL::Kind::Buffer;
            v->device = Dev(device);
            v->buffer = Buf(desc->buffer)->buffer;
            v->bufferOffset = desc->offset;
            v->bufferRange = desc->size;
            *out = ToHandle(v);
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateTextureView(VriDevice* device, const VriTextureViewDesc* desc, VriDescriptor** out)
        {
            TextureMTL* t = Tex(desc->texture);
            const MTLPixelFormat fmt = desc->format == VriFormat_Unknown ? t->mtlFormat : ToMtlFormat(desc->format);
            MTLTextureType vt = ToMtlViewType(desc->viewType);
            // A view of an MSAA texture must keep the multisample type (a 2D view of a
            // 2DMultisample texture is rejected by Metal) - the view dims carry no sample count.
            if (t->sampleNum > 1 && vt == MTLTextureType2D)
                vt = MTLTextureType2DMultisample;
            const uint32_t mipNum = desc->mipNum ? desc->mipNum : (t->mipNum - desc->baseMip);
            const uint32_t layerNum = desc->layerNum ? desc->layerNum : (t->layerNum - desc->baseLayer);

            DescriptorMTL* v = new DescriptorMTL{};
            v->kind = DescriptorMTL::Kind::Texture;
            v->device = Dev(device);
            // A full-range, same-format/type view can reuse the base texture (cheaper, and
            // avoids a needless view object); otherwise create a reinterpreting view.
            const bool full = fmt == t->mtlFormat && vt == [t->texture textureType] &&
                              desc->baseMip == 0 && mipNum == t->mipNum &&
                              desc->baseLayer == 0 && layerNum == t->layerNum;
            if (full)
            {
                v->texture = t->texture;
                v->ownsTexture = false;
            }
            else
            {
                v->texture = [t->texture newTextureViewWithPixelFormat:fmt
                                                           textureType:vt
                                                                levels:NSMakeRange(desc->baseMip, mipNum)
                                                                slices:NSMakeRange(desc->baseLayer, layerNum)];
                v->ownsTexture = true;
                if (!v->texture) { delete v; return VriResult_Failure; }
            }
            *out = ToHandle(v);
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateSampler(VriDevice* device, const VriSamplerDesc* desc, VriDescriptor** out)
        {
            MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
            sd.minFilter = ToMtlFilter(desc->minFilter);
            sd.magFilter = ToMtlFilter(desc->magFilter);
            sd.mipFilter = ToMtlMipFilter(desc->mipmapMode);
            sd.sAddressMode = ToMtlAddress(desc->addressModeU);
            sd.tAddressMode = ToMtlAddress(desc->addressModeV);
            sd.rAddressMode = ToMtlAddress(desc->addressModeW);
            sd.lodMinClamp = desc->minLod;
            sd.lodMaxClamp = desc->maxLod <= 0.0f ? FLT_MAX : desc->maxLod;
            sd.maxAnisotropy = desc->anisotropyEnable && desc->maxAnisotropy >= 1.0f ? (NSUInteger)desc->maxAnisotropy : 1;
            sd.borderColor = ToMtlBorderColor(desc->borderColor);
            if (desc->compareEnable)
                sd.compareFunction = ToMtlCompare(desc->compareOp);
            sd.supportArgumentBuffers = NO;

            id<MTLSamplerState> sampler = [Dev(device)->Device() newSamplerStateWithDescriptor:sd];
            [sd release];
            if (!sampler) return VriResult_Failure;

            DescriptorMTL* v = new DescriptorMTL{};
            v->kind = DescriptorMTL::Kind::Sampler;
            v->device = Dev(device);
            v->sampler = sampler; // +1 owned
            *out = ToHandle(v);
            return VriResult_Success;
        }

        void VRI_CALL DestroyDescriptor(VriDescriptor* descriptor)
        {
            if (!descriptor) return;
            DescriptorMTL* v = Desc(descriptor);
            if (v->ownsTexture && v->texture) [v->texture release];
            if (v->sampler) [v->sampler release];
            delete v;
        }

        // ---- pipeline layout & pipelines -----------------------------------
        VriResult VRI_CALL CreatePipelineLayout(VriDevice* device, const VriPipelineLayoutDesc* desc, VriPipelineLayout** out)
        {
            PipelineLayoutMTL* layout = new PipelineLayoutMTL{};
            layout->device = Dev(device);
            // Assign Metal argument-table slots: separate counters per resource class,
            // flat across all sets (Metal tables are flat per stage). Push constants take
            // the next buffer slot after all descriptor buffers.
            uint32_t bufferCounter = 0, textureCounter = 0, samplerCounter = 0;
            layout->setBindings.resize(desc->descriptorSetNum);
            for (uint32_t s = 0; s < desc->descriptorSetNum; ++s)
            {
                const VriDescriptorSetDesc& set = desc->descriptorSets[s];
                const uint32_t setIdx = set.registerSpace < desc->descriptorSetNum ? set.registerSpace : s;
                std::vector<BindingMTL>& dst = layout->setBindings[setIdx];
                for (uint32_t r = 0; r < set.rangeNum; ++r)
                {
                    const VriDescriptorRangeDesc& range = set.ranges[r];
                    BindingMTL b{};
                    b.baseRegister = range.baseRegister;
                    b.type = range.descriptorType;
                    b.count = range.descriptorNum ? range.descriptorNum : 1;
                    b.stages = range.shaderStages;
                    if (IsBufferType(range.descriptorType))      { b.mslIndex = bufferCounter;  bufferCounter  += b.count; }
                    else if (IsTextureType(range.descriptorType)) { b.mslIndex = textureCounter; textureCounter += b.count; }
                    else                                          { b.mslIndex = samplerCounter; samplerCounter += b.count; }
                    dst.push_back(b);
                }
            }
            if (desc->pushConstantNum > 0)
            {
                layout->hasPush = true;
                layout->pushBufferIndex = bufferCounter++;
                layout->pushSize = desc->pushConstants[0].size;
                layout->pushStages = desc->pushConstants[0].shaderStages;
                if (layout->pushStages == 0)
                    layout->pushStages = VriShaderStage_Vertex | VriShaderStage_Fragment | VriShaderStage_Compute;
            }
            *out = ToHandle(layout);
            return VriResult_Success;
        }

        void VRI_CALL DestroyPipelineLayout(VriPipelineLayout* layout) { delete PL(layout); }

        VriResult VRI_CALL CreateGraphicsPipeline(VriDevice* device, const VriGraphicsPipelineDesc* desc, VriPipeline** out)
        {
            DeviceMTL* d = Dev(device);
            const PipelineLayoutMTL* pl = desc->pipelineLayout ? PLc(desc->pipelineLayout) : nullptr;

            id<MTLFunction> vsFn = nil, fsFn = nil;
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                const VriShaderDesc& s = desc->shaders[i];
                if (s.stage == VriShaderStage_Vertex)
                    vsFn = MakeFunction(d, SpirvToMsl(d, pl, s, spv::ExecutionModelVertex));
                else if (s.stage == VriShaderStage_Fragment)
                    fsFn = MakeFunction(d, SpirvToMsl(d, pl, s, spv::ExecutionModelFragment));
                else if (s.stage == VriShaderStage_Geometry || s.stage == VriShaderStage_TessControl || s.stage == VriShaderStage_TessEval)
                    { if (vsFn) [vsFn release]; if (fsFn) [fsFn release]; return VriResult_Unsupported; }
            }
            if (!vsFn) { if (fsFn) [fsFn release]; return VriResult_Failure; }

            MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
            pd.vertexFunction = vsFn;
            pd.fragmentFunction = fsFn;
            pd.rasterSampleCount = desc->multisample.sampleNum ? desc->multisample.sampleNum : 1;
            pd.inputPrimitiveTopology = ToMtlTopologyClass(desc->inputAssembly.topology);

            // Vertex input: one MTLVertexBufferLayout per stream at its reserved (high) buffer
            // index; attribute index == its array index (== SPIR-V location, as the GL/WGPU
            // backends assume).
            if (desc->vertexInput.streamNum > 0)
            {
                MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
                for (uint32_t s = 0; s < desc->vertexInput.streamNum; ++s)
                {
                    const VriVertexStreamDesc& stream = desc->vertexInput.streams[s];
                    const uint32_t bufIdx = VertexBufferIndex(stream.bindingSlot);
                    vd.layouts[bufIdx].stride = stream.stride;
                    vd.layouts[bufIdx].stepFunction = stream.stepRate == VriVertexStepRate_PerInstance
                                                    ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
                    vd.layouts[bufIdx].stepRate = 1;
                }
                for (uint32_t a = 0; a < desc->vertexInput.attributeNum; ++a)
                {
                    const VriVertexAttributeDesc& attr = desc->vertexInput.attributes[a];
                    const VriVertexStreamDesc& stream = desc->vertexInput.streams[attr.streamIndex];
                    vd.attributes[a].format = ToMtlVertexFormat(attr.format);
                    vd.attributes[a].offset = attr.offset;
                    vd.attributes[a].bufferIndex = VertexBufferIndex(stream.bindingSlot);
                }
                pd.vertexDescriptor = vd;
            }

            for (uint32_t i = 0; i < desc->outputMerger.colorNum; ++i)
            {
                const VriColorAttachmentDesc& c = desc->outputMerger.colors[i];
                MTLRenderPipelineColorAttachmentDescriptor* ca = pd.colorAttachments[i];
                ca.pixelFormat = ToMtlFormat(c.format);
                ca.writeMask = ToMtlColorWriteMask(c.colorWriteMask);
                if (c.blend.enable)
                {
                    ca.blendingEnabled = YES;
                    ca.sourceRGBBlendFactor = ToMtlBlendFactor(c.blend.srcColor);
                    ca.destinationRGBBlendFactor = ToMtlBlendFactor(c.blend.dstColor);
                    ca.rgbBlendOperation = ToMtlBlendOp(c.blend.colorOp);
                    ca.sourceAlphaBlendFactor = ToMtlBlendFactor(c.blend.srcAlpha);
                    ca.destinationAlphaBlendFactor = ToMtlBlendFactor(c.blend.dstAlpha);
                    ca.alphaBlendOperation = ToMtlBlendOp(c.blend.alphaOp);
                }
            }
            if (desc->outputMerger.depthStencilFormat != VriFormat_Unknown)
            {
                const MTLPixelFormat dsf = ToMtlFormat(desc->outputMerger.depthStencilFormat);
                if (FormatHasDepth(desc->outputMerger.depthStencilFormat))   pd.depthAttachmentPixelFormat = dsf;
                if (FormatHasStencil(desc->outputMerger.depthStencilFormat)) pd.stencilAttachmentPixelFormat = dsf;
            }

            NSError* err = nil;
            id<MTLRenderPipelineState> rps = [d->Device() newRenderPipelineStateWithDescriptor:pd error:&err];
            [pd release];
            if (vsFn) [vsFn release];
            if (fsFn) [fsFn release];
            if (!rps)
            {
                std::string m = "newRenderPipelineState failed: ";
                m += err ? [[err localizedDescription] UTF8String] : "unknown";
                d->ReportError(m.c_str());
                return VriResult_Failure;
            }

            // Depth/stencil state object (separate from the pipeline in Metal). Always built and
            // always bound (see CmdSetPipeline): a Metal render encoder keeps the last-set
            // depth-stencil state across draws, so a depth-test-off pipeline (e.g. ImGui drawn
            // after a 3D pass) must explicitly install a compare=Always / no-write state or it
            // would inherit the previous pipeline's test and get depth-culled.
            id<MTLDepthStencilState> dss = nil;
            const VriDepthStencilDesc& ds = desc->depthStencil;
            {
                MTLDepthStencilDescriptor* dsd = [[MTLDepthStencilDescriptor alloc] init];
                dsd.depthCompareFunction = ds.depthTest ? ToMtlCompare(ds.depthCompareOp) : MTLCompareFunctionAlways;
                dsd.depthWriteEnabled = ds.depthWrite ? YES : NO;
                if (ds.stencilTest)
                {
                    auto face = [](const VriStencilOpDesc& s) {
                        MTLStencilDescriptor* f = [[MTLStencilDescriptor alloc] init];
                        f.stencilCompareFunction = ToMtlCompare(s.compareOp);
                        f.stencilFailureOperation = ToMtlStencilOp(s.failOp);
                        f.depthFailureOperation = ToMtlStencilOp(s.depthFailOp);
                        f.depthStencilPassOperation = ToMtlStencilOp(s.passOp);
                        f.readMask = s.compareMask;
                        f.writeMask = s.writeMask;
                        return f;
                    };
                    MTLStencilDescriptor* frontF = face(ds.front);
                    MTLStencilDescriptor* backF = face(ds.back);
                    dsd.frontFaceStencil = frontF;
                    dsd.backFaceStencil = backF;
                    [frontF release];
                    [backF release];
                }
                dss = [d->Device() newDepthStencilStateWithDescriptor:dsd];
                [dsd release];
            }

            PipelineMTL* p = new PipelineMTL{};
            p->device = d;
            p->isCompute = false;
            p->render = rps;
            p->depthStencil = dss;
            p->primType = ToMtlPrimitive(desc->inputAssembly.topology);
            p->cull = ToMtlCull(desc->rasterization.cullMode);
            p->winding = ToMtlWinding(desc->rasterization.frontFace);
            p->fill = desc->rasterization.polygonMode == VriPolygonMode_Line ? MTLTriangleFillModeLines : MTLTriangleFillModeFill;
            p->depthClampEnable = desc->rasterization.depthClamp != VRI_FALSE;
            p->stencilTest = ds.stencilTest != VRI_FALSE;
            p->stencilReference = ds.front.reference;
            *out = ToHandle(p);
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateComputePipeline(VriDevice* device, const VriComputePipelineDesc* desc, VriPipeline** out)
        {
            DeviceMTL* d = Dev(device);
            const PipelineLayoutMTL* pl = desc->pipelineLayout ? PLc(desc->pipelineLayout) : nullptr;
            MslResult msl = SpirvToMsl(d, pl, desc->shader, spv::ExecutionModelGLCompute);
            id<MTLFunction> fn = MakeFunction(d, msl);
            if (!fn) return VriResult_Failure;

            NSError* err = nil;
            id<MTLComputePipelineState> cps = [d->Device() newComputePipelineStateWithFunction:fn error:&err];
            [fn release];
            if (!cps)
            {
                std::string m = "newComputePipelineState failed: ";
                m += err ? [[err localizedDescription] UTF8String] : "unknown";
                d->ReportError(m.c_str());
                return VriResult_Failure;
            }
            PipelineMTL* p = new PipelineMTL{};
            p->device = d;
            p->isCompute = true;
            p->compute = cps;
            p->threadsPerThreadgroup = msl.localSize;
            *out = ToHandle(p);
            return VriResult_Success;
        }

        void VRI_CALL DestroyPipeline(VriPipeline* pipeline)
        {
            if (!pipeline) return;
            PipelineMTL* p = Pipe(pipeline);
            if (p->render) [p->render release];
            if (p->depthStencil) [p->depthStencil release];
            if (p->compute) [p->compute release];
            delete p;
        }

        // ---- descriptor pools / sets ---------------------------------------
        VriResult VRI_CALL CreateDescriptorPool(VriDevice* device, const VriDescriptorPoolDesc*, VriDescriptorPool** out)
        {
            *out = ToHandle(new DescriptorPoolMTL{Dev(device), {}});
            return VriResult_Success;
        }
        void VRI_CALL ResetDescriptorPool(VriDescriptorPool* pool)
        {
            DescriptorPoolMTL* p = DPool(pool);
            for (DescriptorSetMTL* s : p->sets) delete s;
            p->sets.clear();
        }
        void VRI_CALL DestroyDescriptorPool(VriDescriptorPool* pool)
        {
            if (!pool) return;
            ResetDescriptorPool(pool);
            delete DPool(pool);
        }
        VriResult VRI_CALL AllocateDescriptorSets(VriDescriptorPool* pool, const VriPipelineLayout* layout, uint32_t setIndex, VriDescriptorSet** outSets, uint32_t setNum)
        {
            DescriptorPoolMTL* p = DPool(pool);
            const PipelineLayoutMTL* l = PLc(layout);
            if (setIndex >= l->setBindings.size())
                return VriResult_InvalidArgument;
            for (uint32_t i = 0; i < setNum; ++i)
            {
                DescriptorSetMTL* s = new DescriptorSetMTL{p->device, l, setIndex, {}};
                p->sets.push_back(s);
                outSets[i] = ToHandle(s);
            }
            return VriResult_Success;
        }
        void VRI_CALL UpdateDescriptorRanges(VriDescriptorSet* set, uint32_t baseRange, uint32_t rangeNum, const VriDescriptorRangeUpdateDesc* updates)
        {
            DescriptorSetMTL* s = DSet(set);
            const std::vector<BindingMTL>& ranges = s->layout->setBindings[s->setIndex];
            for (uint32_t r = 0; r < rangeNum; ++r)
            {
                const VriDescriptorRangeUpdateDesc& u = updates[r];
                const BindingMTL& info = ranges[baseRange + r];
                for (uint32_t k = 0; k < u.descriptorNum; ++k)
                {
                    DescriptorSetMTL::Bound b{};
                    b.mslIndex = info.mslIndex + u.baseDescriptor + k;
                    b.type = info.type;
                    b.stages = info.stages;
                    b.desc = Desc(u.descriptors[k]);
                    // Replace an existing binding at this slot, else append.
                    bool replaced = false;
                    for (DescriptorSetMTL::Bound& e : s->bound)
                        if (e.mslIndex == b.mslIndex && e.type == b.type) { e = b; replaced = true; break; }
                    if (!replaced) s->bound.push_back(b);
                }
            }
        }

        // ---- synchronization (MTLSharedEvent timeline) ---------------------
        VriResult VRI_CALL CreateFence(VriDevice* device, uint64_t initialValue, VriFence** out)
        {
            id<MTLSharedEvent> ev = [Dev(device)->Device() newSharedEvent];
            if (!ev) return VriResult_Failure;
            ev.signaledValue = initialValue;
            *out = ToHandle(new FenceMTL{Dev(device), ev});
            return VriResult_Success;
        }
        void VRI_CALL DestroyFence(VriFence* fence)
        {
            if (!fence) return;
            FenceMTL* f = Fen(fence);
            if (f->event) [f->event release];
            delete f;
        }
        uint64_t VRI_CALL GetFenceValue(VriFence* fence) { return [Fen(fence)->event signaledValue]; }
        void VRI_CALL Wait(VriFence* fence, uint64_t value)
        {
            FenceMTL* f = Fen(fence);
            if ([f->event signaledValue] >= value)
                return;
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            [f->event notifyListener:Listener()
                             atValue:value
                               block:^(id<MTLSharedEvent>, uint64_t) { dispatch_semaphore_signal(sem); }];
            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
            dispatch_release(sem);
        }

        // ---- render pass / draw --------------------------------------------
        void VRI_CALL CmdBeginRendering(VriCommandBuffer* cmd, const VriAttachmentsDesc* a)
        {
            CommandBufferMTL* c = CB(cmd);
            EndAux(c);
            MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor]; // autoreleased
            for (uint32_t i = 0; i < a->colorNum; ++i)
            {
                const VriAttachmentDesc& src = a->colors[i];
                MTLRenderPassColorAttachmentDescriptor* ca = rp.colorAttachments[i];
                ca.texture = Desc(src.view)->texture;
                ca.loadAction = ToMtlLoadAction(src.loadOp);
                ca.clearColor = MTLClearColorMake(src.clearValue.color.f32[0], src.clearValue.color.f32[1],
                                                  src.clearValue.color.f32[2], src.clearValue.color.f32[3]);
                if (src.resolveView)
                {
                    ca.resolveTexture = Desc(src.resolveView)->texture;
                    ca.storeAction = src.storeOp == VriAttachmentStoreOp_DontCare
                                   ? MTLStoreActionMultisampleResolve : MTLStoreActionStoreAndMultisampleResolve;
                }
                else
                    ca.storeAction = ToMtlStoreAction(src.storeOp);
            }
            if (a->depth)
            {
                MTLRenderPassDepthAttachmentDescriptor* da = rp.depthAttachment;
                da.texture = Desc(a->depth->view)->texture;
                da.loadAction = ToMtlLoadAction(a->depth->loadOp);
                da.storeAction = ToMtlStoreAction(a->depth->storeOp);
                da.clearDepth = a->depth->clearValue.depthStencil.depth;
            }
            auto mtlHasStencil = [](MTLPixelFormat f) {
                return f == MTLPixelFormatStencil8 || f == MTLPixelFormatDepth32Float_Stencil8 ||
                       f == MTLPixelFormatDepth24Unorm_Stencil8 || f == MTLPixelFormatX32_Stencil8 ||
                       f == MTLPixelFormatX24_Stencil8;
            };
            if (a->stencil)
            {
                MTLRenderPassStencilAttachmentDescriptor* sa = rp.stencilAttachment;
                sa.texture = Desc(a->stencil->view)->texture;
                sa.loadAction = ToMtlLoadAction(a->stencil->loadOp);
                sa.storeAction = ToMtlStoreAction(a->stencil->storeOp);
                sa.clearStencil = a->stencil->clearValue.depthStencil.stencil;
            }
            else if (a->depth)
            {
                // Combined depth-stencil view: mirror the depth attachment onto the stencil slot.
                id<MTLTexture> dt = Desc(a->depth->view)->texture;
                if (dt && mtlHasStencil([dt pixelFormat]))
                {
                    MTLRenderPassStencilAttachmentDescriptor* sa = rp.stencilAttachment;
                    sa.texture = dt;
                    sa.loadAction = ToMtlLoadAction(a->depth->loadOp);
                    sa.storeAction = ToMtlStoreAction(a->depth->storeOp);
                    sa.clearStencil = a->depth->clearValue.depthStencil.stencil;
                }
            }

            c->renderEnc = [[c->cmd renderCommandEncoderWithDescriptor:rp] retain];
        }

        void VRI_CALL CmdEndRendering(VriCommandBuffer* cmd)
        {
            CommandBufferMTL* c = CB(cmd);
            if (c->renderEnc) { [c->renderEnc endEncoding]; [c->renderEnc release]; c->renderEnc = nil; }
        }

        void VRI_CALL CmdSetViewports(VriCommandBuffer* cmd, const VriViewport* vps, uint32_t num)
        {
            if (num == 0 || !CB(cmd)->renderEnc) return;
            std::vector<MTLViewport> v(num);
            for (uint32_t i = 0; i < num; ++i)
                v[i] = (MTLViewport){vps[i].x, vps[i].y, vps[i].width, vps[i].height, vps[i].minDepth, vps[i].maxDepth};
            [CB(cmd)->renderEnc setViewports:v.data() count:num];
        }

        void VRI_CALL CmdSetScissors(VriCommandBuffer* cmd, const VriRect* rects, uint32_t num)
        {
            if (num == 0 || !CB(cmd)->renderEnc) return;
            std::vector<MTLScissorRect> r(num);
            for (uint32_t i = 0; i < num; ++i)
                r[i] = (MTLScissorRect){(NSUInteger)(rects[i].x < 0 ? 0 : rects[i].x),
                                        (NSUInteger)(rects[i].y < 0 ? 0 : rects[i].y),
                                        rects[i].width, rects[i].height};
            [CB(cmd)->renderEnc setScissorRects:r.data() count:num];
        }

        void VRI_CALL CmdSetPipelineLayout(VriCommandBuffer* cmd, VriPipelineLayout* layout) { CB(cmd)->boundLayout = PLc(layout); }

        void VRI_CALL CmdSetPipeline(VriCommandBuffer* cmd, VriPipeline* pipeline)
        {
            CommandBufferMTL* c = CB(cmd);
            PipelineMTL* p = Pipe(pipeline);
            c->boundPipeline = p;
            if (p->isCompute)
            {
                EnsureCompute(c);
                [c->computeEnc setComputePipelineState:p->compute];
                return;
            }
            if (!c->renderEnc) return;
            [c->renderEnc setRenderPipelineState:p->render];
            [c->renderEnc setCullMode:p->cull];
            [c->renderEnc setFrontFacingWinding:p->winding];
            [c->renderEnc setTriangleFillMode:p->fill];
            [c->renderEnc setDepthClipMode:p->depthClampEnable ? MTLDepthClipModeClamp : MTLDepthClipModeClip];
            if (p->depthStencil) [c->renderEnc setDepthStencilState:p->depthStencil];
            if (p->stencilTest)  [c->renderEnc setStencilReferenceValue:p->stencilReference];
        }

        void BindOne(CommandBufferMTL* c, const DescriptorSetMTL::Bound& b)
        {
            const DescriptorMTL* dsc = b.desc;
            if (!dsc) return;
            if (c->computeEnc)
            {
                if (b.type == VriDescriptorType_Sampler && dsc->sampler)
                    [c->computeEnc setSamplerState:dsc->sampler atIndex:b.mslIndex];
                else if (IsTextureType(b.type) && dsc->texture)
                    [c->computeEnc setTexture:dsc->texture atIndex:b.mslIndex];
                else if (IsBufferType(b.type) && dsc->buffer)
                    [c->computeEnc setBuffer:dsc->buffer offset:dsc->bufferOffset atIndex:b.mslIndex];
                return;
            }
            if (!c->renderEnc) return;
            if (b.type == VriDescriptorType_Sampler && dsc->sampler)
            {
                if (b.stages & VriShaderStage_Vertex)   [c->renderEnc setVertexSamplerState:dsc->sampler atIndex:b.mslIndex];
                if (b.stages & VriShaderStage_Fragment) [c->renderEnc setFragmentSamplerState:dsc->sampler atIndex:b.mslIndex];
            }
            else if (IsTextureType(b.type) && dsc->texture)
            {
                if (b.stages & VriShaderStage_Vertex)   [c->renderEnc setVertexTexture:dsc->texture atIndex:b.mslIndex];
                if (b.stages & VriShaderStage_Fragment) [c->renderEnc setFragmentTexture:dsc->texture atIndex:b.mslIndex];
            }
            else if (IsBufferType(b.type) && dsc->buffer)
            {
                if (b.stages & VriShaderStage_Vertex)   [c->renderEnc setVertexBuffer:dsc->buffer offset:dsc->bufferOffset atIndex:b.mslIndex];
                if (b.stages & VriShaderStage_Fragment) [c->renderEnc setFragmentBuffer:dsc->buffer offset:dsc->bufferOffset atIndex:b.mslIndex];
            }
        }

        void VRI_CALL CmdSetDescriptorSet(VriCommandBuffer* cmd, uint32_t, const VriDescriptorSet* set)
        {
            if (!set) return;
            CommandBufferMTL* c = CB(cmd);
            const DescriptorSetMTL* s = reinterpret_cast<const DescriptorSetMTL*>(set);
            for (const DescriptorSetMTL::Bound& b : s->bound)
                BindOne(c, b);
        }

        void VRI_CALL CmdSetConstants(VriCommandBuffer* cmd, uint32_t, const void* data, uint32_t size)
        {
            CommandBufferMTL* c = CB(cmd);
            if (!c->boundLayout || !c->boundLayout->hasPush || !data || !size) return;
            const uint32_t idx = c->boundLayout->pushBufferIndex;
            const VriShaderStageFlags st = c->boundLayout->pushStages;
            if (c->computeEnc)
                [c->computeEnc setBytes:data length:size atIndex:idx];
            else if (c->renderEnc)
            {
                if (st & VriShaderStage_Vertex)   [c->renderEnc setVertexBytes:data length:size atIndex:idx];
                if (st & VriShaderStage_Fragment) [c->renderEnc setFragmentBytes:data length:size atIndex:idx];
            }
        }

        void VRI_CALL CmdSetVertexBuffers(VriCommandBuffer* cmd, uint32_t baseSlot, const VriVertexBufferBinding* bindings, uint32_t num)
        {
            CommandBufferMTL* c = CB(cmd);
            if (!c->renderEnc) return;
            for (uint32_t i = 0; i < num; ++i)
                [c->renderEnc setVertexBuffer:Buf(bindings[i].buffer)->buffer
                                       offset:bindings[i].offset
                                      atIndex:VertexBufferIndex(baseSlot + i)];
        }

        void VRI_CALL CmdSetIndexBuffer(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, VriIndexType type)
        {
            CommandBufferMTL* c = CB(cmd);
            c->indexBuffer = Buf(buffer)->buffer;
            c->indexOffset = offset;
            c->indexType = type == VriIndexType_UInt16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
        }

        void VRI_CALL CmdDraw(VriCommandBuffer* cmd, const VriDrawDesc* d)
        {
            CommandBufferMTL* c = CB(cmd);
            if (!c->renderEnc || !c->boundPipeline) return;
            [c->renderEnc drawPrimitives:c->boundPipeline->primType
                             vertexStart:d->baseVertex
                             vertexCount:d->vertexNum
                           instanceCount:d->instanceNum ? d->instanceNum : 1
                            baseInstance:d->baseInstance];
        }

        void VRI_CALL CmdDrawIndexed(VriCommandBuffer* cmd, const VriDrawIndexedDesc* d)
        {
            CommandBufferMTL* c = CB(cmd);
            if (!c->renderEnc || !c->boundPipeline || !c->indexBuffer) return;
            const uint32_t idxSize = c->indexType == MTLIndexTypeUInt16 ? 2 : 4;
            [c->renderEnc drawIndexedPrimitives:c->boundPipeline->primType
                                     indexCount:d->indexNum
                                      indexType:c->indexType
                                    indexBuffer:c->indexBuffer
                              indexBufferOffset:c->indexOffset + (uint64_t)d->baseIndex * idxSize
                                  instanceCount:d->instanceNum ? d->instanceNum : 1
                                     baseVertex:d->vertexOffset
                                   baseInstance:d->baseInstance];
        }

        void VRI_CALL CmdDrawIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint32_t drawNum, uint32_t stride)
        {
            CommandBufferMTL* c = CB(cmd);
            if (!c->renderEnc || !c->boundPipeline) return;
            for (uint32_t i = 0; i < drawNum; ++i)
                [c->renderEnc drawPrimitives:c->boundPipeline->primType
                              indirectBuffer:Buf(buffer)->buffer
                        indirectBufferOffset:offset + (uint64_t)i * stride];
        }

        void VRI_CALL CmdDispatch(VriCommandBuffer* cmd, const VriDispatchDesc* d)
        {
            CommandBufferMTL* c = CB(cmd);
            if (!c->boundPipeline) return;
            EnsureCompute(c);
            [c->computeEnc dispatchThreadgroups:MTLSizeMake(d->x, d->y, d->z)
                          threadsPerThreadgroup:c->boundPipeline->threadsPerThreadgroup];
        }

        void VRI_CALL CmdDispatchIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset)
        {
            CommandBufferMTL* c = CB(cmd);
            if (!c->boundPipeline) return;
            EnsureCompute(c);
            [c->computeEnc dispatchThreadgroupsWithIndirectBuffer:Buf(buffer)->buffer
                                             indirectBufferOffset:offset
                                            threadsPerThreadgroup:c->boundPipeline->threadsPerThreadgroup];
        }

        void VRI_CALL CmdBarrier(VriCommandBuffer*, const VriBarrierGroupDesc*) {} // Metal auto-tracks hazards

        // ---- copies (blit encoder) -----------------------------------------
        void VRI_CALL CmdCopyBuffer(VriCommandBuffer* cmd, VriBuffer* dst, VriBuffer* src, const VriBufferCopyDesc* r)
        {
            CommandBufferMTL* c = CB(cmd);
            EnsureBlit(c);
            [c->blitEnc copyFromBuffer:Buf(src)->buffer sourceOffset:r->srcOffset
                              toBuffer:Buf(dst)->buffer destinationOffset:r->dstOffset size:r->size];
        }

        void VRI_CALL CmdCopyTexture(VriCommandBuffer* cmd, VriTexture* dst, VriTexture* src, const VriTextureCopyDesc* region)
        {
            CommandBufferMTL* c = CB(cmd);
            TextureMTL* s = Tex(src);
            TextureMTL* dt = Tex(dst);
            EnsureBlit(c);
            const uint32_t srcMip = region ? region->src.mip : 0;
            const uint32_t dstMip = region ? region->dst.mip : 0;
            const uint32_t srcSlice = region ? region->src.baseLayer : 0;
            const uint32_t dstSlice = region ? region->dst.baseLayer : 0;
            MTLOrigin so = region ? MTLOriginMake(region->src.x, region->src.y, region->src.z) : MTLOriginMake(0, 0, 0);
            MTLOrigin dorg = region ? MTLOriginMake(region->dst.x, region->dst.y, region->dst.z) : MTLOriginMake(0, 0, 0);
            uint32_t w = (region && region->src.width)  ? region->src.width  : s->width;
            uint32_t h = (region && region->src.height) ? region->src.height : s->height;
            uint32_t d = (region && region->src.depth)  ? region->src.depth  : (s->type == VriTextureType_3D ? s->depth : 1);
            [c->blitEnc copyFromTexture:s->texture sourceSlice:srcSlice sourceLevel:srcMip sourceOrigin:so sourceSize:MTLSizeMake(w, h, d)
                              toTexture:dt->texture destinationSlice:dstSlice destinationLevel:dstMip destinationOrigin:dorg];
        }

        void VRI_CALL CmdUploadBufferToTexture(VriCommandBuffer* cmd, VriTexture* dst, VriBuffer* src, const VriBufferTextureCopyDesc* region)
        {
            CommandBufferMTL* c = CB(cmd);
            TextureMTL* t = Tex(dst);
            EnsureBlit(c);
            const uint32_t w = region->texture.width ? region->texture.width : t->width;
            const uint32_t h = region->texture.height ? region->texture.height : t->height;
            const uint32_t d = region->texture.depth ? region->texture.depth : 1;
            const uint32_t pitch = (region->bufferRowLength ? region->bufferRowLength : w) * t->texelSize;
            const uint32_t imgHeight = region->bufferImageHeight ? region->bufferImageHeight : h;
            MTLOrigin origin = MTLOriginMake(region->texture.x, region->texture.y, region->texture.z);
            [c->blitEnc copyFromBuffer:Buf(src)->buffer
                          sourceOffset:region->bufferOffset
                     sourceBytesPerRow:pitch
                   sourceBytesPerImage:(NSUInteger)pitch * imgHeight
                            sourceSize:MTLSizeMake(w, h, d)
                             toTexture:t->texture
                      destinationSlice:region->texture.baseLayer
                      destinationLevel:region->texture.mip
                     destinationOrigin:origin];
        }

        void VRI_CALL CmdReadbackTextureToBuffer(VriCommandBuffer* cmd, VriBuffer* dst, VriTexture* src, const VriBufferTextureCopyDesc* region)
        {
            CommandBufferMTL* c = CB(cmd);
            TextureMTL* t = Tex(src);
            EnsureBlit(c);
            const uint32_t w = region->texture.width ? region->texture.width : t->width;
            const uint32_t h = region->texture.height ? region->texture.height : t->height;
            const uint32_t d = region->texture.depth ? region->texture.depth : 1;
            const uint32_t pitch = (region->bufferRowLength ? region->bufferRowLength : w) * t->texelSize;
            const uint32_t imgHeight = region->bufferImageHeight ? region->bufferImageHeight : h;
            MTLOrigin origin = MTLOriginMake(region->texture.x, region->texture.y, region->texture.z);
            [c->blitEnc copyFromTexture:t->texture
                            sourceSlice:region->texture.baseLayer
                            sourceLevel:region->texture.mip
                           sourceOrigin:origin
                             sourceSize:MTLSizeMake(w, h, d)
                               toBuffer:Buf(dst)->buffer
                      destinationOffset:region->bufferOffset
                 destinationBytesPerRow:pitch
               destinationBytesPerImage:(NSUInteger)pitch * imgHeight];
        }

        void VRI_CALL CmdBeginDebugGroup(VriCommandBuffer* cmd, const char* name)
        {
            CommandBufferMTL* c = CB(cmd);
            NSString* s = [NSString stringWithUTF8String:name ? name : ""];
            if (c->renderEnc)       [c->renderEnc pushDebugGroup:s];
            else if (c->computeEnc) [c->computeEnc pushDebugGroup:s];
            else if (c->blitEnc)    [c->blitEnc pushDebugGroup:s];
            else if (c->cmd)        [c->cmd pushDebugGroup:s];
        }
        void VRI_CALL CmdEndDebugGroup(VriCommandBuffer* cmd)
        {
            CommandBufferMTL* c = CB(cmd);
            if (c->renderEnc)       [c->renderEnc popDebugGroup];
            else if (c->computeEnc) [c->computeEnc popDebugGroup];
            else if (c->blitEnc)    [c->blitEnc popDebugGroup];
            else if (c->cmd)        [c->cmd popDebugGroup];
        }

        // ---- submission ----------------------------------------------------
        void VRI_CALL QueueSubmit(VriQueue* queue, const VriQueueSubmitDesc* submit)
        {
            QueueMTL* q = Q(queue);
            // Wait fences must gate the queue BEFORE the work runs. Encoders are already
            // closed by submit time, so a wait can't be prepended onto the work command
            // buffers; commit a tiny wait-only buffer first (queue executes in commit order).
            if (submit->waitFenceNum > 0)
            {
                id<MTLCommandBuffer> wb = [q->queue commandBuffer];
                for (uint32_t i = 0; i < submit->waitFenceNum; ++i)
                    [wb encodeWaitForEvent:Fen(submit->waitFences[i].fence)->event value:submit->waitFences[i].value];
                [wb commit];
            }
            for (uint32_t i = 0; i < submit->commandBufferNum; ++i)
            {
                CommandBufferMTL* c = CB(submit->commandBuffers[i]);
                if (!c->cmd) continue;
                // Signal all fences on the last command buffer (after its GPU work completes).
                if (i + 1 == submit->commandBufferNum)
                    for (uint32_t f = 0; f < submit->signalFenceNum; ++f)
                        [c->cmd encodeSignalEvent:Fen(submit->signalFences[f].fence)->event value:submit->signalFences[f].value];
                [c->cmd commit];
                [c->cmd release];
                c->cmd = nil;
            }
            // No command buffers but signal requested: signal via an empty buffer.
            if (submit->commandBufferNum == 0 && submit->signalFenceNum > 0)
            {
                id<MTLCommandBuffer> sb = [q->queue commandBuffer];
                for (uint32_t f = 0; f < submit->signalFenceNum; ++f)
                    [sb encodeSignalEvent:Fen(submit->signalFences[f].fence)->event value:submit->signalFences[f].value];
                [sb commit];
            }
        }

        void VRI_CALL QueueWaitIdle(VriQueue* queue)
        {
            id<MTLCommandBuffer> c = [Q(queue)->queue commandBuffer];
            [c commit];
            [c waitUntilCompleted];
        }
        void VRI_CALL DeviceWaitIdle(VriDevice* device)
        {
            id<MTLCommandBuffer> c = [Dev(device)->Queue() commandBuffer];
            [c commit];
            [c waitUntilCompleted];
        }
        void VRI_CALL SetDebugName(void*, const char*) {} // type-erased handle; no-op for MVP

        VriCoreInterface MakeTable()
        {
            VriCoreInterface t = {};
            t.GetDeviceDesc = GetDeviceDesc;
            t.GetFormatSupport = GetFormatSupport;
            t.GetQueue = GetQueue;
            t.CreateCommandAllocator = CreateCommandAllocator;
            t.ResetCommandAllocator = ResetCommandAllocator;
            t.DestroyCommandAllocator = DestroyCommandAllocator;
            t.CreateCommandBuffer = CreateCommandBuffer;
            t.BeginCommandBuffer = BeginCommandBuffer;
            t.EndCommandBuffer = EndCommandBuffer;
            t.CreateBuffer = CreateBuffer;
            t.DestroyBuffer = DestroyBuffer;
            t.MapBuffer = MapBuffer;
            t.UnmapBuffer = UnmapBuffer;
            t.GetBufferDeviceAddress = GetBufferDeviceAddress;
            t.CreateTexture = CreateTexture;
            t.DestroyTexture = DestroyTexture;
            t.GetBufferMemoryDesc = GetBufferMemoryDesc;
            t.GetTextureMemoryDesc = GetTextureMemoryDesc;
            t.AllocateMemory = AllocateMemory;
            t.FreeMemory = FreeMemory;
            t.BindBufferMemory = BindBufferMemory;
            t.BindTextureMemory = BindTextureMemory;
            t.CreateBufferView = CreateBufferView;
            t.CreateTextureView = CreateTextureView;
            t.CreateSampler = CreateSampler;
            t.DestroyDescriptor = DestroyDescriptor;
            t.CreatePipelineLayout = CreatePipelineLayout;
            t.DestroyPipelineLayout = DestroyPipelineLayout;
            t.CreateGraphicsPipeline = CreateGraphicsPipeline;
            t.CreateComputePipeline = CreateComputePipeline;
            t.DestroyPipeline = DestroyPipeline;
            t.CreateDescriptorPool = CreateDescriptorPool;
            t.ResetDescriptorPool = ResetDescriptorPool;
            t.DestroyDescriptorPool = DestroyDescriptorPool;
            t.AllocateDescriptorSets = AllocateDescriptorSets;
            t.UpdateDescriptorRanges = UpdateDescriptorRanges;
            t.CreateFence = CreateFence;
            t.DestroyFence = DestroyFence;
            t.GetFenceValue = GetFenceValue;
            t.Wait = Wait;
            t.CmdBeginRendering = CmdBeginRendering;
            t.CmdEndRendering = CmdEndRendering;
            t.CmdSetViewports = CmdSetViewports;
            t.CmdSetScissors = CmdSetScissors;
            t.CmdSetPipelineLayout = CmdSetPipelineLayout;
            t.CmdSetPipeline = CmdSetPipeline;
            t.CmdSetDescriptorSet = CmdSetDescriptorSet;
            t.CmdSetConstants = CmdSetConstants;
            t.CmdSetVertexBuffers = CmdSetVertexBuffers;
            t.CmdSetIndexBuffer = CmdSetIndexBuffer;
            t.CmdDraw = CmdDraw;
            t.CmdDrawIndexed = CmdDrawIndexed;
            t.CmdDrawIndirect = CmdDrawIndirect;
            t.CmdDispatch = CmdDispatch;
            t.CmdDispatchIndirect = CmdDispatchIndirect;
            t.CmdBarrier = CmdBarrier;
            t.CmdCopyBuffer = CmdCopyBuffer;
            t.CmdCopyTexture = CmdCopyTexture;
            t.CmdUploadBufferToTexture = CmdUploadBufferToTexture;
            t.CmdReadbackTextureToBuffer = CmdReadbackTextureToBuffer;
            t.CmdBeginDebugGroup = CmdBeginDebugGroup;
            t.CmdEndDebugGroup = CmdEndDebugGroup;
            t.QueueSubmit = QueueSubmit;
            t.QueueWaitIdle = QueueWaitIdle;
            t.DeviceWaitIdle = DeviceWaitIdle;
            t.SetDebugName = SetDebugName;
            return t;
        }

        const VriCoreInterface g_coreMTL = MakeTable();
    } // namespace

    const VriCoreInterface* GetCoreInterfaceMTL() { return &g_coreMTL; }
} // namespace vri::mtl
