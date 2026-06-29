// objects_mtl.h - concrete Metal objects behind the opaque VRI handles.
//
// Objective-C++ header (imports <Metal/Metal.h>); only the backend's .mm files
// include it. Mirrors objects_wgpu.h / objects_d3d12.h. Manual retain/release
// (no ARC) matches the project's other .mm files: newXxx returns +1 (released in
// Destroy*), autoreleased returns are retained when stored past their pool.
#pragma once

#import <Metal/Metal.h>

#include <vector>

#include <vri/vri.h>

namespace vri::mtl
{
    class DeviceMTL;
    struct QueryPoolMTL;

    // Metal exposes 31 buffer argument slots per stage. Descriptor buffers + push
    // constants grow up from 0; vertex stream buffers grow down from 30 so the two
    // pools can't collide for any realistic layout (see CreatePipelineLayout /
    // CreateGraphicsPipeline / CmdSetVertexBuffers).
    static constexpr uint32_t kMaxMtlBuffers = 31;
    inline uint32_t VertexBufferIndex(uint32_t streamSlot) { return kMaxMtlBuffers - 1 - streamSlot; }

    // Max occlusion queries per command buffer (one visibility-result slot each). The buffer is
    // bound at every render pass, so this caps total occlusion Begin/End pairs between submits.
    static constexpr uint32_t kMtlMaxOcclusionSlots = 4096;

    struct QueueMTL
    {
        DeviceMTL*           device;
        id<MTLCommandQueue>  queue;
    };

    struct CommandAllocatorMTL
    {
        DeviceMTL*  device; // Metal has no command pool; command buffers come from the queue
        VriQueueType type;
    };

    struct CommandBufferMTL
    {
        DeviceMTL*                  device;
        id<MTLCommandQueue>         queue;      // the per-type queue this buffer is allocated against
        id<MTLCommandBuffer>        cmd;        // retained for the Begin..Submit span
        id<MTLRenderCommandEncoder> renderEnc;  // valid between BeginRendering/EndRendering
        id<MTLComputeCommandEncoder> computeEnc; // lazily opened for dispatch
        id<MTLBlitCommandEncoder>   blitEnc;     // lazily opened for copies
        // bound state, applied at draw/dispatch
        const struct PipelineLayoutMTL* boundLayout;
        const struct PipelineMTL*       boundPipeline;
        id<MTLBuffer>               indexBuffer;
        uint64_t                    indexOffset;
        MTLIndexType                indexType;

        // ---- query state (ext/vri_ext_query.h) ----
        // Occlusion: a single per-command-buffer visibility-result buffer is bound on every render
        // pass; each CmdBeginQuery grabs a fresh uint64 slot (visNextSlot) and CmdCopyQueries blits
        // the recorded slots into the destination (all GPU-side).
        id<MTLBuffer>               visBuffer;   // lazily created; reused across Begin..Submit resets
        uint32_t                    visNextSlot; // next free slot index, reset each BeginCommandBuffer
        id<MTLBuffer>               tsScratch;   // tiny buffer a timestamp blit encoder fills (must be non-empty)
        struct OcclusionMark { QueryPoolMTL* pool; uint32_t index; uint32_t visSlot; };
        std::vector<OcclusionMark>  occMarks;
        // Timestamp: Apple Silicon only resolves counter sample buffers CPU-side, so CmdCopyQueries
        // records the resolve and QueueSubmit executes it in the command buffer's completion handler.
        struct TimestampResolve { id<MTLCounterSampleBuffer> sb; uint32_t srcIndex; uint32_t num; void* dst; };
        std::vector<TimestampResolve> tsResolves;
    };

    struct BufferMTL
    {
        DeviceMTL*       device;
        id<MTLBuffer>    buffer;      // nil until BindBufferMemory for the explicit (Undefined) path
        uint64_t         size;
        MTLResourceOptions options;   // remembered for the deferred (heap-bound) path
        bool             owned;       // created via newBufferWithLength (release on destroy)
    };

    struct TextureMTL
    {
        DeviceMTL*       device;
        id<MTLTexture>   texture;
        MTLTextureDescriptor* descriptor; // retained for the deferred (heap-bound) path; else nil
        VriFormat        format;
        MTLPixelFormat   mtlFormat;
        VriTextureType   type;
        uint32_t         width;
        uint32_t         height;
        uint32_t         depth;
        uint32_t         mipNum;
        uint32_t         layerNum;
        uint32_t         sampleNum;
        uint32_t         texelSize;
        bool             owned;
    };

    struct DescriptorMTL
    {
        enum class Kind { Buffer, Texture, Sampler, AccelStruct } kind;
        DeviceMTL*               device;
        id<MTLBuffer>            buffer;       // Kind::Buffer
        uint64_t                 bufferOffset;
        uint64_t                 bufferRange;
        id<MTLTexture>          texture;      // Kind::Texture (a view; released on destroy if owned)
        bool                     ownsTexture;
        id<MTLSamplerState>     sampler;      // Kind::Sampler
        id<MTLAccelerationStructure> accel;   // Kind::AccelStruct (borrowed; owned by the AS object)
        struct AccelerationStructureMTL* accelObj = nullptr; // TLAS: source object, to keep its BLASes resident
    };

    // One descriptor range resolved to its Metal argument-table slot(s).
    struct BindingMTL
    {
        uint32_t            baseRegister;
        VriDescriptorType   type;
        uint32_t            mslIndex; // buffer / texture / sampler slot, per type
        uint32_t            count;
        VriShaderStageFlags stages;
    };

    struct PipelineLayoutMTL
    {
        DeviceMTL*                            device;
        std::vector<std::vector<BindingMTL>>  setBindings; // [set][range]
        bool                                  hasPush = false;
        uint32_t                              pushBufferIndex = 0;
        uint32_t                              pushSize = 0;
        VriShaderStageFlags                   pushStages = 0;

        const BindingMTL* Find(uint32_t set, uint32_t binding) const
        {
            if (set >= setBindings.size())
                return nullptr;
            for (const BindingMTL& b : setBindings[set])
                if (b.baseRegister == binding)
                    return &b;
            return nullptr;
        }
    };

    struct DescriptorSetMTL
    {
        DeviceMTL*               device;
        const PipelineLayoutMTL* layout;
        uint32_t                 setIndex;
        struct Bound
        {
            uint32_t            mslIndex;
            VriDescriptorType   type;
            VriShaderStageFlags stages;
            const DescriptorMTL* desc;
        };
        std::vector<Bound>       bound; // resolved on UpdateDescriptorRanges
    };

    struct DescriptorPoolMTL
    {
        DeviceMTL*                     device; // Metal has no pool; owns the allocated sets
        std::vector<DescriptorSetMTL*> sets;
    };

    struct PipelineMTL
    {
        DeviceMTL*                  device;
        bool                        isCompute;
        id<MTLRenderPipelineState>  render;
        id<MTLDepthStencilState>    depthStencil;
        id<MTLComputePipelineState> compute;
        MTLPrimitiveType            primType;
        MTLCullMode                 cull;
        MTLWinding                  winding;
        MTLTriangleFillMode         fill;
        bool                        depthClampEnable;
        bool                        stencilTest;
        uint32_t                    stencilReference;
        MTLSize                     threadsPerThreadgroup; // compute local size (from SPIR-V)
        // Mesh-shader pipeline (object/mesh/fragment): threadgroup sizes for drawMeshThreadgroups.
        bool                        isMesh;
        MTLSize                     objectTG;  // task/object stage local size (1,1,1 if mesh-only)
        MTLSize                     meshTG;    // mesh stage local size
        // Multiview (single-pass stereo): 0 = off. viewCount = #views, viewBase = first view index;
        // bound as the spvViewMask buffer (see CmdSetPipeline) and used to scale the instance count.
        uint32_t                    viewCount;
        uint32_t                    viewBase;
    };

    struct FenceMTL
    {
        DeviceMTL*          device;
        id<MTLSharedEvent>  event; // native timeline
    };

    struct MemoryMTL
    {
        DeviceMTL*        device;
        id<MTLHeap>       heap;
        uint64_t          size;
        VriMemoryLocation location;
    };

    struct QueryPoolMTL
    {
        DeviceMTL*                 device;
        VriQueryType               type;
        uint32_t                   count;
        id<MTLCounterSampleBuffer> sampleBuf; // Timestamp: GPU clock samples (nil otherwise)
        id<MTLBuffer>              results;    // Occlusion: count * uint64 visibility results (nil otherwise)
    };

    struct PipelineCacheMTL
    {
        DeviceMTL*            device;
        id<MTLBinaryArchive>  archive; // pipeline binaries; populated at pipeline creation, serialized to a blob
        // serializeToURL: is reliable only on the first call per archive object (a Metal quirk), so the
        // serialized bytes are cached and only refreshed when a new pipeline marks the archive dirty.
        std::vector<uint8_t>  blob;
        bool                  dirty;
    };

    inline VriQueue*            ToHandle(QueueMTL* q)            { return reinterpret_cast<VriQueue*>(q); }
    inline VriCommandAllocator* ToHandle(CommandAllocatorMTL* a) { return reinterpret_cast<VriCommandAllocator*>(a); }
    inline VriCommandBuffer*    ToHandle(CommandBufferMTL* c)    { return reinterpret_cast<VriCommandBuffer*>(c); }
    inline VriBuffer*           ToHandle(BufferMTL* b)           { return reinterpret_cast<VriBuffer*>(b); }
    inline VriTexture*          ToHandle(TextureMTL* t)          { return reinterpret_cast<VriTexture*>(t); }
    inline VriDescriptor*       ToHandle(DescriptorMTL* d)       { return reinterpret_cast<VriDescriptor*>(d); }
    inline VriPipelineLayout*   ToHandle(PipelineLayoutMTL* p)   { return reinterpret_cast<VriPipelineLayout*>(p); }
    inline VriPipeline*         ToHandle(PipelineMTL* p)         { return reinterpret_cast<VriPipeline*>(p); }
    inline VriFence*            ToHandle(FenceMTL* f)            { return reinterpret_cast<VriFence*>(f); }
    inline VriDescriptorPool*   ToHandle(DescriptorPoolMTL* p)   { return reinterpret_cast<VriDescriptorPool*>(p); }
    inline VriDescriptorSet*    ToHandle(DescriptorSetMTL* s)    { return reinterpret_cast<VriDescriptorSet*>(s); }
    inline VriMemory*           ToHandle(MemoryMTL* m)           { return reinterpret_cast<VriMemory*>(m); }
    inline VriQueryPool*        ToHandle(QueryPoolMTL* q)        { return reinterpret_cast<VriQueryPool*>(q); }
    inline VriPipelineCache*    ToHandle(PipelineCacheMTL* p)    { return reinterpret_cast<VriPipelineCache*>(p); }
} // namespace vri::mtl
