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

    // Directly-bound argument-table capacity per stage. These are the limits that make a
    // descriptor set ineligible for direct binding and push it onto the argument-buffer
    // path (see SetNeedsArgumentBuffer); they are NOT a cap on what a bindless range may
    // declare - an argument buffer's capacity is reported as VriDeviceDesc::bindless*MaxNum.
    static constexpr uint32_t kMaxMtlDirectTextures = 128;
    static constexpr uint32_t kMaxMtlDirectSamplers = 16;

    // Argument-buffer (bindless) encoding. Metal 3 / Tier 2 argument buffers are plain
    // buffers of 8-byte entries, one per [[id(n)]]: MTLResourceID for a texture/sampler/
    // acceleration structure, a 64-bit GPU address for a `constant T*`. SPIRV-Cross emits
    // exactly that struct (see SpirvToMsl), so entry n lives at byte offset n * 8.
    static constexpr uint32_t kMtlArgBufferEntrySize = 8;

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
        bool                        renderPipelineDeferred; // graphics pipeline set before an encoder existed
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

    // One descriptor range resolved to its Metal slot(s). For a directly-bound set mslIndex is
    // the argument-table slot for the range's type; for an argument-buffer set it is the
    // [[id(n)]] of the range's first element in that set's single id namespace.
    struct BindingMTL
    {
        uint32_t            baseRegister;
        VriDescriptorType   type;
        uint32_t            mslIndex; // direct: per-type table slot / argument buffer: [[id(n)]]
        uint32_t            count;
        VriShaderStageFlags stages;
    };

    // Per-set binding model. A set is promoted to ArgumentBuffer when it declares bindless
    // intent (VariableSized / PartiallyBound) or simply does not fit the direct argument
    // table - that promotion is what removes the per-stage texture ceiling.
    enum class SetBindingModel : uint8_t
    {
        Direct,
        ArgumentBuffer,
    };

    struct SetLayoutMTL
    {
        SetBindingModel     model = SetBindingModel::Direct;
        uint32_t            argBufferSlot = 0; // MTLBuffer index the argument buffer binds to
        uint32_t            argEntryNum = 0;   // number of [[id(n)]] entries (buffer size / 8)
        VriShaderStageFlags argStages = 0;     // union of the set's range stages
    };

    struct PipelineLayoutMTL
    {
        DeviceMTL*                            device;
        std::vector<std::vector<BindingMTL>>  setBindings; // [set][range]
        std::vector<SetLayoutMTL>             setLayouts;  // [set], parallel to setBindings
        bool                                  hasPush = false;
        bool                                  hasArgBufferSet = false; // any set on the AB path
        uint32_t                              pushBufferIndex = 0;
        uint32_t                              pushSize = 0;
        VriShaderStageFlags                   pushStages = 0;

        bool IsArgBuffer(uint32_t set) const
        {
            return set < setLayouts.size() && setLayouts[set].model == SetBindingModel::ArgumentBuffer;
        }

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
        std::vector<Bound>       bound; // Direct sets: resolved on UpdateDescriptorRanges

        // ---- argument-buffer sets (SetBindingModel::ArgumentBuffer) ----
        // Metal does not make resources referenced through an argument buffer resident, so every
        // referenced MTLResource has to be replayed as useResource at bind time. Residency is
        // tracked per [[id(n)]] entry rather than appended per update, so re-updating a set
        // replaces an entry instead of accumulating a stale duplicate of it.
        struct Entry
        {
            id<MTLResource>                  resource = nil;   // nil: sampler or unpopulated
            bool                             writable = false; // storage texture/buffer
            struct AccelerationStructureMTL* tlas = nullptr;   // TLAS: also make its BLASes resident
        };
        id<MTLBuffer>                argBuffer = nil; // nil on a Direct set
        std::vector<Entry>           entries;         // sized argEntryNum on allocation
        // Flattened from `entries` on demand - useResources: takes one array per usage class.
        std::vector<id<MTLResource>> residentRead;
        std::vector<id<MTLResource>> residentReadWrite;
        std::vector<struct AccelerationStructureMTL*> residentTlas;
        bool                         residencyDirty = true;
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
        // Stages this pipeline was actually built from. An argument buffer is declared by every
        // compiled stage (force_active_argument_buffer_resources), including stages that read
        // nothing from it, so this is the exact set that must have it bound (see
        // BindArgumentBufferSet) - the set's own shaderStages would leave some of them unbound.
        VriShaderStageFlags         stageMask;
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
