/*
 * vri_ext_query.h - GPU query pools (timestamps + occlusion).
 *
 * Queried via vriGetInterface(device, VRI_INTERFACE_QUERY, ...). Registered by backends that
 * support it. Timestamp availability + tick scale are reported in
 * VriDeviceDesc (hasTimestampQueries or hasTimestampRangeQueries, timestampPeriodNanoseconds).
 *
 * Usage (timestamps, the basis of a GPU profiler): CmdResetQueries once, CmdWriteTimestamp at
 * each measurement point, then CmdCopyQueries to resolve the raw uint64 ticks into a buffer.
 * Metal uses TimestampRange scopes instead of arbitrary timestamp points.
 * After the submit's fence signals, map the buffer; a section's GPU time in nanoseconds is
 * (tickEnd - tickStart) * VriDeviceDesc::timestampPeriodNanoseconds.
 *
 * Occlusion queries bracket draws inside a render pass with CmdBeginQuery/CmdEndQuery; the
 * result is the number of samples that passed the depth/stencil test (0 = fully occluded).
 *
 * Point timestamps/occlusion are uint64 results resolved GPU-side into a VriBuffer.
 * Metal execution ranges contain three uint64 fields and resolve in the completion handler
 * before the submission fence signals. Always use GetQuerySize for the result stride.
 * (Pipeline-statistics query types are a planned addition.)
 */
#ifndef VRI_EXT_QUERY_H
#define VRI_EXT_QUERY_H

#include "../vri_base.h"
#include "../vri_handles.h"

VRI_EXTERN_C_BEGIN

typedef struct VriQueryPool VriQueryPool;

typedef enum VriQueryType
{
    VriQueryType_Timestamp          = 0, /* GPU clock tick (scale by timestampPeriodNanoseconds) */
    VriQueryType_Occlusion          = 1, /* samples that passed depth/stencil between Begin/End */
    VriQueryType_PipelineStatistics = 2, /* per-stage invocation counts (VriPipelineStatistics) */
    /* Native execution envelope of a logical scope. Currently supported by Metal, whose GPU
     * counters sample encoder stages rather than arbitrary bottom-of-pipe command points. */
    VriQueryType_TimestampRange    = 3,
    VriQueryType_MaxEnum            = 0x7fffffff
} VriQueryType;

/* PipelineStatistics result: one VriPipelineStatistics (each GetQuerySize == sizeof this).
 * The field order is the portable one shared by Vulkan (all stats enabled) and D3D12. A stage
 * that does not run (no geometry/tessellation/compute in the draw) reports 0. Check
 * VriDeviceDesc::hasPipelineStatistics; not supported on OpenGL/WebGPU (CreateQueryPool fails). */
typedef struct VriPipelineStatistics
{
    uint64_t inputVertices;
    uint64_t inputPrimitives;
    uint64_t vertexShaderInvocations;
    uint64_t geometryShaderInvocations;
    uint64_t geometryShaderPrimitives;
    uint64_t clippingInvocations;
    uint64_t clippingPrimitives;
    uint64_t fragmentShaderInvocations;
    uint64_t tessControlShaderPatches;        /* D3D12 hull-shader invocations */
    uint64_t tessEvaluationShaderInvocations; /* D3D12 domain-shader invocations */
    uint64_t computeShaderInvocations;
} VriPipelineStatistics;

/* TimestampRange result, in the device's timestamp ticks. Begin/EndQuery use the same
 * index and bracket whole encoders outside render passes; nested scopes are supported.
 * Metal samples the real vertex/fragment/compute/blit/AS encoder boundaries, without
 * inserting work or serializing independent encoders. The result spans the earliest start
 * to latest completion, including scheduling gaps; overlapping scopes must not be added.
 * A scope with no executed stages returns {0,0,0}; UINT64_MAX reports unavailable data.
 * active is the union of measured stage intervals inside this scope; end-begin is its
 * elapsed envelope. Neither active values nor envelopes of overlapping scopes are additive.
 * On Metal, CmdCopyQueries for this type requires a HostReadback destination and completes
 * its host-side resolve before the submission's signal fence. Keep pools/buffers/command
 * allocators alive until completion. Timestamp point pools are unsupported on Metal.
 */
typedef struct VriTimestampRange
{
    uint64_t begin;
    uint64_t end;
    uint64_t active; /* Union of valid native stage intervals; excludes gaps between stages. */
} VriTimestampRange;

typedef struct VriQueryPoolDesc
{
    VriQueryType type;
    uint32_t     queryCount;
} VriQueryPoolDesc;

/* A GPU + CPU timestamp captured at (nearly) the same instant, for aligning GPU spans with a
 * CPU timeline. gpuTimestamp is in GPU ticks (scale by VriDeviceDesc::timestampPeriodNanoseconds);
 * cpuTimestamp is the host performance counter (QueryPerformanceCounter ticks on Windows,
 * CLOCK_MONOTONIC nanoseconds on Linux). Available when VriDeviceDesc::hasCalibratedTimestamps. */
typedef struct VriCalibratedTimestamps
{
    uint64_t gpuTimestamp;
    uint64_t cpuTimestamp;
} VriCalibratedTimestamps;

typedef struct VriQueryInterface
{
    VriResult (VRI_CALL *CreateQueryPool)(VriDevice* device, const VriQueryPoolDesc* desc, VriQueryPool** outPool);
    void      (VRI_CALL *DestroyQueryPool)(VriQueryPool* pool);
    /* Bytes one query occupies in the CmdCopyQueries destination buffer: 8 for timestamp /
     * occlusion, sizeof(VriTimestampRange) for ranges, sizeof(VriPipelineStatistics) for statistics. */
    uint32_t  (VRI_CALL *GetQuerySize)(const VriQueryPool* pool);

    /* Reset [offset, offset+num) before (re)writing. REQUIRED on Vulkan; a no-op on D3D12. */
    void (VRI_CALL *CmdResetQueries)(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t offset, uint32_t num);
    /* Timestamp query: record the GPU clock once prior work has reached the bottom of the pipe. */
    void (VRI_CALL *CmdWriteTimestamp)(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index);
    /* TimestampRange: bracket whole encoders outside render passes, with the same slot
     * at begin/end. Occlusion query: bracket draws inside a render pass. Not supported on WebGPU (its
     * occlusion model binds the query set at pass-begin, which this pool-at-query-time API
     * does not express); CreateQueryPool returns Unsupported there. */
    void (VRI_CALL *CmdBeginQuery)(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index);
    void (VRI_CALL *CmdEndQuery)(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index);
    /* Resolve num results into dstBuffer; read it back after the submit's fence.
     * dstBuffer needs num*GetQuerySize bytes and TransferDst usage (a HostReadback buffer works). */
    void (VRI_CALL *CmdCopyQueries)(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t offset, uint32_t num,
                                    VriBuffer* dstBuffer, uint64_t dstOffset);

    /* Capture a correlated GPU+CPU timestamp pair (no command buffer; immediate device call).
     * Returns Unsupported where VriDeviceDesc::hasCalibratedTimestamps is false. */
    VriResult (VRI_CALL *GetCalibratedTimestamps)(VriDevice* device, VriCalibratedTimestamps* out);
} VriQueryInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_QUERY_H */
