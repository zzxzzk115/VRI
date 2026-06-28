/*
 * vri_ext_query.h - GPU query pools (timestamps + occlusion).
 *
 * Queried via vriGetInterface(device, VRI_INTERFACE_QUERY, ...). Registered by backends that
 * support it (Vulkan, D3D12). Timestamp availability + tick scale are reported in
 * VriDeviceDesc (hasTimestampQueries, timestampPeriodNanoseconds).
 *
 * Usage (timestamps, the basis of a GPU profiler): CmdResetQueries once, CmdWriteTimestamp at
 * each measurement point, then CmdCopyQueries to resolve the raw uint64 ticks into a buffer.
 * After the submit's fence signals, map the buffer; a section's GPU time in nanoseconds is
 * (tickEnd - tickStart) * VriDeviceDesc::timestampPeriodNanoseconds.
 *
 * Occlusion queries bracket draws inside a render pass with CmdBeginQuery/CmdEndQuery; the
 * result is the number of samples that passed the depth/stencil test (0 = fully occluded).
 *
 * Results are always 64-bit unsigned and resolved GPU-side into a VriBuffer (no host-side
 * read-out call); this maps cleanly onto vkCmdCopyQueryPoolResults and D3D12 ResolveQueryData.
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
    VriQueryType_Timestamp = 0, /* GPU clock tick (scale by timestampPeriodNanoseconds) */
    VriQueryType_Occlusion = 1, /* samples that passed depth/stencil between Begin/End */
    VriQueryType_MaxEnum   = 0x7fffffff
} VriQueryType;

typedef struct VriQueryPoolDesc
{
    VriQueryType type;
    uint32_t     queryCount;
} VriQueryPoolDesc;

typedef struct VriQueryInterface
{
    VriResult (VRI_CALL *CreateQueryPool)(VriDevice* device, const VriQueryPoolDesc* desc, VriQueryPool** outPool);
    void      (VRI_CALL *DestroyQueryPool)(VriQueryPool* pool);
    /* Bytes one query occupies in the CmdCopyQueries destination buffer (8 for both types). */
    uint32_t  (VRI_CALL *GetQuerySize)(const VriQueryPool* pool);

    /* Reset [offset, offset+num) before (re)writing. REQUIRED on Vulkan; a no-op on D3D12. */
    void (VRI_CALL *CmdResetQueries)(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t offset, uint32_t num);
    /* Timestamp query: record the GPU clock once prior work has reached the bottom of the pipe. */
    void (VRI_CALL *CmdWriteTimestamp)(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index);
    /* Occlusion query: bracket draws inside a render pass. Not supported on WebGPU (its
     * occlusion model binds the query set at pass-begin, which this pool-at-query-time API
     * does not express); CreateQueryPool returns Unsupported there. */
    void (VRI_CALL *CmdBeginQuery)(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index);
    void (VRI_CALL *CmdEndQuery)(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index);
    /* Resolve num results (uint64 each) into dstBuffer; read it back after the submit's fence.
     * dstBuffer needs num*GetQuerySize bytes and TransferDst usage (a HostReadback buffer works). */
    void (VRI_CALL *CmdCopyQueries)(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t offset, uint32_t num,
                                    VriBuffer* dstBuffer, uint64_t dstOffset);
} VriQueryInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_QUERY_H */
