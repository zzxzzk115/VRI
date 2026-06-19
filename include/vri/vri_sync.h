/*
 * vri_sync.h - synchronization descriptors.
 *
 * Fences are timeline-style (monotonic uint64 value), matching modern Vulkan /
 * D3D12. Legacy backends emulate the timeline.
 */
#ifndef VRI_SYNC_H
#define VRI_SYNC_H

#include "vri_base.h"
#include "vri_handles.h"
#include "vri_flags.h"

VRI_EXTERN_C_BEGIN

typedef struct VriFenceSubmitDesc
{
    VriFence*             fence;
    uint64_t              value;  /* value to wait-for / signal-to */
    VriPipelineStageFlags stages; /* stages that wait / that must complete */
} VriFenceSubmitDesc;

typedef struct VriQueueSubmitDesc
{
    const VriFenceSubmitDesc*       waitFences;
    uint32_t                        waitFenceNum;
    VriCommandBuffer* const*        commandBuffers;
    uint32_t                        commandBufferNum;
    const VriFenceSubmitDesc*       signalFences;
    uint32_t                        signalFenceNum;
} VriQueueSubmitDesc;

VRI_EXTERN_C_END

#endif /* VRI_SYNC_H */
