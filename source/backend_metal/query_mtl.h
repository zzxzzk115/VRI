// query_mtl.h - accessor for the native Metal VriQueryInterface (timestamps + occlusion).
#pragma once

#include <vri/vri.h>
#import <Metal/Metal.h>

namespace vri::mtl
{
    struct CommandBufferMTL;
    void AttachTimestampSamples(CommandBufferMTL*, MTLRenderPassDescriptor*);
    void AttachTimestampSamples(CommandBufferMTL*, MTLComputePassDescriptor*);
    void AttachTimestampSamples(CommandBufferMTL*, MTLBlitPassDescriptor*);
    void AttachTimestampSamples(CommandBufferMTL*, MTLAccelerationStructurePassDescriptor*);
    void ResolveTimestampRanges(CommandBufferMTL*, id<MTLCommandBuffer>);
    void DestroyTimestampSamples(CommandBufferMTL*);

    // Returns the query function table. Registered by every Metal device (Apple GPUs have a
    // timestamp counter set + occlusion visibility-result buffers; pipeline statistics are absent).
    const VriQueryInterface* GetQueryInterfaceMTL();
} // namespace vri::mtl
