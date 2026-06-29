// query_mtl.mm - VriQueryInterface for native Metal (timestamps + occlusion).
//
// Timestamps: Apple GPUs sample the GPU clock only at *encoder stage boundaries* (M-series report
// no draw/blit/dispatch-boundary sampling), and a counter sample buffer is resolved CPU-side. So
// CmdWriteTimestamp opens a tiny empty blit encoder whose start-of-encoder boundary samples the
// clock into the pool's MTLCounterSampleBuffer, and CmdCopyQueries records a resolve that
// QueueSubmit runs in the command buffer's completion handler (see QueueSubmit in core_mtl.mm).
//
// Occlusion: native. A per-command-buffer visibility-result buffer is bound on every render pass
// (see CmdBeginRendering); CmdBeginQuery enables MTLVisibilityResultModeCounting into a fresh slot
// and CmdCopyQueries blits the recorded slots into the destination buffer (all GPU-side).
//
// Pipeline statistics: Apple GPUs expose no statistic counter set, so that pool type is Unsupported.

#include "query_mtl.h"
#include "device_mtl.h"
#include "objects_mtl.h"

namespace vri::mtl
{
    namespace
    {
        inline DeviceMTL*        Dev(VriDevice* h) { return reinterpret_cast<DeviceMTL*>(h); }
        inline CommandBufferMTL* CB(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferMTL*>(h); }
        inline BufferMTL*        Buf(VriBuffer* h) { return reinterpret_cast<BufferMTL*>(h); }
        inline QueryPoolMTL*     QP(VriQueryPool* h) { return reinterpret_cast<QueryPoolMTL*>(h); }

        // The standard timestamp counter set ("timestamp"); nil if the device lacks it.
        id<MTLCounterSet> FindTimestampCounterSet(id<MTLDevice> dev)
        {
            for (id<MTLCounterSet> cs in [dev counterSets])
                if ([[cs name] isEqualToString:MTLCommonCounterSetTimestamp])
                    return cs;
            return nil;
        }

        VriResult VRI_CALL CreateQueryPool(VriDevice* device, const VriQueryPoolDesc* desc, VriQueryPool** out)
        {
            if (!desc || !out || desc->queryCount == 0)
                return VriResult_InvalidArgument;
            DeviceMTL*   d = Dev(device);
            id<MTLDevice> mdev = d->Device();

            QueryPoolMTL* q = new QueryPoolMTL{};
            q->device = d;
            q->type   = desc->type;
            q->count  = desc->queryCount;

            if (desc->type == VriQueryType_Timestamp)
            {
                id<MTLCounterSet> set = FindTimestampCounterSet(mdev);
                if (!set)
                {
                    delete q;
                    return VriResult_Unsupported;
                }
                MTLCounterSampleBufferDescriptor* sd = [[MTLCounterSampleBufferDescriptor alloc] init];
                sd.counterSet  = set;
                sd.storageMode = MTLStorageModeShared;
                sd.sampleCount = desc->queryCount;
                NSError* err   = nil;
                q->sampleBuf   = [mdev newCounterSampleBufferWithDescriptor:sd error:&err]; // +1
                [sd release];
                if (!q->sampleBuf)
                {
                    d->ReportError("newCounterSampleBufferWithDescriptor failed");
                    delete q;
                    return VriResult_Failure;
                }
            }
            else if (desc->type == VriQueryType_Occlusion)
            {
                q->results = [mdev newBufferWithLength:desc->queryCount * sizeof(uint64_t)
                                               options:MTLResourceStorageModeShared]; // +1
                if (!q->results)
                {
                    delete q;
                    return VriResult_Failure;
                }
            }
            else
            {
                // Pipeline statistics: no Apple statistic counter set.
                delete q;
                return VriResult_Unsupported;
            }
            *out = reinterpret_cast<VriQueryPool*>(q);
            return VriResult_Success;
        }

        void VRI_CALL DestroyQueryPool(VriQueryPool* pool)
        {
            if (!pool)
                return;
            QueryPoolMTL* q = QP(pool);
            if (q->sampleBuf) [q->sampleBuf release];
            if (q->results)   [q->results release];
            delete q;
        }

        uint32_t VRI_CALL GetQuerySize(const VriQueryPool*) { return sizeof(uint64_t); }

        // No reset needed: each timestamp re-samples its slot and each occlusion query grabs a fresh
        // visibility slot (matches the D3D12 no-op).
        void VRI_CALL CmdResetQueries(VriCommandBuffer*, VriQueryPool*, uint32_t, uint32_t) {}

        void VRI_CALL CmdWriteTimestamp(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index)
        {
            CommandBufferMTL* c = CB(cmd);
            QueryPoolMTL*     q = QP(pool);
            if (!q->sampleBuf)
                return;
            if (c->renderEnc)
            {
                c->device->ReportError("CmdWriteTimestamp inside a render pass is unsupported on Metal");
                return;
            }
            // Close any open aux encoder, then sample the GPU clock at the start boundary of a fresh
            // (empty) blit encoder - the only counter-sampling point Apple Silicon supports.
            if (c->blitEnc)    { [c->blitEnc endEncoding];    [c->blitEnc release];    c->blitEnc = nil; }
            if (c->computeEnc) { [c->computeEnc endEncoding]; [c->computeEnc release]; c->computeEnc = nil; }

            MTLBlitPassDescriptor* bpd = [MTLBlitPassDescriptor blitPassDescriptor]; // autoreleased
            MTLBlitPassSampleBufferAttachmentDescriptor* a = bpd.sampleBufferAttachments[0];
            a.sampleBuffer              = q->sampleBuf;
            a.startOfEncoderSampleIndex = index;
            a.endOfEncoderSampleIndex   = MTLCounterDontSample;
            id<MTLBlitCommandEncoder> e = [c->cmd blitCommandEncoderWithDescriptor:bpd];
            // A start-of-encoder sample is only recorded if the encoder actually runs, so give it a
            // trivial fill (the GPU clock is captured at the start boundary, before this fill).
            if (!c->tsScratch)
                c->tsScratch = [c->device->Device() newBufferWithLength:4 options:MTLResourceStorageModePrivate];
            [e fillBuffer:c->tsScratch range:NSMakeRange(0, 4) value:0];
            [e endEncoding];
        }

        void VRI_CALL CmdBeginQuery(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index)
        {
            CommandBufferMTL* c = CB(cmd);
            QueryPoolMTL*     q = QP(pool);
            if (!q->results || !c->renderEnc)
                return;
            if (c->visNextSlot >= kMtlMaxOcclusionSlots)
            {
                c->device->ReportError("occlusion queries exceeded kMtlMaxOcclusionSlots per command buffer");
                return;
            }
            uint32_t slot = c->visNextSlot++;
            c->occMarks.push_back({q, index, slot});
            [c->renderEnc setVisibilityResultMode:MTLVisibilityResultModeCounting offset:slot * sizeof(uint64_t)];
        }

        void VRI_CALL CmdEndQuery(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t)
        {
            CommandBufferMTL* c = CB(cmd);
            if (!QP(pool)->results || !c->renderEnc)
                return;
            [c->renderEnc setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
        }

        void VRI_CALL CmdCopyQueries(VriCommandBuffer* cmd,
                                     VriQueryPool*     pool,
                                     uint32_t          offset,
                                     uint32_t          num,
                                     VriBuffer*        dstBuffer,
                                     uint64_t          dstOffset)
        {
            CommandBufferMTL* c = CB(cmd);
            QueryPoolMTL*     q = QP(pool);

            if (q->sampleBuf)
            {
                // Resolve is CPU-side; record it and let QueueSubmit run it after GPU completion.
                void* dst = static_cast<char*>([Buf(dstBuffer)->buffer contents]) + dstOffset;
                c->tsResolves.push_back({q->sampleBuf, offset, num, dst});
                return;
            }
            if (!q->results || !c->visBuffer)
                return;
            // Occlusion: blit each recorded slot into the destination, in query order.
            if (c->computeEnc) { [c->computeEnc endEncoding]; [c->computeEnc release]; c->computeEnc = nil; }
            if (!c->blitEnc) c->blitEnc = [[c->cmd blitCommandEncoder] retain];
            id<MTLBuffer> dst = Buf(dstBuffer)->buffer;
            for (uint32_t k = 0; k < num; ++k)
            {
                for (const auto& m : c->occMarks)
                {
                    if (m.pool == q && m.index == offset + k)
                    {
                        [c->blitEnc copyFromBuffer:c->visBuffer
                                      sourceOffset:m.visSlot * sizeof(uint64_t)
                                          toBuffer:dst
                                 destinationOffset:dstOffset + k * sizeof(uint64_t)
                                              size:sizeof(uint64_t)];
                        break;
                    }
                }
            }
        }

        VriResult VRI_CALL GetCalibratedTimestamps(VriDevice* device, VriCalibratedTimestamps* out)
        {
            if (!out)
                return VriResult_InvalidArgument;
            MTLTimestamp cpu = 0, gpu = 0;
            [Dev(device)->Device() sampleTimestamps:&cpu gpuTimestamp:&gpu]; // both in nanoseconds
            out->gpuTimestamp = gpu;
            out->cpuTimestamp = cpu;
            return VriResult_Success;
        }

        const VriQueryInterface g_queryMTL = {
            CreateQueryPool,
            DestroyQueryPool,
            GetQuerySize,
            CmdResetQueries,
            CmdWriteTimestamp,
            CmdBeginQuery,
            CmdEndQuery,
            CmdCopyQueries,
            GetCalibratedTimestamps,
        };
    } // namespace

    const VriQueryInterface* GetQueryInterfaceMTL() { return &g_queryMTL; }
} // namespace vri::mtl
