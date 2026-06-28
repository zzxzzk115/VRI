// query_wgpu.cpp - VriQueryInterface for WebGPU. Timestamp queries via wgpuCommandEncoderWriteTimestamp
// (wgpu-native supports writing inside the encoder, which matches VRI's arbitrary-point model).
// Results are resolved into an internal QUERY_RESOLVE buffer and then copied into the caller's
// buffer (WebGPU's resolve target needs QUERY_RESOLVE usage, which can't coexist with MAP_READ).
// Registered by every WebGPU device; CreateQueryPool returns Unsupported without the feature.

#include "query_wgpu.h"
#include "device_wgpu.h"
#include "objects_wgpu.h"

namespace vri::wgpu
{
    namespace
    {
        inline DeviceWGPU*        Dev(VriDevice* h) { return reinterpret_cast<DeviceWGPU*>(h); }
        inline CommandBufferWGPU* Cmd(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferWGPU*>(h); }
        inline BufferWGPU*        Buf(VriBuffer* h) { return reinterpret_cast<BufferWGPU*>(h); }
        inline QueryPoolWGPU*     QP(VriQueryPool* h) { return reinterpret_cast<QueryPoolWGPU*>(h); }

        // Timestamp writes + query resolves are encoder-level, so a lazily-open compute pass
        // (begun for a prior dispatch) must be closed first -- mirrors EndComputePass in core_wgpu.
        void EndOpenComputePass(CommandBufferWGPU* c)
        {
            if (c->computePass)
            {
                wgpuComputePassEncoderEnd(c->computePass);
                wgpuComputePassEncoderRelease(c->computePass);
                c->computePass = nullptr;
            }
        }

        VriResult VRI_CALL CreateQueryPool(VriDevice* device, const VriQueryPoolDesc* desc, VriQueryPool** out)
        {
            if (!desc || !out || desc->queryCount == 0)
                return VriResult_InvalidArgument;
            DeviceWGPU* d = Dev(device);
            // WebGPU occlusion binds the query set at render-pass-begin, which VRI's
            // pool-at-query-time API does not express -> only timestamps are supported here.
            if (desc->type != VriQueryType_Timestamp)
                return VriResult_Unsupported;
            if (!d->HasTimestamp())
                return VriResult_Unsupported;

            WGPUQuerySetDescriptor qsd = {};
            qsd.type                   = WGPUQueryType_Timestamp;
            qsd.count                  = desc->queryCount;
            WGPUQuerySet set           = wgpuDeviceCreateQuerySet(d->Device(), &qsd);
            if (!set)
                return VriResult_Failure;

            WGPUBufferDescriptor bd = {};
            bd.usage                = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
            bd.size                 = static_cast<uint64_t>(desc->queryCount) * sizeof(uint64_t);
            WGPUBuffer resolve      = wgpuDeviceCreateBuffer(d->Device(), &bd);
            if (!resolve)
            {
                wgpuQuerySetRelease(set);
                return VriResult_Failure;
            }
            *out = ToHandle(new QueryPoolWGPU {d, set, resolve, desc->queryCount});
            return VriResult_Success;
        }

        void VRI_CALL DestroyQueryPool(VriQueryPool* pool)
        {
            if (!pool)
                return;
            QueryPoolWGPU* q = QP(pool);
            wgpuBufferRelease(q->resolveBuffer);
            wgpuQuerySetRelease(q->querySet);
            delete q;
        }

        uint32_t VRI_CALL GetQuerySize(const VriQueryPool*) { return sizeof(uint64_t); }

        // WebGPU query sets need no reset.
        void VRI_CALL CmdResetQueries(VriCommandBuffer*, VriQueryPool*, uint32_t, uint32_t) {}

        // Occlusion is unsupported on WebGPU (no occlusion pool can be created), so these are
        // never reached -- present only to fill the interface table.
        void VRI_CALL CmdBeginQuery(VriCommandBuffer*, VriQueryPool*, uint32_t) {}
        void VRI_CALL CmdEndQuery(VriCommandBuffer*, VriQueryPool*, uint32_t) {}

        void VRI_CALL CmdWriteTimestamp(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index)
        {
            CommandBufferWGPU* c = Cmd(cmd);
            EndOpenComputePass(c);
            wgpuCommandEncoderWriteTimestamp(c->encoder, QP(pool)->querySet, index);
        }

        void VRI_CALL CmdCopyQueries(VriCommandBuffer* cmd,
                                     VriQueryPool*     pool,
                                     uint32_t          offset,
                                     uint32_t          num,
                                     VriBuffer*        dstBuffer,
                                     uint64_t          dstOffset)
        {
            CommandBufferWGPU* c = Cmd(cmd);
            QueryPoolWGPU*     q = QP(pool);
            EndOpenComputePass(c);
            // Resolve into the QUERY_RESOLVE scratch, then copy into the caller's (read-back) buffer.
            wgpuCommandEncoderResolveQuerySet(c->encoder, q->querySet, offset, num, q->resolveBuffer, 0);
            wgpuCommandEncoderCopyBufferToBuffer(c->encoder,
                                                 q->resolveBuffer,
                                                 0,
                                                 Buf(dstBuffer)->buffer,
                                                 dstOffset,
                                                 static_cast<uint64_t>(num) * sizeof(uint64_t));
        }

        const VriQueryInterface g_queryWGPU = {
            CreateQueryPool,
            DestroyQueryPool,
            GetQuerySize,
            CmdResetQueries,
            CmdWriteTimestamp,
            CmdBeginQuery,
            CmdEndQuery,
            CmdCopyQueries,
        };
    } // namespace

    const VriQueryInterface* GetQueryInterfaceWGPU() { return &g_queryWGPU; }
} // namespace vri::wgpu
