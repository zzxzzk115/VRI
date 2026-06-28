// query_d3d12.cpp - VriQueryInterface for D3D12. Timestamp + occlusion query heaps, resolved
// GPU-side into a buffer via ResolveQueryData (the app reads it back after the submit's fence).
// D3D12 needs no query reset, so CmdResetQueries is a no-op. Registered by every D3D12 device.

#include "query_d3d12.h"
#include "device_d3d12.h"
#include "objects_d3d12.h"

namespace vri::d3d12
{
    namespace
    {
        inline DeviceD3D12*        Dev(VriDevice* h) { return reinterpret_cast<DeviceD3D12*>(h); }
        inline CommandBufferD3D12* Cmd(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferD3D12*>(h); }
        inline BufferD3D12*        Buf(VriBuffer* h) { return reinterpret_cast<BufferD3D12*>(h); }
        inline QueryPoolD3D12*     QP(VriQueryPool* h) { return reinterpret_cast<QueryPoolD3D12*>(h); }

        VriResult VRI_CALL CreateQueryPool(VriDevice* device, const VriQueryPoolDesc* desc, VriQueryPool** out)
        {
            if (!desc || !out || desc->queryCount == 0)
                return VriResult_InvalidArgument;
            DeviceD3D12*          d  = Dev(device);
            D3D12_QUERY_HEAP_DESC hd = {};
            hd.Count                 = desc->queryCount;
            D3D12_QUERY_TYPE qt      = D3D12_QUERY_TYPE_TIMESTAMP;
            switch (desc->type)
            {
                case VriQueryType_Occlusion:
                    hd.Type = D3D12_QUERY_HEAP_TYPE_OCCLUSION;
                    qt      = D3D12_QUERY_TYPE_OCCLUSION;
                    break;
                case VriQueryType_PipelineStatistics:
                    hd.Type = D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS;
                    qt      = D3D12_QUERY_TYPE_PIPELINE_STATISTICS;
                    break;
                default:
                    hd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
                    qt      = D3D12_QUERY_TYPE_TIMESTAMP;
                    break;
            }

            QueryPoolD3D12* q = new QueryPoolD3D12 {};
            if (FAILED(d->Device()->CreateQueryHeap(&hd, IID_PPV_ARGS(&q->heap))))
            {
                delete q;
                d->ReportError("CreateQueryHeap failed");
                return VriResult_Failure;
            }
            q->device    = d;
            q->queryType = qt;
            q->type      = desc->type;
            q->count     = desc->queryCount;
            *out         = ToHandle(q);
            return VriResult_Success;
        }

        void VRI_CALL DestroyQueryPool(VriQueryPool* pool)
        {
            if (pool)
                delete QP(pool);
        }

        uint32_t VRI_CALL GetQuerySize(const VriQueryPool* pool)
        {
            const QueryPoolD3D12* q = reinterpret_cast<const QueryPoolD3D12*>(pool);
            return q->type == VriQueryType_PipelineStatistics ? sizeof(VriPipelineStatistics) : sizeof(uint64_t);
        }

        // D3D12 needs no reset: each EndQuery overwrites and ResolveQueryData reads the latest.
        void VRI_CALL CmdResetQueries(VriCommandBuffer*, VriQueryPool*, uint32_t, uint32_t) {}

        void VRI_CALL CmdWriteTimestamp(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index)
        {
            Cmd(cmd)->list->EndQuery(QP(pool)->heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, index);
        }

        void VRI_CALL CmdBeginQuery(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index)
        {
            QueryPoolD3D12* q = QP(pool);
            Cmd(cmd)->list->BeginQuery(q->heap.Get(), q->queryType, index);
        }

        void VRI_CALL CmdEndQuery(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index)
        {
            QueryPoolD3D12* q = QP(pool);
            Cmd(cmd)->list->EndQuery(q->heap.Get(), q->queryType, index);
        }

        void VRI_CALL CmdCopyQueries(VriCommandBuffer* cmd,
                                     VriQueryPool*     pool,
                                     uint32_t          offset,
                                     uint32_t          num,
                                     VriBuffer*        dstBuffer,
                                     uint64_t          dstOffset)
        {
            // ResolveQueryData requires the destination in COPY_DEST; a HostReadback buffer (the
            // common read-back target) is created there and stays put.
            QueryPoolD3D12* q = QP(pool);
            Cmd(cmd)->list->ResolveQueryData(
                q->heap.Get(), q->queryType, offset, num, Buf(dstBuffer)->resource.Get(), dstOffset);
        }

        const VriQueryInterface g_queryD3D12 = {
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

    const VriQueryInterface* GetQueryInterfaceD3D12() { return &g_queryD3D12; }
} // namespace vri::d3d12
