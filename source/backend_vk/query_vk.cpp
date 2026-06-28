// query_vk.cpp - VriQueryInterface for Vulkan. Timestamp + occlusion query pools, resolved
// GPU-side into a buffer via vkCmdCopyQueryPoolResults (the app reads the buffer back after
// the submit's fence). Registered by every Vulkan device.

#include "query_vk.h"
#include "device_vk.h"
#include "objects_vk.h"

namespace vri::vk
{
    namespace
    {
        inline DeviceVK*        Dev(VriDevice* h) { return reinterpret_cast<DeviceVK*>(h); }
        inline CommandBufferVK* Cmd(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferVK*>(h); }
        inline BufferVK*        Buf(VriBuffer* h) { return reinterpret_cast<BufferVK*>(h); }
        inline QueryPoolVK*     QP(VriQueryPool* h) { return reinterpret_cast<QueryPoolVK*>(h); }

        VriResult VRI_CALL CreateQueryPool(VriDevice* device, const VriQueryPoolDesc* desc, VriQueryPool** out)
        {
            if (!desc || !out || desc->queryCount == 0)
                return VriResult_InvalidArgument;
            DeviceVK*             d  = Dev(device);
            VkQueryPoolCreateInfo ci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            ci.queryCount            = desc->queryCount;
            if (desc->type == VriQueryType_Occlusion)
                ci.queryType = VK_QUERY_TYPE_OCCLUSION;
            else if (desc->type == VriQueryType_PipelineStatistics)
            {
                if (d->Desc().hasPipelineStatistics == VRI_FALSE)
                    return VriResult_Unsupported;
                ci.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
                // All 11 statistics, in the bit order that matches VriPipelineStatistics / D3D12.
                ci.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
                                        VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
                                        VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
                                        VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
                                        VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT |
                                        VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
                                        VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
                                        VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT |
                                        VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT |
                                        VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT |
                                        VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
            }
            else
                ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            const VkQueryType vt = ci.queryType;

            VkQueryPool pool = VK_NULL_HANDLE;
            if (vkCreateQueryPool(d->Device(), &ci, nullptr, &pool) != VK_SUCCESS)
                return VriResult_Failure;
            *out = ToHandle(new QueryPoolVK {d, pool, vt, desc->queryCount});
            return VriResult_Success;
        }

        void VRI_CALL DestroyQueryPool(VriQueryPool* pool)
        {
            if (!pool)
                return;
            QueryPoolVK* q = QP(pool);
            vkDestroyQueryPool(q->device->Device(), q->pool, nullptr);
            delete q;
        }

        uint32_t VRI_CALL GetQuerySize(const VriQueryPool* pool)
        {
            const QueryPoolVK* q = reinterpret_cast<const QueryPoolVK*>(pool);
            return q->type == VK_QUERY_TYPE_PIPELINE_STATISTICS ? sizeof(VriPipelineStatistics) : sizeof(uint64_t);
        }

        void VRI_CALL CmdResetQueries(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t offset, uint32_t num)
        {
            vkCmdResetQueryPool(Cmd(cmd)->cmd, QP(pool)->pool, offset, num);
        }

        void VRI_CALL CmdWriteTimestamp(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index)
        {
            // BOTTOM_OF_PIPE: the timestamp lands once all prior work has finished reaching here.
            vkCmdWriteTimestamp(Cmd(cmd)->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, QP(pool)->pool, index);
        }

        void VRI_CALL CmdBeginQuery(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index)
        {
            vkCmdBeginQuery(Cmd(cmd)->cmd, QP(pool)->pool, index, 0); // non-precise: >0 if any sample passed
        }

        void VRI_CALL CmdEndQuery(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index)
        {
            vkCmdEndQuery(Cmd(cmd)->cmd, QP(pool)->pool, index);
        }

        void VRI_CALL CmdCopyQueries(VriCommandBuffer* cmd,
                                     VriQueryPool*     pool,
                                     uint32_t          offset,
                                     uint32_t          num,
                                     VriBuffer*        dstBuffer,
                                     uint64_t          dstOffset)
        {
            // WAIT_BIT so the copy blocks on the queries' availability; 64_BIT for uint64 results.
            const VkDeviceSize stride =
                QP(pool)->type == VK_QUERY_TYPE_PIPELINE_STATISTICS ? sizeof(VriPipelineStatistics) : sizeof(uint64_t);
            vkCmdCopyQueryPoolResults(Cmd(cmd)->cmd,
                                      QP(pool)->pool,
                                      offset,
                                      num,
                                      Buf(dstBuffer)->buffer,
                                      dstOffset,
                                      stride,
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        }

        VriResult VRI_CALL GetCalibratedTimestamps(VriDevice* device, VriCalibratedTimestamps* out)
        {
            if (!out)
                return VriResult_InvalidArgument;
            DeviceVK* d = Dev(device);
            if (!d->Ext().GetCalibratedTimestamps || d->Desc().hasCalibratedTimestamps == VRI_FALSE)
                return VriResult_Unsupported;
            VkCalibratedTimestampInfoKHR infos[2] = {};
            infos[0].sType                        = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR;
            infos[0].timeDomain                   = VK_TIME_DOMAIN_DEVICE_KHR;
            infos[1].sType                        = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR;
            infos[1].timeDomain                   = d->HostTimeDomain();
            uint64_t ts[2]                        = {0, 0};
            uint64_t maxDeviation                 = 0;
            if (d->Ext().GetCalibratedTimestamps(d->Device(), 2, infos, ts, &maxDeviation) != VK_SUCCESS)
                return VriResult_Failure;
            out->gpuTimestamp = ts[0];
            out->cpuTimestamp = ts[1];
            return VriResult_Success;
        }

        const VriQueryInterface g_queryVK = {
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

    const VriQueryInterface* GetQueryInterfaceVK() { return &g_queryVK; }
} // namespace vri::vk
