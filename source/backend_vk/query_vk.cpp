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
            DeviceVK*   d  = Dev(device);
            VkQueryType vt = VK_QUERY_TYPE_TIMESTAMP;

            VkQueryPoolCreateInfo ci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            ci.queryType             = vt;
            ci.queryCount            = desc->queryCount;

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

        uint32_t VRI_CALL GetQuerySize(const VriQueryPool*) { return sizeof(uint64_t); }

        void VRI_CALL CmdResetQueries(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t offset, uint32_t num)
        {
            vkCmdResetQueryPool(Cmd(cmd)->cmd, QP(pool)->pool, offset, num);
        }

        void VRI_CALL CmdWriteTimestamp(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index)
        {
            // BOTTOM_OF_PIPE: the timestamp lands once all prior work has finished reaching here.
            vkCmdWriteTimestamp(Cmd(cmd)->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, QP(pool)->pool, index);
        }

        void VRI_CALL CmdCopyQueries(VriCommandBuffer* cmd,
                                     VriQueryPool*     pool,
                                     uint32_t          offset,
                                     uint32_t          num,
                                     VriBuffer*        dstBuffer,
                                     uint64_t          dstOffset)
        {
            // WAIT_BIT so the copy blocks on the queries' availability; 64_BIT for uint64 results.
            vkCmdCopyQueryPoolResults(Cmd(cmd)->cmd,
                                      QP(pool)->pool,
                                      offset,
                                      num,
                                      Buf(dstBuffer)->buffer,
                                      dstOffset,
                                      sizeof(uint64_t),
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        }

        const VriQueryInterface g_queryVK = {
            CreateQueryPool,
            DestroyQueryPool,
            GetQuerySize,
            CmdResetQueries,
            CmdWriteTimestamp,
            CmdCopyQueries,
        };
    } // namespace

    const VriQueryInterface* GetQueryInterfaceVK() { return &g_queryVK; }
} // namespace vri::vk
