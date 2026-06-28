// query_gl.cpp - VriQueryInterface for OpenGL. Desktop GL has native timer queries
// (glQueryCounter(GL_TIMESTAMP), core 3.3 / ARB_timer_query; values are nanoseconds).
// GLES/WebGL have no core timer query, so the interface reports Unsupported there.
//
// CmdCopyQueries resolves GPU-side into the destination buffer via a query buffer object
// (ARB_query_buffer_object, core GL 4.4): with the buffer bound to GL_QUERY_BUFFER,
// glGetQueryObjectui64v's last argument is a byte offset and the result lands in the buffer on
// the GPU timeline (like vkCmdCopyQueryPoolResults) -- no CPU stall, and no client-side
// glBufferSubData (which the immutable read-back buffer rejects). Gated on GL 4.4 in the device.

#include "query_gl.h"
#include "device_gl.h"
#include "objects_gl.h"

namespace vri::gl
{
    namespace
    {
        inline DeviceGL*    Dev(VriDevice* h) { return reinterpret_cast<DeviceGL*>(h); }
        inline BufferGL*    Buf(VriBuffer* h) { return reinterpret_cast<BufferGL*>(h); }
        inline QueryPoolGL* QP(VriQueryPool* h) { return reinterpret_cast<QueryPoolGL*>(h); }

#if !defined(VRI_GL_ES_HEADERS)
        VriResult VRI_CALL CreateQueryPool(VriDevice* device, const VriQueryPoolDesc* desc, VriQueryPool** out)
        {
            if (!desc || !out || desc->queryCount == 0)
                return VriResult_InvalidArgument;
            DeviceGL* d = Dev(device);
            if (d->Desc().hasTimestampQueries == VRI_FALSE)
                return VriResult_Unsupported;
            // Pipeline statistics on GL need one query object per stat (separate targets) -- a
            // follow-up; timestamps + occlusion are single-object queries.
            if (desc->type == VriQueryType_PipelineStatistics)
                return VriResult_Unsupported;
            QueryPoolGL* q = new QueryPoolGL {};
            q->device      = d;
            q->ids.resize(desc->queryCount);
            glGenQueries(static_cast<GLsizei>(desc->queryCount), q->ids.data());
            *out = ToHandle(q);
            return VriResult_Success;
        }

        void VRI_CALL DestroyQueryPool(VriQueryPool* pool)
        {
            if (!pool)
                return;
            QueryPoolGL* q = QP(pool);
            glDeleteQueries(static_cast<GLsizei>(q->ids.size()), q->ids.data());
            delete q;
        }

        uint32_t VRI_CALL GetQuerySize(const VriQueryPool*) { return sizeof(uint64_t); }

        // GL timestamp queries need no reset: glQueryCounter just (re)writes the slot.
        void VRI_CALL CmdResetQueries(VriCommandBuffer*, VriQueryPool*, uint32_t, uint32_t) {}

        void VRI_CALL CmdWriteTimestamp(VriCommandBuffer*, VriQueryPool* pool, uint32_t index)
        {
            glQueryCounter(QP(pool)->ids[index], GL_TIMESTAMP);
        }

        void VRI_CALL CmdBeginQuery(VriCommandBuffer*, VriQueryPool* pool, uint32_t index)
        {
            glBeginQuery(GL_SAMPLES_PASSED, QP(pool)->ids[index]); // occlusion: exact sample count
        }

        void VRI_CALL CmdEndQuery(VriCommandBuffer*, VriQueryPool*, uint32_t) { glEndQuery(GL_SAMPLES_PASSED); }

        void VRI_CALL CmdCopyQueries(VriCommandBuffer*,
                                     VriQueryPool* pool,
                                     uint32_t      offset,
                                     uint32_t      num,
                                     VriBuffer*    dstBuffer,
                                     uint64_t      dstOffset)
        {
            QueryPoolGL* q = QP(pool);
            BufferGL*    b = Buf(dstBuffer);
            // Query buffer object: with the buffer bound to GL_QUERY_BUFFER, the result pointer is
            // a byte offset and the GPU writes each result into the buffer (waits for availability).
            glBindBuffer(GL_QUERY_BUFFER, b->id);
            for (uint32_t i = 0; i < num; ++i)
            {
                const uintptr_t byteOffset = static_cast<uintptr_t>(dstOffset) + i * sizeof(uint64_t);
                glGetQueryObjectui64v(q->ids[offset + i], GL_QUERY_RESULT, reinterpret_cast<GLuint64*>(byteOffset));
            }
            glBindBuffer(GL_QUERY_BUFFER, 0);
        }
#else // GLES / WebGL: no core timer query.
        VriResult VRI_CALL CreateQueryPool(VriDevice*, const VriQueryPoolDesc*, VriQueryPool**)
        {
            return VriResult_Unsupported;
        }
        void VRI_CALL     DestroyQueryPool(VriQueryPool*) {}
        uint32_t VRI_CALL GetQuerySize(const VriQueryPool*) { return sizeof(uint64_t); }
        void VRI_CALL     CmdResetQueries(VriCommandBuffer*, VriQueryPool*, uint32_t, uint32_t) {}
        void VRI_CALL     CmdWriteTimestamp(VriCommandBuffer*, VriQueryPool*, uint32_t) {}
        void VRI_CALL     CmdBeginQuery(VriCommandBuffer*, VriQueryPool*, uint32_t) {}
        void VRI_CALL     CmdEndQuery(VriCommandBuffer*, VriQueryPool*, uint32_t) {}
        void VRI_CALL     CmdCopyQueries(VriCommandBuffer*, VriQueryPool*, uint32_t, uint32_t, VriBuffer*, uint64_t) {}
#endif

        const VriQueryInterface g_queryGL = {
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

    const VriQueryInterface* GetQueryInterfaceGL() { return &g_queryGL; }
} // namespace vri::gl
