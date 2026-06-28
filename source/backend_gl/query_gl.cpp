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
        // The 11 pipeline-statistics query targets (ARB_pipeline_statistics_query, core 4.6) in
        // VriPipelineStatistics field order. A pipeline-stats slot uses one GL query object per stat.
        constexpr int kStatNum               = 11;
        const GLenum  kStatTargets[kStatNum] = {
            GL_VERTICES_SUBMITTED,
            GL_PRIMITIVES_SUBMITTED,
            GL_VERTEX_SHADER_INVOCATIONS,
            GL_GEOMETRY_SHADER_INVOCATIONS,
            GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED,
            GL_CLIPPING_INPUT_PRIMITIVES,
            GL_CLIPPING_OUTPUT_PRIMITIVES,
            GL_FRAGMENT_SHADER_INVOCATIONS,
            GL_TESS_CONTROL_SHADER_PATCHES,
            GL_TESS_EVALUATION_SHADER_INVOCATIONS,
            GL_COMPUTE_SHADER_INVOCATIONS,
        };

        VriResult VRI_CALL CreateQueryPool(VriDevice* device, const VriQueryPoolDesc* desc, VriQueryPool** out)
        {
            if (!desc || !out || desc->queryCount == 0)
                return VriResult_InvalidArgument;
            DeviceGL*  d     = Dev(device);
            const bool stats = desc->type == VriQueryType_PipelineStatistics;
            if (stats ? d->Desc().hasPipelineStatistics == VRI_FALSE : d->Desc().hasTimestampQueries == VRI_FALSE)
                return VriResult_Unsupported;
            QueryPoolGL* q   = new QueryPoolGL {};
            q->device        = d;
            q->type          = desc->type;
            const uint32_t n = desc->queryCount * (stats ? kStatNum : 1u); // 11 GL queries per stats slot
            q->ids.resize(n);
            glGenQueries(static_cast<GLsizei>(n), q->ids.data());
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

        uint32_t VRI_CALL GetQuerySize(const VriQueryPool* pool)
        {
            return reinterpret_cast<const QueryPoolGL*>(pool)->type == VriQueryType_PipelineStatistics ?
                       sizeof(VriPipelineStatistics) :
                       sizeof(uint64_t);
        }

        // GL timestamp queries need no reset: glQueryCounter just (re)writes the slot.
        void VRI_CALL CmdResetQueries(VriCommandBuffer*, VriQueryPool*, uint32_t, uint32_t) {}

        void VRI_CALL CmdWriteTimestamp(VriCommandBuffer*, VriQueryPool* pool, uint32_t index)
        {
            glQueryCounter(QP(pool)->ids[index], GL_TIMESTAMP);
        }

        void VRI_CALL CmdBeginQuery(VriCommandBuffer*, VriQueryPool* pool, uint32_t index)
        {
            QueryPoolGL* q = QP(pool);
            if (q->type == VriQueryType_PipelineStatistics)
                for (int k = 0; k < kStatNum; ++k)
                    glBeginQuery(kStatTargets[k], q->ids[index * kStatNum + k]);
            else
                glBeginQuery(GL_SAMPLES_PASSED, q->ids[index]); // occlusion: exact sample count
        }

        void VRI_CALL CmdEndQuery(VriCommandBuffer*, VriQueryPool* pool, uint32_t)
        {
            if (QP(pool)->type == VriQueryType_PipelineStatistics)
                for (int k = 0; k < kStatNum; ++k)
                    glEndQuery(kStatTargets[k]);
            else
                glEndQuery(GL_SAMPLES_PASSED);
        }

        void VRI_CALL CmdCopyQueries(VriCommandBuffer*,
                                     VriQueryPool* pool,
                                     uint32_t      offset,
                                     uint32_t      num,
                                     VriBuffer*    dstBuffer,
                                     uint64_t      dstOffset)
        {
            QueryPoolGL*   q       = QP(pool);
            BufferGL*      b       = Buf(dstBuffer);
            const bool     stats   = q->type == VriQueryType_PipelineStatistics;
            const uint32_t perSlot = stats ? kStatNum : 1u;
            const uint64_t stride  = stats ? sizeof(VriPipelineStatistics) : sizeof(uint64_t);
            // Query buffer object: with the buffer bound to GL_QUERY_BUFFER, the result pointer is a
            // byte offset and the GPU writes each result into the buffer (waits for availability).
            glBindBuffer(GL_QUERY_BUFFER, b->id);
            for (uint32_t i = 0; i < num; ++i)
                for (uint32_t k = 0; k < perSlot; ++k)
                {
                    const uintptr_t byteOffset =
                        static_cast<uintptr_t>(dstOffset) + i * stride + static_cast<uint64_t>(k) * sizeof(uint64_t);
                    glGetQueryObjectui64v(
                        q->ids[(offset + i) * perSlot + k], GL_QUERY_RESULT, reinterpret_cast<GLuint64*>(byteOffset));
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

        // OpenGL has no calibrated GPU+CPU timestamp pairing.
        VriResult VRI_CALL GetCalibratedTimestamps(VriDevice*, VriCalibratedTimestamps*)
        {
            return VriResult_Unsupported;
        }

        const VriQueryInterface g_queryGL = {
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

    const VriQueryInterface* GetQueryInterfaceGL() { return &g_queryGL; }
} // namespace vri::gl
