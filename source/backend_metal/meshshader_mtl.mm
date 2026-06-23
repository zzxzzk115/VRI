// meshshader_mtl.mm - mesh/task shader draws via drawMeshThreadgroups.
//
// (x,y,z) are threadgroup counts (the "mesh tasks"); the per-stage threads-per-
// threadgroup come from the bound pipeline (reflected from the SPIR-V local size
// in CreateGraphicsPipeline). For a mesh-only pipeline the object threadgroup is
// (1,1,1).

#include "meshshader_mtl.h"
#include "device_mtl.h"
#include "objects_mtl.h"

#import <Metal/Metal.h>

namespace vri::mtl
{
    namespace
    {
        inline CommandBufferMTL* CB(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferMTL*>(h); }
        inline BufferMTL*        Buf(VriBuffer* h)       { return reinterpret_cast<BufferMTL*>(h); }

        void VRI_CALL CmdDrawMeshTasks(VriCommandBuffer* cmd, uint32_t x, uint32_t y, uint32_t z)
        {
            CommandBufferMTL* c = CB(cmd);
            if (!c->renderEnc || !c->boundPipeline || !c->boundPipeline->isMesh) return;
            [c->renderEnc drawMeshThreadgroups:MTLSizeMake(x ? x : 1, y ? y : 1, z ? z : 1)
                       threadsPerObjectThreadgroup:c->boundPipeline->objectTG
                         threadsPerMeshThreadgroup:c->boundPipeline->meshTG];
        }

        void VRI_CALL CmdDrawMeshTasksIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint32_t drawNum, uint32_t stride)
        {
            CommandBufferMTL* c = CB(cmd);
            if (!c->renderEnc || !c->boundPipeline || !c->boundPipeline->isMesh) return;
            for (uint32_t i = 0; i < drawNum; ++i)
                [c->renderEnc drawMeshThreadgroupsWithIndirectBuffer:Buf(buffer)->buffer
                                               indirectBufferOffset:offset + (uint64_t)i * stride
                                        threadsPerObjectThreadgroup:c->boundPipeline->objectTG
                                          threadsPerMeshThreadgroup:c->boundPipeline->meshTG];
        }

        const VriMeshShaderInterface g_meshMTL = {
            CmdDrawMeshTasks,
            CmdDrawMeshTasksIndirect,
        };
    } // namespace

    const VriMeshShaderInterface* GetMeshShaderInterfaceMTL() { return &g_meshMTL; }
} // namespace vri::mtl
