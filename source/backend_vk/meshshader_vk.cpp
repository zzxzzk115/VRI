// meshshader_vk.cpp - mesh/task shader draws via VK_EXT_mesh_shader.
//
// The pipeline (task?/mesh/fragment stages) is built by the core
// CreateGraphicsPipeline - Vulkan ignores vertex-input/input-assembly state when a
// mesh shader is present. This interface only adds the dispatch-style draws.
#include "meshshader_vk.h"

#include "device_vk.h"
#include "objects_vk.h"

namespace vri::vk
{
    namespace
    {
        CommandBufferVK* CB(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferVK*>(h); }
        BufferVK*        BUF(VriBuffer* h)       { return reinterpret_cast<BufferVK*>(h); }

        void VRI_CALL CmdDrawMeshTasks(VriCommandBuffer* cmd, uint32_t x, uint32_t y, uint32_t z)
        {
            CommandBufferVK* c = CB(cmd);
            if (c->device->Ext().CmdDrawMeshTasks)
                c->device->Ext().CmdDrawMeshTasks(c->cmd, x, y, z);
        }

        void VRI_CALL CmdDrawMeshTasksIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint32_t drawNum, uint32_t stride)
        {
            CommandBufferVK* c = CB(cmd);
            if (c->device->Ext().CmdDrawMeshTasksIndirect)
                c->device->Ext().CmdDrawMeshTasksIndirect(c->cmd, BUF(buffer)->buffer, offset, drawNum, stride);
        }

        const VriMeshShaderInterface g_meshVK = {
            CmdDrawMeshTasks,
            CmdDrawMeshTasksIndirect,
        };
    } // namespace

    const VriMeshShaderInterface* GetMeshShaderInterfaceVK() { return &g_meshVK; }
} // namespace vri::vk
