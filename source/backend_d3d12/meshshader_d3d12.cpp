// meshshader_d3d12.cpp - mesh/task draws via ID3D12GraphicsCommandList6::DispatchMesh.
//
// The pipeline (MS [+ AS] + PS) is built by the core CreateGraphicsPipeline mesh path
// (pipeline-state stream). This interface adds the dispatch-style draws.
#include "meshshader_d3d12.h"

#include "device_d3d12.h"
#include "objects_d3d12.h"

namespace vri::d3d12
{
    namespace
    {
        CommandBufferD3D12* CB(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferD3D12*>(h); }
        BufferD3D12*        BUF(VriBuffer* h)       { return reinterpret_cast<BufferD3D12*>(h); }

        void VRI_CALL CmdDrawMeshTasks(VriCommandBuffer* cmd, uint32_t x, uint32_t y, uint32_t z)
        {
            CommandBufferD3D12* c = CB(cmd);
            Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList6> list6;
            if (SUCCEEDED(c->list.As(&list6)))
                list6->DispatchMesh(x ? x : 1, y ? y : 1, z ? z : 1);
        }

        void VRI_CALL CmdDrawMeshTasksIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint32_t drawNum, uint32_t /*stride*/)
        {
            CommandBufferD3D12* c = CB(cmd);
            // The command signature is fixed at the natural 12-byte (x,y,z) stride; the
            // arg buffer must be tightly packed (matches the cross-backend SBT/indirect use).
            ID3D12CommandSignature* sig = c->device->DispatchMeshSignature();
            if (sig)
                c->list->ExecuteIndirect(sig, drawNum, BUF(buffer)->resource.Get(), offset, nullptr, 0);
        }

        const VriMeshShaderInterface g_meshD3D12 = {
            CmdDrawMeshTasks,
            CmdDrawMeshTasksIndirect,
        };
    } // namespace

    const VriMeshShaderInterface* GetMeshShaderInterfaceD3D12() { return &g_meshD3D12; }
} // namespace vri::d3d12
