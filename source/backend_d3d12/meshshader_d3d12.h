// meshshader_d3d12.h - Direct3D 12 mesh/task shader interface (DispatchMesh).
#pragma once

#include <vri/vri.h>

namespace vri::d3d12
{
    // Returns the mesh-shader function table. Registered by the device only when
    // VriFeature_MeshShader was granted (Mesh Shader Tier 1+). Mesh/task pipelines
    // are created via the core CreateGraphicsPipeline (mesh/task stages).
    const VriMeshShaderInterface* GetMeshShaderInterfaceD3D12();
} // namespace vri::d3d12
