// meshshader_vk.h - Vulkan mesh/task shader interface (VK_EXT_mesh_shader).
#pragma once

#include <vri/vri.h>

namespace vri::vk
{
    // Returns the mesh-shader function table. Registered by the device only when
    // VriFeature_MeshShader was granted at creation. Mesh/task pipelines are
    // created via the core CreateGraphicsPipeline (mesh/task stages); this table
    // adds the dispatch-style draws.
    const VriMeshShaderInterface* GetMeshShaderInterfaceVK();
} // namespace vri::vk
