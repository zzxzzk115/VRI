// meshshader_mtl.h - Metal mesh/task shader draws (drawMeshThreadgroups).
//
// The object/mesh/fragment pipeline is built by the core CreateGraphicsPipeline
// (it detects the mesh/task stages and uses MTLMeshRenderPipelineDescriptor).
// This interface only adds the dispatch-style draws.
#pragma once

#include <vri/vri.h>

namespace vri::mtl
{
    const VriMeshShaderInterface* GetMeshShaderInterfaceMTL();
} // namespace vri::mtl
