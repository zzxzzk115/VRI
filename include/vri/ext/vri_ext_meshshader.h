/*
 * vri_ext_meshshader.h - mesh/task shader extension (SKELETON, Phase 7).
 *
 * Queried via vriGetInterface(device, VRI_INTERFACE_MESHSHADER, ...). Registered
 * only by backends that support mesh shaders. Pipelines using mesh/task stages
 * are created through the core CreateGraphicsPipeline (the stages exist in
 * VriShaderStageBits); this interface adds only the dispatch-style draw.
 */
#ifndef VRI_EXT_MESHSHADER_H
#define VRI_EXT_MESHSHADER_H

#include "../vri_base.h"
#include "../vri_handles.h"

VRI_EXTERN_C_BEGIN

typedef struct VriMeshShaderInterface
{
    void (VRI_CALL *CmdDrawMeshTasks)(VriCommandBuffer* cmd, uint32_t x, uint32_t y, uint32_t z);
    void (VRI_CALL *CmdDrawMeshTasksIndirect)(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint32_t drawNum, uint32_t stride);
} VriMeshShaderInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_MESHSHADER_H */
