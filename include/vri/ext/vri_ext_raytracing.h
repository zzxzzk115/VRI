/*
 * vri_ext_raytracing.h - ray tracing extension (SKELETON, Phase 7).
 *
 * Queried via vriGetInterface(device, VRI_INTERFACE_RAYTRACING, ...). Registered
 * only by backends that support it (Vulkan/D3D12); GL/GLES/D3D11 never register
 * it, so the table is simply unavailable there. The full desc set and method
 * bodies land in Phase 7; this fixes the interface shape and key handle.
 */
#ifndef VRI_EXT_RAYTRACING_H
#define VRI_EXT_RAYTRACING_H

#include "../vri_base.h"
#include "../vri_handles.h"
#include "../vri_command.h"

VRI_EXTERN_C_BEGIN

typedef struct VriAccelerationStructure VriAccelerationStructure;

typedef enum VriAccelerationStructureType
{
    VriAccelerationStructureType_BottomLevel = 0,
    VriAccelerationStructureType_TopLevel,
    VriAccelerationStructureType_MaxEnum = 0x7fffffff
} VriAccelerationStructureType;

typedef struct VriRayTracingInterface
{
    /* Skeleton: representative entry points; descriptors completed in Phase 7. */
    VriResult (VRI_CALL *CreateAccelerationStructure)(VriDevice* device, VriAccelerationStructureType type, uint64_t size, VriAccelerationStructure** outAS);
    void      (VRI_CALL *DestroyAccelerationStructure)(VriAccelerationStructure* as);
    uint64_t  (VRI_CALL *GetAccelerationStructureDeviceAddress)(const VriAccelerationStructure* as);

    VriResult (VRI_CALL *CreateRayTracingPipeline)(VriDevice* device, const void* rayTracingPipelineDesc, VriPipeline** outPipeline);
    void      (VRI_CALL *CmdBuildAccelerationStructure)(VriCommandBuffer* cmd, const void* buildDesc);
    void      (VRI_CALL *CmdTraceRays)(VriCommandBuffer* cmd, const void* dispatchRaysDesc);
} VriRayTracingInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_RAYTRACING_H */
