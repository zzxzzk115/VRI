/*
 * vri_ext_raytracing.h - ray tracing extension.
 *
 * Queried via vriGetInterface(device, VRI_INTERFACE_RAYTRACING, ...). Registered
 * only by backends that support it (Vulkan KHR ray tracing pipeline + acceleration
 * structure); GL/GLES/D3D11 never register it, so the table is simply unavailable
 * there. Enable with VriFeature_RayTracing at device creation and check
 * VriDeviceDesc::hasRayTracing.
 *
 * Model (Vulkan/D3D12-style, explicit): build bottom-level acceleration structures
 * (BLAS) from geometry and a top-level (TLAS) from instances; create a ray-tracing
 * pipeline of shader stages grouped into raygen/miss/hit groups; build a shader
 * binding table (SBT) from the group handles; CmdTraceRays.
 */
#ifndef VRI_EXT_RAYTRACING_H
#define VRI_EXT_RAYTRACING_H

#include "../vri_base.h"
#include "../vri_handles.h"
#include "../vri_enums.h"
#include "../vri_flags.h"
#include "../vri_format.h"
#include "../vri_pipeline.h"
#include "../vri_command.h"

VRI_EXTERN_C_BEGIN

typedef struct VriAccelerationStructure VriAccelerationStructure;
typedef struct VriMicromap VriMicromap; /* opacity micromap (see ext/vri_ext_omm.h) */

#define VRI_SHADER_UNUSED 0xFFFFFFFFu

typedef enum VriAccelerationStructureType
{
    VriAccelerationStructureType_BottomLevel = 0,
    VriAccelerationStructureType_TopLevel,
    VriAccelerationStructureType_MaxEnum = 0x7fffffff
} VriAccelerationStructureType;

typedef uint32_t VriAccelerationStructureBuildFlags;
typedef enum VriAccelerationStructureBuildBits
{
    VriAccelerationStructureBuild_None            = 0,
    VriAccelerationStructureBuild_AllowUpdate     = 1u << 0,
    VriAccelerationStructureBuild_AllowCompaction = 1u << 1,
    VriAccelerationStructureBuild_PreferFastTrace = 1u << 2,
    VriAccelerationStructureBuild_PreferFastBuild = 1u << 3,
    VriAccelerationStructureBuild_LowMemory       = 1u << 4,
    VriAccelerationStructureBuild_MaxEnum         = 0x7fffffff
} VriAccelerationStructureBuildBits;

typedef enum VriAsGeometryType
{
    VriAsGeometryType_Triangles = 0,
    VriAsGeometryType_Instances,        /* TLAS: a buffer of VkAccelerationStructureInstance-layout records */
    VriAsGeometryType_MaxEnum = 0x7fffffff
} VriAsGeometryType;

typedef uint32_t VriAsGeometryFlags;
typedef enum VriAsGeometryBits
{
    VriAsGeometry_None              = 0,
    VriAsGeometry_Opaque            = 1u << 0,
    VriAsGeometry_NoDuplicateAnyHit = 1u << 1,
    VriAsGeometry_MaxEnum           = 0x7fffffff
} VriAsGeometryBits;

typedef struct VriAsTrianglesDesc
{
    VriBuffer*   vertexBuffer;
    uint64_t     vertexOffset;
    uint32_t     vertexCount;
    uint32_t     vertexStride;
    VriFormat    vertexFormat;   /* e.g. VriFormat_RGB32_SFLOAT */
    VriBuffer*   indexBuffer;    /* optional (NULL => non-indexed) */
    uint64_t     indexOffset;
    uint32_t     indexCount;
    VriIndexType indexType;
    VriBuffer*   transformBuffer; /* optional 3x4 row-major float transform */
    uint64_t     transformOffset;
    /* optional opacity micromap attachment (requires VriFeature_OpacityMicromap) */
    VriMicromap* micromap;         /* NULL => no OMM */
    VriBuffer*   ommIndexBuffer;   /* per-triangle index into the micromap */
    uint64_t     ommIndexOffset;
    VriIndexType ommIndexType;
} VriAsTrianglesDesc;

typedef struct VriAsInstancesDesc
{
    VriBuffer* instanceBuffer; /* array of VkAccelerationStructureInstanceKHR-layout records */
    uint64_t   offset;
    uint32_t   instanceCount;
} VriAsInstancesDesc;

typedef struct VriAsGeometryDesc
{
    VriAsGeometryType  type;
    VriAsGeometryFlags flags;
    VriAsTrianglesDesc triangles; /* used when type == Triangles */
    VriAsInstancesDesc instances; /* used when type == Instances */
} VriAsGeometryDesc;

typedef struct VriAccelerationStructureDesc
{
    VriAccelerationStructureType       type;
    VriAccelerationStructureBuildFlags flags;
    uint32_t                           geometryCount;
    const VriAsGeometryDesc*           geometries;
} VriAccelerationStructureDesc;

typedef struct VriBuildAccelerationStructureDesc
{
    VriAccelerationStructure*           dst;      /* the AS to build into (from CreateAccelerationStructure) */
    const VriAccelerationStructureDesc* geometry; /* geometries + buffers + primitive counts */
} VriBuildAccelerationStructureDesc;

/* ---- ray-tracing pipeline ---- */

typedef enum VriShaderGroupType
{
    VriShaderGroupType_General = 0,       /* raygen / miss / callable: generalShader set */
    VriShaderGroupType_TrianglesHitGroup,/* closestHit (+ optional anyHit) */
    VriShaderGroupType_ProceduralHitGroup,
    VriShaderGroupType_MaxEnum = 0x7fffffff
} VriShaderGroupType;

typedef struct VriShaderGroupDesc
{
    VriShaderGroupType type;
    uint32_t           generalShader;      /* index into shaders[] or VRI_SHADER_UNUSED */
    uint32_t           closestHitShader;
    uint32_t           anyHitShader;
    uint32_t           intersectionShader;
} VriShaderGroupDesc;

typedef struct VriRayTracingPipelineDesc
{
    VriPipelineLayout*        pipelineLayout;
    const VriShaderDesc*      shaders;     /* raygen/miss/hit/... stages */
    uint32_t                  shaderNum;
    const VriShaderGroupDesc* groups;
    uint32_t                  groupNum;
    uint32_t                  maxRecursionDepth;
} VriRayTracingPipelineDesc;

/* A region of the shader binding table for one shader category. */
typedef struct VriStridedBufferRegion
{
    VriBuffer* buffer;
    uint64_t   offset;
    uint64_t   stride;
    uint64_t   size;
} VriStridedBufferRegion;

typedef struct VriDispatchRaysDesc
{
    VriStridedBufferRegion raygen;
    VriStridedBufferRegion miss;
    VriStridedBufferRegion hit;
    VriStridedBufferRegion callable;
    uint32_t               width;
    uint32_t               height;
    uint32_t               depth;
} VriDispatchRaysDesc;

typedef struct VriRayTracingInterface
{
    /* Create an AS sized for the given geometry (counts/formats). Owns its backing
       store + scratch; build it later with CmdBuildAccelerationStructure. */
    VriResult (VRI_CALL *CreateAccelerationStructure)(VriDevice* device, const VriAccelerationStructureDesc* desc, VriAccelerationStructure** outAS);
    void      (VRI_CALL *DestroyAccelerationStructure)(VriAccelerationStructure* as);
    uint64_t  (VRI_CALL *GetAccelerationStructureDeviceAddress)(const VriAccelerationStructure* as);

    /* Create a descriptor for binding a TLAS (VriDescriptorType_AccelerationStructure).
       Destroy it with the core DestroyDescriptor, like any other VriDescriptor. */
    VriResult (VRI_CALL *CreateAccelerationStructureDescriptor)(VriDevice* device, VriAccelerationStructure* as, VriDescriptor** outDescriptor);

    VriResult (VRI_CALL *CreateRayTracingPipeline)(VriDevice* device, const VriRayTracingPipelineDesc* desc, VriPipeline** outPipeline);
    /* Copy the opaque shader-group handles for [firstGroup, firstGroup+groupNum)
       into `dst` (dstSize must be >= groupNum * shaderGroupHandleSize). The app
       writes these into the SBT buffer at shaderGroupBaseAlignment strides. */
    VriResult (VRI_CALL *GetShaderGroupHandles)(VriPipeline* pipeline, uint32_t firstGroup, uint32_t groupNum, size_t dstSize, void* dst);

    void (VRI_CALL *CmdBuildAccelerationStructure)(VriCommandBuffer* cmd, const VriBuildAccelerationStructureDesc* desc);
    void (VRI_CALL *CmdTraceRays)(VriCommandBuffer* cmd, const VriDispatchRaysDesc* desc);

    /* ---- compaction (shrink a built AS to its exact size) ---------------------
       Build the source AS with VriAccelerationStructureBuild_AllowCompaction, then:
         1. CmdWriteAccelerationStructureCompactedSize -> a uint64 size in a buffer
            (read it back after the submit's fence),
         2. CreateAccelerationStructureCompacted(size) for the destination,
         3. CmdCopyAccelerationStructure(dst, src, compact=VRI_TRUE),
         4. use dst (smaller) and DestroyAccelerationStructure(src). */

    /* Write src's compacted size (uint64) into dstBuffer at dstOffset. dstBuffer must be a
       device buffer usable as the size sink (StorageBuffer + TransferSrc); copy it to a
       HostReadback buffer to read the value. */
    void (VRI_CALL *CmdWriteAccelerationStructureCompactedSize)(VriCommandBuffer* cmd, VriAccelerationStructure* as,
                                                                VriBuffer* dstBuffer, uint64_t dstOffset);
    /* Create an AS with an explicit backing size and no geometry/scratch -- the destination of a
       compacting copy. */
    VriResult (VRI_CALL *CreateAccelerationStructureCompacted)(VriDevice* device, VriAccelerationStructureType type,
                                                               uint64_t size, VriAccelerationStructure** outAS);
    /* Copy src into dst; compact == VRI_TRUE shrinks (dst sized from the compacted-size query,
       src built with AllowCompaction). */
    void (VRI_CALL *CmdCopyAccelerationStructure)(VriCommandBuffer* cmd, VriAccelerationStructure* dst,
                                                  VriAccelerationStructure* src, VriBool compact);
} VriRayTracingInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_RAYTRACING_H */
