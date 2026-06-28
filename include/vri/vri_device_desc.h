/*
 * vri_device_desc.h - VriDeviceDesc: the scalar/bitfield capability struct
 * returned by VriCoreInterface::GetDeviceDesc. Mirrors NRI's DeviceDesc.
 *
 * Phase-0 stub: a representative subset of capabilities. Tiered/optional
 * features that add entry points are exposed as separate queryable interfaces
 * (see include/vri/ext/), not here.
 */
#ifndef VRI_DEVICE_DESC_H
#define VRI_DEVICE_DESC_H

#include "vri_base.h"
#include "vri_enums.h"

VRI_EXTERN_C_BEGIN

/* Live video-memory budget/usage for a memory location (VriCoreInterface::GetVideoMemoryInfo).
 * `budget` is how much the OS currently lets this process use; `usage` is how much it is using.
 * Both in bytes; useful for streaming/eviction decisions. */
typedef struct VriVideoMemoryInfo
{
    uint64_t budget;
    uint64_t usage;
} VriVideoMemoryInfo;

typedef struct VriAdapterDesc
{
    char           name[256];
    uint64_t       videoMemorySize;
    uint64_t       sharedMemorySize;
    uint32_t       deviceId;
    uint32_t       vendorId;
    VriAdapterType type;
} VriAdapterDesc;

typedef struct VriDeviceDesc
{
    VriAdapterDesc adapter;
    VriGraphicsAPI graphicsAPI;
    uint32_t       apiVersionMajor;
    uint32_t       apiVersionMinor;

    /* viewport / attachment limits */
    uint32_t       viewportMaxNum;
    uint32_t       attachmentColorMaxNum;
    uint32_t       attachmentMaxDim;

    /* resource limits */
    uint32_t       texture1DMaxDim;
    uint32_t       texture2DMaxDim;
    uint32_t       texture3DMaxDim;
    uint32_t       textureArrayLayerMaxNum;
    uint64_t       bufferMaxSize;

    /* alignment requirements (kept Vulkan-compatible across backends) */
    uint32_t       uploadBufferTextureRowAlignment;
    uint32_t       constantBufferOffsetAlignment;
    uint32_t       storageBufferOffsetAlignment;

    /* always-present-but-varying feature tiers / flags */
    uint32_t       queueCount[VriQueueType_Count];
    VriBool        hasTessellation;
    VriBool        hasGeometryShader;
    VriBool        hasComputeShader;

    /* optional features actually granted at device creation (subset of the
       requested VriFeatureBits; see VriDeviceCreationDesc::enabledFeatures). */
    uint64_t       enabledFeatures;
    VriBool        hasRayTracing;
    VriBool        hasMeshShader;
    VriBool        hasBindless;
    VriBool        hasVariableShadingRate;
    VriBool        hasOpacityMicromap;
    VriBool        hasRayQuery; /* inline ray tracing in any shader stage */
    VriBool        hasConservativeRaster;
    VriBool        hasFragmentShaderBarycentric;
    VriBool        hasCustomBorderColor;
    VriBool        hasShaderWaveOps; /* subgroup/wave intrinsics (WaveActiveSum, ballots, ...) */
    VriBool        hasExternalMemory; /* external memory/semaphore export (see ext/vri_ext_external.h) */
    VriBool        hasTimestampQueries;        /* GPU timestamp query pools (see ext/vri_ext_query.h) */
    VriBool        hasPipelineStatistics;      /* pipeline-statistics query pools (see ext/vri_ext_query.h) */
    VriBool        hasCalibratedTimestamps;    /* correlated GPU+CPU timestamps (see ext/vri_ext_query.h) */
    VriBool        hasDrawIndirectCount;       /* CmdDraw[Indexed]IndirectCount (GPU-driven draw count) */
    VriBool        hasClearStorageBuffer;      /* CmdClearStorageBuffer (fill a storage buffer with a uint32) */
    float          timestampPeriodNanoseconds; /* nanoseconds per timestamp tick (0 if unsupported) */
    uint32_t       subgroupSize;     /* lanes per subgroup/wave (0 if unknown) */

    /* ray-tracing shader-binding-table layout (valid when hasRayTracing) */
    uint32_t       rtShaderGroupHandleSize;
    uint32_t       rtShaderGroupBaseAlignment;
    uint32_t       rtShaderGroupHandleAlignment;
} VriDeviceDesc;

VRI_EXTERN_C_END

#endif /* VRI_DEVICE_DESC_H */
