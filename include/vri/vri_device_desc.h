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
#include "vri_format.h"

VRI_EXTERN_C_BEGIN

/* Live video-memory budget/usage for a memory location (VriCoreInterface::GetVideoMemoryInfo).
 * `budget` is how much the OS currently lets this process use; `usage` is how much it is using.
 * Both in bytes; useful for streaming/eviction decisions. */
/* ---- object tracking -------------------------------------------------------
 * What a device currently owns, with the memory each object actually cost. The
 * driver's own budget query (GetVideoMemoryInfo) says how much is in use but
 * never what is using it, and a renderer that is near its VRAM budget needs the
 * second question answered far more than the first.
 *
 * Objects are tracked from creation to destruction; SetDebugName attaches the
 * label that makes a listing readable. */
typedef enum VriObjectType
{
    VriObjectType_Unknown = 0,
    VriObjectType_Buffer,
    VriObjectType_Texture,
    VriObjectType_AccelerationStructure,
    VriObjectType_Micromap,
    VriObjectType_MaxEnum = 0x7fffffff
} VriObjectType;

/* Names are copied into a fixed field rather than returned as pointers: a
 * snapshot has to stay valid after the lock is released, and a C caller has no
 * way to know when the backend would free a string. */
#define VRI_OBJECT_NAME_MAX 64

typedef struct VriObjectInfo
{
    const void*   handle; /* the VRI object pointer, as passed to SetDebugName */
    VriObjectType type;
    char          name[VRI_OBJECT_NAME_MAX]; /* "" until SetDebugName */
    /* Device memory this object actually holds, from the allocator rather than
     * recomputed from the desc, so alignment and tiling padding are included.
     * 0 for a wrapped/aliased object that owns no allocation of its own. */
    uint64_t memoryBytes;
    /* Textures: the created dimensions. Buffers: width = size, rest 0.
     * Acceleration structures: width = store bytes, height = scratch bytes. */
    uint32_t  width;
    uint32_t  height;
    uint32_t  depth;
    uint32_t  mipNum;
    uint32_t  layerNum;
    VriFormat format;
} VriObjectInfo;

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
    uint32_t viewportMaxNum;
    uint32_t attachmentColorMaxNum;
    uint32_t attachmentMaxDim;

    /* resource limits */
    uint32_t texture1DMaxDim;
    uint32_t texture2DMaxDim;
    uint32_t texture3DMaxDim;
    uint32_t textureArrayLayerMaxNum;
    uint64_t bufferMaxSize;

    /* alignment requirements (kept Vulkan-compatible across backends) */
    uint32_t uploadBufferTextureRowAlignment;
    uint32_t constantBufferOffsetAlignment;
    uint32_t storageBufferOffsetAlignment;

    /* always-present-but-varying feature tiers / flags */
    uint32_t queueCount[VriQueueType_Count];
    VriBool  hasTessellation;
    VriBool  hasGeometryShader;
    VriBool  hasComputeShader;

    /* optional features actually granted at device creation (subset of the
       requested VriFeatureBits; see VriDeviceCreationDesc::enabledFeatures). */
    uint64_t enabledFeatures;
    VriBool  hasRayTracing;
    VriBool  hasMeshShader;
    VriBool  hasBindless;
    VriBool  hasVariableShadingRate;
    VriBool  hasOpacityMicromap;
    VriBool  hasRayQuery; /* inline ray tracing in any shader stage */
    VriBool  hasConservativeRaster;
    VriBool  hasFragmentShaderBarycentric;
    VriBool  hasCustomBorderColor;
    VriBool  hasShaderWaveOps;           /* subgroup/wave intrinsics (WaveActiveSum, ballots, ...) */
    VriBool  hasExternalMemory;          /* external memory/semaphore export (see ext/vri_ext_external.h) */
    VriBool  hasTimestampQueries;        /* GPU timestamp query pools (see ext/vri_ext_query.h) */
    VriBool  hasPipelineStatistics;      /* pipeline-statistics query pools (see ext/vri_ext_query.h) */
    VriBool  hasCalibratedTimestamps;    /* correlated GPU+CPU timestamps (see ext/vri_ext_query.h) */
    VriBool  hasDrawIndirectCount;       /* CmdDraw[Indexed]IndirectCount (GPU-driven draw count) */
    VriBool  hasClearStorageBuffer;      /* CmdClearStorageBuffer (fill a storage buffer with a uint32) */
    VriBool  hasClearStorageTexture;     /* CmdClearStorageTexture (clear a storage texture to a color) */
    VriBool  hasMultiview;               /* single-pass layered rendering via viewMask (VR stereo) */
    uint32_t maxViewCount;               /* max simultaneous views in a viewMask (0 if no multiview) */
    float    timestampPeriodNanoseconds; /* nanoseconds per timestamp tick (0 if unsupported) */
    uint32_t subgroupSize;               /* lanes per subgroup/wave (0 if unknown) */

    /* ray-tracing shader-binding-table layout (valid when hasRayTracing) */
    uint32_t rtShaderGroupHandleSize;
    uint32_t rtShaderGroupBaseAlignment;
    uint32_t rtShaderGroupHandleAlignment;

    /* Bindless capacity: the largest descriptorNum a single VriDescriptorRangeDesc
       may declare for that type. These are device properties, not compile-time
       constants - they differ by orders of magnitude across backends and hardware,
       so a renderer sizing a material-texture table should clamp against them
       rather than hardcode a limit that happens to hold on one backend.
       0 means "this backend does not report a capacity": always the case when
       hasBindless is false, and also on backends that have not filled it in yet
       (currently reported by Vulkan and Metal). */
    uint32_t bindlessTextureMaxNum;
    uint32_t bindlessSamplerMaxNum;
} VriDeviceDesc;

VRI_EXTERN_C_END

#endif /* VRI_DEVICE_DESC_H */
