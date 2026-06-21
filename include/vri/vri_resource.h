/*
 * vri_resource.h - descriptor structs for buffers, textures, samplers,
 * resource views, and memory.
 */
#ifndef VRI_RESOURCE_H
#define VRI_RESOURCE_H

#include "vri_base.h"
#include "vri_handles.h"
#include "vri_enums.h"
#include "vri_flags.h"
#include "vri_format.h"

VRI_EXTERN_C_BEGIN

/* ---- buffer ----------------------------------------------------------- */
typedef struct VriBufferDesc
{
    uint64_t            size;
    uint32_t            structureStride; /* 0 = raw/typed; >0 = structured */
    VriBufferUsageFlags usage;
    /* Convenience: when used with CreateBuffer the resource is allocated and
       bound here. (The explicit GetBufferMemoryDesc/AllocateMemory/Bind path
       ignores this field.) */
    VriMemoryLocation   memoryLocation;
} VriBufferDesc;

/* ---- texture ---------------------------------------------------------- */
typedef struct VriTextureDesc
{
    VriTextureType       type;
    VriFormat            format;
    uint32_t             width;
    uint32_t             height;     /* 1 for 1D */
    uint32_t             depth;      /* 1 unless 3D */
    uint32_t             mipNum;     /* number of mip levels */
    uint32_t             layerNum;   /* array layers (cube = 6 * arraySize) */
    uint32_t             sampleNum;  /* 1 = no MSAA */
    VriTextureUsageFlags usage;
    VriMemoryLocation    memoryLocation; /* see VriBufferDesc::memoryLocation */
} VriTextureDesc;

/* ---- sampler (aligned with libvultra SamplerInfo) --------------------- */
typedef struct VriSamplerDesc
{
    VriFilter      magFilter;
    VriFilter      minFilter;
    VriMipmapMode  mipmapMode;
    VriAddressMode addressModeU;
    VriAddressMode addressModeV;
    VriAddressMode addressModeW;
    float          mipLodBias;
    VriBool        anisotropyEnable;
    float          maxAnisotropy;
    VriBool        compareEnable;
    VriCompareOp   compareOp;
    float          minLod;
    float          maxLod;
    VriBorderColor borderColor;
    /* Custom (arbitrary) border color for ClampToBorder addressing. When
       useCustomBorderColor is set, customBorderColor (RGBA) overrides borderColor.
       Requires VriDeviceDesc::hasCustomBorderColor. */
    VriBool        useCustomBorderColor;
    float          customBorderColor[4];
} VriSamplerDesc;

/* ---- resource views (a VriDescriptor is created from one of these) ---- */
typedef struct VriBufferViewDesc
{
    VriBuffer*        buffer;
    VriDescriptorType viewType;  /* ConstantBuffer / StructuredBuffer / StorageBuffer */
    VriFormat         format;    /* for typed buffer views; Unknown otherwise */
    uint64_t          offset;
    uint64_t          size;      /* 0 = whole remaining range */
} VriBufferViewDesc;

typedef struct VriTextureViewDesc
{
    VriTexture*         texture;
    VriTextureViewType  viewType;
    VriFormat           format;     /* Unknown = inherit texture format */
    VriImageAspectFlags aspect;
    uint32_t            baseMip;
    uint32_t            mipNum;     /* 0 = all remaining */
    uint32_t            baseLayer;
    uint32_t            layerNum;   /* 0 = all remaining */
} VriTextureViewDesc;

/* ---- memory ----------------------------------------------------------- */
typedef struct VriMemoryDesc
{
    uint64_t          size;
    uint64_t          alignment;
    VriMemoryLocation location;
    /* Backend-opaque: the set of memory types a resource accepts. Produced by
       GetBufferMemoryDesc/GetTextureMemoryDesc and consumed by AllocateMemory.
       Do not interpret or set by hand. */
    uint32_t          memoryTypeMask;
} VriMemoryDesc;

VRI_EXTERN_C_END

#endif /* VRI_RESOURCE_H */
