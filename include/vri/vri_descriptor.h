/*
 * vri_descriptor.h - pipeline layout + descriptor set/pool/update descs.
 *
 * Model (NRI-style, explicit): a VriDescriptor is a single resource view (made
 * via CreateBufferView / CreateTextureView / CreateSampler). A pipeline layout
 * declares descriptor-set layouts as arrays of ranges. Descriptor sets are
 * allocated from a pool and then updated with VriDescriptor handles.
 */
#ifndef VRI_DESCRIPTOR_H
#define VRI_DESCRIPTOR_H

#include "vri_base.h"
#include "vri_handles.h"
#include "vri_enums.h"
#include "vri_flags.h"

VRI_EXTERN_C_BEGIN

typedef uint32_t VriDescriptorRangeFlags;
typedef enum VriDescriptorRangeBits
{
    VriDescriptorRange_None          = 0,
    VriDescriptorRange_PartiallyBound = 1u << 0, /* not every descriptor need be set */
    VriDescriptorRange_VariableSized  = 1u << 1, /* bindless / runtime array (last range) */
    VriDescriptorRange_MaxEnum       = 0x7fffffff
} VriDescriptorRangeBits;

/* One contiguous range of same-typed descriptors within a set. */
typedef struct VriDescriptorRangeDesc
{
    uint32_t                baseRegister;  /* binding slot within the set */
    uint32_t                descriptorNum; /* array size (>=1) */
    VriDescriptorType       descriptorType;
    VriShaderStageFlags     shaderStages;
    VriDescriptorRangeFlags flags;
} VriDescriptorRangeDesc;

/* A descriptor-set layout = a register space + its ranges. */
typedef struct VriDescriptorSetDesc
{
    uint32_t                      registerSpace; /* set index */
    const VriDescriptorRangeDesc* ranges;
    uint32_t                      rangeNum;
} VriDescriptorSetDesc;

typedef struct VriPushConstantDesc
{
    uint32_t            baseRegister;
    uint32_t            size;          /* bytes */
    VriShaderStageFlags shaderStages;
} VriPushConstantDesc;

typedef struct VriPipelineLayoutDesc
{
    const VriDescriptorSetDesc* descriptorSets;
    uint32_t                    descriptorSetNum;
    const VriPushConstantDesc*  pushConstants;
    uint32_t                    pushConstantNum;
    VriShaderStageFlags         shaderStages;  /* stages the layout is visible to */
} VriPipelineLayoutDesc;

/* ---- descriptor pool -------------------------------------------------- */
typedef struct VriDescriptorPoolDesc
{
    uint32_t descriptorSetMaxNum;
    uint32_t samplerMaxNum;
    uint32_t textureMaxNum;
    uint32_t storageTextureMaxNum;
    uint32_t constantBufferMaxNum;
    uint32_t structuredBufferMaxNum;
    uint32_t storageBufferMaxNum;
    uint32_t accelerationStructureMaxNum;
} VriDescriptorPoolDesc;

/* ---- descriptor set update -------------------------------------------- */
/* Writes `descriptorNum` views starting at `baseDescriptor` within a range. */
typedef struct VriDescriptorRangeUpdateDesc
{
    const VriDescriptor* const* descriptors;
    uint32_t                    descriptorNum;
    uint32_t                    baseDescriptor;
} VriDescriptorRangeUpdateDesc;

VRI_EXTERN_C_END

#endif /* VRI_DESCRIPTOR_H */
