/*
 * vri_ext_vrs.h - Variable Rate Shading (VRS) extension.
 *
 * Queried via vriGetInterface(device, VRI_INTERFACE_VRS, ...). Registered only by
 * backends that support per-draw shading rate (Vulkan VK_KHR_fragment_shading_rate,
 * D3D12 Tier 1+); never registered elsewhere, so the table is simply unavailable.
 *
 * This covers the per-draw (pipeline) shading rate set dynamically before a draw.
 * Enable the feature at device creation with VriFeature_VariableShadingRate; check
 * VriDeviceDesc::hasVariableShadingRate (and that the interface is queryable).
 */
#ifndef VRI_EXT_VRS_H
#define VRI_EXT_VRS_H

#include "../vri_base.h"
#include "../vri_handles.h"

VRI_EXTERN_C_BEGIN

/* Coarse fragment size: WxH pixels shaded as one. 1x1 == full rate (off). */
typedef enum VriShadingRate
{
    VriShadingRate_1x1 = 0,
    VriShadingRate_1x2,
    VriShadingRate_2x1,
    VriShadingRate_2x2,
    VriShadingRate_2x4,
    VriShadingRate_4x2,
    VriShadingRate_4x4,
    VriShadingRate_MaxEnum = 0x7fffffff
} VriShadingRate;

/* How the pipeline rate combines with the per-primitive rate / the shading-rate
 * attachment rate. Keep == take the pipeline (this) rate, ignoring the other. */
typedef enum VriShadingRateCombiner
{
    VriShadingRateCombiner_Keep = 0,
    VriShadingRateCombiner_Replace,
    VriShadingRateCombiner_Min,
    VriShadingRateCombiner_Max,
    VriShadingRateCombiner_Sum,
    VriShadingRateCombiner_MaxEnum = 0x7fffffff
} VriShadingRateCombiner;

typedef struct VriShadingRateDesc
{
    VriShadingRate         shadingRate;        /* the per-draw (pipeline) rate */
    VriShadingRateCombiner primitiveCombiner;  /* combine with per-primitive rate */
    VriShadingRateCombiner attachmentCombiner; /* combine with shading-rate image  */
} VriShadingRateDesc;

typedef struct VriShadingRateInterface
{
    /* Set the shading rate for subsequent draws in this command buffer. */
    void (VRI_CALL *CmdSetShadingRate)(VriCommandBuffer* cmd, const VriShadingRateDesc* desc);
} VriShadingRateInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_VRS_H */
