/*
 * vri_ext_omm.h - opacity micromap (OMM) extension.
 *
 * Queried via vriGetInterface(device, VRI_INTERFACE_OMM, ...). Registered only by
 * backends that support it (Vulkan VK_EXT_opacity_micromap). Enable with
 * VriFeature_OpacityMicromap at device creation and check VriDeviceDesc::hasOpacityMicromap.
 *
 * An opacity micromap encodes per-microtriangle opacity (transparent/opaque, or a
 * 4-state variant) so the ray tracer can skip or shortcut any-hit evaluation. Build
 * a micromap, then attach it to a BLAS triangle geometry via the OMM fields on
 * VriAsTrianglesDesc (see ext/vri_ext_raytracing.h).
 */
#ifndef VRI_EXT_OMM_H
#define VRI_EXT_OMM_H

#include "../vri_base.h"
#include "../vri_handles.h"
#include "vri_ext_raytracing.h" /* VriMicromap forward decl */

VRI_EXTERN_C_BEGIN

typedef enum VriOpacityMicromapFormat
{
    VriOpacityMicromapFormat_2State = 1, /* opaque / transparent (1 bit per microtriangle) */
    VriOpacityMicromapFormat_4State = 2, /* + unknown-opaque / unknown-transparent (2 bits) */
    VriOpacityMicromapFormat_MaxEnum = 0x7fffffff
} VriOpacityMicromapFormat;

/* Sizing info: one usage group (count triangles at a subdivision level + format). */
typedef struct VriMicromapDesc
{
    VriOpacityMicromapFormat format;
    uint32_t                 subdivisionLevel;
    uint32_t                 triangleCount;
} VriMicromapDesc;

typedef struct VriBuildMicromapDesc
{
    VriMicromap* dst;
    VriBuffer*   dataBuffer;     /* packed opacity values */
    uint64_t     dataOffset;
    VriBuffer*   triangleBuffer; /* array of micromap-triangle records (VkMicromapTriangleEXT layout) */
    uint64_t     triangleOffset;
} VriBuildMicromapDesc;

typedef struct VriOpacityMicromapInterface
{
    VriResult (VRI_CALL *CreateMicromap)(VriDevice* device, const VriMicromapDesc* desc, VriMicromap** outMicromap);
    void      (VRI_CALL *DestroyMicromap)(VriMicromap* micromap);
    void      (VRI_CALL *CmdBuildMicromap)(VriCommandBuffer* cmd, const VriBuildMicromapDesc* desc);
} VriOpacityMicromapInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_OMM_H */
