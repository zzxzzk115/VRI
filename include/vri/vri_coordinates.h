/* Device-owned authoring conventions and explicit shader/projection conversion. */
#ifndef VRI_COORDINATES_H
#define VRI_COORDINATES_H

#include "vri_base.h"
#include "vri_handles.h"

VRI_EXTERN_C_BEGIN

typedef enum VriClipSpaceY
{
    VriClipSpaceY_Down = 0,
    VriClipSpaceY_Up,
    VriClipSpaceY_MaxEnum = 0x7fffffff
} VriClipSpaceY;

typedef enum VriClipDepthRange
{
    VriClipDepthRange_ZeroToOne = 0,
    VriClipDepthRange_NegativeOneToOne,
    VriClipDepthRange_MaxEnum = 0x7fffffff
} VriClipDepthRange;

typedef enum VriTextureOrigin
{
    VriTextureOrigin_TopLeft = 0,
    VriTextureOrigin_BottomLeft,
    VriTextureOrigin_MaxEnum = 0x7fffffff
} VriTextureOrigin;

typedef enum VriMatrixLayout
{
    VriMatrixLayout_ColumnMajor = 0,
    VriMatrixLayout_RowMajor,
    VriMatrixLayout_MaxEnum = 0x7fffffff
} VriMatrixLayout;

typedef enum VriVectorConvention
{
    VriVectorConvention_ColumnVector = 0,
    VriVectorConvention_RowVector,
    VriVectorConvention_MaxEnum = 0x7fffffff
} VriVectorConvention;

typedef enum VriHandedness
{
    VriHandedness_RightHanded = 0,
    VriHandedness_LeftHanded,
    VriHandedness_MaxEnum = 0x7fffffff
} VriHandedness;

typedef enum VriConventionPreset
{
    VriConventionPreset_Vulkan = 0,
    VriConventionPreset_Direct3D,
    VriConventionPreset_Metal,
    VriConventionPreset_MaxEnum = 0x7fffffff
} VriConventionPreset;

/* Zero initialization selects Vulkan clip conventions and the explicit math
 * policies below. Vulkan itself does not prescribe matrix layout or handedness.
 * Presets are editable values, not global state. */
typedef struct VriCoordinateConventions
{
    VriClipSpaceY       clipSpaceY;
    VriClipDepthRange   clipDepthRange;
    VriTextureOrigin    textureOrigin;
    VriMatrixLayout     matrixLayout;
    VriVectorConvention vectorConvention;
    VriHandedness       handedness;
} VriCoordinateConventions;

/* Apply once at the shader output/input boundary:
 *   position.y *= clipYScale;
 *   position.z = position.z * clipZScale + position.w * clipZBias;
 *   uv.y = uv.y * uvYScale + uvYBias;
 * Texture upload/readback row zero always stays the top row. UV conversion
 * changes the author's coordinates, never the stored pixel order. */
typedef struct VriCoordinateTransform
{
    float clipYScale;
    float clipZScale;
    float clipZBias;
    float uvYScale;
    float uvYBias;
} VriCoordinateTransform;

typedef struct VriPerspectiveProjectionDesc
{
    float   verticalFovRadians;
    float   aspectRatio;
    float   nearPlane;
    float   farPlane;
    VriBool reversedDepth;
} VriPerspectiveProjectionDesc;

VRI_API VriResult VRI_CALL vriGetConventionPreset(VriConventionPreset preset, VriCoordinateConventions* out);

/* Configuration is copied at device creation. The returned copy can be used as
 * a source convention but cannot mutate the live device. */
VRI_API VriResult VRI_CALL vriGetCoordinateConventions(const VriDevice* device, VriCoordinateConventions* out);

/* NULL source means the device's authoring convention. An explicit source lets
 * a Y-up UI coexist with a differently authored scene. This conversion targets
 * the actual backend rasterizer. It does not patch arbitrary shader bytecode. */
VRI_API VriResult VRI_CALL vriGetCoordinateTransform(const VriDevice*                device,
                                                     const VriCoordinateConventions* source,
                                                     VriCoordinateTransform*         out);

/* Produces a projection in the supplied authoring convention. Apply the device
 * transform afterwards to make it raster-ready. All sixteen elements are written. */
VRI_API VriResult VRI_CALL vriMakePerspectiveProjection(const VriCoordinateConventions*     conventions,
                                                        const VriPerspectiveProjectionDesc* desc,
                                                        float                               out[16]);

/* In-place clip conversion, including off-center/jitter terms. Memory layout and
 * multiplication order remain those of `conventions`; world handedness is unchanged. */
VRI_API VriResult VRI_CALL vriTransformProjection(const VriCoordinateConventions* conventions,
                                                  const VriCoordinateTransform*   transform,
                                                  float                           matrix[16]);

VRI_EXTERN_C_END

#endif /* VRI_COORDINATES_H */
