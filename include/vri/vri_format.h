/*
 * vri_format.h - VriFormat enumeration (aligned with libvultra PixelFormat)
 * and per-format capability bits.
 */
#ifndef VRI_FORMAT_H
#define VRI_FORMAT_H

#include "vri_base.h"

VRI_EXTERN_C_BEGIN

typedef enum VriFormat
{
    VriFormat_Unknown = 0,

    /* 8-bit */
    VriFormat_R8_UNORM,
    VriFormat_R8_SNORM,
    VriFormat_R8_UINT,
    VriFormat_R8_SINT,
    VriFormat_RG8_UNORM,
    VriFormat_RG8_SNORM,
    VriFormat_RG8_UINT,
    VriFormat_RG8_SINT,
    VriFormat_RGBA8_UNORM,
    VriFormat_RGBA8_SRGB,
    VriFormat_RGBA8_UINT,
    VriFormat_RGBA8_SINT,
    VriFormat_BGRA8_UNORM,
    VriFormat_BGRA8_SRGB,

    /* 16-bit */
    VriFormat_R16_UNORM,
    VriFormat_R16_SNORM,
    VriFormat_R16_UINT,
    VriFormat_R16_SINT,
    VriFormat_R16_SFLOAT,
    VriFormat_RG16_UNORM,
    VriFormat_RG16_SNORM,
    VriFormat_RG16_UINT,
    VriFormat_RG16_SINT,
    VriFormat_RG16_SFLOAT,
    VriFormat_RGBA16_UNORM,
    VriFormat_RGBA16_SNORM,
    VriFormat_RGBA16_UINT,
    VriFormat_RGBA16_SINT,
    VriFormat_RGBA16_SFLOAT,

    /* 32-bit */
    VriFormat_R32_UINT,
    VriFormat_R32_SINT,
    VriFormat_R32_SFLOAT,
    VriFormat_RG32_UINT,
    VriFormat_RG32_SINT,
    VriFormat_RG32_SFLOAT,
    VriFormat_RGB32_UINT,
    VriFormat_RGB32_SINT,
    VriFormat_RGB32_SFLOAT,
    VriFormat_RGBA32_UINT,
    VriFormat_RGBA32_SINT,
    VriFormat_RGBA32_SFLOAT,

    /* packed */
    VriFormat_RGB10A2_UNORM,
    VriFormat_RG11B10_UFLOAT,

    /* block-compressed */
    VriFormat_BC1_UNORM,
    VriFormat_BC2_UNORM,
    VriFormat_BC3_UNORM,
    VriFormat_BC4_UNORM,
    VriFormat_BC5_UNORM,
    VriFormat_BC6H_UFLOAT,
    VriFormat_BC7_UNORM,

    /* depth / stencil */
    VriFormat_D16_UNORM,
    VriFormat_D32_SFLOAT,
    VriFormat_S8_UINT,
    VriFormat_D24_UNORM_S8_UINT,
    VriFormat_D32_SFLOAT_S8_UINT,

    VriFormat_Count,
    VriFormat_MaxEnum = 0x7fffffff
} VriFormat;

/* Per-format capability bits returned by GetFormatSupport. */
typedef uint32_t VriFormatSupportFlags;
typedef enum VriFormatSupportBits
{
    VriFormatSupport_None            = 0,
    VriFormatSupport_Texture         = 1u << 0,
    VriFormatSupport_StorageTexture  = 1u << 1,
    VriFormatSupport_ColorAttachment = 1u << 2,
    VriFormatSupport_DepthStencil    = 1u << 3,
    VriFormatSupport_Blend           = 1u << 4,
    VriFormatSupport_VertexBuffer    = 1u << 5,
    VriFormatSupport_MaxEnum         = 0x7fffffff
} VriFormatSupportBits;

VRI_EXTERN_C_END

#endif /* VRI_FORMAT_H */
