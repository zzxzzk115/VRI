// conversions_mtl.h - VRI enum/format <-> Metal enum mapping.
//
// Header is Objective-C++ (imports <Metal/Metal.h>); only the backend's .mm
// translation units include it. Mirrors conversions_wgpu.h / conversions_d3d12.h.
#pragma once

#import <Metal/Metal.h>

#include <vri/vri.h>

#include "core/conversion_map.h"
#include "core/format_traits.h"

namespace vri::mtl
{
    // VriFormat -> MTLPixelFormat. Returns MTLPixelFormatInvalid for unmapped.
    // Note: D24_UNORM_S8 is not available on Apple Silicon, so it (and the
    // combined D32S8) both map to Depth32Float_Stencil8 - the portable choice.
    inline MTLPixelFormat ToMtlFormat(VriFormat f)
    {
        static constexpr vri::ConvRow<VriFormat, MTLPixelFormat> kTable[] = {
            {VriFormat_R8_UNORM,        MTLPixelFormatR8Unorm},
            {VriFormat_R8_SNORM,        MTLPixelFormatR8Snorm},
            {VriFormat_R8_UINT,         MTLPixelFormatR8Uint},
            {VriFormat_R8_SINT,         MTLPixelFormatR8Sint},
            {VriFormat_RG8_UNORM,       MTLPixelFormatRG8Unorm},
            {VriFormat_RG8_SNORM,       MTLPixelFormatRG8Snorm},
            {VriFormat_RG8_UINT,        MTLPixelFormatRG8Uint},
            {VriFormat_RG8_SINT,        MTLPixelFormatRG8Sint},
            {VriFormat_RGBA8_UNORM,     MTLPixelFormatRGBA8Unorm},
            {VriFormat_RGBA8_SRGB,      MTLPixelFormatRGBA8Unorm_sRGB},
            {VriFormat_RGBA8_UINT,      MTLPixelFormatRGBA8Uint},
            {VriFormat_RGBA8_SINT,      MTLPixelFormatRGBA8Sint},
            {VriFormat_BGRA8_UNORM,     MTLPixelFormatBGRA8Unorm},
            {VriFormat_BGRA8_SRGB,      MTLPixelFormatBGRA8Unorm_sRGB},

            {VriFormat_R16_UNORM,       MTLPixelFormatR16Unorm},
            {VriFormat_R16_SNORM,       MTLPixelFormatR16Snorm},
            {VriFormat_R16_UINT,        MTLPixelFormatR16Uint},
            {VriFormat_R16_SINT,        MTLPixelFormatR16Sint},
            {VriFormat_R16_SFLOAT,      MTLPixelFormatR16Float},
            {VriFormat_RG16_UNORM,      MTLPixelFormatRG16Unorm},
            {VriFormat_RG16_SNORM,      MTLPixelFormatRG16Snorm},
            {VriFormat_RG16_UINT,       MTLPixelFormatRG16Uint},
            {VriFormat_RG16_SINT,       MTLPixelFormatRG16Sint},
            {VriFormat_RG16_SFLOAT,     MTLPixelFormatRG16Float},
            {VriFormat_RGBA16_UNORM,    MTLPixelFormatRGBA16Unorm},
            {VriFormat_RGBA16_SNORM,    MTLPixelFormatRGBA16Snorm},
            {VriFormat_RGBA16_UINT,     MTLPixelFormatRGBA16Uint},
            {VriFormat_RGBA16_SINT,     MTLPixelFormatRGBA16Sint},
            {VriFormat_RGBA16_SFLOAT,   MTLPixelFormatRGBA16Float},

            {VriFormat_R32_UINT,        MTLPixelFormatR32Uint},
            {VriFormat_R32_SINT,        MTLPixelFormatR32Sint},
            {VriFormat_R32_SFLOAT,      MTLPixelFormatR32Float},
            {VriFormat_RG32_UINT,       MTLPixelFormatRG32Uint},
            {VriFormat_RG32_SINT,       MTLPixelFormatRG32Sint},
            {VriFormat_RG32_SFLOAT,     MTLPixelFormatRG32Float},
            {VriFormat_RGBA32_UINT,     MTLPixelFormatRGBA32Uint},
            {VriFormat_RGBA32_SINT,     MTLPixelFormatRGBA32Sint},
            {VriFormat_RGBA32_SFLOAT,   MTLPixelFormatRGBA32Float},
            // Metal has no 3-component pixel formats; callers use those for vertex
            // attributes only (ToMtlVertexFormat handles RGB32).

            {VriFormat_RGB10A2_UNORM,   MTLPixelFormatRGB10A2Unorm},
            {VriFormat_RG11B10_UFLOAT,  MTLPixelFormatRG11B10Float},

            {VriFormat_BC1_UNORM,       MTLPixelFormatBC1_RGBA},
            {VriFormat_BC2_UNORM,       MTLPixelFormatBC2_RGBA},
            {VriFormat_BC3_UNORM,       MTLPixelFormatBC3_RGBA},
            {VriFormat_BC4_UNORM,       MTLPixelFormatBC4_RUnorm},
            {VriFormat_BC5_UNORM,       MTLPixelFormatBC5_RGUnorm},
            {VriFormat_BC6H_UFLOAT,     MTLPixelFormatBC6H_RGBUfloat},
            {VriFormat_BC7_UNORM,       MTLPixelFormatBC7_RGBAUnorm},

            {VriFormat_D16_UNORM,       MTLPixelFormatDepth16Unorm},
            {VriFormat_D32_SFLOAT,      MTLPixelFormatDepth32Float},
            {VriFormat_S8_UINT,         MTLPixelFormatStencil8},
            {VriFormat_D24_UNORM_S8_UINT,  MTLPixelFormatDepth32Float_Stencil8},
            {VriFormat_D32_SFLOAT_S8_UINT, MTLPixelFormatDepth32Float_Stencil8},
        };
        return vri::MapOr(f, kTable, MTLPixelFormatInvalid);
    }

    // VriFormat -> MTLVertexFormat (vertex-attribute use; includes 3-component).
    inline MTLVertexFormat ToMtlVertexFormat(VriFormat f)
    {
        static constexpr vri::ConvRow<VriFormat, MTLVertexFormat> kTable[] = {
            {VriFormat_R8_UNORM,      MTLVertexFormatUCharNormalized},
            {VriFormat_RG8_UNORM,     MTLVertexFormatUChar2Normalized},
            {VriFormat_RGBA8_UNORM,   MTLVertexFormatUChar4Normalized},
            {VriFormat_R8_UINT,       MTLVertexFormatUChar},
            {VriFormat_RGBA8_UINT,    MTLVertexFormatUChar4},
            {VriFormat_R16_SFLOAT,    MTLVertexFormatHalf},
            {VriFormat_RG16_SFLOAT,   MTLVertexFormatHalf2},
            {VriFormat_RGBA16_SFLOAT, MTLVertexFormatHalf4},
            {VriFormat_R32_SFLOAT,    MTLVertexFormatFloat},
            {VriFormat_RG32_SFLOAT,   MTLVertexFormatFloat2},
            {VriFormat_RGB32_SFLOAT,  MTLVertexFormatFloat3},
            {VriFormat_RGBA32_SFLOAT, MTLVertexFormatFloat4},
            {VriFormat_R32_UINT,      MTLVertexFormatUInt},
            {VriFormat_RG32_UINT,     MTLVertexFormatUInt2},
            {VriFormat_RGB32_UINT,    MTLVertexFormatUInt3},
            {VriFormat_RGBA32_UINT,   MTLVertexFormatUInt4},
            {VriFormat_R32_SINT,      MTLVertexFormatInt},
            {VriFormat_RG32_SINT,     MTLVertexFormatInt2},
            {VriFormat_RGB32_SINT,    MTLVertexFormatInt3},
            {VriFormat_RGBA32_SINT,   MTLVertexFormatInt4},
        };
        return vri::MapOr(f, kTable, MTLVertexFormatInvalid);
    }

    // Bytes per (uncompressed) texel - drives copy bytesPerRow.
    inline uint32_t TexelSize(VriFormat f)
    {
        switch (f)
        {
            case VriFormat_R8_UNORM: case VriFormat_R8_SNORM: case VriFormat_R8_UINT:
            case VriFormat_R8_SINT:  case VriFormat_S8_UINT:
                return 1;
            case VriFormat_RG8_UNORM: case VriFormat_RG8_SNORM: case VriFormat_RG8_UINT: case VriFormat_RG8_SINT:
            case VriFormat_R16_UNORM: case VriFormat_R16_SNORM: case VriFormat_R16_UINT: case VriFormat_R16_SINT:
            case VriFormat_R16_SFLOAT: case VriFormat_D16_UNORM:
                return 2;
            case VriFormat_RGBA8_UNORM: case VriFormat_RGBA8_SRGB: case VriFormat_RGBA8_UINT: case VriFormat_RGBA8_SINT:
            case VriFormat_BGRA8_UNORM: case VriFormat_BGRA8_SRGB:
            case VriFormat_RG16_UNORM: case VriFormat_RG16_SNORM: case VriFormat_RG16_UINT: case VriFormat_RG16_SINT: case VriFormat_RG16_SFLOAT:
            case VriFormat_R32_UINT: case VriFormat_R32_SINT: case VriFormat_R32_SFLOAT:
            case VriFormat_RGB10A2_UNORM: case VriFormat_RG11B10_UFLOAT:
            case VriFormat_D32_SFLOAT: case VriFormat_D24_UNORM_S8_UINT:
                return 4;
            case VriFormat_RGBA16_UNORM: case VriFormat_RGBA16_SNORM: case VriFormat_RGBA16_UINT:
            case VriFormat_RGBA16_SINT:  case VriFormat_RGBA16_SFLOAT:
            case VriFormat_RG32_UINT: case VriFormat_RG32_SINT: case VriFormat_RG32_SFLOAT:
            case VriFormat_D32_SFLOAT_S8_UINT:
                return 8;
            case VriFormat_RGBA32_UINT: case VriFormat_RGBA32_SINT: case VriFormat_RGBA32_SFLOAT:
                return 16;
            default:
                return 4;
        }
    }

    // The depth/stencil predicates are backend-neutral; see core/format_traits.h.
    using vri::FormatHasDepth;
    using vri::FormatHasStencil;

    inline MTLCompareFunction ToMtlCompare(VriCompareOp op)
    {
        switch (op)
        {
            case VriCompareOp_Never:          return MTLCompareFunctionNever;
            case VriCompareOp_Less:           return MTLCompareFunctionLess;
            case VriCompareOp_Equal:          return MTLCompareFunctionEqual;
            case VriCompareOp_LessOrEqual:    return MTLCompareFunctionLessEqual;
            case VriCompareOp_Greater:        return MTLCompareFunctionGreater;
            case VriCompareOp_NotEqual:       return MTLCompareFunctionNotEqual;
            case VriCompareOp_GreaterOrEqual: return MTLCompareFunctionGreaterEqual;
            case VriCompareOp_Always:         return MTLCompareFunctionAlways;
            default:                          return MTLCompareFunctionAlways;
        }
    }

    inline MTLStencilOperation ToMtlStencilOp(VriStencilOp op)
    {
        switch (op)
        {
            case VriStencilOp_Keep:              return MTLStencilOperationKeep;
            case VriStencilOp_Zero:              return MTLStencilOperationZero;
            case VriStencilOp_Replace:           return MTLStencilOperationReplace;
            case VriStencilOp_IncrementAndClamp: return MTLStencilOperationIncrementClamp;
            case VriStencilOp_DecrementAndClamp: return MTLStencilOperationDecrementClamp;
            case VriStencilOp_Invert:            return MTLStencilOperationInvert;
            case VriStencilOp_IncrementAndWrap:  return MTLStencilOperationIncrementWrap;
            case VriStencilOp_DecrementAndWrap:  return MTLStencilOperationDecrementWrap;
            default:                             return MTLStencilOperationKeep;
        }
    }

    inline MTLBlendOperation ToMtlBlendOp(VriBlendOp op)
    {
        switch (op)
        {
            case VriBlendOp_Add:             return MTLBlendOperationAdd;
            case VriBlendOp_Subtract:        return MTLBlendOperationSubtract;
            case VriBlendOp_ReverseSubtract: return MTLBlendOperationReverseSubtract;
            case VriBlendOp_Min:             return MTLBlendOperationMin;
            case VriBlendOp_Max:             return MTLBlendOperationMax;
            default:                         return MTLBlendOperationAdd;
        }
    }

    inline MTLBlendFactor ToMtlBlendFactor(VriBlendFactor f)
    {
        switch (f)
        {
            case VriBlendFactor_Zero:                 return MTLBlendFactorZero;
            case VriBlendFactor_One:                  return MTLBlendFactorOne;
            case VriBlendFactor_SrcColor:             return MTLBlendFactorSourceColor;
            case VriBlendFactor_OneMinusSrcColor:     return MTLBlendFactorOneMinusSourceColor;
            case VriBlendFactor_DstColor:             return MTLBlendFactorDestinationColor;
            case VriBlendFactor_OneMinusDstColor:     return MTLBlendFactorOneMinusDestinationColor;
            case VriBlendFactor_SrcAlpha:             return MTLBlendFactorSourceAlpha;
            case VriBlendFactor_OneMinusSrcAlpha:     return MTLBlendFactorOneMinusSourceAlpha;
            case VriBlendFactor_DstAlpha:             return MTLBlendFactorDestinationAlpha;
            case VriBlendFactor_OneMinusDstAlpha:     return MTLBlendFactorOneMinusDestinationAlpha;
            case VriBlendFactor_ConstantColor:        return MTLBlendFactorBlendColor;
            case VriBlendFactor_OneMinusConstantColor:return MTLBlendFactorOneMinusBlendColor;
            case VriBlendFactor_ConstantAlpha:        return MTLBlendFactorBlendAlpha;
            case VriBlendFactor_OneMinusConstantAlpha:return MTLBlendFactorOneMinusBlendAlpha;
            case VriBlendFactor_SrcAlphaSaturate:     return MTLBlendFactorSourceAlphaSaturated;
            case VriBlendFactor_Src1Color:            return MTLBlendFactorSource1Color;
            case VriBlendFactor_OneMinusSrc1Color:    return MTLBlendFactorOneMinusSource1Color;
            case VriBlendFactor_Src1Alpha:            return MTLBlendFactorSource1Alpha;
            case VriBlendFactor_OneMinusSrc1Alpha:    return MTLBlendFactorOneMinusSource1Alpha;
            default:                                  return MTLBlendFactorZero;
        }
    }

    inline MTLColorWriteMask ToMtlColorWriteMask(VriColorWriteFlags m)
    {
        if (m == 0 || m == VriColorWrite_RGBA)
            return MTLColorWriteMaskAll;
        MTLColorWriteMask r = MTLColorWriteMaskNone;
        if (m & VriColorWrite_R) r |= MTLColorWriteMaskRed;
        if (m & VriColorWrite_G) r |= MTLColorWriteMaskGreen;
        if (m & VriColorWrite_B) r |= MTLColorWriteMaskBlue;
        if (m & VriColorWrite_A) r |= MTLColorWriteMaskAlpha;
        return r;
    }

    inline MTLCullMode ToMtlCull(VriCullMode m)
    {
        switch (m)
        {
            case VriCullMode_Front: return MTLCullModeFront;
            case VriCullMode_Back:  return MTLCullModeBack;
            default:                return MTLCullModeNone;
        }
    }

    inline MTLWinding ToMtlWinding(VriFrontFace f)
    {
        // VRI clip space is Y-Up, same as Metal's native NDC, so the viewport needs no
        // flip (unlike the Vulkan backend) and CCW maps straight to CCW front-facing.
        return f == VriFrontFace_Clockwise ? MTLWindingClockwise : MTLWindingCounterClockwise;
    }

    inline MTLPrimitiveType ToMtlPrimitive(VriPrimitiveTopology t)
    {
        switch (t)
        {
            case VriPrimitiveTopology_PointList:     return MTLPrimitiveTypePoint;
            case VriPrimitiveTopology_LineList:      return MTLPrimitiveTypeLine;
            case VriPrimitiveTopology_LineStrip:     return MTLPrimitiveTypeLineStrip;
            case VriPrimitiveTopology_TriangleList:  return MTLPrimitiveTypeTriangle;
            case VriPrimitiveTopology_TriangleStrip: return MTLPrimitiveTypeTriangleStrip;
            default:                                 return MTLPrimitiveTypeTriangle;
        }
    }

    inline MTLPrimitiveTopologyClass ToMtlTopologyClass(VriPrimitiveTopology t)
    {
        switch (t)
        {
            case VriPrimitiveTopology_PointList: return MTLPrimitiveTopologyClassPoint;
            case VriPrimitiveTopology_LineList:
            case VriPrimitiveTopology_LineStrip: return MTLPrimitiveTopologyClassLine;
            default:                             return MTLPrimitiveTopologyClassTriangle;
        }
    }

    inline MTLSamplerMinMagFilter ToMtlFilter(VriFilter f)
    {
        return f == VriFilter_Linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    }
    inline MTLSamplerMipFilter ToMtlMipFilter(VriMipmapMode m)
    {
        return m == VriMipmapMode_Linear ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
    }
    inline MTLSamplerAddressMode ToMtlAddress(VriAddressMode m)
    {
        switch (m)
        {
            case VriAddressMode_Repeat:            return MTLSamplerAddressModeRepeat;
            case VriAddressMode_MirroredRepeat:    return MTLSamplerAddressModeMirrorRepeat;
            case VriAddressMode_ClampToEdge:       return MTLSamplerAddressModeClampToEdge;
            case VriAddressMode_ClampToBorder:     return MTLSamplerAddressModeClampToBorderColor;
            case VriAddressMode_MirrorClampToEdge: return MTLSamplerAddressModeMirrorClampToEdge;
            default:                               return MTLSamplerAddressModeRepeat;
        }
    }
    inline MTLSamplerBorderColor ToMtlBorderColor(VriBorderColor c)
    {
        switch (c)
        {
            case VriBorderColor_FloatOpaqueWhite:
            case VriBorderColor_IntOpaqueWhite:  return MTLSamplerBorderColorOpaqueWhite;
            case VriBorderColor_FloatOpaqueBlack:
            case VriBorderColor_IntOpaqueBlack:  return MTLSamplerBorderColorOpaqueBlack;
            default:                             return MTLSamplerBorderColorTransparentBlack;
        }
    }

    inline MTLLoadAction ToMtlLoadAction(VriAttachmentLoadOp op)
    {
        switch (op)
        {
            case VriAttachmentLoadOp_Load:     return MTLLoadActionLoad;
            case VriAttachmentLoadOp_Clear:    return MTLLoadActionClear;
            default:                           return MTLLoadActionDontCare;
        }
    }
    inline MTLStoreAction ToMtlStoreAction(VriAttachmentStoreOp op)
    {
        return op == VriAttachmentStoreOp_DontCare ? MTLStoreActionDontCare : MTLStoreActionStore;
    }

    inline MTLTextureType ToMtlTextureType(VriTextureType t, uint32_t sampleNum)
    {
        const bool ms = sampleNum > 1;
        switch (t)
        {
            case VriTextureType_1D:        return MTLTextureType1D;
            case VriTextureType_1DArray:   return MTLTextureType1DArray;
            case VriTextureType_2D:        return ms ? MTLTextureType2DMultisample : MTLTextureType2D;
            case VriTextureType_2DArray:   return MTLTextureType2DArray;
            case VriTextureType_3D:        return MTLTextureType3D;
            case VriTextureType_Cube:      return MTLTextureTypeCube;
            case VriTextureType_CubeArray: return MTLTextureTypeCubeArray;
            default:                       return MTLTextureType2D;
        }
    }

    inline MTLTextureType ToMtlViewType(VriTextureViewType t)
    {
        switch (t)
        {
            case VriTextureViewType_1D:        return MTLTextureType1D;
            case VriTextureViewType_1DArray:   return MTLTextureType1DArray;
            case VriTextureViewType_2D:        return MTLTextureType2D;
            case VriTextureViewType_2DArray:   return MTLTextureType2DArray;
            case VriTextureViewType_3D:        return MTLTextureType3D;
            case VriTextureViewType_Cube:      return MTLTextureTypeCube;
            case VriTextureViewType_CubeArray: return MTLTextureTypeCubeArray;
            default:                           return MTLTextureType2D;
        }
    }
} // namespace vri::mtl
