// conversions_d3d12.h - VRI enum -> D3D12/DXGI mappings.
#pragma once

#include <d3d12.h>
#include <dxgi.h>

#include <algorithm>

#include <vri/vri.h>

#include "core/conversion_map.h"

namespace vri::d3d12
{
    struct DxgiFormatInfo
    {
        DXGI_FORMAT format;
        uint32_t    texelSize; // bytes per texel (for readback row math)
    };

    inline constexpr vri::ConvRow<VriFormat, DxgiFormatInfo> kDxgiFormatTable[] = {
        {VriFormat_R8_UNORM, {DXGI_FORMAT_R8_UNORM, 1}},
        {VriFormat_R8_SNORM, {DXGI_FORMAT_R8_SNORM, 1}},
        {VriFormat_R8_UINT, {DXGI_FORMAT_R8_UINT, 1}},
        {VriFormat_R8_SINT, {DXGI_FORMAT_R8_SINT, 1}},
        {VriFormat_RG8_UNORM, {DXGI_FORMAT_R8G8_UNORM, 2}},
        {VriFormat_RG8_SNORM, {DXGI_FORMAT_R8G8_SNORM, 2}},
        {VriFormat_RG8_UINT, {DXGI_FORMAT_R8G8_UINT, 2}},
        {VriFormat_RG8_SINT, {DXGI_FORMAT_R8G8_SINT, 2}},
        {VriFormat_RGBA8_UNORM, {DXGI_FORMAT_R8G8B8A8_UNORM, 4}},
        {VriFormat_RGBA8_SRGB, {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 4}},
        {VriFormat_RGBA8_UINT, {DXGI_FORMAT_R8G8B8A8_UINT, 4}},
        {VriFormat_RGBA8_SINT, {DXGI_FORMAT_R8G8B8A8_SINT, 4}},
        {VriFormat_BGRA8_UNORM, {DXGI_FORMAT_B8G8R8A8_UNORM, 4}},
        {VriFormat_BGRA8_SRGB, {DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, 4}},
        {VriFormat_R16_UNORM, {DXGI_FORMAT_R16_UNORM, 2}},
        {VriFormat_R16_SNORM, {DXGI_FORMAT_R16_SNORM, 2}},
        {VriFormat_R16_UINT, {DXGI_FORMAT_R16_UINT, 2}},
        {VriFormat_R16_SINT, {DXGI_FORMAT_R16_SINT, 2}},
        {VriFormat_R16_SFLOAT, {DXGI_FORMAT_R16_FLOAT, 2}},
        {VriFormat_RG16_UNORM, {DXGI_FORMAT_R16G16_UNORM, 4}},
        {VriFormat_RG16_SNORM, {DXGI_FORMAT_R16G16_SNORM, 4}},
        {VriFormat_RG16_UINT, {DXGI_FORMAT_R16G16_UINT, 4}},
        {VriFormat_RG16_SINT, {DXGI_FORMAT_R16G16_SINT, 4}},
        {VriFormat_RG16_SFLOAT, {DXGI_FORMAT_R16G16_FLOAT, 4}},
        {VriFormat_RGBA16_UNORM, {DXGI_FORMAT_R16G16B16A16_UNORM, 8}},
        {VriFormat_RGBA16_SNORM, {DXGI_FORMAT_R16G16B16A16_SNORM, 8}},
        {VriFormat_RGBA16_UINT, {DXGI_FORMAT_R16G16B16A16_UINT, 8}},
        {VriFormat_RGBA16_SINT, {DXGI_FORMAT_R16G16B16A16_SINT, 8}},
        {VriFormat_RGBA16_SFLOAT, {DXGI_FORMAT_R16G16B16A16_FLOAT, 8}},
        {VriFormat_R32_UINT, {DXGI_FORMAT_R32_UINT, 4}},
        {VriFormat_R32_SINT, {DXGI_FORMAT_R32_SINT, 4}},
        {VriFormat_R32_SFLOAT, {DXGI_FORMAT_R32_FLOAT, 4}},
        {VriFormat_RG32_UINT, {DXGI_FORMAT_R32G32_UINT, 8}},
        {VriFormat_RG32_SINT, {DXGI_FORMAT_R32G32_SINT, 8}},
        {VriFormat_RG32_SFLOAT, {DXGI_FORMAT_R32G32_FLOAT, 8}},
        {VriFormat_RGB32_UINT, {DXGI_FORMAT_R32G32B32_UINT, 12}},
        {VriFormat_RGB32_SINT, {DXGI_FORMAT_R32G32B32_SINT, 12}},
        {VriFormat_RGB32_SFLOAT, {DXGI_FORMAT_R32G32B32_FLOAT, 12}},
        {VriFormat_RGBA32_UINT, {DXGI_FORMAT_R32G32B32A32_UINT, 16}},
        {VriFormat_RGBA32_SINT, {DXGI_FORMAT_R32G32B32A32_SINT, 16}},
        {VriFormat_RGBA32_SFLOAT, {DXGI_FORMAT_R32G32B32A32_FLOAT, 16}},
        {VriFormat_RGB10A2_UNORM, {DXGI_FORMAT_R10G10B10A2_UNORM, 4}},
        {VriFormat_RG11B10_UFLOAT, {DXGI_FORMAT_R11G11B10_FLOAT, 4}},
        // Compressed storage is measured in blocks, not integer bytes per texel.
        // Copy/readback pitches come from GetCopyableFootprints.
        {VriFormat_BC1_UNORM, {DXGI_FORMAT_BC1_UNORM, 0}},
        {VriFormat_BC2_UNORM, {DXGI_FORMAT_BC2_UNORM, 0}},
        {VriFormat_BC3_UNORM, {DXGI_FORMAT_BC3_UNORM, 0}},
        {VriFormat_BC4_UNORM, {DXGI_FORMAT_BC4_UNORM, 0}},
        {VriFormat_BC5_UNORM, {DXGI_FORMAT_BC5_UNORM, 0}},
        {VriFormat_BC6H_UFLOAT, {DXGI_FORMAT_BC6H_UF16, 0}},
        {VriFormat_BC7_UNORM, {DXGI_FORMAT_BC7_UNORM, 0}},
        {VriFormat_BC1_SRGB, {DXGI_FORMAT_BC1_UNORM_SRGB, 0}},
        {VriFormat_BC2_SRGB, {DXGI_FORMAT_BC2_UNORM_SRGB, 0}},
        {VriFormat_BC3_SRGB, {DXGI_FORMAT_BC3_UNORM_SRGB, 0}},
        {VriFormat_BC7_SRGB, {DXGI_FORMAT_BC7_UNORM_SRGB, 0}},
        {VriFormat_BC6H_SFLOAT, {DXGI_FORMAT_BC6H_SF16, 0}},
        {VriFormat_BC4_SNORM, {DXGI_FORMAT_BC4_SNORM, 0}},
        {VriFormat_BC5_SNORM, {DXGI_FORMAT_BC5_SNORM, 0}},
        // ETC2 and ASTC have no DXGI format: unmapped, reported unsupported.
        {VriFormat_D16_UNORM, {DXGI_FORMAT_D16_UNORM, 2}},
        {VriFormat_D32_SFLOAT, {DXGI_FORMAT_D32_FLOAT, 4}},
        {VriFormat_D24_UNORM_S8_UINT, {DXGI_FORMAT_D24_UNORM_S8_UINT, 4}},
        {VriFormat_D32_SFLOAT_S8_UINT, {DXGI_FORMAT_D32_FLOAT_S8X24_UINT, 8}},
        // VriFormat_S8_UINT has no DXGI equivalent (D3D12 has no stencil-only format): unmapped.
    };

    // DXGI_FORMAT_UNKNOWN means the backend has no DXGI format for `f`. Every caller that
    // creates an object reports that as an error instead of substituting another format.
    inline DxgiFormatInfo ToDxgiFormat(VriFormat f)
    {
        return vri::MapOr(f, kDxgiFormatTable, DxgiFormatInfo {DXGI_FORMAT_UNKNOWN, 0});
    }

    inline D3D12_PRIMITIVE_TOPOLOGY_TYPE ToD3DTopologyType(VriPrimitiveTopology t)
    {
        switch (t)
        {
            case VriPrimitiveTopology_PointList:
                return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
            case VriPrimitiveTopology_LineList:
            case VriPrimitiveTopology_LineStrip:
                return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
            case VriPrimitiveTopology_PatchList:
                return D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH;
            default:
                return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        }
    }

    inline D3D_PRIMITIVE_TOPOLOGY ToD3DTopology(VriPrimitiveTopology t, uint32_t patchControlPoints)
    {
        switch (t)
        {
            case VriPrimitiveTopology_PointList:
                return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
            case VriPrimitiveTopology_LineList:
                return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
            case VriPrimitiveTopology_LineStrip:
                return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
            case VriPrimitiveTopology_TriangleStrip:
                return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
            case VriPrimitiveTopology_PatchList:
                return static_cast<D3D_PRIMITIVE_TOPOLOGY>(D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST +
                                                           (patchControlPoints ? patchControlPoints - 1 : 0));
            default:
                return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        }
    }

    inline D3D12_CULL_MODE ToD3DCull(VriCullMode m)
    {
        switch (m)
        {
            case VriCullMode_Front:
                return D3D12_CULL_MODE_FRONT;
            case VriCullMode_Back:
                return D3D12_CULL_MODE_BACK;
            default:
                return D3D12_CULL_MODE_NONE;
        }
    }

    inline D3D12_COMPARISON_FUNC ToD3DCompare(VriCompareOp op)
    {
        switch (op)
        {
            case VriCompareOp_Never:
                return D3D12_COMPARISON_FUNC_NEVER;
            case VriCompareOp_Less:
                return D3D12_COMPARISON_FUNC_LESS;
            case VriCompareOp_Equal:
                return D3D12_COMPARISON_FUNC_EQUAL;
            case VriCompareOp_LessOrEqual:
                return D3D12_COMPARISON_FUNC_LESS_EQUAL;
            case VriCompareOp_Greater:
                return D3D12_COMPARISON_FUNC_GREATER;
            case VriCompareOp_NotEqual:
                return D3D12_COMPARISON_FUNC_NOT_EQUAL;
            case VriCompareOp_GreaterOrEqual:
                return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
            default:
                return D3D12_COMPARISON_FUNC_ALWAYS;
        }
    }

    inline D3D12_STENCIL_OP ToD3DStencilOp(VriStencilOp op)
    {
        switch (op)
        {
            case VriStencilOp_Zero:
                return D3D12_STENCIL_OP_ZERO;
            case VriStencilOp_Replace:
                return D3D12_STENCIL_OP_REPLACE;
            case VriStencilOp_IncrementAndClamp:
                return D3D12_STENCIL_OP_INCR_SAT;
            case VriStencilOp_DecrementAndClamp:
                return D3D12_STENCIL_OP_DECR_SAT;
            case VriStencilOp_Invert:
                return D3D12_STENCIL_OP_INVERT;
            case VriStencilOp_IncrementAndWrap:
                return D3D12_STENCIL_OP_INCR;
            case VriStencilOp_DecrementAndWrap:
                return D3D12_STENCIL_OP_DECR;
            default:
                return D3D12_STENCIL_OP_KEEP;
        }
    }

    inline D3D12_DEPTH_STENCILOP_DESC ToD3DStencilFace(const VriStencilOpDesc& f)
    {
        D3D12_DEPTH_STENCILOP_DESC d = {};
        d.StencilFailOp              = ToD3DStencilOp(f.failOp);
        d.StencilDepthFailOp         = ToD3DStencilOp(f.depthFailOp);
        d.StencilPassOp              = ToD3DStencilOp(f.passOp);
        d.StencilFunc                = ToD3DCompare(f.compareOp);
        return d;
    }

    inline D3D12_BLEND ToD3DBlend(VriBlendFactor f)
    {
        switch (f)
        {
            case VriBlendFactor_Zero:
                return D3D12_BLEND_ZERO;
            case VriBlendFactor_One:
                return D3D12_BLEND_ONE;
            case VriBlendFactor_SrcColor:
                return D3D12_BLEND_SRC_COLOR;
            case VriBlendFactor_OneMinusSrcColor:
                return D3D12_BLEND_INV_SRC_COLOR;
            case VriBlendFactor_DstColor:
                return D3D12_BLEND_DEST_COLOR;
            case VriBlendFactor_OneMinusDstColor:
                return D3D12_BLEND_INV_DEST_COLOR;
            case VriBlendFactor_SrcAlpha:
                return D3D12_BLEND_SRC_ALPHA;
            case VriBlendFactor_OneMinusSrcAlpha:
                return D3D12_BLEND_INV_SRC_ALPHA;
            case VriBlendFactor_DstAlpha:
                return D3D12_BLEND_DEST_ALPHA;
            case VriBlendFactor_OneMinusDstAlpha:
                return D3D12_BLEND_INV_DEST_ALPHA;
            default:
                return D3D12_BLEND_ONE;
        }
    }

    inline D3D12_TEXTURE_ADDRESS_MODE ToD3DAddress(VriAddressMode m)
    {
        switch (m)
        {
            case VriAddressMode_MirroredRepeat:
                return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
            case VriAddressMode_ClampToEdge:
                return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            case VriAddressMode_ClampToBorder:
                return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            default:
                return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        }
    }

    inline D3D12_SAMPLER_DESC ToD3DSampler(const VriSamplerDesc& s)
    {
        D3D12_SAMPLER_DESC d = {};
        // Minification, magnification and mip filtering are chosen separately, as on the other backends.
        // A comparison sampler (shadow PCF) needs a COMPARISON filter variant + the compare func.
        const auto type = [](bool linear) { return linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT; };
        const D3D12_FILTER_REDUCTION_TYPE reduction =
            s.compareEnable ? D3D12_FILTER_REDUCTION_TYPE_COMPARISON : D3D12_FILTER_REDUCTION_TYPE_STANDARD;
        if (s.anisotropyEnable)
            d.Filter = D3D12_ENCODE_ANISOTROPIC_FILTER(reduction);
        else
            d.Filter = D3D12_ENCODE_BASIC_FILTER(type(s.minFilter == VriFilter_Linear),
                                                 type(s.magFilter == VriFilter_Linear),
                                                 type(s.mipmapMode == VriMipmapMode_Linear),
                                                 reduction);
        d.AddressU   = ToD3DAddress(s.addressModeU);
        d.AddressV   = ToD3DAddress(s.addressModeV);
        d.AddressW   = ToD3DAddress(s.addressModeW);
        d.MipLODBias = s.mipLodBias;
        // Every D3D12 device supports anisotropy up to 16.
        d.MaxAnisotropy =
            s.anisotropyEnable ? static_cast<UINT>((std::min)((std::max)(s.maxAnisotropy, 1.0f), 16.0f)) : 1u;
        // Non-comparison samplers must use COMPARISON_FUNC_NONE: a non-NONE func on a non-comparison
        // filter trips a debug-layer warning (the func is ignored, but it flags likely-unintended use).
        d.ComparisonFunc = s.compareEnable ? ToD3DCompare(s.compareOp) : D3D12_COMPARISON_FUNC_NONE;
        d.MinLOD         = s.minLod;
        d.MaxLOD         = s.maxLod > 0.0f ? s.maxLod : D3D12_FLOAT32_MAX;
        // D3D12 border color is always an arbitrary float4 (native custom support).
        if (s.useCustomBorderColor)
            for (int i = 0; i < 4; ++i)
                d.BorderColor[i] = s.customBorderColor[i];
        else if (s.borderColor == VriBorderColor_FloatOpaqueWhite || s.borderColor == VriBorderColor_IntOpaqueWhite)
        {
            d.BorderColor[0] = d.BorderColor[1] = d.BorderColor[2] = d.BorderColor[3] = 1.0f;
        }
        else if (s.borderColor == VriBorderColor_FloatOpaqueBlack || s.borderColor == VriBorderColor_IntOpaqueBlack)
        {
            d.BorderColor[3] = 1.0f;
        } // opaque black (rgb 0, a 1)
        return d;
    }

    inline D3D12_BLEND_OP ToD3DBlendOp(VriBlendOp op)
    {
        switch (op)
        {
            case VriBlendOp_Subtract:
                return D3D12_BLEND_OP_SUBTRACT;
            case VriBlendOp_ReverseSubtract:
                return D3D12_BLEND_OP_REV_SUBTRACT;
            case VriBlendOp_Min:
                return D3D12_BLEND_OP_MIN;
            case VriBlendOp_Max:
                return D3D12_BLEND_OP_MAX;
            default:
                return D3D12_BLEND_OP_ADD;
        }
    }
} // namespace vri::d3d12
