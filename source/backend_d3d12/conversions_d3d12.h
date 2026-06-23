// conversions_d3d12.h - VRI enum -> D3D12/DXGI mappings.
#pragma once

#include <d3d12.h>
#include <dxgi.h>

#include <vri/vri.h>

namespace vri::d3d12
{
    struct DxgiFormatInfo
    {
        DXGI_FORMAT format;
        uint32_t    texelSize; // bytes per texel (for readback row math)
    };

    inline DxgiFormatInfo ToDxgiFormat(VriFormat f)
    {
        switch (f)
        {
            case VriFormat_RGBA8_UNORM:  return {DXGI_FORMAT_R8G8B8A8_UNORM, 4};
            case VriFormat_RGBA8_SRGB:   return {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 4};
            case VriFormat_BGRA8_UNORM:  return {DXGI_FORMAT_B8G8R8A8_UNORM, 4};
            case VriFormat_RG8_UNORM:    return {DXGI_FORMAT_R8G8_UNORM, 2};
            case VriFormat_R8_UNORM:     return {DXGI_FORMAT_R8_UNORM, 1};
            case VriFormat_RGBA16_SFLOAT:return {DXGI_FORMAT_R16G16B16A16_FLOAT, 8};
            case VriFormat_RGBA32_SFLOAT:return {DXGI_FORMAT_R32G32B32A32_FLOAT, 16};
            case VriFormat_RG32_SFLOAT:  return {DXGI_FORMAT_R32G32_FLOAT, 8};
            case VriFormat_RGB32_SFLOAT: return {DXGI_FORMAT_R32G32B32_FLOAT, 12};
            case VriFormat_R32_SFLOAT:   return {DXGI_FORMAT_R32_FLOAT, 4};
            case VriFormat_R32_UINT:     return {DXGI_FORMAT_R32_UINT, 4};
            case VriFormat_R32_SINT:     return {DXGI_FORMAT_R32_SINT, 4};
            case VriFormat_RG32_UINT:    return {DXGI_FORMAT_R32G32_UINT, 8};
            case VriFormat_RG32_SINT:    return {DXGI_FORMAT_R32G32_SINT, 8};
            case VriFormat_RGBA32_UINT:  return {DXGI_FORMAT_R32G32B32A32_UINT, 16};
            case VriFormat_RGBA32_SINT:  return {DXGI_FORMAT_R32G32B32A32_SINT, 16};
            case VriFormat_D32_SFLOAT:   return {DXGI_FORMAT_D32_FLOAT, 4};
            case VriFormat_D24_UNORM_S8_UINT: return {DXGI_FORMAT_D24_UNORM_S8_UINT, 4};
            default:                     return {DXGI_FORMAT_R8G8B8A8_UNORM, 4};
        }
    }

    inline DXGI_FORMAT ToDxgiVertexFormat(VriFormat f) { return ToDxgiFormat(f).format; }

    inline D3D12_PRIMITIVE_TOPOLOGY_TYPE ToD3DTopologyType(VriPrimitiveTopology t)
    {
        switch (t)
        {
            case VriPrimitiveTopology_PointList:     return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
            case VriPrimitiveTopology_LineList:
            case VriPrimitiveTopology_LineStrip:     return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
            case VriPrimitiveTopology_PatchList:     return D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH;
            default:                                 return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        }
    }

    inline D3D_PRIMITIVE_TOPOLOGY ToD3DTopology(VriPrimitiveTopology t, uint32_t patchControlPoints)
    {
        switch (t)
        {
            case VriPrimitiveTopology_PointList:     return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
            case VriPrimitiveTopology_LineList:      return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
            case VriPrimitiveTopology_LineStrip:     return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
            case VriPrimitiveTopology_TriangleStrip: return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
            case VriPrimitiveTopology_PatchList:
                return static_cast<D3D_PRIMITIVE_TOPOLOGY>(D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST + (patchControlPoints ? patchControlPoints - 1 : 0));
            default:                                 return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        }
    }

    inline D3D12_CULL_MODE ToD3DCull(VriCullMode m)
    {
        switch (m)
        {
            case VriCullMode_Front: return D3D12_CULL_MODE_FRONT;
            case VriCullMode_Back:  return D3D12_CULL_MODE_BACK;
            default:                return D3D12_CULL_MODE_NONE;
        }
    }

    inline D3D12_COMPARISON_FUNC ToD3DCompare(VriCompareOp op)
    {
        switch (op)
        {
            case VriCompareOp_Never:        return D3D12_COMPARISON_FUNC_NEVER;
            case VriCompareOp_Less:         return D3D12_COMPARISON_FUNC_LESS;
            case VriCompareOp_Equal:        return D3D12_COMPARISON_FUNC_EQUAL;
            case VriCompareOp_LessOrEqual:  return D3D12_COMPARISON_FUNC_LESS_EQUAL;
            case VriCompareOp_Greater:      return D3D12_COMPARISON_FUNC_GREATER;
            case VriCompareOp_NotEqual:     return D3D12_COMPARISON_FUNC_NOT_EQUAL;
            case VriCompareOp_GreaterOrEqual: return D3D12_COMPARISON_FUNC_GREATER_EQUAL;
            default:                        return D3D12_COMPARISON_FUNC_ALWAYS;
        }
    }

    inline D3D12_STENCIL_OP ToD3DStencilOp(VriStencilOp op)
    {
        switch (op)
        {
            case VriStencilOp_Zero:              return D3D12_STENCIL_OP_ZERO;
            case VriStencilOp_Replace:           return D3D12_STENCIL_OP_REPLACE;
            case VriStencilOp_IncrementAndClamp: return D3D12_STENCIL_OP_INCR_SAT;
            case VriStencilOp_DecrementAndClamp: return D3D12_STENCIL_OP_DECR_SAT;
            case VriStencilOp_Invert:            return D3D12_STENCIL_OP_INVERT;
            case VriStencilOp_IncrementAndWrap:  return D3D12_STENCIL_OP_INCR;
            case VriStencilOp_DecrementAndWrap:  return D3D12_STENCIL_OP_DECR;
            default:                             return D3D12_STENCIL_OP_KEEP;
        }
    }

    inline D3D12_DEPTH_STENCILOP_DESC ToD3DStencilFace(const VriStencilOpDesc& f)
    {
        D3D12_DEPTH_STENCILOP_DESC d = {};
        d.StencilFailOp = ToD3DStencilOp(f.failOp);
        d.StencilDepthFailOp = ToD3DStencilOp(f.depthFailOp);
        d.StencilPassOp = ToD3DStencilOp(f.passOp);
        d.StencilFunc = ToD3DCompare(f.compareOp);
        return d;
    }

    inline D3D12_BLEND ToD3DBlend(VriBlendFactor f)
    {
        switch (f)
        {
            case VriBlendFactor_Zero:             return D3D12_BLEND_ZERO;
            case VriBlendFactor_One:              return D3D12_BLEND_ONE;
            case VriBlendFactor_SrcColor:         return D3D12_BLEND_SRC_COLOR;
            case VriBlendFactor_OneMinusSrcColor: return D3D12_BLEND_INV_SRC_COLOR;
            case VriBlendFactor_DstColor:         return D3D12_BLEND_DEST_COLOR;
            case VriBlendFactor_OneMinusDstColor: return D3D12_BLEND_INV_DEST_COLOR;
            case VriBlendFactor_SrcAlpha:         return D3D12_BLEND_SRC_ALPHA;
            case VriBlendFactor_OneMinusSrcAlpha: return D3D12_BLEND_INV_SRC_ALPHA;
            case VriBlendFactor_DstAlpha:         return D3D12_BLEND_DEST_ALPHA;
            case VriBlendFactor_OneMinusDstAlpha: return D3D12_BLEND_INV_DEST_ALPHA;
            default:                              return D3D12_BLEND_ONE;
        }
    }

    inline D3D12_TEXTURE_ADDRESS_MODE ToD3DAddress(VriAddressMode m)
    {
        switch (m)
        {
            case VriAddressMode_MirroredRepeat: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
            case VriAddressMode_ClampToEdge:    return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            case VriAddressMode_ClampToBorder:  return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            default:                            return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        }
    }

    inline D3D12_SAMPLER_DESC ToD3DSampler(const VriSamplerDesc& s)
    {
        D3D12_SAMPLER_DESC d = {};
        const bool linear = s.magFilter == VriFilter_Linear || s.minFilter == VriFilter_Linear;
        // A comparison sampler (shadow PCF) needs a COMPARISON filter variant + the compare func.
        if (s.compareEnable)
            d.Filter = linear ? D3D12_FILTER_COMPARISON_MIN_MAG_MIP_LINEAR : D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
        else
            d.Filter = linear ? D3D12_FILTER_MIN_MAG_MIP_LINEAR : D3D12_FILTER_MIN_MAG_MIP_POINT;
        d.AddressU = ToD3DAddress(s.addressModeU); d.AddressV = ToD3DAddress(s.addressModeV); d.AddressW = ToD3DAddress(s.addressModeW);
        d.MipLODBias = s.mipLodBias; d.MaxAnisotropy = 1;
        d.ComparisonFunc = s.compareEnable ? ToD3DCompare(s.compareOp) : D3D12_COMPARISON_FUNC_ALWAYS;
        d.MinLOD = s.minLod; d.MaxLOD = s.maxLod > 0.0f ? s.maxLod : D3D12_FLOAT32_MAX;
        // D3D12 border color is always an arbitrary float4 (native custom support).
        if (s.useCustomBorderColor)
            for (int i = 0; i < 4; ++i) d.BorderColor[i] = s.customBorderColor[i];
        else if (s.borderColor == VriBorderColor_FloatOpaqueWhite || s.borderColor == VriBorderColor_IntOpaqueWhite)
        { d.BorderColor[0] = d.BorderColor[1] = d.BorderColor[2] = d.BorderColor[3] = 1.0f; }
        else if (s.borderColor == VriBorderColor_FloatOpaqueBlack || s.borderColor == VriBorderColor_IntOpaqueBlack)
        { d.BorderColor[3] = 1.0f; } // opaque black (rgb 0, a 1)
        return d;
    }

    inline D3D12_BLEND_OP ToD3DBlendOp(VriBlendOp op)
    {
        switch (op)
        {
            case VriBlendOp_Subtract:        return D3D12_BLEND_OP_SUBTRACT;
            case VriBlendOp_ReverseSubtract: return D3D12_BLEND_OP_REV_SUBTRACT;
            case VriBlendOp_Min:             return D3D12_BLEND_OP_MIN;
            case VriBlendOp_Max:             return D3D12_BLEND_OP_MAX;
            default:                         return D3D12_BLEND_OP_ADD;
        }
    }
} // namespace vri::d3d12
