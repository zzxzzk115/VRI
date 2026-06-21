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
