// conversions_wgpu.h - VRI enum -> WebGPU enum conversions (triangle subset).
#pragma once

#include <webgpu/webgpu.h>

#include <vri/vri.h>

#include "core/conversion_map.h"
#include "core/format_traits.h"

namespace vri::wgpu
{
    inline WGPUTextureFormat ToWgpuFormat(VriFormat f)
    {
        static constexpr vri::ConvRow<VriFormat, WGPUTextureFormat> kTable[] = {
            {VriFormat_R8_UNORM, WGPUTextureFormat_R8Unorm},
            {VriFormat_R8_UINT, WGPUTextureFormat_R8Uint},
            {VriFormat_RG8_UNORM, WGPUTextureFormat_RG8Unorm},
            {VriFormat_RGBA8_UNORM, WGPUTextureFormat_RGBA8Unorm},
            {VriFormat_RGBA8_SRGB, WGPUTextureFormat_RGBA8UnormSrgb},
            {VriFormat_BGRA8_UNORM, WGPUTextureFormat_BGRA8Unorm},
            {VriFormat_BGRA8_SRGB, WGPUTextureFormat_BGRA8UnormSrgb},
            {VriFormat_R16_SFLOAT, WGPUTextureFormat_R16Float},
            {VriFormat_RG16_SFLOAT, WGPUTextureFormat_RG16Float},
            {VriFormat_RGBA16_SFLOAT, WGPUTextureFormat_RGBA16Float},
            {VriFormat_R32_UINT, WGPUTextureFormat_R32Uint},
            {VriFormat_R32_SFLOAT, WGPUTextureFormat_R32Float},
            {VriFormat_RG32_SFLOAT, WGPUTextureFormat_RG32Float},
            {VriFormat_RGBA32_SFLOAT, WGPUTextureFormat_RGBA32Float},
            {VriFormat_RGB10A2_UNORM, WGPUTextureFormat_RGB10A2Unorm},
            {VriFormat_D16_UNORM, WGPUTextureFormat_Depth16Unorm},
            {VriFormat_D32_SFLOAT, WGPUTextureFormat_Depth32Float},
            {VriFormat_D24_UNORM_S8_UINT, WGPUTextureFormat_Depth24PlusStencil8},
            {VriFormat_D32_SFLOAT_S8_UINT, WGPUTextureFormat_Depth32FloatStencil8},
        };
        return vri::MapOr(f, kTable, WGPUTextureFormat_Undefined);
    }

    inline WGPUTextureUsage ToWgpuTextureUsage(VriTextureUsageFlags u)
    {
        static constexpr vri::ConvRow<VriTextureUsageFlags, WGPUTextureUsage> kTable[] = {
            {VriTextureUsage_TransferSrc, WGPUTextureUsage_CopySrc},
            {VriTextureUsage_TransferDst, WGPUTextureUsage_CopyDst},
            {VriTextureUsage_ShaderResource, WGPUTextureUsage_TextureBinding},
            {VriTextureUsage_ShaderResourceStorage, WGPUTextureUsage_StorageBinding},
            {VriTextureUsage_ColorAttachment, WGPUTextureUsage_RenderAttachment},
            {VriTextureUsage_DepthStencilAttachment, WGPUTextureUsage_RenderAttachment},
        };
        // No explicit seed: To{} is 0 == WGPUTextureUsage_None, matching the old r init.
        return vri::MapFlags(u, kTable);
    }

    inline WGPUBufferUsage ToWgpuBufferUsage(VriBufferUsageFlags u)
    {
        static constexpr vri::ConvRow<VriBufferUsageFlags, WGPUBufferUsage> kTable[] = {
            {VriBufferUsage_TransferSrc, WGPUBufferUsage_CopySrc},
            {VriBufferUsage_TransferDst, WGPUBufferUsage_CopyDst},
            {VriBufferUsage_VertexBuffer, WGPUBufferUsage_Vertex},
            {VriBufferUsage_IndexBuffer, WGPUBufferUsage_Index},
            {VriBufferUsage_ConstantBuffer, WGPUBufferUsage_Uniform},
            {VriBufferUsage_StorageBuffer, WGPUBufferUsage_Storage},
            {VriBufferUsage_IndirectBuffer, WGPUBufferUsage_Indirect},
        };
        // No explicit seed: To{} is 0 == WGPUBufferUsage_None, matching the old r init.
        return vri::MapFlags(u, kTable);
    }

    inline WGPUPrimitiveTopology ToWgpuTopology(VriPrimitiveTopology t)
    {
        switch (t)
        {
            case VriPrimitiveTopology_PointList:
                return WGPUPrimitiveTopology_PointList;
            case VriPrimitiveTopology_LineList:
                return WGPUPrimitiveTopology_LineList;
            case VriPrimitiveTopology_LineStrip:
                return WGPUPrimitiveTopology_LineStrip;
            case VriPrimitiveTopology_TriangleStrip:
                return WGPUPrimitiveTopology_TriangleStrip;
            default:
                return WGPUPrimitiveTopology_TriangleList;
        }
    }

    inline WGPUCullMode ToWgpuCullMode(VriCullMode m)
    {
        switch (m)
        {
            case VriCullMode_Front:
                return WGPUCullMode_Front;
            case VriCullMode_Back:
                return WGPUCullMode_Back;
            default:
                return WGPUCullMode_None;
        }
    }

    inline WGPUFrontFace ToWgpuFrontFace(VriFrontFace f)
    {
        return f == VriFrontFace_Clockwise ? WGPUFrontFace_CW : WGPUFrontFace_CCW;
    }

    inline WGPUCompareFunction ToWgpuCompareOp(VriCompareOp o)
    {
        switch (o)
        {
            case VriCompareOp_Never:
                return WGPUCompareFunction_Never;
            case VriCompareOp_Less:
                return WGPUCompareFunction_Less;
            case VriCompareOp_Equal:
                return WGPUCompareFunction_Equal;
            case VriCompareOp_LessOrEqual:
                return WGPUCompareFunction_LessEqual;
            case VriCompareOp_Greater:
                return WGPUCompareFunction_Greater;
            case VriCompareOp_NotEqual:
                return WGPUCompareFunction_NotEqual;
            case VriCompareOp_GreaterOrEqual:
                return WGPUCompareFunction_GreaterEqual;
            case VriCompareOp_Always:
                return WGPUCompareFunction_Always;
            default:
                return WGPUCompareFunction_Always;
        }
    }

    inline WGPUStencilOperation ToWgpuStencilOp(VriStencilOp o)
    {
        switch (o)
        {
            case VriStencilOp_Keep:
                return WGPUStencilOperation_Keep;
            case VriStencilOp_Zero:
                return WGPUStencilOperation_Zero;
            case VriStencilOp_Replace:
                return WGPUStencilOperation_Replace;
            case VriStencilOp_IncrementAndClamp:
                return WGPUStencilOperation_IncrementClamp;
            case VriStencilOp_DecrementAndClamp:
                return WGPUStencilOperation_DecrementClamp;
            case VriStencilOp_Invert:
                return WGPUStencilOperation_Invert;
            case VriStencilOp_IncrementAndWrap:
                return WGPUStencilOperation_IncrementWrap;
            case VriStencilOp_DecrementAndWrap:
                return WGPUStencilOperation_DecrementWrap;
            default:
                return WGPUStencilOperation_Keep;
        }
    }

    inline constexpr vri::ConvRow<VriFormat, WGPUVertexFormat> kWgpuVertexFormatTable[] = {
        {VriFormat_R32_SFLOAT, WGPUVertexFormat_Float32},
        {VriFormat_RG32_SFLOAT, WGPUVertexFormat_Float32x2},
        {VriFormat_RGB32_SFLOAT, WGPUVertexFormat_Float32x3},
        {VriFormat_RGBA32_SFLOAT, WGPUVertexFormat_Float32x4},
        {VriFormat_RGBA8_UNORM, WGPUVertexFormat_Unorm8x4},
        {VriFormat_R32_UINT, WGPUVertexFormat_Uint32},
        {VriFormat_R32_SINT, WGPUVertexFormat_Sint32},
        {VriFormat_RG32_UINT, WGPUVertexFormat_Uint32x2},
        {VriFormat_RG32_SINT, WGPUVertexFormat_Sint32x2},
        {VriFormat_RGBA32_UINT, WGPUVertexFormat_Uint32x4},
        {VriFormat_RGBA32_SINT, WGPUVertexFormat_Sint32x4},
    };

    // Look `f` up WITHOUT ToWgpuVertexFormat's Float32x4 fallback: Undefined
    // means the backend has no vertex format for it. GetFormatSupport needs the
    // difference so it does not advertise a vertex format that would silently
    // reach the shader as a float4.
    inline WGPUVertexFormat ToWgpuVertexFormatOrNone(VriFormat f)
    {
        return vri::MapOr(f, kWgpuVertexFormatTable, WGPUVertexFormat_Undefined);
    }

    inline WGPUVertexFormat ToWgpuVertexFormat(VriFormat f)
    {
        const WGPUVertexFormat mapped = ToWgpuVertexFormatOrNone(f);
        return mapped != WGPUVertexFormat_Undefined ? mapped : WGPUVertexFormat_Float32x4;
    }

    inline WGPUBlendFactor ToWgpuBlendFactor(VriBlendFactor f)
    {
        switch (f)
        {
            case VriBlendFactor_Zero:
                return WGPUBlendFactor_Zero;
            case VriBlendFactor_One:
                return WGPUBlendFactor_One;
            case VriBlendFactor_SrcColor:
                return WGPUBlendFactor_Src;
            case VriBlendFactor_OneMinusSrcColor:
                return WGPUBlendFactor_OneMinusSrc;
            case VriBlendFactor_DstColor:
                return WGPUBlendFactor_Dst;
            case VriBlendFactor_OneMinusDstColor:
                return WGPUBlendFactor_OneMinusDst;
            case VriBlendFactor_SrcAlpha:
                return WGPUBlendFactor_SrcAlpha;
            case VriBlendFactor_OneMinusSrcAlpha:
                return WGPUBlendFactor_OneMinusSrcAlpha;
            case VriBlendFactor_DstAlpha:
                return WGPUBlendFactor_DstAlpha;
            case VriBlendFactor_OneMinusDstAlpha:
                return WGPUBlendFactor_OneMinusDstAlpha;
            default:
                return WGPUBlendFactor_Zero;
        }
    }

    inline WGPUBlendOperation ToWgpuBlendOp(VriBlendOp o)
    {
        switch (o)
        {
            case VriBlendOp_Subtract:
                return WGPUBlendOperation_Subtract;
            case VriBlendOp_ReverseSubtract:
                return WGPUBlendOperation_ReverseSubtract;
            case VriBlendOp_Min:
                return WGPUBlendOperation_Min;
            case VriBlendOp_Max:
                return WGPUBlendOperation_Max;
            default:
                return WGPUBlendOperation_Add;
        }
    }
} // namespace vri::wgpu
