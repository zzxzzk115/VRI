// conversions_wgpu.h - VRI enum -> WebGPU enum conversions (triangle subset).
#pragma once

#include <webgpu/webgpu.h>

#include <vri/vri.h>

namespace vri::wgpu
{
    inline WGPUTextureFormat ToWgpuFormat(VriFormat f)
    {
        switch (f)
        {
            case VriFormat_R8_UNORM:        return WGPUTextureFormat_R8Unorm;
            case VriFormat_R8_UINT:         return WGPUTextureFormat_R8Uint;
            case VriFormat_RG8_UNORM:       return WGPUTextureFormat_RG8Unorm;
            case VriFormat_RGBA8_UNORM:     return WGPUTextureFormat_RGBA8Unorm;
            case VriFormat_RGBA8_SRGB:      return WGPUTextureFormat_RGBA8UnormSrgb;
            case VriFormat_BGRA8_UNORM:     return WGPUTextureFormat_BGRA8Unorm;
            case VriFormat_BGRA8_SRGB:      return WGPUTextureFormat_BGRA8UnormSrgb;
            case VriFormat_R16_SFLOAT:      return WGPUTextureFormat_R16Float;
            case VriFormat_RG16_SFLOAT:     return WGPUTextureFormat_RG16Float;
            case VriFormat_RGBA16_SFLOAT:   return WGPUTextureFormat_RGBA16Float;
            case VriFormat_R32_UINT:        return WGPUTextureFormat_R32Uint;
            case VriFormat_R32_SFLOAT:      return WGPUTextureFormat_R32Float;
            case VriFormat_RG32_SFLOAT:     return WGPUTextureFormat_RG32Float;
            case VriFormat_RGBA32_SFLOAT:   return WGPUTextureFormat_RGBA32Float;
            case VriFormat_RGB10A2_UNORM:   return WGPUTextureFormat_RGB10A2Unorm;
            case VriFormat_D16_UNORM:       return WGPUTextureFormat_Depth16Unorm;
            case VriFormat_D32_SFLOAT:      return WGPUTextureFormat_Depth32Float;
            case VriFormat_D24_UNORM_S8_UINT:  return WGPUTextureFormat_Depth24PlusStencil8;
            case VriFormat_D32_SFLOAT_S8_UINT: return WGPUTextureFormat_Depth32FloatStencil8;
            default:                        return WGPUTextureFormat_Undefined;
        }
    }

    inline WGPUTextureUsage ToWgpuTextureUsage(VriTextureUsageFlags u)
    {
        WGPUTextureUsage r = WGPUTextureUsage_None;
        if (u & VriTextureUsage_TransferSrc)            r |= WGPUTextureUsage_CopySrc;
        if (u & VriTextureUsage_TransferDst)            r |= WGPUTextureUsage_CopyDst;
        if (u & VriTextureUsage_ShaderResource)         r |= WGPUTextureUsage_TextureBinding;
        if (u & VriTextureUsage_ShaderResourceStorage)  r |= WGPUTextureUsage_StorageBinding;
        if (u & VriTextureUsage_ColorAttachment)        r |= WGPUTextureUsage_RenderAttachment;
        if (u & VriTextureUsage_DepthStencilAttachment) r |= WGPUTextureUsage_RenderAttachment;
        return r;
    }

    inline WGPUBufferUsage ToWgpuBufferUsage(VriBufferUsageFlags u)
    {
        WGPUBufferUsage r = WGPUBufferUsage_None;
        if (u & VriBufferUsage_TransferSrc)    r |= WGPUBufferUsage_CopySrc;
        if (u & VriBufferUsage_TransferDst)    r |= WGPUBufferUsage_CopyDst;
        if (u & VriBufferUsage_VertexBuffer)   r |= WGPUBufferUsage_Vertex;
        if (u & VriBufferUsage_IndexBuffer)    r |= WGPUBufferUsage_Index;
        if (u & VriBufferUsage_ConstantBuffer) r |= WGPUBufferUsage_Uniform;
        if (u & VriBufferUsage_StorageBuffer)  r |= WGPUBufferUsage_Storage;
        if (u & VriBufferUsage_IndirectBuffer) r |= WGPUBufferUsage_Indirect;
        return r;
    }

    inline WGPUPrimitiveTopology ToWgpuTopology(VriPrimitiveTopology t)
    {
        switch (t)
        {
            case VriPrimitiveTopology_PointList:     return WGPUPrimitiveTopology_PointList;
            case VriPrimitiveTopology_LineList:      return WGPUPrimitiveTopology_LineList;
            case VriPrimitiveTopology_LineStrip:     return WGPUPrimitiveTopology_LineStrip;
            case VriPrimitiveTopology_TriangleStrip: return WGPUPrimitiveTopology_TriangleStrip;
            default:                                 return WGPUPrimitiveTopology_TriangleList;
        }
    }

    inline WGPUCullMode ToWgpuCullMode(VriCullMode m)
    {
        switch (m)
        {
            case VriCullMode_Front: return WGPUCullMode_Front;
            case VriCullMode_Back:  return WGPUCullMode_Back;
            default:                return WGPUCullMode_None;
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
            case VriCompareOp_Never:        return WGPUCompareFunction_Never;
            case VriCompareOp_Less:         return WGPUCompareFunction_Less;
            case VriCompareOp_Equal:        return WGPUCompareFunction_Equal;
            case VriCompareOp_LessOrEqual:    return WGPUCompareFunction_LessEqual;
            case VriCompareOp_Greater:        return WGPUCompareFunction_Greater;
            case VriCompareOp_NotEqual:       return WGPUCompareFunction_NotEqual;
            case VriCompareOp_GreaterOrEqual: return WGPUCompareFunction_GreaterEqual;
            case VriCompareOp_Always:       return WGPUCompareFunction_Always;
            default:                        return WGPUCompareFunction_Always;
        }
    }

    inline WGPUStencilOperation ToWgpuStencilOp(VriStencilOp o)
    {
        switch (o)
        {
            case VriStencilOp_Keep:              return WGPUStencilOperation_Keep;
            case VriStencilOp_Zero:              return WGPUStencilOperation_Zero;
            case VriStencilOp_Replace:           return WGPUStencilOperation_Replace;
            case VriStencilOp_IncrementAndClamp: return WGPUStencilOperation_IncrementClamp;
            case VriStencilOp_DecrementAndClamp: return WGPUStencilOperation_DecrementClamp;
            case VriStencilOp_Invert:            return WGPUStencilOperation_Invert;
            case VriStencilOp_IncrementAndWrap:  return WGPUStencilOperation_IncrementWrap;
            case VriStencilOp_DecrementAndWrap:  return WGPUStencilOperation_DecrementWrap;
            default:                             return WGPUStencilOperation_Keep;
        }
    }

    inline WGPUVertexFormat ToWgpuVertexFormat(VriFormat f)
    {
        switch (f)
        {
            case VriFormat_R32_SFLOAT:    return WGPUVertexFormat_Float32;
            case VriFormat_RG32_SFLOAT:   return WGPUVertexFormat_Float32x2;
            case VriFormat_RGB32_SFLOAT:  return WGPUVertexFormat_Float32x3;
            case VriFormat_RGBA32_SFLOAT: return WGPUVertexFormat_Float32x4;
            case VriFormat_RGBA8_UNORM:   return WGPUVertexFormat_Unorm8x4;
            case VriFormat_R32_UINT:      return WGPUVertexFormat_Uint32;
            case VriFormat_R32_SINT:      return WGPUVertexFormat_Sint32;
            case VriFormat_RG32_UINT:     return WGPUVertexFormat_Uint32x2;
            case VriFormat_RG32_SINT:     return WGPUVertexFormat_Sint32x2;
            case VriFormat_RGBA32_UINT:   return WGPUVertexFormat_Uint32x4;
            case VriFormat_RGBA32_SINT:   return WGPUVertexFormat_Sint32x4;
            default:                      return WGPUVertexFormat_Float32x4;
        }
    }

    inline WGPUBlendFactor ToWgpuBlendFactor(VriBlendFactor f)
    {
        switch (f)
        {
            case VriBlendFactor_Zero:              return WGPUBlendFactor_Zero;
            case VriBlendFactor_One:               return WGPUBlendFactor_One;
            case VriBlendFactor_SrcColor:          return WGPUBlendFactor_Src;
            case VriBlendFactor_OneMinusSrcColor:  return WGPUBlendFactor_OneMinusSrc;
            case VriBlendFactor_DstColor:          return WGPUBlendFactor_Dst;
            case VriBlendFactor_OneMinusDstColor:  return WGPUBlendFactor_OneMinusDst;
            case VriBlendFactor_SrcAlpha:          return WGPUBlendFactor_SrcAlpha;
            case VriBlendFactor_OneMinusSrcAlpha:  return WGPUBlendFactor_OneMinusSrcAlpha;
            case VriBlendFactor_DstAlpha:          return WGPUBlendFactor_DstAlpha;
            case VriBlendFactor_OneMinusDstAlpha:  return WGPUBlendFactor_OneMinusDstAlpha;
            default:                               return WGPUBlendFactor_Zero;
        }
    }

    inline WGPUBlendOperation ToWgpuBlendOp(VriBlendOp o)
    {
        switch (o)
        {
            case VriBlendOp_Subtract:        return WGPUBlendOperation_Subtract;
            case VriBlendOp_ReverseSubtract: return WGPUBlendOperation_ReverseSubtract;
            case VriBlendOp_Min:             return WGPUBlendOperation_Min;
            case VriBlendOp_Max:             return WGPUBlendOperation_Max;
            default:                         return WGPUBlendOperation_Add;
        }
    }
} // namespace vri::wgpu
