// ConversionsVK.h - VRI enum -> Vulkan enum conversions (Phase 1 subset).
#pragma once

#include <vulkan/vulkan.h>

#include <vri/vri.h>

namespace vri::vk
{
    inline VkFormat ToVkFormat(VriFormat f)
    {
        switch (f)
        {
            case VriFormat_R8_UNORM:        return VK_FORMAT_R8_UNORM;
            case VriFormat_R8_SNORM:        return VK_FORMAT_R8_SNORM;
            case VriFormat_R8_UINT:         return VK_FORMAT_R8_UINT;
            case VriFormat_R8_SINT:         return VK_FORMAT_R8_SINT;
            case VriFormat_RG8_UNORM:       return VK_FORMAT_R8G8_UNORM;
            case VriFormat_RG8_SNORM:       return VK_FORMAT_R8G8_SNORM;
            case VriFormat_RG8_UINT:        return VK_FORMAT_R8G8_UINT;
            case VriFormat_RG8_SINT:        return VK_FORMAT_R8G8_SINT;
            case VriFormat_RGBA8_UNORM:     return VK_FORMAT_R8G8B8A8_UNORM;
            case VriFormat_RGBA8_SRGB:      return VK_FORMAT_R8G8B8A8_SRGB;
            case VriFormat_RGBA8_UINT:      return VK_FORMAT_R8G8B8A8_UINT;
            case VriFormat_RGBA8_SINT:      return VK_FORMAT_R8G8B8A8_SINT;
            case VriFormat_BGRA8_UNORM:     return VK_FORMAT_B8G8R8A8_UNORM;
            case VriFormat_BGRA8_SRGB:      return VK_FORMAT_B8G8R8A8_SRGB;
            case VriFormat_R16_UNORM:       return VK_FORMAT_R16_UNORM;
            case VriFormat_R16_SNORM:       return VK_FORMAT_R16_SNORM;
            case VriFormat_R16_UINT:        return VK_FORMAT_R16_UINT;
            case VriFormat_R16_SINT:        return VK_FORMAT_R16_SINT;
            case VriFormat_R16_SFLOAT:      return VK_FORMAT_R16_SFLOAT;
            case VriFormat_RG16_UNORM:      return VK_FORMAT_R16G16_UNORM;
            case VriFormat_RG16_SNORM:      return VK_FORMAT_R16G16_SNORM;
            case VriFormat_RG16_UINT:       return VK_FORMAT_R16G16_UINT;
            case VriFormat_RG16_SINT:       return VK_FORMAT_R16G16_SINT;
            case VriFormat_RG16_SFLOAT:     return VK_FORMAT_R16G16_SFLOAT;
            case VriFormat_RGBA16_UNORM:    return VK_FORMAT_R16G16B16A16_UNORM;
            case VriFormat_RGBA16_SNORM:    return VK_FORMAT_R16G16B16A16_SNORM;
            case VriFormat_RGBA16_UINT:     return VK_FORMAT_R16G16B16A16_UINT;
            case VriFormat_RGBA16_SINT:     return VK_FORMAT_R16G16B16A16_SINT;
            case VriFormat_RGBA16_SFLOAT:   return VK_FORMAT_R16G16B16A16_SFLOAT;
            case VriFormat_R32_UINT:        return VK_FORMAT_R32_UINT;
            case VriFormat_R32_SINT:        return VK_FORMAT_R32_SINT;
            case VriFormat_R32_SFLOAT:      return VK_FORMAT_R32_SFLOAT;
            case VriFormat_RG32_UINT:       return VK_FORMAT_R32G32_UINT;
            case VriFormat_RG32_SINT:       return VK_FORMAT_R32G32_SINT;
            case VriFormat_RG32_SFLOAT:     return VK_FORMAT_R32G32_SFLOAT;
            case VriFormat_RGB32_UINT:      return VK_FORMAT_R32G32B32_UINT;
            case VriFormat_RGB32_SINT:      return VK_FORMAT_R32G32B32_SINT;
            case VriFormat_RGB32_SFLOAT:    return VK_FORMAT_R32G32B32_SFLOAT;
            case VriFormat_RGBA32_UINT:     return VK_FORMAT_R32G32B32A32_UINT;
            case VriFormat_RGBA32_SINT:     return VK_FORMAT_R32G32B32A32_SINT;
            case VriFormat_RGBA32_SFLOAT:   return VK_FORMAT_R32G32B32A32_SFLOAT;
            case VriFormat_RGB10A2_UNORM:   return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
            case VriFormat_RG11B10_UFLOAT:  return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
            case VriFormat_BC1_UNORM:       return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
            case VriFormat_BC2_UNORM:       return VK_FORMAT_BC2_UNORM_BLOCK;
            case VriFormat_BC3_UNORM:       return VK_FORMAT_BC3_UNORM_BLOCK;
            case VriFormat_BC4_UNORM:       return VK_FORMAT_BC4_UNORM_BLOCK;
            case VriFormat_BC5_UNORM:       return VK_FORMAT_BC5_UNORM_BLOCK;
            case VriFormat_BC6H_UFLOAT:     return VK_FORMAT_BC6H_UFLOAT_BLOCK;
            case VriFormat_BC7_UNORM:       return VK_FORMAT_BC7_UNORM_BLOCK;
            case VriFormat_D16_UNORM:       return VK_FORMAT_D16_UNORM;
            case VriFormat_D32_SFLOAT:      return VK_FORMAT_D32_SFLOAT;
            case VriFormat_S8_UINT:         return VK_FORMAT_S8_UINT;
            case VriFormat_D24_UNORM_S8_UINT:   return VK_FORMAT_D24_UNORM_S8_UINT;
            case VriFormat_D32_SFLOAT_S8_UINT:  return VK_FORMAT_D32_SFLOAT_S8_UINT;
            default:                        return VK_FORMAT_UNDEFINED;
        }
    }

    inline VkImageType ToVkImageType(VriTextureType t)
    {
        switch (t)
        {
            case VriTextureType_1D:
            case VriTextureType_1DArray: return VK_IMAGE_TYPE_1D;
            case VriTextureType_3D:      return VK_IMAGE_TYPE_3D;
            default:                     return VK_IMAGE_TYPE_2D;
        }
    }

    inline VkImageViewType ToVkImageViewType(VriTextureViewType t)
    {
        switch (t)
        {
            case VriTextureViewType_1D:        return VK_IMAGE_VIEW_TYPE_1D;
            case VriTextureViewType_1DArray:   return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
            case VriTextureViewType_2DArray:   return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            case VriTextureViewType_3D:        return VK_IMAGE_VIEW_TYPE_3D;
            case VriTextureViewType_Cube:      return VK_IMAGE_VIEW_TYPE_CUBE;
            case VriTextureViewType_CubeArray: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
            default:                           return VK_IMAGE_VIEW_TYPE_2D;
        }
    }

    inline VkImageAspectFlags ToVkAspect(VriImageAspectFlags a)
    {
        VkImageAspectFlags r = 0;
        if (a & VriImageAspect_Color)   r |= VK_IMAGE_ASPECT_COLOR_BIT;
        if (a & VriImageAspect_Depth)   r |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (a & VriImageAspect_Stencil) r |= VK_IMAGE_ASPECT_STENCIL_BIT;
        return r;
    }

    inline VkBufferUsageFlags ToVkBufferUsage(VriBufferUsageFlags u)
    {
        VkBufferUsageFlags r = 0;
        if (u & VriBufferUsage_TransferSrc)    r |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (u & VriBufferUsage_TransferDst)    r |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (u & VriBufferUsage_VertexBuffer)   r |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        if (u & VriBufferUsage_IndexBuffer)    r |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        if (u & VriBufferUsage_ConstantBuffer) r |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (u & VriBufferUsage_StorageBuffer)  r |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (u & VriBufferUsage_IndirectBuffer) r |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        if (u & VriBufferUsage_ShaderDeviceAddress) r |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        return r;
    }

    inline VkImageUsageFlags ToVkImageUsage(VriTextureUsageFlags u)
    {
        VkImageUsageFlags r = 0;
        if (u & VriTextureUsage_TransferSrc)            r |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (u & VriTextureUsage_TransferDst)            r |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (u & VriTextureUsage_ShaderResource)         r |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if (u & VriTextureUsage_ShaderResourceStorage)  r |= VK_IMAGE_USAGE_STORAGE_BIT;
        if (u & VriTextureUsage_ColorAttachment)        r |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (u & VriTextureUsage_DepthStencilAttachment) r |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        return r;
    }

    inline VkPrimitiveTopology ToVkTopology(VriPrimitiveTopology t)
    {
        switch (t)
        {
            case VriPrimitiveTopology_PointList:     return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
            case VriPrimitiveTopology_LineList:      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            case VriPrimitiveTopology_LineStrip:     return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
            case VriPrimitiveTopology_TriangleStrip: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
            default:                                 return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        }
    }

    inline VkPolygonMode ToVkPolygonMode(VriPolygonMode m)
    {
        switch (m)
        {
            case VriPolygonMode_Line:  return VK_POLYGON_MODE_LINE;
            case VriPolygonMode_Point: return VK_POLYGON_MODE_POINT;
            default:                   return VK_POLYGON_MODE_FILL;
        }
    }

    inline VkCullModeFlags ToVkCullMode(VriCullMode m)
    {
        switch (m)
        {
            case VriCullMode_Front: return VK_CULL_MODE_FRONT_BIT;
            case VriCullMode_Back:  return VK_CULL_MODE_BACK_BIT;
            default:                return VK_CULL_MODE_NONE;
        }
    }

    inline VkCompareOp ToVkCompareOp(VriCompareOp o)
    {
        switch (o)
        {
            case VriCompareOp_Never:          return VK_COMPARE_OP_NEVER;
            case VriCompareOp_Less:           return VK_COMPARE_OP_LESS;
            case VriCompareOp_Equal:          return VK_COMPARE_OP_EQUAL;
            case VriCompareOp_LessOrEqual:    return VK_COMPARE_OP_LESS_OR_EQUAL;
            case VriCompareOp_Greater:        return VK_COMPARE_OP_GREATER;
            case VriCompareOp_NotEqual:       return VK_COMPARE_OP_NOT_EQUAL;
            case VriCompareOp_GreaterOrEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
            default:                          return VK_COMPARE_OP_ALWAYS;
        }
    }

    inline VkShaderStageFlagBits ToVkShaderStage(VriShaderStageBits s)
    {
        switch (s)
        {
            case VriShaderStage_Vertex:       return VK_SHADER_STAGE_VERTEX_BIT;
            case VriShaderStage_TessControl:  return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
            case VriShaderStage_TessEval:     return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
            case VriShaderStage_Geometry:     return VK_SHADER_STAGE_GEOMETRY_BIT;
            case VriShaderStage_Fragment:     return VK_SHADER_STAGE_FRAGMENT_BIT;
            case VriShaderStage_Compute:      return VK_SHADER_STAGE_COMPUTE_BIT;
            default:                          return VK_SHADER_STAGE_ALL;
        }
    }

    inline VkImageLayout ToVkLayout(VriLayout l)
    {
        switch (l)
        {
            case VriLayout_General:                 return VK_IMAGE_LAYOUT_GENERAL;
            case VriLayout_ColorAttachment:         return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            case VriLayout_DepthStencilAttachment:  return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            case VriLayout_DepthStencilReadOnly:    return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            case VriLayout_ShaderResource:          return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            case VriLayout_ShaderResourceStorage:   return VK_IMAGE_LAYOUT_GENERAL;
            case VriLayout_CopySource:              return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            case VriLayout_CopyDestination:         return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            case VriLayout_Present:                 return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            default:                                return VK_IMAGE_LAYOUT_UNDEFINED;
        }
    }

    inline VkPipelineStageFlags2 ToVkStages(VriPipelineStageFlags s)
    {
        if (s == VriPipelineStage_None)
            return VK_PIPELINE_STAGE_2_NONE;

        VkPipelineStageFlags2 r = 0;
        if (s & VriPipelineStage_DrawIndirect)          r |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        if (s & VriPipelineStage_VertexInput)           r |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
        if (s & VriPipelineStage_VertexShader)          r |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        if (s & VriPipelineStage_FragmentShader)        r |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        if (s & VriPipelineStage_EarlyFragmentTests)    r |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        if (s & VriPipelineStage_LateFragmentTests)     r |= VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        if (s & VriPipelineStage_ColorAttachmentOutput) r |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        if (s & VriPipelineStage_ComputeShader)         r |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        if (s & VriPipelineStage_Transfer)              r |= VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        if (s & VriPipelineStage_AllGraphics)           r |= VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
        if (s & VriPipelineStage_AllCommands)           r |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        return r;
    }

    inline VkFrontFace ToVkFrontFace(VriFrontFace f)
    {
        return f == VriFrontFace_Clockwise ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    }

    inline VkDescriptorType ToVkDescriptorType(VriDescriptorType t)
    {
        switch (t)
        {
            case VriDescriptorType_Sampler:               return VK_DESCRIPTOR_TYPE_SAMPLER;
            case VriDescriptorType_Texture:               return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            case VriDescriptorType_StorageTexture:        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            case VriDescriptorType_ConstantBuffer:        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            case VriDescriptorType_StructuredBuffer:      return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            case VriDescriptorType_StorageBuffer:         return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            case VriDescriptorType_AccelerationStructure: return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            default:                                      return VK_DESCRIPTOR_TYPE_SAMPLER;
        }
    }

    inline VkShaderStageFlags ToVkShaderStageFlags(VriShaderStageFlags s)
    {
        VkShaderStageFlags r = 0;
        if (s & VriShaderStage_Vertex)      r |= VK_SHADER_STAGE_VERTEX_BIT;
        if (s & VriShaderStage_TessControl) r |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        if (s & VriShaderStage_TessEval)    r |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        if (s & VriShaderStage_Geometry)    r |= VK_SHADER_STAGE_GEOMETRY_BIT;
        if (s & VriShaderStage_Fragment)    r |= VK_SHADER_STAGE_FRAGMENT_BIT;
        if (s & VriShaderStage_Compute)     r |= VK_SHADER_STAGE_COMPUTE_BIT;
        if (r == 0) r = VK_SHADER_STAGE_ALL;
        return r;
    }

    inline VkBlendFactor ToVkBlendFactor(VriBlendFactor f)
    {
        switch (f)
        {
            case VriBlendFactor_Zero:                  return VK_BLEND_FACTOR_ZERO;
            case VriBlendFactor_One:                   return VK_BLEND_FACTOR_ONE;
            case VriBlendFactor_SrcColor:              return VK_BLEND_FACTOR_SRC_COLOR;
            case VriBlendFactor_OneMinusSrcColor:      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
            case VriBlendFactor_DstColor:              return VK_BLEND_FACTOR_DST_COLOR;
            case VriBlendFactor_OneMinusDstColor:      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
            case VriBlendFactor_SrcAlpha:              return VK_BLEND_FACTOR_SRC_ALPHA;
            case VriBlendFactor_OneMinusSrcAlpha:      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            case VriBlendFactor_DstAlpha:              return VK_BLEND_FACTOR_DST_ALPHA;
            case VriBlendFactor_OneMinusDstAlpha:      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
            case VriBlendFactor_ConstantColor:         return VK_BLEND_FACTOR_CONSTANT_COLOR;
            case VriBlendFactor_OneMinusConstantColor: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
            case VriBlendFactor_ConstantAlpha:         return VK_BLEND_FACTOR_CONSTANT_ALPHA;
            case VriBlendFactor_OneMinusConstantAlpha: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
            case VriBlendFactor_SrcAlphaSaturate:      return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
            case VriBlendFactor_Src1Color:             return VK_BLEND_FACTOR_SRC1_COLOR;
            case VriBlendFactor_OneMinusSrc1Color:     return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
            case VriBlendFactor_Src1Alpha:             return VK_BLEND_FACTOR_SRC1_ALPHA;
            case VriBlendFactor_OneMinusSrc1Alpha:     return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
            default:                                   return VK_BLEND_FACTOR_ZERO;
        }
    }

    inline VkBlendOp ToVkBlendOp(VriBlendOp o)
    {
        switch (o)
        {
            case VriBlendOp_Subtract:        return VK_BLEND_OP_SUBTRACT;
            case VriBlendOp_ReverseSubtract: return VK_BLEND_OP_REVERSE_SUBTRACT;
            case VriBlendOp_Min:             return VK_BLEND_OP_MIN;
            case VriBlendOp_Max:             return VK_BLEND_OP_MAX;
            default:                         return VK_BLEND_OP_ADD;
        }
    }

    inline VkAccessFlags2 ToVkAccess(VriAccessFlags a)
    {
        VkAccessFlags2 r = 0;
        if (a & VriAccess_IndexBufferRead)            r |= VK_ACCESS_2_INDEX_READ_BIT;
        if (a & VriAccess_VertexBufferRead)           r |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
        if (a & VriAccess_IndirectBufferRead)         r |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        if (a & VriAccess_ConstantBufferRead)         r |= VK_ACCESS_2_UNIFORM_READ_BIT;
        if (a & VriAccess_ShaderResourceRead)         r |= VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        if (a & VriAccess_ShaderResourceStorageRead)  r |= VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        if (a & VriAccess_ShaderResourceStorageWrite) r |= VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        if (a & VriAccess_ColorAttachmentRead)        r |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
        if (a & VriAccess_ColorAttachmentWrite)       r |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        if (a & VriAccess_DepthStencilAttachmentRead) r |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        if (a & VriAccess_DepthStencilAttachmentWrite)r |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        if (a & VriAccess_CopySourceRead)             r |= VK_ACCESS_2_TRANSFER_READ_BIT;
        if (a & VriAccess_CopyDestinationWrite)       r |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
        return r;
    }
} // namespace vri::vk
