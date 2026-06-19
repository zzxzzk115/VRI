// conversions_gl.h - VRI enum -> OpenGL enum conversions (triangle subset).
#pragma once

#include <glad/glad.h>

#include <vri/vri.h>

namespace vri::gl
{
    struct GLFormat
    {
        GLenum internalFormat;
        GLenum format;
        GLenum type;
        uint32_t texelSize;
    };

    inline GLFormat ToGLFormat(VriFormat f)
    {
        switch (f)
        {
            case VriFormat_R8_UNORM:      return {GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1};
            case VriFormat_RG8_UNORM:     return {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, 2};
            case VriFormat_RGBA8_UNORM:   return {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4};
            case VriFormat_RGBA8_SRGB:    return {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, 4};
            case VriFormat_BGRA8_UNORM:   return {GL_RGBA8, GL_BGRA, GL_UNSIGNED_BYTE, 4};
            case VriFormat_R16_SFLOAT:    return {GL_R16F, GL_RED, GL_HALF_FLOAT, 2};
            case VriFormat_RG16_SFLOAT:   return {GL_RG16F, GL_RG, GL_HALF_FLOAT, 4};
            case VriFormat_RGBA16_SFLOAT: return {GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, 8};
            case VriFormat_R32_SFLOAT:    return {GL_R32F, GL_RED, GL_FLOAT, 4};
            case VriFormat_RG32_SFLOAT:   return {GL_RG32F, GL_RG, GL_FLOAT, 8};
            case VriFormat_RGBA32_SFLOAT: return {GL_RGBA32F, GL_RGBA, GL_FLOAT, 16};
            case VriFormat_D32_SFLOAT:    return {GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, 4};
            case VriFormat_D24_UNORM_S8_UINT: return {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, 4};
            default:                      return {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4};
        }
    }

    inline GLenum ToGLTopology(VriPrimitiveTopology t)
    {
        switch (t)
        {
            case VriPrimitiveTopology_PointList:     return GL_POINTS;
            case VriPrimitiveTopology_LineList:      return GL_LINES;
            case VriPrimitiveTopology_LineStrip:     return GL_LINE_STRIP;
            case VriPrimitiveTopology_TriangleStrip: return GL_TRIANGLE_STRIP;
            default:                                 return GL_TRIANGLES;
        }
    }

    inline GLenum ToGLCompareOp(VriCompareOp o)
    {
        switch (o)
        {
            case VriCompareOp_Never:          return GL_NEVER;
            case VriCompareOp_Less:           return GL_LESS;
            case VriCompareOp_Equal:          return GL_EQUAL;
            case VriCompareOp_LessOrEqual:    return GL_LEQUAL;
            case VriCompareOp_Greater:        return GL_GREATER;
            case VriCompareOp_NotEqual:       return GL_NOTEQUAL;
            case VriCompareOp_GreaterOrEqual: return GL_GEQUAL;
            default:                          return GL_ALWAYS;
        }
    }

    inline GLenum ToGLBlendFactor(VriBlendFactor f)
    {
        switch (f)
        {
            case VriBlendFactor_Zero:             return GL_ZERO;
            case VriBlendFactor_One:              return GL_ONE;
            case VriBlendFactor_SrcColor:         return GL_SRC_COLOR;
            case VriBlendFactor_OneMinusSrcColor: return GL_ONE_MINUS_SRC_COLOR;
            case VriBlendFactor_DstColor:         return GL_DST_COLOR;
            case VriBlendFactor_OneMinusDstColor: return GL_ONE_MINUS_DST_COLOR;
            case VriBlendFactor_SrcAlpha:         return GL_SRC_ALPHA;
            case VriBlendFactor_OneMinusSrcAlpha: return GL_ONE_MINUS_SRC_ALPHA;
            case VriBlendFactor_DstAlpha:         return GL_DST_ALPHA;
            case VriBlendFactor_OneMinusDstAlpha: return GL_ONE_MINUS_DST_ALPHA;
            default:                              return GL_ZERO;
        }
    }

    inline GLenum ToGLBlendOp(VriBlendOp o)
    {
        switch (o)
        {
            case VriBlendOp_Subtract:        return GL_FUNC_SUBTRACT;
            case VriBlendOp_ReverseSubtract: return GL_FUNC_REVERSE_SUBTRACT;
            case VriBlendOp_Min:             return GL_MIN;
            case VriBlendOp_Max:             return GL_MAX;
            default:                         return GL_FUNC_ADD;
        }
    }
} // namespace vri::gl
