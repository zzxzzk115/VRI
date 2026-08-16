// conversions_gl.h - VRI enum -> OpenGL enum conversions (triangle subset).
#pragma once

#include "gl_loader.h"

#include <vri/vri.h>

#include "core/conversion_map.h"

namespace vri::gl
{
    struct GLFormat
    {
        GLenum   internalFormat;
        GLenum   format;
        GLenum   type;
        uint32_t texelSize;
    };

    inline constexpr vri::ConvRow<VriFormat, GLFormat> kGLFormatTable[] = {
        {VriFormat_R8_UNORM, {GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1}},
        {VriFormat_RG8_UNORM, {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, 2}},
        {VriFormat_RGBA8_UNORM, {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4}},
        {VriFormat_RGBA8_SRGB, {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, 4}},
        {VriFormat_BGRA8_UNORM, {GL_RGBA8, GL_BGRA, GL_UNSIGNED_BYTE, 4}},
        {VriFormat_R16_SFLOAT, {GL_R16F, GL_RED, GL_HALF_FLOAT, 2}},
        {VriFormat_RG16_SFLOAT, {GL_RG16F, GL_RG, GL_HALF_FLOAT, 4}},
        {VriFormat_RGBA16_SFLOAT, {GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, 8}},
        {VriFormat_R32_SFLOAT, {GL_R32F, GL_RED, GL_FLOAT, 4}},
        {VriFormat_RG32_SFLOAT, {GL_RG32F, GL_RG, GL_FLOAT, 8}},
        {VriFormat_RGBA32_SFLOAT, {GL_RGBA32F, GL_RGBA, GL_FLOAT, 16}},
        {VriFormat_D32_SFLOAT, {GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, 4}},
        {VriFormat_D24_UNORM_S8_UINT, {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, 4}},
    };

    // Look `f` up WITHOUT ToGLFormat's RGBA8 fallback: internalFormat == 0 means
    // the backend has no GL format for it. GetFormatSupport needs the difference
    // - "GL honors this format" is not the same as "ToGLFormat returned
    // something" - because the fallback would otherwise let a caller be told a
    // depth format is fine and then be handed an RGBA8 texture for it.
    inline GLFormat ToGLFormatOrNone(VriFormat f) { return vri::MapOr(f, kGLFormatTable, GLFormat {0, 0, 0, 0}); }

    inline GLFormat ToGLFormat(VriFormat f)
    {
        const GLFormat mapped = ToGLFormatOrNone(f);
        return mapped.internalFormat != 0 ? mapped : GLFormat {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4};
    }

    // Whether the GL format carries depth (and so belongs on a depth/stencil
    // attachment rather than a color one).
    inline bool IsGLDepthFormat(const GLFormat& g)
    {
        return g.format == GL_DEPTH_COMPONENT || g.format == GL_DEPTH_STENCIL;
    }

    // Whether the color format holds floats. Renderability of these is not
    // universal: ES/WebGL2 gate it behind EXT_color_buffer_float.
    inline bool IsGLFloatFormat(const GLFormat& g) { return g.type == GL_HALF_FLOAT || g.type == GL_FLOAT; }

    // Vertex-attribute format: component count + GL component type + normalized.
    struct GLVertexFormat
    {
        GLint     size; // components (1..4)
        GLenum    type; // GL_FLOAT / GL_UNSIGNED_BYTE / ...
        GLboolean normalized;
        bool      integer; // true -> bind with glVertexAttribIPointer (no float conversion)
    };

    inline constexpr vri::ConvRow<VriFormat, GLVertexFormat> kGLVertexFormatTable[] = {
        {VriFormat_R32_SFLOAT, {1, GL_FLOAT, GL_FALSE, false}},
        {VriFormat_RG32_SFLOAT, {2, GL_FLOAT, GL_FALSE, false}},
        {VriFormat_RGB32_SFLOAT, {3, GL_FLOAT, GL_FALSE, false}},
        {VriFormat_RGBA32_SFLOAT, {4, GL_FLOAT, GL_FALSE, false}},
        {VriFormat_RGBA8_UNORM, {4, GL_UNSIGNED_BYTE, GL_TRUE, false}},
        {VriFormat_RG8_UNORM, {2, GL_UNSIGNED_BYTE, GL_TRUE, false}},
        {VriFormat_R8_UNORM, {1, GL_UNSIGNED_BYTE, GL_TRUE, false}},
        {VriFormat_R32_UINT, {1, GL_UNSIGNED_INT, GL_FALSE, true}},
        {VriFormat_R32_SINT, {1, GL_INT, GL_FALSE, true}},
        {VriFormat_RG32_UINT, {2, GL_UNSIGNED_INT, GL_FALSE, true}},
        {VriFormat_RG32_SINT, {2, GL_INT, GL_FALSE, true}},
        {VriFormat_RGBA32_UINT, {4, GL_UNSIGNED_INT, GL_FALSE, true}},
        {VriFormat_RGBA32_SINT, {4, GL_INT, GL_FALSE, true}},
    };

    inline GLVertexFormat ToGLVertexFormat(VriFormat f)
    {
        return vri::MapOr(f, kGLVertexFormatTable, GLVertexFormat {4, GL_FLOAT, GL_FALSE, false});
    }

    // Whether the table really has `f`, as opposed to ToGLVertexFormat
    // answering with its float4 fallback - the same distinction ToGLFormatOrNone
    // draws for textures.
    inline bool HasGLVertexFormat(VriFormat f) { return vri::Contains(f, kGLVertexFormatTable); }

    inline GLenum ToGLTopology(VriPrimitiveTopology t)
    {
        switch (t)
        {
            case VriPrimitiveTopology_PointList:
                return GL_POINTS;
            case VriPrimitiveTopology_LineList:
                return GL_LINES;
            case VriPrimitiveTopology_LineStrip:
                return GL_LINE_STRIP;
            case VriPrimitiveTopology_TriangleStrip:
                return GL_TRIANGLE_STRIP;
#if defined(GL_PATCHES) // tessellation is desktop-GL only (absent in GLES3/WebGL2)
            case VriPrimitiveTopology_PatchList:
                return GL_PATCHES;
#endif
            default:
                return GL_TRIANGLES;
        }
    }

    inline GLenum ToGLCompareOp(VriCompareOp o)
    {
        switch (o)
        {
            case VriCompareOp_Never:
                return GL_NEVER;
            case VriCompareOp_Less:
                return GL_LESS;
            case VriCompareOp_Equal:
                return GL_EQUAL;
            case VriCompareOp_LessOrEqual:
                return GL_LEQUAL;
            case VriCompareOp_Greater:
                return GL_GREATER;
            case VriCompareOp_NotEqual:
                return GL_NOTEQUAL;
            case VriCompareOp_GreaterOrEqual:
                return GL_GEQUAL;
            default:
                return GL_ALWAYS;
        }
    }

    inline GLenum ToGLAddress(VriAddressMode m)
    {
        switch (m)
        {
            case VriAddressMode_MirroredRepeat:
                return GL_MIRRORED_REPEAT;
            case VriAddressMode_ClampToEdge:
                return GL_CLAMP_TO_EDGE;
#if !defined(VRI_GL_ES_HEADERS)
            case VriAddressMode_ClampToBorder:
                return GL_CLAMP_TO_BORDER; // WebGL2/GLES has no border clamp
#else
            case VriAddressMode_ClampToBorder:
                return GL_CLAMP_TO_EDGE;
#endif
            default:
                return GL_REPEAT;
        }
    }

    inline GLenum ToGLStencilOp(VriStencilOp o)
    {
        switch (o)
        {
            case VriStencilOp_Keep:
                return GL_KEEP;
            case VriStencilOp_Zero:
                return GL_ZERO;
            case VriStencilOp_Replace:
                return GL_REPLACE;
            case VriStencilOp_IncrementAndClamp:
                return GL_INCR;
            case VriStencilOp_DecrementAndClamp:
                return GL_DECR;
            case VriStencilOp_Invert:
                return GL_INVERT;
            case VriStencilOp_IncrementAndWrap:
                return GL_INCR_WRAP;
            case VriStencilOp_DecrementAndWrap:
                return GL_DECR_WRAP;
            default:
                return GL_KEEP;
        }
    }

    inline GLenum ToGLBlendFactor(VriBlendFactor f)
    {
        switch (f)
        {
            case VriBlendFactor_Zero:
                return GL_ZERO;
            case VriBlendFactor_One:
                return GL_ONE;
            case VriBlendFactor_SrcColor:
                return GL_SRC_COLOR;
            case VriBlendFactor_OneMinusSrcColor:
                return GL_ONE_MINUS_SRC_COLOR;
            case VriBlendFactor_DstColor:
                return GL_DST_COLOR;
            case VriBlendFactor_OneMinusDstColor:
                return GL_ONE_MINUS_DST_COLOR;
            case VriBlendFactor_SrcAlpha:
                return GL_SRC_ALPHA;
            case VriBlendFactor_OneMinusSrcAlpha:
                return GL_ONE_MINUS_SRC_ALPHA;
            case VriBlendFactor_DstAlpha:
                return GL_DST_ALPHA;
            case VriBlendFactor_OneMinusDstAlpha:
                return GL_ONE_MINUS_DST_ALPHA;
            default:
                return GL_ZERO;
        }
    }

    inline GLenum ToGLBlendOp(VriBlendOp o)
    {
        switch (o)
        {
            case VriBlendOp_Subtract:
                return GL_FUNC_SUBTRACT;
            case VriBlendOp_ReverseSubtract:
                return GL_FUNC_REVERSE_SUBTRACT;
            case VriBlendOp_Min:
                return GL_MIN;
            case VriBlendOp_Max:
                return GL_MAX;
            default:
                return GL_FUNC_ADD;
        }
    }
} // namespace vri::gl
