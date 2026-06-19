// objects_gl.h - concrete OpenGL objects behind the opaque VRI handles.
#pragma once

#include <glad/glad.h>

#include <vri/vri.h>

namespace vri::gl
{
    class DeviceGL;

    struct QueueGL
    {
        DeviceGL* device;
    };

    struct CommandAllocatorGL
    {
        DeviceGL* device; // GL has no command pool
    };

    // Commands execute immediately on the (single, thread-bound) GL context;
    // a deferred stream for multi-threaded recording is a later enhancement.
    struct CommandBufferGL
    {
        DeviceGL* device;
        GLuint    fbo;      // transient FBO for the active render pass
        GLenum    topology; // from the bound pipeline
    };

    struct BufferGL
    {
        DeviceGL*  device;
        GLuint     id;
        uint64_t   size;
        GLbitfield mapAccess; // GL_MAP_READ_BIT / GL_MAP_WRITE_BIT for MapBuffer
    };

    struct TextureGL
    {
        DeviceGL* device;
        GLuint    id;
        GLenum    target;
        GLenum    glFormat; // for read/write (e.g. GL_RGBA)
        GLenum    glType;   // e.g. GL_UNSIGNED_BYTE
        uint32_t  width;
        uint32_t  height;
        uint32_t  depth;
        uint32_t  mipNum;
        uint32_t  layerNum;
        uint32_t  texelSize;
    };

    struct DescriptorGL
    {
        enum class Kind { TextureView, BufferView, Sampler } kind;
        DeviceGL*        device;
        const TextureGL* texture; // for TextureView (FBO attachment / sampling)
        uint32_t         mip;
        uint32_t         layer;
        GLuint           sampler;
        const BufferGL*  buffer;
        uint64_t         bufferOffset;
        uint64_t         bufferRange;
    };

    struct PipelineLayoutGL
    {
        DeviceGL* device;
    };

    struct PipelineGL
    {
        DeviceGL* device;
        GLuint    program;
        GLenum    topology;
        bool      cullEnable;
        GLenum    cullFace;
        GLenum    frontFace;
        bool      depthTest;
        bool      depthWrite;
        GLenum    depthFunc;
        bool      blendEnable;
        GLenum    srcRGB, dstRGB, rgbOp, srcA, dstA, aOp;
        GLboolean colorMask[4];
    };

    struct FenceGL
    {
        DeviceGL* device;
        uint64_t  value;
    };

    inline VriQueue*            ToHandle(QueueGL* q)            { return reinterpret_cast<VriQueue*>(q); }
    inline VriCommandAllocator* ToHandle(CommandAllocatorGL* a) { return reinterpret_cast<VriCommandAllocator*>(a); }
    inline VriCommandBuffer*    ToHandle(CommandBufferGL* c)    { return reinterpret_cast<VriCommandBuffer*>(c); }
    inline VriBuffer*           ToHandle(BufferGL* b)           { return reinterpret_cast<VriBuffer*>(b); }
    inline VriTexture*          ToHandle(TextureGL* t)          { return reinterpret_cast<VriTexture*>(t); }
    inline VriDescriptor*       ToHandle(DescriptorGL* d)       { return reinterpret_cast<VriDescriptor*>(d); }
    inline VriPipelineLayout*   ToHandle(PipelineLayoutGL* p)   { return reinterpret_cast<VriPipelineLayout*>(p); }
    inline VriPipeline*         ToHandle(PipelineGL* p)         { return reinterpret_cast<VriPipeline*>(p); }
    inline VriFence*            ToHandle(FenceGL* f)            { return reinterpret_cast<VriFence*>(f); }
} // namespace vri::gl
