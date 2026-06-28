// objects_gl.h - concrete OpenGL objects behind the opaque VRI handles.
#pragma once

#include "gl_loader.h"

#include <vri/vri.h>

#include <string>
#include <unordered_map>
#include <vector>

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
    struct PipelineLayoutGL;

    struct PipelineGL;

    struct VertexBufferBindingGL
    {
        GLuint   id;
        uint64_t offset;
    };

    struct TextureGL;
    struct PendingResolveGL // MSAA color attachment -> single-sample resolve target (blit at EndRendering)
    {
        GLenum           srcAttachment;
        const TextureGL* dst;
        GLuint           dstMip;
    };

    struct CommandBufferGL
    {
        DeviceGL*               device;
        GLuint                  fbo;           // transient FBO for the active render pass
        GLenum                  topology;      // from the bound pipeline
        const PipelineLayoutGL* boundLayout;   // from CmdSetPipelineLayout (descriptor remap)
        const PipelineGL*       boundPipeline; // from CmdSetPipeline (combined-sampler pairing)
        std::unordered_map<uint32_t, VertexBufferBindingGL> vertexBuffers; // slot -> buffer
        GLuint                                              indexBuffer = 0;
        uint64_t                                            indexOffset = 0;
        GLenum                                              indexType   = 0x1405; // GL_UNSIGNED_INT
        uint32_t                      enabledAttribMask = 0; // attrib locations enabled by the last draw
        std::vector<PendingResolveGL> pendingResolves;       // MSAA resolves to run at EndRendering
    };

    struct BufferGL
    {
        DeviceGL* device;
        GLuint    id;
        uint64_t  size;
        // WebGL forbids ever binding an ELEMENT_ARRAY_BUFFER to a non-element target
        // (and vice versa), so index buffers must use GL_ELEMENT_ARRAY_BUFFER for all
        // create/map/copy binds. Other buffers use the neutral GL_COPY_WRITE_BUFFER.
        GLenum     target;
        GLbitfield mapAccess; // GL_MAP_READ_BIT / GL_MAP_WRITE_BIT for MapBuffer
        // WebGL2 has no real buffer mapping, so on ES we emulate Map/Unmap with a CPU
        // shadow filled via glGetBufferSubData and flushed back via glBufferSubData.
        void*    shadow    = nullptr;
        uint64_t mapOffset = 0;
        uint64_t mapLen    = 0;
        bool     mapWrite  = false;
        // GL 4.4 immutable storage + persistent-coherent mapping: a mappable buffer is
        // mapped once at creation and stays mapped; MapBuffer returns persistentPtr+offset
        // and UnmapBuffer is a no-op (coherent). Null when not persistently mapped.
        void* persistentPtr = nullptr;
    };

    struct TextureGL
    {
        DeviceGL* device;
        GLuint    id;
        GLenum    target;
        GLenum    glFormat;           // for read/write (e.g. GL_RGBA)
        GLenum    glType;             // e.g. GL_UNSIGNED_BYTE
        GLenum    internalFormat = 0; // e.g. GL_RGBA8 (for glBindImageTexture on storage images)
        uint32_t  width;
        uint32_t  height;
        uint32_t  depth;
        uint32_t  mipNum;
        uint32_t  layerNum;
        uint32_t  texelSize;
        bool      isRenderbuffer = false; // MSAA targets are multisample renderbuffers (desktop GL + WebGL2)
    };

    struct DescriptorGL
    {
        enum class Kind
        {
            TextureView,
            BufferView,
            Sampler
        } kind;
        DeviceGL*        device;
        const TextureGL* texture; // for TextureView (FBO attachment / sampling)
        uint32_t         mip;
        uint32_t         layer;
        GLuint           sampler;
        const BufferGL*  buffer;
        uint64_t         bufferOffset;
        uint64_t         bufferRange;
    };

    // Flattened binding: each (set, binding) maps to a per-type GL unit (uniform
    // buffer binding point / SSBO binding point / texture unit). This is the
    // legacy-backend descriptor remap from the design plan.
    struct LayoutBindingGL
    {
        uint32_t          set;
        uint32_t          binding;
        VriDescriptorType type;
        uint32_t          glUnit;
    };

    // Push constants on GL: SPIRV-Cross (vulkan_semantics=false) lowers a push_constant
    // block to a default-block struct uniform (`uniform T vriPush;`), so each member is set
    // individually via glUniform. One entry per leaf member, captured at pipeline build.
    struct PushMemberGL
    {
        std::string name;          // GLSL uniform name, e.g. "vriPush.color"
        uint32_t    offset;        // byte offset within the push-constant blob
        uint32_t    basetype;      // 0 = float, 1 = int, 2 = uint
        uint32_t    vecsize;       // 1..4
        uint32_t    columns;       // 1 = vector, >1 = matrix (NxN)
        uint32_t    count;         // array length (1 if not an array)
        int         location = -1; // glGetUniformLocation after link
    };

    struct PipelineLayoutGL
    {
        DeviceGL*                    device;
        std::vector<LayoutBindingGL> bindings;

        const LayoutBindingGL* Find(uint32_t set, uint32_t binding) const
        {
            for (const LayoutBindingGL& b : bindings)
                if (b.set == set && b.binding == binding)
                    return &b;
            return nullptr;
        }
    };

    struct DescriptorPoolGL
    {
        DeviceGL* device;
    };

    // CPU-side descriptor set: records which view sits at each binding; bound to
    // GL units at CmdSetDescriptorSet time via the pipeline layout's mapping.
    struct DescriptorSetGL
    {
        DeviceGL*                                         device;
        const PipelineLayoutGL*                           layout;
        uint32_t                                          setIndex;
        std::unordered_map<uint32_t, const DescriptorGL*> bound; // binding -> view
    };

    // A SPIRV-Cross-combined sampler (GLSL has no separate texture/sampler): the
    // VRI texture at (texSet,texBinding) and sampler at (smpSet,smpBinding) both
    // bind to GL texture unit `unit`. The combined sampler uniform is pointed at
    // `unit` via glUniform1i at pipeline creation (name kept for that lookup).
    struct CombinedSamplerGL
    {
        uint32_t    unit;
        uint32_t    texSet, texBinding;
        uint32_t    smpSet, smpBinding;
        std::string name;
    };

    // One vertex attribute, flattened for the classic glVertexAttribPointer path
    // (couples format+stream+stride; GLES3/WebGL2 has no separate-format DSA).
    struct VertexAttribGL
    {
        GLuint    location; // GLSL attribute location (= attribute index)
        GLint     size;     // components
        GLenum    type;
        GLboolean normalized;
        bool      integer;     // true -> glVertexAttribIPointer (integer attribute, no float cast)
        GLuint    offset;      // within the vertex
        GLsizei   stride;      // of the owning stream
        uint32_t  bindingSlot; // CmdSetVertexBuffers slot to source from
        uint32_t  divisor;     // 0 = per-vertex, 1 = per-instance (VriVertexStepRate)
    };

    // A vertex-buffer binding point used by the pipeline's separate-format VAO:
    // the stream slot (CmdSetVertexBuffers slot) and its stride. The VAO bakes in
    // the attribute *format* once; only the buffer+offset+stride is set per draw.
    struct VboBindingGL
    {
        uint32_t slot;
        GLsizei  stride;
        uint32_t divisor; // 0 = per-vertex, 1 = per-instance
    };

    struct PipelineGL
    {
        DeviceGL* device;
        GLuint    program;
        bool      isCompute = false;
        // Separate attribute format (GL 4.3+): a per-pipeline VAO with the vertex
        // format baked in, replacing per-draw glVertexAttribPointer churn. 0 when
        // the context lacks separate-format support (classic path uses the default VAO).
        GLuint                    formatVao = 0;
        std::vector<VboBindingGL> vboBindings; // unique stream slots + strides for glBindVertexBuffer
        // Base-vertex/base-instance parity (VK's VertexIndex/InstanceIndex include the
        // base offset; GL's gl_VertexID/gl_InstanceID do not). SPIRV-Cross bridges this
        // with gl_BaseVertexARB/gl_BaseInstanceARB (auto-set by *BaseVertex/*BaseInstance
        // draws when ARB_shader_draw_parameters is present) OR a uniform fallback. These
        // hold the fallback uniform locations (-1 when the ARB builtin path is used).
        GLint     baseVertexLoc   = -1;
        GLint     baseInstanceLoc = -1;
        GLenum    topology;
        uint32_t  patchVertices = 0; // tessellation: glPatchParameteri(GL_PATCH_VERTICES) when topology == GL_PATCHES
        bool      cullEnable;
        GLenum    cullFace;
        GLenum    frontFace;
        bool      depthTest;
        bool      depthWrite;
        GLenum    depthFunc;
        bool      blendEnable;
        GLenum    srcRGB, dstRGB, rgbOp, srcA, dstA, aOp;
        GLboolean colorMask[4];
        // stencil: per-face func (compareOp), ops (fail/depthFail/pass), masks, ref
        bool stencilEnable = false;
        struct StencilFaceGL
        {
            GLenum func, sfail, dpfail, dppass;
            GLuint compareMask, writeMask;
            GLint  ref;
        } stencilFront, stencilBack;
        std::vector<CombinedSamplerGL> combinedSamplers;
        std::vector<VertexAttribGL>    vertexAttribs;
        std::vector<PushMemberGL>      pushMembers; // emulated push constants (set via glUniform)
    };

    struct FenceGL
    {
        DeviceGL* device;
        uint64_t  value;
    };

    struct QueryPoolGL
    {
        DeviceGL*           device;
        std::vector<GLuint> ids; // one GL query object per timestamp slot
    };
    inline VriQueryPool* ToHandle(QueryPoolGL* q) { return reinterpret_cast<VriQueryPool*>(q); }

    inline VriQueue*            ToHandle(QueueGL* q) { return reinterpret_cast<VriQueue*>(q); }
    inline VriCommandAllocator* ToHandle(CommandAllocatorGL* a) { return reinterpret_cast<VriCommandAllocator*>(a); }
    inline VriCommandBuffer*    ToHandle(CommandBufferGL* c) { return reinterpret_cast<VriCommandBuffer*>(c); }
    inline VriBuffer*           ToHandle(BufferGL* b) { return reinterpret_cast<VriBuffer*>(b); }
    inline VriTexture*          ToHandle(TextureGL* t) { return reinterpret_cast<VriTexture*>(t); }
    inline VriDescriptor*       ToHandle(DescriptorGL* d) { return reinterpret_cast<VriDescriptor*>(d); }
    inline VriPipelineLayout*   ToHandle(PipelineLayoutGL* p) { return reinterpret_cast<VriPipelineLayout*>(p); }
    inline VriPipeline*         ToHandle(PipelineGL* p) { return reinterpret_cast<VriPipeline*>(p); }
    inline VriFence*            ToHandle(FenceGL* f) { return reinterpret_cast<VriFence*>(f); }
    inline VriDescriptorPool*   ToHandle(DescriptorPoolGL* p) { return reinterpret_cast<VriDescriptorPool*>(p); }
    inline VriDescriptorSet*    ToHandle(DescriptorSetGL* s) { return reinterpret_cast<VriDescriptorSet*>(s); }
} // namespace vri::gl
