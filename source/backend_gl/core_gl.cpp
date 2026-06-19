// core_gl.cpp - OpenGL / OpenGL ES / WebGL implementation of VriCoreInterface.
//
// This uses the GLES3 / WebGL2-compatible NON-DSA subset (bind-then-modify) so a
// single command path serves desktop GL, desktop GLES, and WebGL (Emscripten).
// Commands execute immediately on the (single, thread-bound) GL context, so the
// VRI command buffer is a thin recorder and QueueSubmit just flushes; a deferred
// stream for multi-threaded recording is a later enhancement.
//
// Coordinate system: GLES/WebGL lack glClipControl, so the VRI Y-up convention is
// honored by flipping clip-space Y in-shader (SPIRV-Cross flip_vert_y). That makes
// GL's bottom-left framebuffer read back top-left, matching Vulkan/WebGPU. Shaders
// are transpiled SPIR-V -> GLSL/ESSL via SPIRV-Cross at pipeline creation.
//
// Explicit memory and descriptor sets are stubbed for now (descriptor flattening
// lands next).

#include "core_gl.h"
#include "conversions_gl.h"
#include "device_gl.h"
#include "objects_gl.h"

#include <glad/glad.h>

#include <spirv_cross/spirv_glsl.hpp>

#include <string>
#include <vector>

namespace vri::gl
{
    namespace
    {
        inline DeviceGL*        Dev(VriDevice* h)        { return reinterpret_cast<DeviceGL*>(h); }
        inline const DeviceGL*  Dev(const VriDevice* h)  { return reinterpret_cast<const DeviceGL*>(h); }
        inline CommandBufferGL* CB(VriCommandBuffer* h)  { return reinterpret_cast<CommandBufferGL*>(h); }
        inline BufferGL*        Buf(VriBuffer* h)        { return reinterpret_cast<BufferGL*>(h); }
        inline TextureGL*       Tex(VriTexture* h)       { return reinterpret_cast<TextureGL*>(h); }
        inline DescriptorGL*    Desc(VriDescriptor* h)   { return reinterpret_cast<DescriptorGL*>(h); }
        inline PipelineGL*      Pipe(VriPipeline* h)     { return reinterpret_cast<PipelineGL*>(h); }
        inline FenceGL*         Fen(VriFence* h)         { return reinterpret_cast<FenceGL*>(h); }

        std::string SpirvToGlsl(const DeviceGL* d, const void* bytecode, size_t bytecodeSize, const char* entry, spv::ExecutionModel model)
        {
            const uint32_t* words = static_cast<const uint32_t*>(bytecode);
            spirv_cross::CompilerGLSL comp(words, bytecodeSize / 4);
            spirv_cross::CompilerGLSL::Options o = comp.get_common_options();
            o.version = d->ShaderVersion();
            o.es = d->IsES();
            o.vulkan_semantics = false;
            // Flip clip-space Y so GL's bottom-left framebuffer matches the VRI
            // (Vulkan/WebGPU) top-left convention without glClipControl.
            o.vertex.flip_vert_y = true;
            comp.set_common_options(o);
            if (entry)
                comp.set_entry_point(entry, model);
            return comp.compile();
        }

        GLuint CompileShader(DeviceGL* d, GLenum type, const std::string& src)
        {
            GLuint s = glCreateShader(type);
            const char* p = src.c_str();
            const GLint len = static_cast<GLint>(src.size());
            glShaderSource(s, 1, &p, &len);
            glCompileShader(s);
            GLint ok = 0;
            glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
            if (!ok)
            {
                char log[2048] = {};
                glGetShaderInfoLog(s, sizeof(log), nullptr, log);
                d->ReportError(log);
                glDeleteShader(s);
                return 0;
            }
            return s;
        }

        // ---- queries -------------------------------------------------------
        const VriDeviceDesc* VRI_CALL GetDeviceDesc(const VriDevice* device) { return &Dev(device)->Desc(); }
        VriFormatSupportFlags VRI_CALL GetFormatSupport(const VriDevice*, VriFormat)
        {
            return VriFormatSupport_Texture | VriFormatSupport_ColorAttachment | VriFormatSupport_Blend | VriFormatSupport_VertexBuffer;
        }
        VriResult VRI_CALL GetQueue(VriDevice* device, VriQueueType type, uint32_t, VriQueue** outQueue)
        {
            if (type >= VriQueueType_Count) return VriResult_InvalidArgument;
            *outQueue = ToHandle(Dev(device)->GetQueue(type));
            return VriResult_Success;
        }

        // ---- command allocation / lifecycle --------------------------------
        VriResult VRI_CALL CreateCommandAllocator(VriDevice* device, VriQueueType, VriCommandAllocator** out)
        {
            *out = ToHandle(new CommandAllocatorGL{Dev(device)});
            return VriResult_Success;
        }
        void VRI_CALL ResetCommandAllocator(VriCommandAllocator*) {}
        void VRI_CALL DestroyCommandAllocator(VriCommandAllocator* a) { delete reinterpret_cast<CommandAllocatorGL*>(a); }
        VriResult VRI_CALL CreateCommandBuffer(VriCommandAllocator* allocator, VriCommandBuffer** out)
        {
            *out = ToHandle(new CommandBufferGL{reinterpret_cast<CommandAllocatorGL*>(allocator)->device, 0, GL_TRIANGLES});
            return VriResult_Success;
        }
        VriResult VRI_CALL BeginCommandBuffer(VriCommandBuffer* cmd) { CB(cmd)->fbo = 0; return VriResult_Success; }
        VriResult VRI_CALL EndCommandBuffer(VriCommandBuffer*) { return VriResult_Success; }

        // ---- resources -----------------------------------------------------
        VriResult VRI_CALL CreateBuffer(VriDevice* device, const VriBufferDesc* desc, VriBuffer** out)
        {
            GLuint id = 0;
            glGenBuffers(1, &id);
            GLenum usage = GL_STATIC_DRAW;
            GLbitfield mapAccess = 0;
            if (desc->memoryLocation == VriMemoryLocation_HostReadback) { usage = GL_STREAM_READ; mapAccess = GL_MAP_READ_BIT; }
            else if (desc->memoryLocation == VriMemoryLocation_HostUpload) { usage = GL_STREAM_DRAW; mapAccess = GL_MAP_WRITE_BIT; }
            // GL_COPY_WRITE_BUFFER is a neutral binding point that doesn't disturb
            // vertex/index/uniform binding state.
            glBindBuffer(GL_COPY_WRITE_BUFFER, id);
            glBufferData(GL_COPY_WRITE_BUFFER, static_cast<GLsizeiptr>(desc->size), nullptr, usage);
            glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
            *out = ToHandle(new BufferGL{Dev(device), id, desc->size, mapAccess});
            return VriResult_Success;
        }
        void VRI_CALL DestroyBuffer(VriBuffer* buffer)
        {
            if (!buffer) return;
            BufferGL* b = Buf(buffer);
            glDeleteBuffers(1, &b->id);
            delete b;
        }
        void* VRI_CALL MapBuffer(VriBuffer* buffer, uint64_t offset, uint64_t size)
        {
            BufferGL* b = Buf(buffer);
            const GLbitfield access = b->mapAccess ? b->mapAccess : GL_MAP_READ_BIT;
            const GLsizeiptr len = static_cast<GLsizeiptr>(size ? size : (b->size - offset));
            glBindBuffer(GL_COPY_WRITE_BUFFER, b->id);
            return glMapBufferRange(GL_COPY_WRITE_BUFFER, static_cast<GLintptr>(offset), len, access);
        }
        void VRI_CALL UnmapBuffer(VriBuffer* buffer)
        {
            glBindBuffer(GL_COPY_WRITE_BUFFER, Buf(buffer)->id);
            glUnmapBuffer(GL_COPY_WRITE_BUFFER);
            glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
        }
        uint64_t VRI_CALL GetBufferDeviceAddress(const VriBuffer*) { return 0; }

        VriResult VRI_CALL CreateTexture(VriDevice* device, const VriTextureDesc* desc, VriTexture** out)
        {
            const GLFormat gf = ToGLFormat(desc->format);
            GLuint id = 0;
            glGenTextures(1, &id);
            const GLsizei mips = static_cast<GLsizei>(desc->mipNum ? desc->mipNum : 1u);
            const GLsizei w = static_cast<GLsizei>(desc->width);
            const GLsizei h = static_cast<GLsizei>(desc->height ? desc->height : 1u);
            glBindTexture(GL_TEXTURE_2D, id);
            glTexStorage2D(GL_TEXTURE_2D, mips, gf.internalFormat, w, h);
            glBindTexture(GL_TEXTURE_2D, 0);

            TextureGL* t = new TextureGL{};
            t->device = Dev(device);
            t->id = id;
            t->target = GL_TEXTURE_2D;
            t->glFormat = gf.format;
            t->glType = gf.type;
            t->width = desc->width;
            t->height = desc->height ? desc->height : 1u;
            t->depth = 1;
            t->mipNum = static_cast<uint32_t>(mips);
            t->layerNum = 1;
            t->texelSize = gf.texelSize;
            *out = ToHandle(t);
            return VriResult_Success;
        }
        void VRI_CALL DestroyTexture(VriTexture* texture)
        {
            if (!texture) return;
            TextureGL* t = Tex(texture);
            glDeleteTextures(1, &t->id);
            delete t;
        }

        // ---- explicit memory: unsupported on GL (no exposed memory objects) -
        void VRI_CALL GetBufferMemoryDesc(const VriDevice*, const VriBufferDesc*, VriMemoryLocation, VriMemoryDesc* o) { if (o) *o = VriMemoryDesc{}; }
        void VRI_CALL GetTextureMemoryDesc(const VriDevice*, const VriTextureDesc*, VriMemoryLocation, VriMemoryDesc* o) { if (o) *o = VriMemoryDesc{}; }
        VriResult VRI_CALL AllocateMemory(VriDevice*, const VriMemoryDesc*, VriMemory**) { return VriResult_Unsupported; }
        void VRI_CALL FreeMemory(VriMemory*) {}
        VriResult VRI_CALL BindBufferMemory(VriDevice*, VriBuffer*, VriMemory*, uint64_t) { return VriResult_Unsupported; }
        VriResult VRI_CALL BindTextureMemory(VriDevice*, VriTexture*, VriMemory*, uint64_t) { return VriResult_Unsupported; }

        // ---- views & samplers ----------------------------------------------
        VriResult VRI_CALL CreateBufferView(VriDevice* device, const VriBufferViewDesc* desc, VriDescriptor** out)
        {
            DescriptorGL* v = new DescriptorGL{};
            v->kind = DescriptorGL::Kind::BufferView;
            v->device = Dev(device);
            v->buffer = Buf(desc->buffer);
            v->bufferOffset = desc->offset;
            v->bufferRange = desc->size;
            *out = ToHandle(v);
            return VriResult_Success;
        }
        VriResult VRI_CALL CreateTextureView(VriDevice* device, const VriTextureViewDesc* desc, VriDescriptor** out)
        {
            DescriptorGL* v = new DescriptorGL{};
            v->kind = DescriptorGL::Kind::TextureView;
            v->device = Dev(device);
            v->texture = reinterpret_cast<const TextureGL*>(desc->texture);
            v->mip = desc->baseMip;
            v->layer = desc->baseLayer;
            *out = ToHandle(v);
            return VriResult_Success;
        }
        VriResult VRI_CALL CreateSampler(VriDevice* device, const VriSamplerDesc* desc, VriDescriptor** out)
        {
            GLuint s = 0;
            glGenSamplers(1, &s);
            glSamplerParameteri(s, GL_TEXTURE_MAG_FILTER, desc->magFilter == VriFilter_Linear ? GL_LINEAR : GL_NEAREST);
            glSamplerParameteri(s, GL_TEXTURE_MIN_FILTER, desc->minFilter == VriFilter_Linear ? GL_LINEAR : GL_NEAREST);
            DescriptorGL* v = new DescriptorGL{};
            v->kind = DescriptorGL::Kind::Sampler;
            v->device = Dev(device);
            v->sampler = s;
            *out = ToHandle(v);
            return VriResult_Success;
        }
        void VRI_CALL DestroyDescriptor(VriDescriptor* descriptor)
        {
            if (!descriptor) return;
            DescriptorGL* v = Desc(descriptor);
            if (v->sampler) glDeleteSamplers(1, &v->sampler);
            delete v;
        }

        // ---- pipeline layout & pipelines -----------------------------------
        VriResult VRI_CALL CreatePipelineLayout(VriDevice* device, const VriPipelineLayoutDesc*, VriPipelineLayout** out)
        {
            *out = ToHandle(new PipelineLayoutGL{Dev(device)});
            return VriResult_Success;
        }
        void VRI_CALL DestroyPipelineLayout(VriPipelineLayout* layout) { delete reinterpret_cast<PipelineLayoutGL*>(layout); }

        VriResult VRI_CALL CreateGraphicsPipeline(VriDevice* device, const VriGraphicsPipelineDesc* desc, VriPipeline** out)
        {
            DeviceGL* d = Dev(device);
            GLuint vs = 0, fs = 0;
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                const VriShaderDesc& s = desc->shaders[i];
                if (s.stage == VriShaderStage_Vertex)
                    vs = CompileShader(d, GL_VERTEX_SHADER, SpirvToGlsl(d, s.bytecode, s.bytecodeSize, s.entryPointName, spv::ExecutionModelVertex));
                else if (s.stage == VriShaderStage_Fragment)
                    fs = CompileShader(d, GL_FRAGMENT_SHADER, SpirvToGlsl(d, s.bytecode, s.bytecodeSize, s.entryPointName, spv::ExecutionModelFragment));
            }
            if (!vs || !fs)
            {
                if (vs) glDeleteShader(vs);
                if (fs) glDeleteShader(fs);
                return VriResult_Failure;
            }
            GLuint program = glCreateProgram();
            glAttachShader(program, vs);
            glAttachShader(program, fs);
            glLinkProgram(program);
            glDetachShader(program, vs);
            glDetachShader(program, fs);
            glDeleteShader(vs);
            glDeleteShader(fs);
            GLint ok = 0;
            glGetProgramiv(program, GL_LINK_STATUS, &ok);
            if (!ok)
            {
                char log[2048] = {};
                glGetProgramInfoLog(program, sizeof(log), nullptr, log);
                d->ReportError(log);
                glDeleteProgram(program);
                return VriResult_Failure;
            }

            PipelineGL* p = new PipelineGL{};
            p->device = d;
            p->program = program;
            p->topology = ToGLTopology(desc->inputAssembly.topology);
            p->cullEnable = desc->rasterization.cullMode != VriCullMode_None;
            p->cullFace = desc->rasterization.cullMode == VriCullMode_Front ? GL_FRONT : GL_BACK;
            // flip_vert_y negates clip Y, reversing window-space winding, so a VRI
            // CCW front face is GL CW (and vice versa).
            p->frontFace = desc->rasterization.frontFace == VriFrontFace_CounterClockwise ? GL_CW : GL_CCW;
            p->depthTest = desc->depthStencil.depthTest != VRI_FALSE;
            p->depthWrite = desc->depthStencil.depthWrite != VRI_FALSE;
            p->depthFunc = ToGLCompareOp(desc->depthStencil.depthCompareOp);
            const VriColorAttachmentDesc* c0 = desc->outputMerger.colorNum ? &desc->outputMerger.colors[0] : nullptr;
            p->blendEnable = c0 && c0->blend.enable;
            p->srcRGB = c0 ? ToGLBlendFactor(c0->blend.srcColor) : GL_ONE;
            p->dstRGB = c0 ? ToGLBlendFactor(c0->blend.dstColor) : GL_ZERO;
            p->rgbOp = c0 ? ToGLBlendOp(c0->blend.colorOp) : GL_FUNC_ADD;
            p->srcA = c0 ? ToGLBlendFactor(c0->blend.srcAlpha) : GL_ONE;
            p->dstA = c0 ? ToGLBlendFactor(c0->blend.dstAlpha) : GL_ZERO;
            p->aOp = c0 ? ToGLBlendOp(c0->blend.alphaOp) : GL_FUNC_ADD;
            const VriColorWriteFlags wm = c0 ? c0->colorWriteMask : VriColorWrite_RGBA;
            const VriColorWriteFlags m = (wm == 0) ? VriColorWrite_RGBA : wm;
            p->colorMask[0] = (m & VriColorWrite_R) ? GL_TRUE : GL_FALSE;
            p->colorMask[1] = (m & VriColorWrite_G) ? GL_TRUE : GL_FALSE;
            p->colorMask[2] = (m & VriColorWrite_B) ? GL_TRUE : GL_FALSE;
            p->colorMask[3] = (m & VriColorWrite_A) ? GL_TRUE : GL_FALSE;
            *out = ToHandle(p);
            return VriResult_Success;
        }
        VriResult VRI_CALL CreateComputePipeline(VriDevice*, const VriComputePipelineDesc*, VriPipeline**) { return VriResult_Unsupported; }
        void VRI_CALL DestroyPipeline(VriPipeline* pipeline)
        {
            if (!pipeline) return;
            PipelineGL* p = Pipe(pipeline);
            glDeleteProgram(p->program);
            delete p;
        }

        // ---- descriptor pools / sets (flattening lands next) ---------------
        VriResult VRI_CALL CreateDescriptorPool(VriDevice*, const VriDescriptorPoolDesc*, VriDescriptorPool**) { return VriResult_Unsupported; }
        void VRI_CALL ResetDescriptorPool(VriDescriptorPool*) {}
        void VRI_CALL DestroyDescriptorPool(VriDescriptorPool*) {}
        VriResult VRI_CALL AllocateDescriptorSets(VriDescriptorPool*, const VriPipelineLayout*, uint32_t, VriDescriptorSet**, uint32_t) { return VriResult_Unsupported; }
        void VRI_CALL UpdateDescriptorRanges(VriDescriptorSet*, uint32_t, uint32_t, const VriDescriptorRangeUpdateDesc*) {}

        // ---- synchronization (coarse; GL is ordered on one context) --------
        VriResult VRI_CALL CreateFence(VriDevice* device, uint64_t initialValue, VriFence** out)
        {
            *out = ToHandle(new FenceGL{Dev(device), initialValue});
            return VriResult_Success;
        }
        void VRI_CALL DestroyFence(VriFence* fence) { delete Fen(fence); }
        uint64_t VRI_CALL GetFenceValue(VriFence* fence) { return Fen(fence)->value; }
        void VRI_CALL Wait(VriFence* fence, uint64_t value)
        {
            glFinish();
            FenceGL* f = Fen(fence);
            if (value > f->value) f->value = value;
        }

        // ---- command recording (immediate) ---------------------------------
        void VRI_CALL CmdBeginRendering(VriCommandBuffer* cmd, const VriAttachmentsDesc* a)
        {
            CommandBufferGL* c = CB(cmd);
            GLuint fbo = 0;
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            std::vector<GLenum> drawBufs(a->colorNum);
            for (uint32_t i = 0; i < a->colorNum; ++i)
            {
                const DescriptorGL* v = Desc(a->colors[i].view);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, v->texture->target, v->texture->id, static_cast<GLint>(v->mip));
                drawBufs[i] = GL_COLOR_ATTACHMENT0 + i;
            }
            glDrawBuffers(static_cast<GLsizei>(drawBufs.size()), drawBufs.data());

            for (uint32_t i = 0; i < a->colorNum; ++i)
                if (a->colors[i].loadOp == VriAttachmentLoadOp_Clear)
                    glClearBufferfv(GL_COLOR, static_cast<GLint>(i), a->colors[i].clearValue.color.f32);

            c->fbo = fbo;
        }
        void VRI_CALL CmdEndRendering(VriCommandBuffer* cmd)
        {
            CommandBufferGL* c = CB(cmd);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            if (c->fbo) { glDeleteFramebuffers(1, &c->fbo); c->fbo = 0; }
        }
        void VRI_CALL CmdSetViewports(VriCommandBuffer*, const VriViewport* vps, uint32_t num)
        {
            if (!num) return;
            glViewport(static_cast<GLint>(vps[0].x), static_cast<GLint>(vps[0].y),
                       static_cast<GLsizei>(vps[0].width), static_cast<GLsizei>(vps[0].height));
            glDepthRangef(vps[0].minDepth, vps[0].maxDepth);
        }
        void VRI_CALL CmdSetScissors(VriCommandBuffer*, const VriRect* r, uint32_t num)
        {
            if (!num) return;
            glEnable(GL_SCISSOR_TEST);
            glScissor(r[0].x, r[0].y, static_cast<GLsizei>(r[0].width), static_cast<GLsizei>(r[0].height));
        }
        void VRI_CALL CmdSetPipelineLayout(VriCommandBuffer*, VriPipelineLayout*) {}
        void VRI_CALL CmdSetPipeline(VriCommandBuffer* cmd, VriPipeline* pipeline)
        {
            CommandBufferGL* c = CB(cmd);
            PipelineGL* p = Pipe(pipeline);
            glUseProgram(p->program);
            if (p->cullEnable) { glEnable(GL_CULL_FACE); glCullFace(p->cullFace); }
            else glDisable(GL_CULL_FACE);
            glFrontFace(p->frontFace);
            if (p->depthTest) { glEnable(GL_DEPTH_TEST); glDepthFunc(p->depthFunc); }
            else glDisable(GL_DEPTH_TEST);
            glDepthMask(p->depthWrite ? GL_TRUE : GL_FALSE);
            if (p->blendEnable)
            {
                glEnable(GL_BLEND);
                glBlendFuncSeparate(p->srcRGB, p->dstRGB, p->srcA, p->dstA);
                glBlendEquationSeparate(p->rgbOp, p->aOp);
            }
            else glDisable(GL_BLEND);
            glColorMask(p->colorMask[0], p->colorMask[1], p->colorMask[2], p->colorMask[3]);
            c->topology = p->topology;
        }
        void VRI_CALL CmdSetDescriptorSet(VriCommandBuffer*, uint32_t, const VriDescriptorSet*) {}
        void VRI_CALL CmdSetConstants(VriCommandBuffer*, uint32_t, const void*, uint32_t) {}
        void VRI_CALL CmdSetVertexBuffers(VriCommandBuffer*, uint32_t, const VriVertexBufferBinding*, uint32_t) {}
        void VRI_CALL CmdSetIndexBuffer(VriCommandBuffer*, VriBuffer*, uint64_t, VriIndexType) {}
        void VRI_CALL CmdDraw(VriCommandBuffer* cmd, const VriDrawDesc* d)
        {
            CommandBufferGL* c = CB(cmd);
            glBindVertexArray(c->device->DefaultVao());
            // baseInstance is GL 4.2+ only (absent on GLES/WebGL); the portable
            // path uses glDrawArraysInstanced. baseInstance!=0 is unsupported here.
            glDrawArraysInstanced(c->topology, static_cast<GLint>(d->baseVertex), static_cast<GLsizei>(d->vertexNum),
                                  static_cast<GLsizei>(d->instanceNum));
        }
        void VRI_CALL CmdDrawIndexed(VriCommandBuffer*, const VriDrawIndexedDesc*) {}
        void VRI_CALL CmdDrawIndirect(VriCommandBuffer*, VriBuffer*, uint64_t, uint32_t, uint32_t) {}
        void VRI_CALL CmdDispatch(VriCommandBuffer*, const VriDispatchDesc*) {}
        void VRI_CALL CmdDispatchIndirect(VriCommandBuffer*, VriBuffer*, uint64_t) {}
        void VRI_CALL CmdBarrier(VriCommandBuffer*, const VriBarrierGroupDesc*) {} // GL is ordered on one context
        void VRI_CALL CmdCopyBuffer(VriCommandBuffer*, VriBuffer* dst, VriBuffer* src, const VriBufferCopyDesc* r)
        {
            glBindBuffer(GL_COPY_READ_BUFFER, Buf(src)->id);
            glBindBuffer(GL_COPY_WRITE_BUFFER, Buf(dst)->id);
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, static_cast<GLintptr>(r->srcOffset), static_cast<GLintptr>(r->dstOffset), static_cast<GLsizeiptr>(r->size));
            glBindBuffer(GL_COPY_READ_BUFFER, 0);
            glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
        }
        void VRI_CALL CmdCopyTexture(VriCommandBuffer*, VriTexture* dst, VriTexture* src, const VriTextureCopyDesc*)
        {
            // glCopyImageSubData is GL 4.3 / GLES 3.2; use a portable framebuffer
            // blit so the path also works on GLES 3.0 / WebGL2.
            const TextureGL* s = reinterpret_cast<const TextureGL*>(src);
            const TextureGL* d = reinterpret_cast<const TextureGL*>(dst);
            GLuint fbos[2] = {0, 0};
            glGenFramebuffers(2, fbos);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, fbos[0]);
            glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, s->target, s->id, 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbos[1]);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, d->target, d->id, 0);
            const GLint w = static_cast<GLint>(s->width), h = static_cast<GLint>(s->height);
            glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glDeleteFramebuffers(2, fbos);
        }
        void VRI_CALL CmdUploadBufferToTexture(VriCommandBuffer*, VriTexture* dst, VriBuffer* src, const VriBufferTextureCopyDesc* region)
        {
            TextureGL* t = Tex(dst);
            const uint32_t w = region->texture.width ? region->texture.width : t->width;
            const uint32_t h = region->texture.height ? region->texture.height : t->height;
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Buf(src)->id);
            glBindTexture(t->target, t->id);
            glTexSubImage2D(t->target, static_cast<GLint>(region->texture.mip), region->texture.x, region->texture.y,
                            static_cast<GLsizei>(w), static_cast<GLsizei>(h), t->glFormat, t->glType,
                            reinterpret_cast<const void*>(static_cast<uintptr_t>(region->bufferOffset)));
            glBindTexture(t->target, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }
        void VRI_CALL CmdReadbackTextureToBuffer(VriCommandBuffer*, VriBuffer* dst, VriTexture* src, const VriBufferTextureCopyDesc* region)
        {
            // No glGetTextureImage on GLES/WebGL: read via a temp FBO + glReadPixels
            // into a PBO. glReadPixels reads bottom-up; combined with the in-shader
            // flip_vert_y this yields VRI top-left row order.
            const TextureGL* t = reinterpret_cast<const TextureGL*>(src);
            BufferGL* b = Buf(dst);
            const GLint mip = static_cast<GLint>(region->texture.mip);
            GLsizei w = static_cast<GLsizei>(t->width >> region->texture.mip); if (w < 1) w = 1;
            GLsizei h = static_cast<GLsizei>(t->height >> region->texture.mip); if (h < 1) h = 1;

            GLuint fbo = 0;
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, t->target, t->id, mip);
            glReadBuffer(GL_COLOR_ATTACHMENT0);

            glBindBuffer(GL_PIXEL_PACK_BUFFER, b->id);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, w, h, t->glFormat, t->glType,
                         reinterpret_cast<void*>(static_cast<uintptr_t>(region->bufferOffset)));
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDeleteFramebuffers(1, &fbo);
        }
        void VRI_CALL CmdBeginDebugGroup(VriCommandBuffer*, const char*) {}
        void VRI_CALL CmdEndDebugGroup(VriCommandBuffer*) {}

        // ---- submission ----------------------------------------------------
        void VRI_CALL QueueSubmit(VriQueue*, const VriQueueSubmitDesc* submit)
        {
            glFlush();
            for (uint32_t i = 0; i < submit->signalFenceNum; ++i)
                Fen(submit->signalFences[i].fence)->value = submit->signalFences[i].value;
        }
        void VRI_CALL QueueWaitIdle(VriQueue*) { glFinish(); }
        void VRI_CALL DeviceWaitIdle(VriDevice*) { glFinish(); }
        void VRI_CALL SetDebugName(void*, const char*) {}

        VriCoreInterface MakeTable()
        {
            VriCoreInterface t = {};
            t.GetDeviceDesc = GetDeviceDesc; t.GetFormatSupport = GetFormatSupport; t.GetQueue = GetQueue;
            t.CreateCommandAllocator = CreateCommandAllocator; t.ResetCommandAllocator = ResetCommandAllocator; t.DestroyCommandAllocator = DestroyCommandAllocator;
            t.CreateCommandBuffer = CreateCommandBuffer; t.BeginCommandBuffer = BeginCommandBuffer; t.EndCommandBuffer = EndCommandBuffer;
            t.CreateBuffer = CreateBuffer; t.DestroyBuffer = DestroyBuffer; t.MapBuffer = MapBuffer; t.UnmapBuffer = UnmapBuffer; t.GetBufferDeviceAddress = GetBufferDeviceAddress;
            t.CreateTexture = CreateTexture; t.DestroyTexture = DestroyTexture;
            t.GetBufferMemoryDesc = GetBufferMemoryDesc; t.GetTextureMemoryDesc = GetTextureMemoryDesc; t.AllocateMemory = AllocateMemory; t.FreeMemory = FreeMemory; t.BindBufferMemory = BindBufferMemory; t.BindTextureMemory = BindTextureMemory;
            t.CreateBufferView = CreateBufferView; t.CreateTextureView = CreateTextureView; t.CreateSampler = CreateSampler; t.DestroyDescriptor = DestroyDescriptor;
            t.CreatePipelineLayout = CreatePipelineLayout; t.DestroyPipelineLayout = DestroyPipelineLayout; t.CreateGraphicsPipeline = CreateGraphicsPipeline; t.CreateComputePipeline = CreateComputePipeline; t.DestroyPipeline = DestroyPipeline;
            t.CreateDescriptorPool = CreateDescriptorPool; t.ResetDescriptorPool = ResetDescriptorPool; t.DestroyDescriptorPool = DestroyDescriptorPool; t.AllocateDescriptorSets = AllocateDescriptorSets; t.UpdateDescriptorRanges = UpdateDescriptorRanges;
            t.CreateFence = CreateFence; t.DestroyFence = DestroyFence; t.GetFenceValue = GetFenceValue; t.Wait = Wait;
            t.CmdBeginRendering = CmdBeginRendering; t.CmdEndRendering = CmdEndRendering; t.CmdSetViewports = CmdSetViewports; t.CmdSetScissors = CmdSetScissors;
            t.CmdSetPipelineLayout = CmdSetPipelineLayout; t.CmdSetPipeline = CmdSetPipeline; t.CmdSetDescriptorSet = CmdSetDescriptorSet; t.CmdSetConstants = CmdSetConstants;
            t.CmdSetVertexBuffers = CmdSetVertexBuffers; t.CmdSetIndexBuffer = CmdSetIndexBuffer;
            t.CmdDraw = CmdDraw; t.CmdDrawIndexed = CmdDrawIndexed; t.CmdDrawIndirect = CmdDrawIndirect; t.CmdDispatch = CmdDispatch; t.CmdDispatchIndirect = CmdDispatchIndirect;
            t.CmdBarrier = CmdBarrier; t.CmdCopyBuffer = CmdCopyBuffer; t.CmdCopyTexture = CmdCopyTexture; t.CmdUploadBufferToTexture = CmdUploadBufferToTexture; t.CmdReadbackTextureToBuffer = CmdReadbackTextureToBuffer;
            t.CmdBeginDebugGroup = CmdBeginDebugGroup; t.CmdEndDebugGroup = CmdEndDebugGroup;
            t.QueueSubmit = QueueSubmit; t.QueueWaitIdle = QueueWaitIdle; t.DeviceWaitIdle = DeviceWaitIdle; t.SetDebugName = SetDebugName;
            return t;
        }

        const VriCoreInterface g_coreGL = MakeTable();
    } // namespace

    const VriCoreInterface* GetCoreInterfaceGL() { return &g_coreGL; }
} // namespace vri::gl
