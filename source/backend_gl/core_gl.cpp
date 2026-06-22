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

#include "gl_loader.h"

#include <spirv_cross/spirv_glsl.hpp>

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
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

        // ESSL/WebGL has no gl_BaseVertex / gl_BaseInstance, but Slang lowers
        // SV_VertexID/SV_InstanceID as (VertexIndex - BaseVertex) etc. and SPIRV-Cross
        // also synthesizes a base for the Vulkan VertexIndex builtin -> it throws for
        // the ES profile. For ES we transform the module so no base is referenced:
        //   * VertexIndex(42)  -> VertexId(5),  InstanceIndex(43) -> InstanceId(6)
        //     (so SPIRV-Cross emits plain gl_VertexID/gl_InstanceID, no base term)
        //   * every OpLoad of a BaseVertex/BaseInstance/DrawIndex builtin variable is
        //     turned into a load of constant 0, leaving those variables unused so
        //     SPIRV-Cross's active-variable analysis drops them entirely.
        // VRI always draws with baseVertex/baseInstance == 0, so results are identical.
        void PatchDrawParamsForES(std::vector<uint32_t>& w)
        {
            constexpr uint32_t kOpNop = 0, kOpConstant = 43, kOpFunction = 54;
            constexpr uint32_t kOpDecorate = 71, kOpLoad = 61, kOpCopyObject = 83;
            constexpr uint32_t kDecBuiltIn = 11;
            constexpr uint32_t kVertexId = 5, kInstanceId = 6, kVertexIndex = 42, kInstanceIndex = 43;
            constexpr uint32_t kBaseVertex = 4424, kBaseInstance = 4425, kDrawIndex = 4426;

            std::unordered_set<uint32_t>          baseVars;   // builtin vars to neutralize
            std::unordered_map<uint32_t, uint32_t> zeroConst; // 32-bit type id -> a 0-constant id
            size_t firstFunc = w.size();

            for (size_t i = 5; i < w.size();)
            {
                const uint32_t op = w[i] & 0xFFFFu;
                const uint32_t len = w[i] >> 16;
                if (len == 0) return; // malformed; leave the module untouched
                if (op == kOpDecorate && len >= 4 && w[i + 2] == kDecBuiltIn)
                {
                    const uint32_t b = w[i + 3];
                    if (b == kBaseVertex || b == kBaseInstance || b == kDrawIndex) baseVars.insert(w[i + 1]);
                    else if (b == kVertexIndex) w[i + 3] = kVertexId;
                    else if (b == kInstanceIndex) w[i + 3] = kInstanceId;
                }
                else if (op == kOpConstant && len == 4 && w[i + 3] == 0)
                    zeroConst.emplace(w[i + 1], w[i + 2]); // first 0-constant per 32-bit type
                else if (op == kOpFunction && firstFunc == w.size())
                    firstFunc = i;
                i += len;
            }
            if (baseVars.empty()) return;

            uint32_t              bound = w[3];
            std::vector<uint32_t> newConstants;
            for (size_t i = 5; i < w.size();)
            {
                const uint32_t op = w[i] & 0xFFFFu;
                const uint32_t len = w[i] >> 16;
                if (len == 0) break;
                if (op == kOpLoad && len >= 4 && baseVars.count(w[i + 3]))
                {
                    const uint32_t type = w[i + 1];
                    uint32_t       c;
                    auto           it = zeroConst.find(type);
                    if (it != zeroConst.end())
                        c = it->second;
                    else
                    {
                        c = bound++;
                        zeroConst.emplace(type, c);
                        newConstants.push_back((4u << 16) | kOpConstant);
                        newConstants.push_back(type);
                        newConstants.push_back(c);
                        newConstants.push_back(0u);
                    }
                    // OpLoad %type %res %ptr -> OpCopyObject %type %res %zeroConst
                    w[i] = (4u << 16) | kOpCopyObject;
                    w[i + 3] = c;
                    for (uint32_t k = 4; k < len; ++k)
                        w[i + k] = (1u << 16) | kOpNop; // keep stream length stable
                }
                i += len;
            }
            w[3] = bound;
            if (!newConstants.empty())
                w.insert(w.begin() + static_cast<long>(firstFunc), newConstants.begin(), newConstants.end());
        }

        // Deterministic GLSL block name for a uniform buffer at a flattened unit, so
        // CreateGraphicsPipeline can glGetUniformBlockIndex it after linking (ESSL 300
        // / WebGL2 can't use layout(binding=) on uniform blocks).
        std::string UboBlockName(uint32_t glUnit) { return "VriUbo_" + std::to_string(glUnit); }

        std::string SpirvToGlsl(const DeviceGL* d, const PipelineLayoutGL* layout, const void* bytecode, size_t bytecodeSize,
                                const char* entry, spv::ExecutionModel model, std::vector<CombinedSamplerGL>* outCombined,
                                std::vector<PushMemberGL>* outPush = nullptr)
        {
            const uint32_t* words = static_cast<const uint32_t*>(bytecode);
            const size_t wordCount = bytecodeSize / 4;
            // SPIRV-Cross signals transpile errors by throwing; turn that into a
            // clean failure (empty source -> shader compile fails -> pipeline
            // returns Failure) instead of an abort, and surface the message.
            try
            {
                std::vector<uint32_t> patched(words, words + wordCount);
                if (d->IsES())
                    PatchDrawParamsForES(patched);
                spirv_cross::CompilerGLSL comp(std::move(patched));
                spirv_cross::CompilerGLSL::Options o = comp.get_common_options();
                o.version = d->ShaderVersion();
                o.es = d->IsES();
                o.vulkan_semantics = false;
                // Y orientation: with glClipControl (GL 4.5+) the context already uses the
                // VRI top-left convention, so NO in-shader flip. Otherwise (ES/WebGL2,
                // pre-4.5 desktop) flip clip-space Y in-shader - but never on the
                // tessellation-control stage (its output is gl_out[].gl_Position; a bare
                // gl_Position there is invalid GLSL and strict drivers reject it).
                o.vertex.flip_vert_y = !d->Features().clipControl && model != spv::ExecutionModelTessellationControl;
                comp.set_common_options(o);
                if (entry)
                    comp.set_entry_point(entry, model);

                // GLSL has no separate texture/sampler: fuse each Vulkan (texture,
                // sampler) pair into one combined sampler2D. Must run before
                // get_shader_resources()/compile().
                comp.build_combined_image_samplers();

                const spirv_cross::ShaderResources res = comp.get_shader_resources();

                // ESSL 300 / WebGL2 match inter-stage varyings by NAME (location
                // qualifiers aren't allowed there), but compiling the stages separately
                // yields mismatched names. Rename each varying deterministically by its
                // location so the vertex output and fragment input agree.
                auto renameByLocation = [&](const spirv_cross::SmallVector<spirv_cross::Resource>& vars) {
                    for (const spirv_cross::Resource& r : vars)
                    {
                        const uint32_t loc = comp.get_decoration(r.id, spv::DecorationLocation);
                        comp.set_name(r.id, "vriVarying" + std::to_string(loc));
                    }
                };
                if (model == spv::ExecutionModelVertex)
                    renameByLocation(res.stage_outputs);
                else if (model == spv::ExecutionModelFragment)
                    renameByLocation(res.stage_inputs);

                // Remap each Vulkan (set, binding) to the flattened GL unit from the
                // pipeline layout, and give resources deterministic names so binding
                // points can be assigned after link (glUniformBlockBinding / glUniform1i).
                if (layout)
                {
                    for (const spirv_cross::Resource& u : res.uniform_buffers)
                    {
                        const uint32_t set = comp.get_decoration(u.id, spv::DecorationDescriptorSet);
                        const uint32_t bind = comp.get_decoration(u.id, spv::DecorationBinding);
                        if (const LayoutBindingGL* lb = layout->Find(set, bind))
                        {
                            comp.unset_decoration(u.id, spv::DecorationDescriptorSet);
                            comp.set_decoration(u.id, spv::DecorationBinding, lb->glUnit);
                            comp.set_name(u.base_type_id, UboBlockName(lb->glUnit));
                        }
                    }
                    // Storage buffers (SSBO / UAV): remap to the flattened SSBO unit;
                    // layout(binding=) on SSBOs works in GLSL 430 (desktop compute).
                    for (const spirv_cross::Resource& s : res.storage_buffers)
                    {
                        const uint32_t set = comp.get_decoration(s.id, spv::DecorationDescriptorSet);
                        const uint32_t bind = comp.get_decoration(s.id, spv::DecorationBinding);
                        if (const LayoutBindingGL* lb = layout->Find(set, bind))
                        {
                            comp.unset_decoration(s.id, spv::DecorationDescriptorSet);
                            comp.set_decoration(s.id, spv::DecorationBinding, lb->glUnit);
                        }
                    }
                    for (const spirv_cross::CombinedImageSampler& cis : comp.get_combined_image_samplers())
                    {
                        const uint32_t texSet = comp.get_decoration(cis.image_id, spv::DecorationDescriptorSet);
                        const uint32_t texBind = comp.get_decoration(cis.image_id, spv::DecorationBinding);
                        const uint32_t smpSet = comp.get_decoration(cis.sampler_id, spv::DecorationDescriptorSet);
                        const uint32_t smpBind = comp.get_decoration(cis.sampler_id, spv::DecorationBinding);
                        const LayoutBindingGL* lb = layout->Find(texSet, texBind);
                        if (!lb)
                            continue;
                        const uint32_t unit = lb->glUnit;
                        const std::string name = "VriTex_" + std::to_string(unit);
                        comp.set_name(cis.combined_id, name);
                        comp.unset_decoration(cis.combined_id, spv::DecorationDescriptorSet);
                        comp.set_decoration(cis.combined_id, spv::DecorationBinding, unit);
                        if (outCombined)
                            outCombined->push_back(CombinedSamplerGL{unit, texSet, texBind, smpSet, smpBind, name});
                    }
                }
                // Push constants: SPIRV-Cross (vulkan_semantics=false) lowers the block to a
                // default-block struct uniform, so rename the instance to "vriPush" and record
                // each leaf member for glUniform-by-name after link (see CmdSetConstants).
                if (outPush)
                    for (const spirv_cross::Resource& pc : res.push_constant_buffers)
                    {
                        comp.set_name(pc.id, "vriPush");
                        const spirv_cross::SPIRType& st = comp.get_type(pc.base_type_id);
                        for (uint32_t m = 0; m < static_cast<uint32_t>(st.member_types.size()); ++m)
                        {
                            const spirv_cross::SPIRType& mt = comp.get_type(st.member_types[m]);
                            PushMemberGL pm{};
                            pm.name = "vriPush." + comp.get_member_name(pc.base_type_id, m);
                            pm.offset = comp.type_struct_member_offset(st, m);
                            pm.basetype = mt.basetype == spirv_cross::SPIRType::Int ? 1u
                                        : (mt.basetype == spirv_cross::SPIRType::UInt ? 2u : 0u);
                            pm.vecsize = mt.vecsize;
                            pm.columns = mt.columns;
                            pm.count = mt.array.empty() ? 1u : mt.array[0];
                            bool dup = false;
                            for (const PushMemberGL& e : *outPush) if (e.name == pm.name) { dup = true; break; }
                            if (!dup) outPush->push_back(pm);
                        }
                    }
                std::string glsl = comp.compile();
                // Tessellation-control outputs may only be indexed by gl_InvocationID;
                // strict drivers (NVIDIA) reject SPIRV-Cross's uint(gl_InvocationID) cast
                // on the output write, so strip it (reads stay valid too).
                if (model == spv::ExecutionModelTessellationControl)
                {
                    const std::string from = "uint(gl_InvocationID)";
                    const std::string to = "gl_InvocationID";
                    for (size_t p = glsl.find(from); p != std::string::npos; p = glsl.find(from, p + to.size()))
                        glsl.replace(p, from.size(), to);
                }
                if (std::getenv("VRI_DUMP_GLSL"))
                    std::fprintf(stderr, "=== VRI GLSL model=%d entry=%s ===\n%s\n", static_cast<int>(model), entry ? entry : "?", glsl.c_str());
                return glsl;
            }
            catch (const std::exception& e)
            {
                std::string msg = "SPIRV-Cross transpile failed: ";
                msg += e.what();
                d->ReportError(msg.c_str());
                return std::string();
            }
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
            GLenum usage = GL_STATIC_DRAW;
            GLbitfield mapAccess = 0;
            if (desc->memoryLocation == VriMemoryLocation_HostReadback) { usage = GL_STREAM_READ; mapAccess = GL_MAP_READ_BIT; }
            else if (desc->memoryLocation == VriMemoryLocation_HostUpload) { usage = GL_STREAM_DRAW; mapAccess = GL_MAP_WRITE_BIT; }
            // Index buffers must stay element-class on WebGL (see BufferGL::target);
            // everything else uses the neutral GL_COPY_WRITE_BUFFER binding point.
            const GLenum target = (desc->usage & VriBufferUsage_IndexBuffer) ? GL_ELEMENT_ARRAY_BUFFER : GL_COPY_WRITE_BUFFER;
            void* persistentPtr = nullptr;
#if !defined(__EMSCRIPTEN__)
            if (Dev(device)->Features().dsa) // GL 4.5 DSA (implies 4.4 immutable buffer storage)
            {
                glCreateBuffers(1, &id);
                // Immutable storage. Mappable buffers are persistently + coherently mapped
                // once here (MapBuffer just returns that pointer; UnmapBuffer is a no-op),
                // avoiding per-map driver overhead; device-local buffers get DYNAMIC_STORAGE
                // so glNamedBufferSubData stays legal (copies work on immutable storage too).
                GLbitfield storageFlags = mapAccess ? (mapAccess | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT)
                                                    : GL_DYNAMIC_STORAGE_BIT;
                glNamedBufferStorage(id, static_cast<GLsizeiptr>(desc->size), nullptr, storageFlags);
                if (mapAccess)
                    persistentPtr = glMapNamedBufferRange(id, 0, static_cast<GLsizeiptr>(desc->size),
                                                          mapAccess | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
            }
            else
#endif
            {
                glGenBuffers(1, &id);
                glBindBuffer(target, id);
                glBufferData(target, static_cast<GLsizeiptr>(desc->size), nullptr, usage);
                glBindBuffer(target, 0);
            }
            BufferGL* b = new BufferGL{};
            b->device = Dev(device); b->id = id; b->size = desc->size; b->target = target; b->mapAccess = mapAccess;
            b->persistentPtr = persistentPtr;
            *out = ToHandle(b);
            return VriResult_Success;
        }
        void VRI_CALL DestroyBuffer(VriBuffer* buffer)
        {
            if (!buffer) return;
            BufferGL* b = Buf(buffer);
#if !defined(__EMSCRIPTEN__)
            if (b->persistentPtr) glUnmapNamedBuffer(b->id); // release the persistent mapping (glDeleteBuffers would too)
#endif
            glDeleteBuffers(1, &b->id);
            delete b;
        }
        void* VRI_CALL MapBuffer(VriBuffer* buffer, uint64_t offset, uint64_t size)
        {
            BufferGL* b = Buf(buffer);
            const GLbitfield access = b->mapAccess ? b->mapAccess : GL_MAP_READ_BIT;
            const uint64_t len = size ? size : (b->size - offset);
            if (b->device->IsES())
            {
                // WebGL2 cannot map buffers: stage through a CPU shadow.
                b->shadow = std::malloc(static_cast<size_t>(len));
                b->mapOffset = offset;
                b->mapLen = len;
                b->mapWrite = (access & GL_MAP_WRITE_BIT) != 0;
                if (access & GL_MAP_READ_BIT)
                {
                    glBindBuffer(b->target, b->id);
                    glGetBufferSubData(b->target, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(len), b->shadow);
                    glBindBuffer(b->target, 0);
                }
                return b->shadow;
            }
            if (b->persistentPtr) // GL 4.4: persistently mapped at creation, return that pointer
                return static_cast<char*>(b->persistentPtr) + offset;
#if !defined(__EMSCRIPTEN__)
            if (b->device->Features().dsa) // GL 4.5: map without binding
                return glMapNamedBufferRange(b->id, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(len), access);
#endif
            glBindBuffer(b->target, b->id);
            return glMapBufferRange(b->target, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(len), access);
        }
        void VRI_CALL UnmapBuffer(VriBuffer* buffer)
        {
            BufferGL* b = Buf(buffer);
            if (b->persistentPtr) // GL 4.4 coherent persistent mapping: nothing to flush/unmap
                return;
            if (b->shadow)
            {
                if (b->mapWrite)
                {
                    glBindBuffer(b->target, b->id);
                    glBufferSubData(b->target, static_cast<GLintptr>(b->mapOffset), static_cast<GLsizeiptr>(b->mapLen), b->shadow);
                    glBindBuffer(b->target, 0);
                }
                std::free(b->shadow);
                b->shadow = nullptr;
                return;
            }
#if !defined(__EMSCRIPTEN__)
            if (b->device->Features().dsa) // GL 4.5: unmap without binding
            {
                glUnmapNamedBuffer(b->id);
                return;
            }
#endif
            glBindBuffer(b->target, b->id);
            glUnmapBuffer(b->target);
            glBindBuffer(b->target, 0);
        }
        uint64_t VRI_CALL GetBufferDeviceAddress(const VriBuffer*) { return 0; }

        VriResult VRI_CALL CreateTexture(VriDevice* device, const VriTextureDesc* desc, VriTexture** out)
        {
            const GLFormat gf = ToGLFormat(desc->format);
            const GLsizei mips = static_cast<GLsizei>(desc->mipNum ? desc->mipNum : 1u);
            const GLsizei w = static_cast<GLsizei>(desc->width);
            const GLsizei h = static_cast<GLsizei>(desc->height ? desc->height : 1u);
            const bool ms = desc->sampleNum > 1;

#if !defined(__EMSCRIPTEN__) // DSA (4.5) entry points are undeclared in the GLES3/WebGL2 headers
            const bool dsa = Dev(device)->Features().dsa; // GL 4.5: glCreateTextures + glTextureStorage, no bind-to-edit
#endif
            GLuint id = 0;
            GLenum target = GL_TEXTURE_2D;
            bool isRenderbuffer = false;
            if (ms)
            {
#if defined(__EMSCRIPTEN__)
                // WebGL2 has no multisample textures - use a multisample renderbuffer
                // (render-only; resolved to a single-sample texture).
                glGenRenderbuffers(1, &id);
                glBindRenderbuffer(GL_RENDERBUFFER, id);
                glRenderbufferStorageMultisample(GL_RENDERBUFFER, static_cast<GLsizei>(desc->sampleNum), gf.internalFormat, w, h);
                glBindRenderbuffer(GL_RENDERBUFFER, 0);
                target = GL_RENDERBUFFER;
                isRenderbuffer = true;
#else
                // Desktop GL: a multisample texture (samplable via sampler2DMS/texelFetch -
                // more capable than a renderbuffer), attached + resolved like any target.
                target = GL_TEXTURE_2D_MULTISAMPLE;
                if (dsa)
                {
                    glCreateTextures(target, 1, &id);
                    glTextureStorage2DMultisample(id, static_cast<GLsizei>(desc->sampleNum), gf.internalFormat, w, h, GL_TRUE);
                }
                else
                {
                    glGenTextures(1, &id);
                    glBindTexture(target, id);
                    glTexStorage2DMultisample(target, static_cast<GLsizei>(desc->sampleNum), gf.internalFormat, w, h, GL_TRUE);
                    glBindTexture(target, 0);
                }
#endif
            }
#if !defined(__EMSCRIPTEN__)
            else if (dsa)
            {
                glCreateTextures(GL_TEXTURE_2D, 1, &id);
                glTextureStorage2D(id, mips, gf.internalFormat, w, h);
            }
#endif
            else
            {
                glGenTextures(1, &id);
                glBindTexture(GL_TEXTURE_2D, id);
                glTexStorage2D(GL_TEXTURE_2D, mips, gf.internalFormat, w, h);
                glBindTexture(GL_TEXTURE_2D, 0);
            }

            TextureGL* t = new TextureGL{};
            t->device = Dev(device);
            t->id = id;
            t->target = target;
            t->isRenderbuffer = isRenderbuffer;
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
            t->device->EvictFbosReferencing(t->id); // drop cached render-pass FBOs using it
            if (t->isRenderbuffer) glDeleteRenderbuffers(1, &t->id);
            else glDeleteTextures(1, &t->id);
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
#if !defined(__EMSCRIPTEN__)
            if (Dev(device)->Features().dsa) glCreateSamplers(1, &s); // GL 4.5: pre-initialized sampler object
            else
#endif
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
        VriResult VRI_CALL CreatePipelineLayout(VriDevice* device, const VriPipelineLayoutDesc* desc, VriPipelineLayout** out)
        {
            PipelineLayoutGL* l = new PipelineLayoutGL{Dev(device), {}};
            // Flatten (set, binding) -> per-type GL unit. GL has independent binding
            // namespaces for uniform buffers, shader-storage buffers and texture
            // units, so each type gets its own running counter.
            uint32_t uboNext = 0, ssboNext = 0, texNext = 0;
            for (uint32_t s = 0; s < desc->descriptorSetNum; ++s)
            {
                const VriDescriptorSetDesc& sd = desc->descriptorSets[s];
                for (uint32_t r = 0; r < sd.rangeNum; ++r)
                {
                    const VriDescriptorRangeDesc& rd = sd.ranges[r];
                    // Samplers don't take a unit of their own: GLSL fuses them with a
                    // texture, so the combined sampler rides the texture's unit.
                    uint32_t* counter = nullptr;
                    if (rd.descriptorType == VriDescriptorType_ConstantBuffer) counter = &uboNext;
                    else if (rd.descriptorType == VriDescriptorType_StorageBuffer ||
                             rd.descriptorType == VriDescriptorType_StructuredBuffer) counter = &ssboNext;
                    else if (rd.descriptorType == VriDescriptorType_Texture ||
                             rd.descriptorType == VriDescriptorType_StorageTexture) counter = &texNext;
                    LayoutBindingGL b{};
                    b.set = sd.registerSpace;
                    b.binding = rd.baseRegister;
                    b.type = rd.descriptorType;
                    b.glUnit = counter ? *counter : 0u;
                    if (counter)
                        *counter += (rd.descriptorNum ? rd.descriptorNum : 1u);
                    l->bindings.push_back(b);
                }
            }
            *out = ToHandle(l);
            return VriResult_Success;
        }
        void VRI_CALL DestroyPipelineLayout(VriPipelineLayout* layout) { delete reinterpret_cast<PipelineLayoutGL*>(layout); }

        VriResult VRI_CALL CreateGraphicsPipeline(VriDevice* device, const VriGraphicsPipelineDesc* desc, VriPipeline** out)
        {
            DeviceGL* d = Dev(device);
            const PipelineLayoutGL* layout = reinterpret_cast<const PipelineLayoutGL*>(desc->pipelineLayout);
            // Legacy stages (geometry/tessellation) exist only on desktop GL, not on
            // GLES3/WebGL2 - reject explicitly rather than silently dropping them.
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                const VriShaderStageBits st = desc->shaders[i].stage;
                if (st == VriShaderStage_Geometry && !d->Desc().hasGeometryShader)
                    return VriResult_Unsupported;
                if ((st == VriShaderStage_TessControl || st == VriShaderStage_TessEval) && !d->Desc().hasTessellation)
                    return VriResult_Unsupported;
            }

            std::vector<CombinedSamplerGL> combined;
            std::vector<PushMemberGL> pushMembers;
            GLuint vs = 0, fs = 0, gs = 0, tcs = 0, tes = 0;
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                const VriShaderDesc& s = desc->shaders[i];
                if (s.stage == VriShaderStage_Vertex)
                    vs = CompileShader(d, GL_VERTEX_SHADER, SpirvToGlsl(d, layout, s.bytecode, s.bytecodeSize, s.entryPointName, spv::ExecutionModelVertex, &combined, &pushMembers));
                else if (s.stage == VriShaderStage_Fragment)
                    fs = CompileShader(d, GL_FRAGMENT_SHADER, SpirvToGlsl(d, layout, s.bytecode, s.bytecodeSize, s.entryPointName, spv::ExecutionModelFragment, &combined, &pushMembers));
#if !defined(__EMSCRIPTEN__) // geometry/tessellation shaders are desktop-only (absent in GLES3/WebGL2 headers)
                else if (s.stage == VriShaderStage_Geometry)
                    gs = CompileShader(d, GL_GEOMETRY_SHADER, SpirvToGlsl(d, layout, s.bytecode, s.bytecodeSize, s.entryPointName, spv::ExecutionModelGeometry, &combined));
                else if (s.stage == VriShaderStage_TessControl)
                    tcs = CompileShader(d, GL_TESS_CONTROL_SHADER, SpirvToGlsl(d, layout, s.bytecode, s.bytecodeSize, s.entryPointName, spv::ExecutionModelTessellationControl, &combined));
                else if (s.stage == VriShaderStage_TessEval)
                    tes = CompileShader(d, GL_TESS_EVALUATION_SHADER, SpirvToGlsl(d, layout, s.bytecode, s.bytecodeSize, s.entryPointName, spv::ExecutionModelTessellationEvaluation, &combined));
#endif
            }
            auto deleteStages = [&] {
                if (vs) glDeleteShader(vs);
                if (fs) glDeleteShader(fs);
                if (gs) glDeleteShader(gs);
                if (tcs) glDeleteShader(tcs);
                if (tes) glDeleteShader(tes);
            };
            if (!vs || !fs)
            {
                deleteStages();
                return VriResult_Failure;
            }
            GLuint program = glCreateProgram();
            glAttachShader(program, vs);
            glAttachShader(program, fs);
            if (gs) glAttachShader(program, gs);
            if (tcs) glAttachShader(program, tcs);
            if (tes) glAttachShader(program, tes);
            glLinkProgram(program);
            glDetachShader(program, vs);
            glDetachShader(program, fs);
            if (gs) glDetachShader(program, gs);
            if (tcs) glDetachShader(program, tcs);
            if (tes) glDetachShader(program, tes);
            glDeleteShader(vs);
            glDeleteShader(fs);
            if (gs) glDeleteShader(gs);
            if (tcs) glDeleteShader(tcs);
            if (tes) glDeleteShader(tes);
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

            // Assign each uniform block its flattened binding point. Portable across
            // desktop GL and ESSL 300 / WebGL2 (which can't use layout(binding=)).
            if (layout)
            {
                for (const LayoutBindingGL& b : layout->bindings)
                    if (b.type == VriDescriptorType_ConstantBuffer)
                    {
                        const GLuint idx = glGetUniformBlockIndex(program, UboBlockName(b.glUnit).c_str());
                        if (idx != GL_INVALID_INDEX)
                            glUniformBlockBinding(program, idx, b.glUnit);
                    }
            }
            // Resolve each emulated push-constant member to its default-block uniform
            // location (set per draw in CmdSetConstants).
            for (PushMemberGL& pm : pushMembers)
                pm.location = glGetUniformLocation(program, pm.name.c_str());
            // Point each combined sampler uniform at its texture unit (glUniform1i is
            // the portable way; ESSL 300 / WebGL2 can't use layout(binding=) either).
            if (!combined.empty())
            {
                glUseProgram(program);
                for (const CombinedSamplerGL& cs : combined)
                {
                    const GLint loc = glGetUniformLocation(program, cs.name.c_str());
                    if (loc >= 0)
                        glUniform1i(loc, static_cast<GLint>(cs.unit));
                }
                glUseProgram(0);
            }

            PipelineGL* p = new PipelineGL{};
            p->device = d;
            p->program = program;
            p->combinedSamplers = std::move(combined);
            p->pushMembers = std::move(pushMembers);
            // Base-offset fallback uniforms (present only when the driver lacks
            // ARB_shader_draw_parameters; -1 means the gl_Base*ARB builtin path is used).
            p->baseVertexLoc = glGetUniformLocation(program, "SPIRV_Cross_BaseVertex");
            p->baseInstanceLoc = glGetUniformLocation(program, "SPIRV_Cross_BaseInstance");
            // Vertex input: attribute index i == GLSL location i (matches the VK
            // backend). Flatten each attribute with its stream's stride + slot for
            // the classic glVertexAttribPointer path used at draw time.
            for (uint32_t i = 0; i < desc->vertexInput.attributeNum; ++i)
            {
                const VriVertexAttributeDesc& a = desc->vertexInput.attributes[i];
                const GLVertexFormat vf = ToGLVertexFormat(a.format);
                GLsizei stride = 0;
                uint32_t bindingSlot = a.streamIndex;
                uint32_t divisor = 0;
                if (a.streamIndex < desc->vertexInput.streamNum)
                {
                    stride = static_cast<GLsizei>(desc->vertexInput.streams[a.streamIndex].stride);
                    bindingSlot = desc->vertexInput.streams[a.streamIndex].bindingSlot;
                    divisor = desc->vertexInput.streams[a.streamIndex].stepRate == VriVertexStepRate_PerInstance ? 1u : 0u;
                }
                p->vertexAttribs.push_back(VertexAttribGL{i, vf.size, vf.type, vf.normalized, a.offset, stride, bindingSlot, divisor});
            }
            // Separate attribute format (GL 4.3+): bake the vertex format into a
            // per-pipeline VAO once. At draw we only point each stream binding at a
            // buffer (glBindVertexBuffer), avoiding per-draw glVertexAttribPointer.
            // Pre-4.3 / GLES / WebGL2 keep the classic path (formatVao stays 0); the
            // 4.3+ entry points are undeclared in the Emscripten GLES3 headers.
#if !defined(__EMSCRIPTEN__)
            if (d->Features().separateAttrib)
            {
                const bool dsa = d->Features().dsa; // 4.5: configure the VAO by name
                GLuint vao = 0;
                if (dsa)
                    glCreateVertexArrays(1, &vao);
                else { glGenVertexArrays(1, &vao); glBindVertexArray(vao); }
                for (const VertexAttribGL& a : p->vertexAttribs)
                {
                    if (dsa)
                    {
                        glEnableVertexArrayAttrib(vao, a.location);
                        glVertexArrayAttribFormat(vao, a.location, a.size, a.type, a.normalized, a.offset);
                        glVertexArrayAttribBinding(vao, a.location, a.bindingSlot);
                        glVertexArrayBindingDivisor(vao, a.bindingSlot, a.divisor);
                    }
                    else
                    {
                        glEnableVertexAttribArray(a.location);
                        glVertexAttribFormat(a.location, a.size, a.type, a.normalized, a.offset);
                        glVertexAttribBinding(a.location, a.bindingSlot);
                        glVertexBindingDivisor(a.bindingSlot, a.divisor);
                    }
                }
                if (!dsa) glBindVertexArray(d->DefaultVao()); // restore; DSA never bound it
                p->formatVao = vao;
                // Unique stream slots (with stride) to rebind per draw.
                for (const VertexAttribGL& a : p->vertexAttribs)
                {
                    bool seen = false;
                    for (const VboBindingGL& vb : p->vboBindings)
                        if (vb.slot == a.bindingSlot) { seen = true; break; }
                    if (!seen) p->vboBindings.push_back(VboBindingGL{a.bindingSlot, a.stride, a.divisor});
                }
            }
#endif // !__EMSCRIPTEN__ (separate-format VAO)
            p->topology = ToGLTopology(desc->inputAssembly.topology);
            p->patchVertices = desc->tessellation.patchControlPoints;
            p->cullEnable = desc->rasterization.cullMode != VriCullMode_None;
            p->cullFace = desc->rasterization.cullMode == VriCullMode_Front ? GL_FRONT : GL_BACK;
            // Window-space winding depends on how VRI's Y is oriented: glClipControl
            // (UPPER_LEFT) makes window Y-down like VRI, so a VRI CCW face stays GL CCW;
            // the in-shader flip_vert_y path negates clip Y, reversing it (CCW -> GL CW).
            const bool ccw = desc->rasterization.frontFace == VriFrontFace_CounterClockwise;
            p->frontFace = d->Features().clipControl ? (ccw ? GL_CCW : GL_CW) : (ccw ? GL_CW : GL_CCW);
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
            p->stencilEnable = desc->depthStencil.stencilTest != VRI_FALSE;
            auto toGLFace = [](const VriStencilOpDesc& s) {
                return PipelineGL::StencilFaceGL{ToGLCompareOp(s.compareOp), ToGLStencilOp(s.failOp), ToGLStencilOp(s.depthFailOp),
                                                 ToGLStencilOp(s.passOp), s.compareMask, s.writeMask, static_cast<GLint>(s.reference)};
            };
            p->stencilFront = toGLFace(desc->depthStencil.front);
            p->stencilBack = toGLFace(desc->depthStencil.back);
            *out = ToHandle(p);
            return VriResult_Success;
        }
        VriResult VRI_CALL CreateComputePipeline(VriDevice* device, const VriComputePipelineDesc* desc, VriPipeline** out)
        {
            DeviceGL* d = Dev(device);
            if (!d->Desc().hasComputeShader)
                return VriResult_Unsupported; // GLES 3.0 / WebGL2 has no compute
#if defined(__EMSCRIPTEN__)
            return VriResult_Unsupported; // GL_COMPUTE_SHADER is not in WebGL2
#else
            const PipelineLayoutGL* layout = reinterpret_cast<const PipelineLayoutGL*>(desc->pipelineLayout);
            const GLuint cs = CompileShader(d, GL_COMPUTE_SHADER, SpirvToGlsl(d, layout, desc->shader.bytecode, desc->shader.bytecodeSize, desc->shader.entryPointName, spv::ExecutionModelGLCompute, nullptr));
            if (!cs)
                return VriResult_Failure;
            GLuint program = glCreateProgram();
            glAttachShader(program, cs);
            glLinkProgram(program);
            glDetachShader(program, cs);
            glDeleteShader(cs);
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
            // Uniform blocks (if any) still need binding points; SSBOs use the
            // layout(binding=) emitted by SPIRV-Cross (GLSL 430).
            if (layout)
            {
                glUseProgram(program);
                for (const LayoutBindingGL& b : layout->bindings)
                    if (b.type == VriDescriptorType_ConstantBuffer)
                    {
                        const GLuint idx = glGetUniformBlockIndex(program, UboBlockName(b.glUnit).c_str());
                        if (idx != GL_INVALID_INDEX)
                            glUniformBlockBinding(program, idx, b.glUnit);
                    }
                glUseProgram(0);
            }
            PipelineGL* p = new PipelineGL{};
            p->device = d;
            p->program = program;
            p->isCompute = true;
            *out = ToHandle(p);
            return VriResult_Success;
#endif // !__EMSCRIPTEN__
        }
        void VRI_CALL DestroyPipeline(VriPipeline* pipeline)
        {
            if (!pipeline) return;
            PipelineGL* p = Pipe(pipeline);
            glDeleteProgram(p->program);
            if (p->formatVao) glDeleteVertexArrays(1, &p->formatVao);
            delete p;
        }

        // ---- descriptor pools / sets ---------------------------------------
        // GL has no descriptor objects: a set is a CPU record of (binding -> view)
        // that CmdSetDescriptorSet replays as glBindBufferRange / texture binds via
        // the pipeline layout's (set,binding) -> GL-unit map.
        VriResult VRI_CALL CreateDescriptorPool(VriDevice* device, const VriDescriptorPoolDesc*, VriDescriptorPool** out)
        {
            *out = ToHandle(new DescriptorPoolGL{Dev(device)});
            return VriResult_Success;
        }
        void VRI_CALL ResetDescriptorPool(VriDescriptorPool*) {}
        void VRI_CALL DestroyDescriptorPool(VriDescriptorPool* pool) { delete reinterpret_cast<DescriptorPoolGL*>(pool); }
        VriResult VRI_CALL AllocateDescriptorSets(VriDescriptorPool* pool, const VriPipelineLayout* layout, uint32_t setIndex, VriDescriptorSet** outSets, uint32_t setNum)
        {
            DescriptorPoolGL* p = reinterpret_cast<DescriptorPoolGL*>(pool);
            const PipelineLayoutGL* l = reinterpret_cast<const PipelineLayoutGL*>(layout);
            for (uint32_t i = 0; i < setNum; ++i)
                outSets[i] = ToHandle(new DescriptorSetGL{p->device, l, setIndex, {}});
            return VriResult_Success;
        }
        void VRI_CALL UpdateDescriptorRanges(VriDescriptorSet* set, uint32_t baseRange, uint32_t rangeNum, const VriDescriptorRangeUpdateDesc* updates)
        {
            DescriptorSetGL* s = reinterpret_cast<DescriptorSetGL*>(set);
            // The layout's bindings for this set, in declaration order, line up with
            // the ranges; range r writes the view at that range's binding.
            std::vector<const LayoutBindingGL*> setBindings;
            for (const LayoutBindingGL& b : s->layout->bindings)
                if (b.set == s->setIndex)
                    setBindings.push_back(&b);
            for (uint32_t r = 0; r < rangeNum; ++r)
            {
                const uint32_t idx = baseRange + r;
                if (idx >= setBindings.size() || updates[r].descriptorNum == 0)
                    continue;
                s->bound[setBindings[idx]->binding] = reinterpret_cast<const DescriptorGL*>(updates[r].descriptors[0]);
            }
        }

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

            // FBO identity = the attachment set. Reuse a cached, already-configured FBO
            // (no per-pass gen/delete); only a freshly-created one needs configuring.
            FboKey key{};
            key.colorCount = a->colorNum <= FboKey::kMaxColor ? a->colorNum : FboKey::kMaxColor;
            for (uint32_t i = 0; i < key.colorCount; ++i)
            {
                const DescriptorGL* v = Desc(a->colors[i].view);
                key.colorId[i] = v->texture->id;
                key.colorMip[i] = v->mip;
            }
            if (a->depth) { const DescriptorGL* dv = Desc(a->depth->view); key.depthId = dv->texture->id; key.depthMip = dv->mip; }

#if !defined(__EMSCRIPTEN__) // GL 4.5: configure + clear the FBO by name (undeclared on GLES3)
            const bool dsa = c->device->Features().dsa;
#endif
            bool isNew = false;
            const GLuint fbo = c->device->AcquireFbo(key, isNew);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            c->fbo = fbo;

            if (isNew) // configure attachments + draw buffers once for this attachment set
            {
                std::vector<GLenum> drawBufs(a->colorNum);
                for (uint32_t i = 0; i < a->colorNum; ++i)
                {
                    const DescriptorGL* v = Desc(a->colors[i].view);
                    const GLenum att = GL_COLOR_ATTACHMENT0 + i;
                    if (v->texture->isRenderbuffer) // MSAA renderbuffer attachment
                    {
#if !defined(__EMSCRIPTEN__)
                        if (dsa) glNamedFramebufferRenderbuffer(fbo, att, GL_RENDERBUFFER, v->texture->id);
                        else
#endif
                        glFramebufferRenderbuffer(GL_FRAMEBUFFER, att, GL_RENDERBUFFER, v->texture->id);
                    }
                    else
                    {
#if !defined(__EMSCRIPTEN__)
                        if (dsa) glNamedFramebufferTexture(fbo, att, v->texture->id, static_cast<GLint>(v->mip));
                        else
#endif
                        glFramebufferTexture2D(GL_FRAMEBUFFER, att, v->texture->target, v->texture->id, static_cast<GLint>(v->mip));
                    }
                    drawBufs[i] = att;
                }
#if !defined(__EMSCRIPTEN__)
                if (dsa) glNamedFramebufferDrawBuffers(fbo, static_cast<GLsizei>(drawBufs.size()), drawBufs.data());
                else
#endif
                glDrawBuffers(static_cast<GLsizei>(drawBufs.size()), drawBufs.data());
                if (a->depth)
                {
                    const DescriptorGL* dv = Desc(a->depth->view);
                    const GLenum attach = dv->texture->glFormat == GL_DEPTH_STENCIL ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT;
#if !defined(__EMSCRIPTEN__)
                    if (dsa) glNamedFramebufferTexture(fbo, attach, dv->texture->id, static_cast<GLint>(dv->mip));
                    else
#endif
                    glFramebufferTexture2D(GL_FRAMEBUFFER, attach, dv->texture->target, dv->texture->id, static_cast<GLint>(dv->mip));
                }
            }

            // Per-pass: MSAA resolve targets (recorded for EndRendering) + load/clear ops.
            c->pendingResolves.clear();
            for (uint32_t i = 0; i < a->colorNum; ++i)
                if (a->colors[i].resolveView)
                {
                    const DescriptorGL* rv = Desc(a->colors[i].resolveView);
                    c->pendingResolves.push_back({GL_COLOR_ATTACHMENT0 + i, rv->texture, static_cast<GLuint>(rv->mip)});
                }
            for (uint32_t i = 0; i < a->colorNum; ++i)
                if (a->colors[i].loadOp == VriAttachmentLoadOp_Clear)
                {
                    const GLfloat* col = a->colors[i].clearValue.color.f32;
#if !defined(__EMSCRIPTEN__)
                    if (dsa) glClearNamedFramebufferfv(fbo, GL_COLOR, static_cast<GLint>(i), col);
                    else
#endif
                    glClearBufferfv(GL_COLOR, static_cast<GLint>(i), col);
                }
            if (a->depth && a->depth->loadOp == VriAttachmentLoadOp_Clear)
            {
                const bool hasStencil = Desc(a->depth->view)->texture->glFormat == GL_DEPTH_STENCIL;
                glDepthMask(GL_TRUE); // clears respect the write masks
                if (hasStencil)
                {
                    glStencilMask(0xFFu);
                    const GLfloat depth = a->depth->clearValue.depthStencil.depth;
                    const GLint stencil = static_cast<GLint>(a->depth->clearValue.depthStencil.stencil);
#if !defined(__EMSCRIPTEN__)
                    if (dsa) glClearNamedFramebufferfi(fbo, GL_DEPTH_STENCIL, 0, depth, stencil);
                    else
#endif
                    glClearBufferfi(GL_DEPTH_STENCIL, 0, depth, stencil);
                }
                else
                {
                    const GLfloat d = a->depth->clearValue.depthStencil.depth;
#if !defined(__EMSCRIPTEN__)
                    if (dsa) glClearNamedFramebufferfv(fbo, GL_DEPTH, 0, &d);
                    else
#endif
                    glClearBufferfv(GL_DEPTH, 0, &d);
                }
            }
        }
        void VRI_CALL CmdEndRendering(VriCommandBuffer* cmd)
        {
            CommandBufferGL* c = CB(cmd);
            // Resolve MSAA color attachments into their single-sample targets via blit
            // (the multisample read FBO -> single-sample draw FBO performs the resolve).
            if (!c->pendingResolves.empty())
            {
                GLuint resolveFbo = 0;
                glGenFramebuffers(1, &resolveFbo);
                for (const PendingResolveGL& r : c->pendingResolves)
                {
                    glBindFramebuffer(GL_READ_FRAMEBUFFER, c->fbo);
                    glReadBuffer(r.srcAttachment);
                    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, resolveFbo);
                    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, r.dst->target, r.dst->id, static_cast<GLint>(r.dstMip));
                    const GLenum drawBuf = GL_COLOR_ATTACHMENT0;
                    glDrawBuffers(1, &drawBuf);
                    glBlitFramebuffer(0, 0, static_cast<GLint>(r.dst->width), static_cast<GLint>(r.dst->height),
                                      0, 0, static_cast<GLint>(r.dst->width), static_cast<GLint>(r.dst->height),
                                      GL_COLOR_BUFFER_BIT, GL_NEAREST);
                }
                glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
                glDeleteFramebuffers(1, &resolveFbo);
                c->pendingResolves.clear();
            }
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            c->fbo = 0; // the render-pass FBO is owned by the device cache, not deleted here
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
        void VRI_CALL CmdSetPipelineLayout(VriCommandBuffer* cmd, VriPipelineLayout* layout)
        {
            CB(cmd)->boundLayout = reinterpret_cast<const PipelineLayoutGL*>(layout);
        }
        void VRI_CALL CmdSetPipeline(VriCommandBuffer* cmd, VriPipeline* pipeline)
        {
            CommandBufferGL* c = CB(cmd);
            PipelineGL* p = Pipe(pipeline);
            c->boundPipeline = p;
            glUseProgram(p->program);
            if (p->isCompute)
                return; // no graphics state to apply
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
            if (p->stencilEnable)
            {
                glEnable(GL_STENCIL_TEST);
                const auto& f = p->stencilFront;
                const auto& b = p->stencilBack;
                glStencilFuncSeparate(GL_FRONT, f.func, f.ref, f.compareMask);
                glStencilOpSeparate(GL_FRONT, f.sfail, f.dpfail, f.dppass);
                glStencilMaskSeparate(GL_FRONT, f.writeMask);
                glStencilFuncSeparate(GL_BACK, b.func, b.ref, b.compareMask);
                glStencilOpSeparate(GL_BACK, b.sfail, b.dpfail, b.dppass);
                glStencilMaskSeparate(GL_BACK, b.writeMask);
            }
            else glDisable(GL_STENCIL_TEST);
            c->topology = p->topology;
#if !defined(__EMSCRIPTEN__) && defined(GL_PATCHES) && defined(GL_PATCH_VERTICES) // tessellation: desktop GL only
            if (p->topology == GL_PATCHES && p->patchVertices > 0)
                glPatchParameteri(GL_PATCH_VERTICES, static_cast<GLint>(p->patchVertices));
#endif
        }
        void VRI_CALL CmdSetDescriptorSet(VriCommandBuffer* cmd, uint32_t setIndex, const VriDescriptorSet* set)
        {
            CommandBufferGL* c = CB(cmd);
            if (!c->boundLayout || !set)
                return;
            const DescriptorSetGL* s = reinterpret_cast<const DescriptorSetGL*>(set);
            for (const auto& [binding, view] : s->bound)
            {
                const LayoutBindingGL* lb = c->boundLayout->Find(setIndex, binding);
                if (!lb || !view)
                    continue;
                if (view->kind == DescriptorGL::Kind::BufferView && view->buffer)
                {
                    const GLenum target = (lb->type == VriDescriptorType_StorageBuffer ||
                                           lb->type == VriDescriptorType_StructuredBuffer)
                                              ? GL_SHADER_STORAGE_BUFFER
                                              : GL_UNIFORM_BUFFER;
                    glBindBufferRange(target, lb->glUnit, view->buffer->id,
                                      static_cast<GLintptr>(view->bufferOffset),
                                      static_cast<GLsizeiptr>(view->bufferRange ? view->bufferRange : view->buffer->size));
                }
            }
            // Textures + samplers are bound per combined sampler the pipeline declared:
            // GLSL fuses (texture, sampler) into one sampler2D at a texture unit.
#if !defined(__EMSCRIPTEN__) // GL 4.5: glBindTextureUnit + glTextureParameteri (undeclared on GLES3)
            const bool dsa = c->device->Features().dsa; // no active-unit churn on the modern path
#endif
            if (c->boundPipeline)
                for (const CombinedSamplerGL& cs : c->boundPipeline->combinedSamplers)
                {
#if !defined(__EMSCRIPTEN__)
                    if (!dsa)
#endif
                        glActiveTexture(GL_TEXTURE0 + cs.unit);
                    if (cs.texSet == setIndex)
                    {
                        auto it = s->bound.find(cs.texBinding);
                        if (it != s->bound.end() && it->second && it->second->texture)
                        {
                            const DescriptorGL* tv = it->second;
                            if (tv->texture->isRenderbuffer) continue; // renderbuffers aren't sampled
                            // Honor the view's base mip so a mip-range view samples that level.
#if !defined(__EMSCRIPTEN__)
                            if (dsa)
                            {
                                glBindTextureUnit(cs.unit, tv->texture->id);
                                glTextureParameteri(tv->texture->id, GL_TEXTURE_BASE_LEVEL, static_cast<GLint>(tv->mip));
                            }
                            else
#endif
                            {
                                glBindTexture(tv->texture->target, tv->texture->id);
                                glTexParameteri(tv->texture->target, GL_TEXTURE_BASE_LEVEL, static_cast<GLint>(tv->mip));
                            }
                        }
                    }
                    if (cs.smpSet == setIndex)
                    {
                        auto it = s->bound.find(cs.smpBinding);
                        if (it != s->bound.end() && it->second && it->second->sampler)
                            glBindSampler(cs.unit, it->second->sampler);
                    }
                }
        }
        // Push constants are emulated as default-block uniforms (SPIRV-Cross flattens the
        // push_constant block). Set each member by location on the bound program; CmdSetPipeline
        // already glUseProgram'd it, and the example records CmdSetConstants after it.
        void VRI_CALL CmdSetConstants(VriCommandBuffer* cmd, uint32_t, const void* data, uint32_t size)
        {
            CommandBufferGL* c = CB(cmd);
            if (!c->boundPipeline || !data)
                return;
            const uint8_t* bytes = static_cast<const uint8_t*>(data);
            for (const PushMemberGL& pm : c->boundPipeline->pushMembers)
            {
                if (pm.location < 0 || pm.offset >= size)
                    continue;
                const void* p = bytes + pm.offset;
                const GLint loc = pm.location;
                const GLsizei n = static_cast<GLsizei>(pm.count);
                const GLfloat* f = static_cast<const GLfloat*>(p);
                if (pm.columns == 4)      glUniformMatrix4fv(loc, n, GL_FALSE, f);
                else if (pm.columns == 3) glUniformMatrix3fv(loc, n, GL_FALSE, f);
                else if (pm.columns == 2) glUniformMatrix2fv(loc, n, GL_FALSE, f);
                else if (pm.basetype == 1) // int
                {
                    const GLint* v = static_cast<const GLint*>(p);
                    if (pm.vecsize == 1) glUniform1iv(loc, n, v); else if (pm.vecsize == 2) glUniform2iv(loc, n, v);
                    else if (pm.vecsize == 3) glUniform3iv(loc, n, v); else glUniform4iv(loc, n, v);
                }
                else if (pm.basetype == 2) // uint
                {
                    const GLuint* v = static_cast<const GLuint*>(p);
                    if (pm.vecsize == 1) glUniform1uiv(loc, n, v); else if (pm.vecsize == 2) glUniform2uiv(loc, n, v);
                    else if (pm.vecsize == 3) glUniform3uiv(loc, n, v); else glUniform4uiv(loc, n, v);
                }
                else // float
                {
                    if (pm.vecsize == 1) glUniform1fv(loc, n, f); else if (pm.vecsize == 2) glUniform2fv(loc, n, f);
                    else if (pm.vecsize == 3) glUniform3fv(loc, n, f); else glUniform4fv(loc, n, f);
                }
            }
        }
        void VRI_CALL CmdSetVertexBuffers(VriCommandBuffer* cmd, uint32_t baseSlot, const VriVertexBufferBinding* bindings, uint32_t num)
        {
            CommandBufferGL* c = CB(cmd);
            for (uint32_t i = 0; i < num; ++i)
                c->vertexBuffers[baseSlot + i] = VertexBufferBindingGL{Buf(bindings[i].buffer)->id, bindings[i].offset};
        }
        void VRI_CALL CmdSetIndexBuffer(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, VriIndexType type)
        {
            CommandBufferGL* c = CB(cmd);
            c->indexBuffer = Buf(buffer)->id;
            c->indexOffset = offset;
            c->indexType = type == VriIndexType_UInt16 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT;
        }
        // Configure the default VAO's attribute pointers for the bound pipeline +
        // vertex buffers (classic glVertexAttribPointer path; GLES3/WebGL2 has no
        // separate attribute format). Disables any attribs the previous draw enabled.
        void SetupVertexAttribs(CommandBufferGL* c)
        {
            // Separate-format path (GL 4.3+): bind the pipeline's pre-baked VAO and
            // only point each stream at its buffer (format is already in the VAO).
            // formatVao is always 0 on Emscripten (the 4.3+ path compiles out there).
#if !defined(__EMSCRIPTEN__)
            if (c->boundPipeline && c->boundPipeline->formatVao)
            {
                const PipelineGL* p = c->boundPipeline;
                const bool dsa = c->device->Features().dsa;
                glBindVertexArray(p->formatVao);
                for (const VboBindingGL& vb : p->vboBindings)
                {
                    auto it = c->vertexBuffers.find(vb.slot);
                    if (it == c->vertexBuffers.end())
                        continue;
                    if (dsa)
                        glVertexArrayVertexBuffer(p->formatVao, vb.slot, it->second.id, static_cast<GLintptr>(it->second.offset), vb.stride);
                    else
                        glBindVertexBuffer(vb.slot, it->second.id, static_cast<GLintptr>(it->second.offset), vb.stride);
                }
                return;
            }
#endif
            glBindVertexArray(c->device->DefaultVao());
            for (uint32_t loc = 0; loc < 32; ++loc)
                if (c->enabledAttribMask & (1u << loc))
                    glDisableVertexAttribArray(loc);
            uint32_t mask = 0;
            if (c->boundPipeline)
                for (const VertexAttribGL& a : c->boundPipeline->vertexAttribs)
                {
                    auto it = c->vertexBuffers.find(a.bindingSlot);
                    if (it == c->vertexBuffers.end())
                        continue;
                    glBindBuffer(GL_ARRAY_BUFFER, it->second.id);
                    glEnableVertexAttribArray(a.location);
                    glVertexAttribPointer(a.location, a.size, a.type, a.normalized, a.stride,
                                          reinterpret_cast<const void*>(static_cast<uintptr_t>(it->second.offset + a.offset)));
                    glVertexAttribDivisor(a.location, a.divisor); // per-instance streams advance per instance
                    mask |= (1u << a.location);
                }
            c->enabledAttribMask = mask;
        }
        // Set the SPIRV-Cross base-offset fallback uniforms on the bound program. No-op
        // on the ARB_shader_draw_parameters path (locations are -1); the *Base* draw
        // call populates gl_Base*ARB there instead. Program is already bound (CmdSetPipeline).
        void ApplyBaseOffsetUniforms(CommandBufferGL* c, int32_t baseVertex, uint32_t baseInstance)
        {
            const PipelineGL* p = c->boundPipeline;
            if (!p) return;
            if (p->baseVertexLoc >= 0) glUniform1i(p->baseVertexLoc, baseVertex);
            if (p->baseInstanceLoc >= 0) glUniform1i(p->baseInstanceLoc, static_cast<GLint>(baseInstance));
        }
        void VRI_CALL CmdDraw(VriCommandBuffer* cmd, const VriDrawDesc* d)
        {
            CommandBufferGL* c = CB(cmd);
            SetupVertexAttribs(c);
            // baseVertex maps to `first` (gl_VertexID already includes it, like VK).
#if !defined(__EMSCRIPTEN__)
            if (d->baseInstance != 0 && c->device->Features().baseInstance) // GL 4.2: honor firstInstance
            {
                ApplyBaseOffsetUniforms(c, static_cast<int32_t>(d->baseVertex), d->baseInstance);
                glDrawArraysInstancedBaseInstance(c->topology, static_cast<GLint>(d->baseVertex), static_cast<GLsizei>(d->vertexNum),
                                                  static_cast<GLsizei>(d->instanceNum), d->baseInstance);
                return;
            }
#endif
            if (d->baseInstance != 0) // requested but unsupported here: explicit, never silent
                c->device->ReportError("CmdDraw: baseInstance != 0 requires GL 4.2 (ARB_base_instance); unavailable on this context");
            glDrawArraysInstanced(c->topology, static_cast<GLint>(d->baseVertex), static_cast<GLsizei>(d->vertexNum),
                                  static_cast<GLsizei>(d->instanceNum));
        }
        void VRI_CALL CmdDrawIndexed(VriCommandBuffer* cmd, const VriDrawIndexedDesc* d)
        {
            CommandBufferGL* c = CB(cmd);
            SetupVertexAttribs(c);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, c->indexBuffer);
            const uint32_t indexSize = c->indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
            const uintptr_t offset = static_cast<uintptr_t>(c->indexOffset) + static_cast<uintptr_t>(d->baseIndex) * indexSize;
            const void* ip = reinterpret_cast<const void*>(offset);
            const GLsizei inst = static_cast<GLsizei>(d->instanceNum ? d->instanceNum : 1u);
#if !defined(__EMSCRIPTEN__)
            // Base offsets: vertexOffset is GL 3.2 (glDrawElements*BaseVertex, all desktop
            // incl. macOS 4.1); baseInstance is GL 4.2. gl_VertexID/gl_InstanceID parity is
            // handled by SPIRV-Cross (gl_Base*ARB or the fallback uniforms set below).
            if (d->vertexOffset != 0 || d->baseInstance != 0)
            {
                const bool canBaseInstance = c->device->Features().baseInstance;
                if (d->baseInstance != 0 && !canBaseInstance) // explicit, never silent
                    c->device->ReportError("CmdDrawIndexed: baseInstance != 0 requires GL 4.2 (ARB_base_instance); unavailable on this context");
                ApplyBaseOffsetUniforms(c, d->vertexOffset, d->baseInstance);
                if (d->baseInstance != 0 && canBaseInstance)
                    glDrawElementsInstancedBaseVertexBaseInstance(c->topology, static_cast<GLsizei>(d->indexNum), c->indexType, ip,
                                                                  inst, d->vertexOffset, d->baseInstance);
                else if (inst > 1)
                    glDrawElementsInstancedBaseVertex(c->topology, static_cast<GLsizei>(d->indexNum), c->indexType, ip, inst, d->vertexOffset);
                else
                    glDrawElementsBaseVertex(c->topology, static_cast<GLsizei>(d->indexNum), c->indexType, ip, d->vertexOffset);
                return;
            }
#else
            if (d->vertexOffset != 0 || d->baseInstance != 0) // WebGL2 has neither base offset
                c->device->ReportError("CmdDrawIndexed: vertexOffset/baseInstance != 0 is unsupported on WebGL2");
#endif
            if (d->instanceNum <= 1)
                glDrawElements(c->topology, static_cast<GLsizei>(d->indexNum), c->indexType, ip);
            else
                glDrawElementsInstanced(c->topology, static_cast<GLsizei>(d->indexNum), c->indexType, ip, static_cast<GLsizei>(d->instanceNum));
        }
        void VRI_CALL CmdDrawIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint32_t drawNum, uint32_t stride)
        {
            CommandBufferGL* c = CB(cmd);
#if !defined(__EMSCRIPTEN__)
            // GL indirect commands match VK/GL layout {count, instanceCount, first,
            // baseInstance}. glDrawArraysIndirect is GL 4.0 (all desktop); >1 draw uses
            // multi-draw-indirect (4.3) where available, else loops single draws.
            SetupVertexAttribs(c);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, Buf(buffer)->id);
            if (drawNum > 1 && c->device->Features().drawIndirect)
                glMultiDrawArraysIndirect(c->topology, reinterpret_cast<const void*>(static_cast<uintptr_t>(offset)),
                                          static_cast<GLsizei>(drawNum), static_cast<GLsizei>(stride));
            else
                for (uint32_t i = 0; i < (drawNum ? drawNum : 1u); ++i)
                    glDrawArraysIndirect(c->topology, reinterpret_cast<const void*>(static_cast<uintptr_t>(offset + static_cast<uint64_t>(i) * stride)));
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
#else
            (void)buffer; (void)offset; (void)drawNum; (void)stride;
            c->device->ReportError("CmdDrawIndirect: indirect draw is unavailable on WebGL2"); // explicit, never silent
#endif
        }
        void VRI_CALL CmdDispatch(VriCommandBuffer*, const VriDispatchDesc* d)
        {
#if defined(__EMSCRIPTEN__)
            (void)d; // compute is unavailable in WebGL2 (no pipeline can be created)
#else
            glDispatchCompute(d->x, d->y, d->z);
            // Make SSBO writes visible to the subsequent copy/readback (single context,
            // but the GPU pipeline still needs an explicit memory barrier).
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
#endif
        }
        void VRI_CALL CmdDispatchIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset)
        {
            CommandBufferGL* c = CB(cmd);
#if !defined(__EMSCRIPTEN__)
            // glDispatchComputeIndirect is GL 4.3 (compute already requires 4.3, so no
            // extra gate). Command layout {x, y, z} matches VK/WebGPU.
            glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, Buf(buffer)->id);
            glDispatchComputeIndirect(static_cast<GLintptr>(offset));
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
            glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
#else
            (void)buffer; (void)offset;
            c->device->ReportError("CmdDispatchIndirect: indirect dispatch is unavailable on WebGL2"); // explicit, never silent
#endif
        }
        void VRI_CALL CmdBarrier(VriCommandBuffer*, const VriBarrierGroupDesc*) {} // GL is ordered on one context
        void VRI_CALL CmdCopyBuffer(VriCommandBuffer*, VriBuffer* dst, VriBuffer* src, const VriBufferCopyDesc* r)
        {
            // Keep an element-class buffer on GL_ELEMENT_ARRAY_BUFFER so WebGL never
            // sees it bound to a non-element target (which would taint it).
            const BufferGL* s = Buf(src);
            const BufferGL* d = Buf(dst);
#if !defined(__EMSCRIPTEN__)
            if (s->device->Features().dsa) // GL 4.5: copy without disturbing binding points
            {
                glCopyNamedBufferSubData(s->id, d->id, static_cast<GLintptr>(r->srcOffset), static_cast<GLintptr>(r->dstOffset), static_cast<GLsizeiptr>(r->size));
                return;
            }
#endif
            // WebGL2/GLES reject glCopyBufferSubData between an element-array buffer and a
            // non-element one ("cannot copy into an element buffer destination from a
            // non-element buffer source"), no matter the targets used. The common case is a
            // staging upload into an index buffer; bounce it through CPU, each buffer bound
            // only to its own (locked) target so neither gets tainted.
            const bool srcEl = s->target == GL_ELEMENT_ARRAY_BUFFER;
            const bool dstEl = d->target == GL_ELEMENT_ARRAY_BUFFER;
            if (s->device->IsES() && srcEl != dstEl)
            {
                void* tmp = std::malloc(static_cast<size_t>(r->size));
                glBindBuffer(s->target, s->id);
                glGetBufferSubData(s->target, static_cast<GLintptr>(r->srcOffset), static_cast<GLsizeiptr>(r->size), tmp);
                glBindBuffer(s->target, 0);
                glBindBuffer(d->target, d->id);
                glBufferSubData(d->target, static_cast<GLintptr>(r->dstOffset), static_cast<GLsizeiptr>(r->size), tmp);
                glBindBuffer(d->target, 0);
                std::free(tmp);
                return;
            }
            const GLenum rt = srcEl ? GL_ELEMENT_ARRAY_BUFFER : GL_COPY_READ_BUFFER;
            const GLenum wt = dstEl ? GL_ELEMENT_ARRAY_BUFFER : GL_COPY_WRITE_BUFFER;
            glBindBuffer(rt, s->id);
            glBindBuffer(wt, d->id);
            glCopyBufferSubData(rt, wt, static_cast<GLintptr>(r->srcOffset), static_cast<GLintptr>(r->dstOffset), static_cast<GLsizeiptr>(r->size));
            glBindBuffer(rt, 0);
            glBindBuffer(wt, 0);
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
            const void* off = reinterpret_cast<const void*>(static_cast<uintptr_t>(region->bufferOffset));
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Buf(src)->id); // PBO source binding stays global even with DSA
#if !defined(__EMSCRIPTEN__)
            if (t->device->Features().dsa) // GL 4.5: upload without binding the texture
                glTextureSubImage2D(t->id, static_cast<GLint>(region->texture.mip), region->texture.x, region->texture.y,
                                    static_cast<GLsizei>(w), static_cast<GLsizei>(h), t->glFormat, t->glType, off);
            else
#endif
            {
                glBindTexture(t->target, t->id);
                glTexSubImage2D(t->target, static_cast<GLint>(region->texture.mip), region->texture.x, region->texture.y,
                                static_cast<GLsizei>(w), static_cast<GLsizei>(h), t->glFormat, t->glType, off);
                glBindTexture(t->target, 0);
            }
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
            // WebGL2 glReadPixels only accepts GL_RGBA (+ GL_UNSIGNED_BYTE); GL_BGRA is
            // rejected (INVALID_ENUM -> zeroed readback). The BGRA8 backbuffer is GL_RGBA8
            // internally, so read it as RGBA. Bytes land R,G,B,A (B/R swapped vs the BGRA
            // label) - callers that need exact BGRA must swizzle; the example self-check is
            // channel-order agnostic.
            GLenum readFmt = t->glFormat;
            if (b->device->IsES() && readFmt == GL_BGRA) readFmt = GL_RGBA;
            glReadPixels(0, 0, w, h, readFmt, t->glType,
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
