        // core_gl_common.inc - handle accessors, SPIR-V->GLSL transpile helpers and shared
        // utilities for the GL backend. #included into core_gl.cpp (one translation unit /
        // anonymous namespace); not compiled on its own. See core_gl.cpp for the split rationale.
        inline DeviceGL*        Dev(VriDevice* h) { return reinterpret_cast<DeviceGL*>(h); }
        inline const DeviceGL*  Dev(const VriDevice* h) { return reinterpret_cast<const DeviceGL*>(h); }
        inline CommandBufferGL* CB(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferGL*>(h); }
        inline BufferGL*        Buf(VriBuffer* h) { return reinterpret_cast<BufferGL*>(h); }
        inline TextureGL*       Tex(VriTexture* h) { return reinterpret_cast<TextureGL*>(h); }
        inline DescriptorGL*    Desc(VriDescriptor* h) { return reinterpret_cast<DescriptorGL*>(h); }
        inline PipelineGL*      Pipe(VriPipeline* h) { return reinterpret_cast<PipelineGL*>(h); }
        inline FenceGL*         Fen(VriFence* h) { return reinterpret_cast<FenceGL*>(h); }

        // Read a range of a buffer (bound to `target`) into host memory. glGetBufferSubData
        // exists on desktop GL and the Emscripten WebGL2 runtime, but NOT in native OpenGL ES
        // - there, read back by mapping the range (ES3 supports glMapBufferRange, unlike WebGL2).
        inline void ReadBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, void* dst)
        {
#if defined(VRI_GL_NATIVE_ES)
            if (void* p = glMapBufferRange(target, offset, size, GL_MAP_READ_BIT))
            {
                std::memcpy(dst, p, static_cast<size_t>(size));
                glUnmapBuffer(target);
            }
#else
            glGetBufferSubData(target, offset, size, dst);
#endif
        }

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

            std::unordered_set<uint32_t>           baseVars;  // builtin vars to neutralize
            std::unordered_map<uint32_t, uint32_t> zeroConst; // 32-bit type id -> a 0-constant id
            size_t                                 firstFunc = w.size();

            for (size_t i = 5; i < w.size();)
            {
                const uint32_t op  = w[i] & 0xFFFFu;
                const uint32_t len = w[i] >> 16;
                if (len == 0)
                    return; // malformed; leave the module untouched
                if (op == kOpDecorate && len >= 4 && w[i + 2] == kDecBuiltIn)
                {
                    const uint32_t b = w[i + 3];
                    if (b == kBaseVertex || b == kBaseInstance || b == kDrawIndex)
                        baseVars.insert(w[i + 1]);
                    else if (b == kVertexIndex)
                        w[i + 3] = kVertexId;
                    else if (b == kInstanceIndex)
                        w[i + 3] = kInstanceId;
                }
                else if (op == kOpConstant && len == 4 && w[i + 3] == 0)
                    zeroConst.emplace(w[i + 1], w[i + 2]); // first 0-constant per 32-bit type
                else if (op == kOpFunction && firstFunc == w.size())
                    firstFunc = i;
                i += len;
            }
            if (baseVars.empty())
                return;

            uint32_t              bound = w[3];
            std::vector<uint32_t> newConstants;
            for (size_t i = 5; i < w.size();)
            {
                const uint32_t op  = w[i] & 0xFFFFu;
                const uint32_t len = w[i] >> 16;
                if (len == 0)
                    break;
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
                    w[i]     = (4u << 16) | kOpCopyObject;
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

        std::string SpirvToGlsl(const DeviceGL*                 d,
                                const PipelineLayoutGL*         layout,
                                const void*                     bytecode,
                                size_t                          bytecodeSize,
                                const char*                     entry,
                                spv::ExecutionModel             model,
                                std::vector<CombinedSamplerGL>* outCombined,
                                std::vector<PushMemberGL>*      outPush     = nullptr,
                                int                             inBoundary  = -1,
                                int                             outBoundary = -1,
                                uint32_t                        viewCount   = 0)
        {
            const uint32_t* words     = static_cast<const uint32_t*>(bytecode);
            const size_t    wordCount = bytecodeSize / 4;
            // SPIRV-Cross signals transpile errors by throwing; turn that into a
            // clean failure (empty source -> shader compile fails -> pipeline
            // returns Failure) instead of an abort, and surface the message.
            try
            {
                std::vector<uint32_t> patched(words, words + wordCount);
                if (d->IsES())
                    PatchDrawParamsForES(patched);
                spirv_cross::CompilerGLSL          comp(std::move(patched));
                spirv_cross::CompilerGLSL::Options o = comp.get_common_options();
                o.version                            = d->ShaderVersion();
                o.es                                 = d->IsES();
                o.vulkan_semantics                   = false;
                // Don't emit layout(binding=) via GL_ARB_shading_language_420pack: it's
                // unavailable below GLSL 420 (macOS GL 4.1 / GLSL 410 rejects it), and this
                // backend assigns binding points through the GL API regardless (see below).
                o.enable_420pack_extension = false;
                // Y orientation: with glClipControl (GL 4.5+) the context already uses the
                // VRI top-left convention, so NO in-shader flip. Otherwise (ES/WebGL2,
                // pre-4.5 desktop) flip clip-space Y in-shader - but never on the
                // tessellation-control stage (its output is gl_out[].gl_Position; a bare
                // gl_Position there is invalid GLSL and strict drivers reject it).
                o.vertex.flip_vert_y = !d->Features().clipControl && model != spv::ExecutionModelTessellationControl;
                // Depth range: with glClipControl(ZERO_TO_ONE) the context already uses [0,1]
                // clip-Z (the Vulkan/D3D convention the shaders are authored in). Without it
                // (ES/WebGL2, pre-4.5 desktop) convert the [0,1] clip-Z to GL's [-1,1] in-shader
                // (gl_Position.z = 2z - w) so the rasterizer produces an UNCOMPRESSED [0,1] window
                // depth. This matters whenever a shader compares against a stored depth value:
                // without it the depth-buffer value is squashed into [0.5,1] while a shader-computed
                // reference (e.g. a shadow-map lookup) stays [0,1], so the comparison misfires
                // (shadows vanish). Plain depth testing is monotonic either way, so it's unaffected.
                o.vertex.fixup_clipspace =
                    !d->Features().clipControl && model != spv::ExecutionModelTessellationControl;
                // Multiview: emit GL_OVR_multiview2 (layout(num_views=N) + gl_ViewID_OVR) so SV_ViewID
                // / SPIR-V ViewIndex selects the eye in a single pass. The num_views declaration is
                // only valid on the vertex stage (per-view logic belongs there); 0/1 = no multiview.
                o.ovr_multiview_view_count = (viewCount > 1 && model == spv::ExecutionModelVertex) ? viewCount : 0;
                comp.set_common_options(o);
                if (entry)
                    comp.set_entry_point(entry, model);

                // Multiview: our shaders are one module with both entry points, so Slang lists the
                // ViewIndex builtin in the fragment entry point's interface even though only the
                // vertex stage reads it. SPIRV-Cross would then try to emit gl_ViewID_OVR in the
                // fragment (illegal without num_views, which is vertex-only) -> prune to the variables
                // the active entry point actually uses, dropping the unused builtin.
                if (viewCount > 1)
                    comp.set_enabled_interface_variables(comp.get_active_interface_variables());

                // A texture accessed only by texelFetch (.Load) - e.g. an integer index
                // texture - carries no sampler to fuse, but GLSL still needs a combined
                // sampler to fetch it. Synthesise a dummy sampler for those images first,
                // or build_combined_image_samplers() aborts ("texelFetch without sampler").
                if (const spirv_cross::VariableID dummy = comp.build_dummy_sampler_for_combined_images())
                {
                    comp.set_decoration(dummy, spv::DecorationDescriptorSet, 0);
                    comp.set_decoration(dummy, spv::DecorationBinding, 0);
                }

                // GLSL has no separate texture/sampler: fuse each Vulkan (texture,
                // sampler) pair into one combined sampler2D. Must run before
                // get_shader_resources()/compile().
                comp.build_combined_image_samplers();

                const spirv_cross::ShaderResources res = comp.get_shader_resources();

                // GL links the separately-compiled stages by matching inter-stage varyings by
                // NAME (ESSL/WebGL2 disallow location qualifiers; macOS desktop GL also matches
                // these by name). Name each varying by its inter-stage "boundary" id + location:
                // the producer's output and the consumer's input on a boundary share a name (so
                // they link), while a middle stage's own input (previous boundary) and output
                // (next boundary) stay distinct - avoiding the in/out name collision SPIRV-Cross
                // would otherwise disambiguate (e.g. TCS out vriVarying0 -> vriVarying0_1, breaking
                // the TCS->TES match). The caller assigns the ids in pipeline order. A stage's
                // inputs are renamed unless it's the vertex stage (vertex attributes); its outputs
                // unless it's the fragment stage (color targets).
                auto renameByLocation = [&](const spirv_cross::SmallVector<spirv_cross::Resource>& vars, int boundary) {
                    for (const spirv_cross::Resource& r : vars)
                    {
                        const uint32_t loc = comp.get_decoration(r.id, spv::DecorationLocation);
                        comp.set_name(r.id, "vriVarying" + std::to_string(boundary) + "_" + std::to_string(loc));
                    }
                };
                if (model != spv::ExecutionModelVertex)
                    renameByLocation(res.stage_inputs, inBoundary);
                if (model != spv::ExecutionModelFragment)
                    renameByLocation(res.stage_outputs, outBoundary);

                // Remap each Vulkan (set, binding) to the flattened GL unit from the
                // pipeline layout, and give resources deterministic names so binding
                // points can be assigned after link (glUniformBlockBinding / glUniform1i).
                if (layout)
                {
                    for (const spirv_cross::Resource& u : res.uniform_buffers)
                    {
                        const uint32_t set  = comp.get_decoration(u.id, spv::DecorationDescriptorSet);
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
                        const uint32_t set  = comp.get_decoration(s.id, spv::DecorationDescriptorSet);
                        const uint32_t bind = comp.get_decoration(s.id, spv::DecorationBinding);
                        if (const LayoutBindingGL* lb = layout->Find(set, bind))
                        {
                            comp.unset_decoration(s.id, spv::DecorationDescriptorSet);
                            comp.set_decoration(s.id, spv::DecorationBinding, lb->glUnit);
                        }
                    }
                    // Storage images (compute UAV writes): remap to the image unit; bound via
                    // glBindImageTexture in CmdSetDescriptorSet.
                    for (const spirv_cross::Resource& im : res.storage_images)
                    {
                        const uint32_t set  = comp.get_decoration(im.id, spv::DecorationDescriptorSet);
                        const uint32_t bind = comp.get_decoration(im.id, spv::DecorationBinding);
                        if (const LayoutBindingGL* lb = layout->Find(set, bind))
                        {
                            comp.unset_decoration(im.id, spv::DecorationDescriptorSet);
                            comp.set_decoration(im.id, spv::DecorationBinding, lb->glUnit);
                        }
                    }
                    for (const spirv_cross::CombinedImageSampler& cis : comp.get_combined_image_samplers())
                    {
                        const uint32_t texSet     = comp.get_decoration(cis.image_id, spv::DecorationDescriptorSet);
                        const uint32_t texBind    = comp.get_decoration(cis.image_id, spv::DecorationBinding);
                        const uint32_t smpSet     = comp.get_decoration(cis.sampler_id, spv::DecorationDescriptorSet);
                        const uint32_t smpBind    = comp.get_decoration(cis.sampler_id, spv::DecorationBinding);
                        const LayoutBindingGL* lb = layout->Find(texSet, texBind);
                        if (!lb)
                            continue;
                        const uint32_t    unit = lb->glUnit;
                        const std::string name = "VriTex_" + std::to_string(unit);
                        comp.set_name(cis.combined_id, name);
                        comp.unset_decoration(cis.combined_id, spv::DecorationDescriptorSet);
                        comp.set_decoration(cis.combined_id, spv::DecorationBinding, unit);
                        if (outCombined)
                            outCombined->push_back(CombinedSamplerGL {unit, texSet, texBind, smpSet, smpBind, name});
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
                            PushMemberGL                 pm {};
                            pm.name     = "vriPush." + comp.get_member_name(pc.base_type_id, m);
                            pm.offset   = comp.type_struct_member_offset(st, m);
                            pm.basetype = mt.basetype == spirv_cross::SPIRType::Int ?
                                              1u :
                                              (mt.basetype == spirv_cross::SPIRType::UInt ? 2u : 0u);
                            pm.vecsize  = mt.vecsize;
                            pm.columns  = mt.columns;
                            pm.count    = mt.array.empty() ? 1u : mt.array[0];
                            // Matrix upload transpose: SPIRV-Cross emits the same `v * M` GLSL for a
                            // matrix regardless of storage, so the push blob (raw bytes, correct on
                            // Vulkan/D3D12) must go to glUniformMatrix with the right transpose flag or
                            // the matrix is loaded transposed (translation lands in the wrong slot and
                            // is lost). A column-major matrix (Slang's default, matching glm) needs
                            // GL_TRUE; a row-major one loads as-is. Note the SPIR-V decoration is
                            // inverted vs Slang's source qualifier: Slang column_major -> SPIR-V
                            // RowMajor, so transpose == "member is RowMajor-decorated".
                            if (mt.columns > 1)
                                pm.transpose = comp.has_member_decoration(pc.base_type_id, m, spv::DecorationRowMajor);
                            bool dup = false;
                            for (const PushMemberGL& e : *outPush)
                                if (e.name == pm.name)
                                {
                                    dup = true;
                                    break;
                                }
                            if (!dup)
                                outPush->push_back(pm);
                        }
                    }
                std::string glsl = comp.compile();
                // Tessellation-control outputs may only be indexed by gl_InvocationID;
                // strict drivers (NVIDIA) reject SPIRV-Cross's uint(gl_InvocationID) cast
                // on the output write, so strip it (reads stay valid too).
                if (model == spv::ExecutionModelTessellationControl)
                {
                    const std::string from = "uint(gl_InvocationID)";
                    const std::string to   = "gl_InvocationID";
                    for (size_t p = glsl.find(from); p != std::string::npos; p = glsl.find(from, p + to.size()))
                        glsl.replace(p, from.size(), to);
                }
                if (std::getenv("VRI_DUMP_GLSL"))
                    std::fprintf(stderr,
                                 "=== VRI GLSL model=%d entry=%s ===\n%s\n",
                                 static_cast<int>(model),
                                 entry ? entry : "?",
                                 glsl.c_str());
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
            GLuint      s   = glCreateShader(type);
            const char* p   = src.c_str();
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

