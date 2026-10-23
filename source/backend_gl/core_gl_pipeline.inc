        // core_gl_pipeline.inc - pipeline layout, graphics/compute pipelines, descriptor pools
        // and sets, and fences. #included into core_gl.cpp (one TU); see core_gl.cpp.
        // ---- pipeline layout & pipelines -----------------------------------
        VriResult VRI_CALL CreatePipelineLayout(VriDevice*                   device,
                                                const VriPipelineLayoutDesc* desc,
                                                VriPipelineLayout**          out)
        {
            if (!device || !desc || !out || (desc->descriptorSetNum && !desc->descriptorSets))
                return VriResult_InvalidArgument;
            *out = nullptr;
            for (uint32_t set = 0; set < desc->descriptorSetNum; ++set)
            {
                const auto& layout = desc->descriptorSets[set];
                if (layout.rangeNum && !layout.ranges)
                    return VriResult_InvalidArgument;
                for (uint32_t range = 0; range < layout.rangeNum; ++range)
                {
                    if (layout.ranges[range].descriptorNum == 0)
                        return VriResult_InvalidArgument;
                    if (layout.ranges[range].descriptorNum > 1)
                        return VriResult_Unsupported;
                }
            }
            PipelineLayoutGL* l = new PipelineLayoutGL {Dev(device), {}};
            l->setCount         = desc->descriptorSetNum;
            // Flatten (set, binding) -> per-type GL unit. GL has independent binding
            // namespaces for uniform buffers, shader-storage buffers and texture
            // units, so each type gets its own running counter.
            uint32_t uboNext = 0, ssboNext = 0, texNext = 0, imgNext = 0;
            for (uint32_t s = 0; s < desc->descriptorSetNum; ++s)
            {
                const VriDescriptorSetDesc& sd = desc->descriptorSets[s];
                for (uint32_t r = 0; r < sd.rangeNum; ++r)
                {
                    const VriDescriptorRangeDesc& rd = sd.ranges[r];
                    // Samplers don't take a unit of their own: GLSL fuses them with a
                    // texture, so the combined sampler rides the texture's unit.
                    uint32_t* counter = nullptr;
                    if (rd.descriptorType == VriDescriptorType_ConstantBuffer)
                        counter = &uboNext;
                    else if (rd.descriptorType == VriDescriptorType_StorageBuffer ||
                             rd.descriptorType == VriDescriptorType_StructuredBuffer)
                        counter = &ssboNext;
                    else if (rd.descriptorType == VriDescriptorType_Texture)
                        counter = &texNext;
                    else if (rd.descriptorType == VriDescriptorType_StorageTexture)
                        counter = &imgNext; // image unit (glBindImageTexture)
                    LayoutBindingGL b {};
                    b.set     = sd.registerSpace;
                    b.binding = rd.baseRegister;
                    b.type    = rd.descriptorType;
                    b.glUnit  = counter ? *counter : 0u;
                    if (counter)
                        *counter += (rd.descriptorNum ? rd.descriptorNum : 1u);
                    l->bindings.push_back(b);
                }
            }
            for (const LayoutBindingGL& b : l->bindings)
            {
                l->hash = Fnv1a(&b.set, sizeof(b.set), l->hash);
                l->hash = Fnv1a(&b.binding, sizeof(b.binding), l->hash);
                l->hash = Fnv1a(&b.type, sizeof(b.type), l->hash);
                l->hash = Fnv1a(&b.glUnit, sizeof(b.glUnit), l->hash);
            }
            *out = ToHandle(l);
            return VriResult_Success;
        }
        void VRI_CALL DestroyPipelineLayout(VriPipelineLayout* layout)
        {
            delete reinterpret_cast<PipelineLayoutGL*>(layout);
        }

        struct GlslStage
        {
            GLenum      gl;
            std::string src;
        };

        // Link a program from the transpiled GLSL stages, reusing the pipeline cache when present:
        // a cache hit restores the linked program via glProgramBinary (skipping compile + link); a
        // miss compiles + links and stores the resulting binary. Returns 0 on failure. The cache is
        // desktop-GL only (program binaries are absent on GLES/WebGL2), so on those builds this just
        // compiles + links (cache is always null there).
        GLuint AcquireProgram(DeviceGL* d, PipelineCacheGL* cache, uint64_t key, const std::vector<GlslStage>& stages)
        {
#if !defined(VRI_GL_ES_HEADERS)
            if (cache)
            {
                auto it = cache->entries.find(key);
                if (it != cache->entries.end())
                {
                    GLuint prog = glCreateProgram();
                    glProgramBinary(
                        prog, it->second.format, it->second.data.data(), static_cast<GLsizei>(it->second.data.size()));
                    GLint ok = 0;
                    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
                    if (ok)
                        return prog;       // hit: skipped SPIR-V compile + link
                    glDeleteProgram(prog); // stale / incompatible binary -> fall back to a normal build
                }
            }
#endif
            GLuint              program = glCreateProgram();
            std::vector<GLuint> shaders;
            for (const GlslStage& s : stages)
            {
                GLuint sh = CompileShader(d, s.gl, s.src);
                if (!sh)
                {
                    for (GLuint x : shaders)
                        glDeleteShader(x);
                    glDeleteProgram(program);
                    return 0;
                }
                glAttachShader(program, sh);
                shaders.push_back(sh);
            }
#if !defined(VRI_GL_ES_HEADERS)
            if (cache)
                glProgramParameteri(program, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE); // before link
#endif
            glLinkProgram(program);
            for (GLuint sh : shaders)
            {
                glDetachShader(program, sh);
                glDeleteShader(sh);
            }
            GLint linked = 0;
            glGetProgramiv(program, GL_LINK_STATUS, &linked);
            if (!linked)
            {
                char log[2048] = {};
                glGetProgramInfoLog(program, sizeof(log), nullptr, log);
                d->ReportError(log);
                glDeleteProgram(program);
                return 0;
            }
#if !defined(VRI_GL_ES_HEADERS)
            if (cache)
            {
                GLint len = 0;
                glGetProgramiv(program, GL_PROGRAM_BINARY_LENGTH, &len);
                if (len > 0)
                {
                    ProgramBinaryGL pb;
                    pb.data.resize(static_cast<size_t>(len));
                    glGetProgramBinary(program, len, nullptr, &pb.format, pb.data.data());
                    cache->entries[key] = std::move(pb);
                }
            }
#endif
            return program;
        }

        VriResult VRI_CALL CreateGraphicsPipeline(VriDevice*                     device,
                                                  const VriGraphicsPipelineDesc* desc,
                                                  VriPipeline**                  out)
        {
            DeviceGL*               d      = Dev(device);
            const PipelineLayoutGL* layout = reinterpret_cast<const PipelineLayoutGL*>(desc->pipelineLayout);
            // Multiview view count (popcount of the viewMask; 0/1 = single view). Drives the
            // GL_OVR_multiview2 emission in each stage's GLSL.
            const uint32_t viewCount = static_cast<uint32_t>(std::popcount(desc->outputMerger.viewMask));
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

            // Assign each gap between consecutive present stages a "boundary" id (in pipeline
            // order VS->TCS->TES->GS->FS) so the varying renamer names producer-out and
            // consumer-in identically while keeping a middle stage's in/out distinct.
            const VriShaderStageBits        kOrder[5] = {VriShaderStage_Vertex,
                                                         VriShaderStage_TessControl,
                                                         VriShaderStage_TessEval,
                                                         VriShaderStage_Geometry,
                                                         VriShaderStage_Fragment};
            std::vector<VriShaderStageBits> present;
            for (VriShaderStageBits m : kOrder)
                for (uint32_t i = 0; i < desc->shaderNum; ++i)
                    if (desc->shaders[i].stage == m)
                    {
                        present.push_back(m);
                        break;
                    }
            auto bIn = [&](VriShaderStageBits m) {
                for (size_t k = 0; k < present.size(); ++k)
                    if (present[k] == m)
                        return int(k) - 1;
                return -1;
            };
            auto bOut = [&](VriShaderStageBits m) {
                for (size_t k = 0; k < present.size(); ++k)
                    if (present[k] == m)
                        return int(k);
                return -1;
            };

            std::vector<CombinedSamplerGL> combined;
            std::vector<PushMemberGL>      pushMembers;
            // Transpile each stage to GLSL (this also fills the draw-time reflection: combined
            // samplers + push-constant members). The compile + link is then done once, by
            // AcquireProgram, which may instead restore a cached program binary.
            std::vector<GlslStage> glslStages;
            bool                   hasVS = false, hasFS = false;
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                const VriShaderDesc& s  = desc->shaders[i];
                GLenum               gl = GL_VERTEX_SHADER;
                spv::ExecutionModel  em = spv::ExecutionModelVertex;
                if (s.stage == VriShaderStage_Vertex)
                {
                    gl    = GL_VERTEX_SHADER;
                    em    = spv::ExecutionModelVertex;
                    hasVS = true;
                }
                else if (s.stage == VriShaderStage_Fragment)
                {
                    gl    = GL_FRAGMENT_SHADER;
                    em    = spv::ExecutionModelFragment;
                    hasFS = true;
                }
#if !defined(VRI_GL_ES_HEADERS) // geometry/tessellation shaders are desktop-only (absent in GLES3/WebGL2 headers)
                else if (s.stage == VriShaderStage_Geometry)
                {
                    gl = GL_GEOMETRY_SHADER;
                    em = spv::ExecutionModelGeometry;
                }
                else if (s.stage == VriShaderStage_TessControl)
                {
                    gl = GL_TESS_CONTROL_SHADER;
                    em = spv::ExecutionModelTessellationControl;
                }
                else if (s.stage == VriShaderStage_TessEval)
                {
                    gl = GL_TESS_EVALUATION_SHADER;
                    em = spv::ExecutionModelTessellationEvaluation;
                }
#endif
                else
                    continue;
                // Push constants are gathered from the vertex + fragment stages only (matching the
                // original build); other stages pass null.
                const bool  takesPush = (s.stage == VriShaderStage_Vertex || s.stage == VriShaderStage_Fragment);
                std::string glsl      = SpirvToGlsl(d,
                                               layout,
                                               s.bytecode,
                                               s.bytecodeSize,
                                               s.entryPointName,
                                               em,
                                               &combined,
                                               takesPush ? &pushMembers : nullptr,
                                               bIn(s.stage),
                                               bOut(s.stage),
                                               viewCount);
                if (glsl.empty()) // SPIRV-Cross transpile error
                    return VriResult_Failure;
                glslStages.push_back({gl, std::move(glsl)});
            }
            if (!hasVS || !hasFS)
                return VriResult_Failure;

            uint64_t key = layout ? layout->hash : 0;
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                key = Fnv1a(&desc->shaders[i].stage, sizeof(desc->shaders[i].stage), key);
                key = Fnv1a(desc->shaders[i].bytecode, desc->shaders[i].bytecodeSize, key);
            }
            GLuint program =
                AcquireProgram(d, desc->pipelineCache ? PipeCacheGL(desc->pipelineCache) : nullptr, key, glslStages);
            if (!program)
                return VriResult_Failure;

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

            PipelineGL* p       = new PipelineGL {};
            p->device           = d;
            p->program          = program;
            p->combinedSamplers = std::move(combined);
            p->pushMembers      = std::move(pushMembers);
            // Base-offset fallback uniforms (present only when the driver lacks
            // ARB_shader_draw_parameters; -1 means the gl_Base*ARB builtin path is used).
            p->baseVertexLoc   = glGetUniformLocation(program, "SPIRV_Cross_BaseVertex");
            p->baseInstanceLoc = glGetUniformLocation(program, "SPIRV_Cross_BaseInstance");
            // Vertex input: attribute index i == GLSL location i (matches the VK
            // backend). Flatten each attribute with its stream's stride + slot for
            // the classic glVertexAttribPointer path used at draw time.
            for (uint32_t i = 0; i < desc->vertexInput.attributeNum; ++i)
            {
                const VriVertexAttributeDesc& a           = desc->vertexInput.attributes[i];
                const GLVertexFormat          vf          = ToGLVertexFormat(a.format);
                GLsizei                       stride      = 0;
                uint32_t                      bindingSlot = a.streamIndex;
                uint32_t                      divisor     = 0;
                if (a.streamIndex < desc->vertexInput.streamNum)
                {
                    stride      = static_cast<GLsizei>(desc->vertexInput.streams[a.streamIndex].stride);
                    bindingSlot = desc->vertexInput.streams[a.streamIndex].bindingSlot;
                    divisor =
                        desc->vertexInput.streams[a.streamIndex].stepRate == VriVertexStepRate_PerInstance ? 1u : 0u;
                }
                p->vertexAttribs.push_back(VertexAttribGL {
                    i, vf.size, vf.type, vf.normalized, vf.integer, a.offset, stride, bindingSlot, divisor});
            }
            // Separate attribute format (GL 4.3+): bake the vertex format into a
            // per-pipeline VAO once. At draw we only point each stream binding at a
            // buffer (glBindVertexBuffer), avoiding per-draw glVertexAttribPointer.
            // Pre-4.3 / GLES / WebGL2 keep the classic path (formatVao stays 0); the
            // 4.3+ entry points are undeclared in the Emscripten GLES3 headers.
#if !defined(VRI_GL_ES_HEADERS)
            if (d->Features().separateAttrib)
            {
                const bool dsa = d->Features().dsa; // 4.5: configure the VAO by name
                GLuint     vao = 0;
                if (dsa)
                    glCreateVertexArrays(1, &vao);
                else
                {
                    glGenVertexArrays(1, &vao);
                    glBindVertexArray(vao);
                }
                for (const VertexAttribGL& a : p->vertexAttribs)
                {
                    if (dsa)
                    {
                        glEnableVertexArrayAttrib(vao, a.location);
                        if (a.integer)
                            glVertexArrayAttribIFormat(vao, a.location, a.size, a.type, a.offset);
                        else
                            glVertexArrayAttribFormat(vao, a.location, a.size, a.type, a.normalized, a.offset);
                        glVertexArrayAttribBinding(vao, a.location, a.bindingSlot);
                        glVertexArrayBindingDivisor(vao, a.bindingSlot, a.divisor);
                    }
                    else
                    {
                        glEnableVertexAttribArray(a.location);
                        if (a.integer)
                            glVertexAttribIFormat(a.location, a.size, a.type, a.offset);
                        else
                            glVertexAttribFormat(a.location, a.size, a.type, a.normalized, a.offset);
                        glVertexAttribBinding(a.location, a.bindingSlot);
                        glVertexBindingDivisor(a.bindingSlot, a.divisor);
                    }
                }
                if (!dsa)
                    glBindVertexArray(d->DefaultVao()); // restore; DSA never bound it
                p->formatVao = vao;
                // Unique stream slots (with stride) to rebind per draw.
                for (const VertexAttribGL& a : p->vertexAttribs)
                {
                    bool seen = false;
                    for (const VboBindingGL& vb : p->vboBindings)
                        if (vb.slot == a.bindingSlot)
                        {
                            seen = true;
                            break;
                        }
                    if (!seen)
                        p->vboBindings.push_back(VboBindingGL {a.bindingSlot, a.stride, a.divisor});
                }
            }
#endif // !VRI_GL_ES_HEADERS (separate-format VAO)
            p->topology      = ToGLTopology(desc->inputAssembly.topology);
            p->patchVertices = desc->tessellation.patchControlPoints;
            p->cullEnable    = desc->rasterization.cullMode != VriCullMode_None;
            p->cullFace      = desc->rasterization.cullMode == VriCullMode_Front ? GL_FRONT : GL_BACK;
            // Window-space winding depends on how VRI's Y is oriented: glClipControl
            // (UPPER_LEFT) makes window Y-down like VRI, so a VRI CCW face stays GL CCW;
            // the in-shader flip_vert_y path negates clip Y, reversing it (CCW -> GL CW).
            const bool ccw = desc->rasterization.frontFace == VriFrontFace_CounterClockwise;
            p->frontFace   = d->Features().clipControl ? (ccw ? GL_CCW : GL_CW) : (ccw ? GL_CW : GL_CCW);
            p->depthTest   = desc->depthStencil.depthTest != VRI_FALSE;
            p->depthWrite  = desc->depthStencil.depthWrite != VRI_FALSE;
            p->depthFunc   = ToGLCompareOp(desc->depthStencil.depthCompareOp);
            const VriColorAttachmentDesc* c0 = desc->outputMerger.colorNum ? &desc->outputMerger.colors[0] : nullptr;
            p->blendEnable                   = c0 && c0->blend.enable;
            p->srcRGB                        = c0 ? ToGLBlendFactor(c0->blend.srcColor) : GL_ONE;
            p->dstRGB                        = c0 ? ToGLBlendFactor(c0->blend.dstColor) : GL_ZERO;
            p->rgbOp                         = c0 ? ToGLBlendOp(c0->blend.colorOp) : GL_FUNC_ADD;
            p->srcA                          = c0 ? ToGLBlendFactor(c0->blend.srcAlpha) : GL_ONE;
            p->dstA                          = c0 ? ToGLBlendFactor(c0->blend.dstAlpha) : GL_ZERO;
            p->aOp                           = c0 ? ToGLBlendOp(c0->blend.alphaOp) : GL_FUNC_ADD;
            const VriColorWriteFlags wm      = c0 ? c0->colorWriteMask : VriColorWrite_RGBA;
            const VriColorWriteFlags m       = (wm == 0) ? VriColorWrite_RGBA : wm;
            p->colorMask[0]                  = (m & VriColorWrite_R) ? GL_TRUE : GL_FALSE;
            p->colorMask[1]                  = (m & VriColorWrite_G) ? GL_TRUE : GL_FALSE;
            p->colorMask[2]                  = (m & VriColorWrite_B) ? GL_TRUE : GL_FALSE;
            p->colorMask[3]                  = (m & VriColorWrite_A) ? GL_TRUE : GL_FALSE;
            p->stencilEnable                 = desc->depthStencil.stencilTest != VRI_FALSE;
            auto toGLFace                    = [](const VriStencilOpDesc& s) {
                return PipelineGL::StencilFaceGL {ToGLCompareOp(s.compareOp),
                                                  ToGLStencilOp(s.failOp),
                                                  ToGLStencilOp(s.depthFailOp),
                                                  ToGLStencilOp(s.passOp),
                                                  s.compareMask,
                                                  s.writeMask,
                                                  static_cast<GLint>(s.reference)};
            };
            p->stencilFront = toGLFace(desc->depthStencil.front);
            p->stencilBack  = toGLFace(desc->depthStencil.back);
            *out            = ToHandle(p);
            return VriResult_Success;
        }
        VriResult VRI_CALL CreateComputePipeline(VriDevice*                    device,
                                                 const VriComputePipelineDesc* desc,
                                                 VriPipeline**                 out)
        {
            DeviceGL* d = Dev(device);
            if (!d->Desc().hasComputeShader)
                return VriResult_Unsupported; // GLES 3.0 / WebGL2 has no compute
#if defined(VRI_GL_ES_HEADERS)
            return VriResult_Unsupported; // GL_COMPUTE_SHADER is not in WebGL2
#else
            const PipelineLayoutGL*        layout = reinterpret_cast<const PipelineLayoutGL*>(desc->pipelineLayout);
            std::vector<CombinedSamplerGL> combined; // sampled textures used by the compute kernel
            std::string                    glsl = SpirvToGlsl(d,
                                           layout,
                                           desc->shader.bytecode,
                                           desc->shader.bytecodeSize,
                                           desc->shader.entryPointName,
                                           spv::ExecutionModelGLCompute,
                                           &combined);
            if (glsl.empty())
                return VriResult_Failure;
            std::vector<GlslStage> glslStages {{GL_COMPUTE_SHADER, std::move(glsl)}};
            uint64_t               key = layout ? layout->hash : 0;
            key                        = Fnv1a(desc->shader.bytecode, desc->shader.bytecodeSize, key);
            GLuint program =
                AcquireProgram(d, desc->pipelineCache ? PipeCacheGL(desc->pipelineCache) : nullptr, key, glslStages);
            if (!program)
                return VriResult_Failure;
            // Uniform blocks (if any) still need binding points; SSBOs use the
            // layout(binding=) emitted by SPIRV-Cross (GLSL 430). Sampled textures need their
            // sampler uniform pointed at its texture unit, same as the graphics path.
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
                for (const CombinedSamplerGL& cs : combined)
                {
                    const GLint loc = glGetUniformLocation(program, cs.name.c_str());
                    if (loc >= 0)
                        glUniform1i(loc, static_cast<GLint>(cs.unit));
                }
                glUseProgram(0);
            }
            PipelineGL* p       = new PipelineGL {};
            p->device           = d;
            p->program          = program;
            p->isCompute        = true;
            p->combinedSamplers = std::move(combined);
            *out                = ToHandle(p);
            return VriResult_Success;
#endif // !VRI_GL_ES_HEADERS
        }
        void VRI_CALL DestroyPipeline(VriPipeline* pipeline)
        {
            if (!pipeline)
                return;
            PipelineGL* p = Pipe(pipeline);
            glDeleteProgram(p->program);
            if (p->formatVao)
                glDeleteVertexArrays(1, &p->formatVao);
            delete p;
        }

        // ---- descriptor pools / sets ---------------------------------------
        // GL has no descriptor objects: a set is a CPU record of (binding -> view)
        // that CmdSetDescriptorSet replays as glBindBufferRange / texture binds via
        // the pipeline layout's (set,binding) -> GL-unit map.
        VriResult VRI_CALL CreateDescriptorPool(VriDevice*                   device,
                                                const VriDescriptorPoolDesc* desc,
                                                VriDescriptorPool**          out)
        {
            if (!device || !desc || !out)
                return VriResult_InvalidArgument;
            *out = ToHandle(new DescriptorPoolGL {Dev(device), desc->descriptorSetMaxNum, {}});
            return VriResult_Success;
        }
        void VRI_CALL ResetDescriptorPool(VriDescriptorPool* pool)
        {
            auto* p = reinterpret_cast<DescriptorPoolGL*>(pool);
            for (auto* set : p->sets)
                delete set;
            p->sets.clear();
        }
        void VRI_CALL DestroyDescriptorPool(VriDescriptorPool* pool)
        {
            if (!pool)
                return;
            ResetDescriptorPool(pool);
            delete reinterpret_cast<DescriptorPoolGL*>(pool);
        }
        VriResult VRI_CALL AllocateDescriptorSets(VriDescriptorPool*       pool,
                                                  const VriPipelineLayout* layout,
                                                  uint32_t                 setIndex,
                                                  VriDescriptorSet**       outSets,
                                                  uint32_t                 setNum)
        {
            if (!pool || !layout || (setNum && !outSets))
                return VriResult_InvalidArgument;
            auto*       p = reinterpret_cast<DescriptorPoolGL*>(pool);
            const auto* l = reinterpret_cast<const PipelineLayoutGL*>(layout);
            if (p->device != l->device)
                return VriResult_InvalidArgument;
            if (setNum > p->capacity - p->sets.size())
                return VriResult_OutOfMemory;
            if (setIndex >= l->setCount)
                return VriResult_InvalidArgument;
            for (uint32_t i = 0; i < setNum; ++i)
            {
                auto* set = new DescriptorSetGL {p->device, l, setIndex, {}};
                p->sets.push_back(set);
                outSets[i] = ToHandle(set);
            }
            return VriResult_Success;
        }
        void VRI_CALL UpdateDescriptorRanges(VriDescriptorSet*                   set,
                                             uint32_t                            baseRange,
                                             uint32_t                            rangeNum,
                                             const VriDescriptorRangeUpdateDesc* updates)
        {
            if (!set)
                return;
            auto*                               s = reinterpret_cast<DescriptorSetGL*>(set);
            std::vector<const LayoutBindingGL*> ranges;
            for (const auto& binding : s->layout->bindings)
                if (binding.set == s->setIndex)
                    ranges.push_back(&binding);
            if (baseRange > ranges.size() || rangeNum > ranges.size() - baseRange || (rangeNum && !updates))
            {
                s->device->ReportError("UpdateDescriptorRanges: range index is outside the set layout");
                return;
            }
            for (uint32_t i = 0; i < rangeNum; ++i)
            {
                const auto& update = updates[i];
                if (update.descriptorNum == 0)
                    continue;
                if (update.baseDescriptor != 0 || update.descriptorNum != 1 || !update.descriptors ||
                    !update.descriptors[0])
                {
                    s->device->ReportError("UpdateDescriptorRanges: fixed descriptor arrays are not supported");
                    return;
                }
                const auto* descriptor = reinterpret_cast<const DescriptorGL*>(update.descriptors[0]);
                if (descriptor->device != s->device)
                {
                    s->device->ReportError("UpdateDescriptorRanges: descriptor belongs to another device");
                    return;
                }
            }
            for (uint32_t i = 0; i < rangeNum; ++i)
            {
                if (updates[i].descriptorNum)
                    s->bound[ranges[baseRange + i]->binding] =
                        reinterpret_cast<const DescriptorGL*>(updates[i].descriptors[0]);
            }
        }

        // ---- synchronization (coarse; GL is ordered on one context) --------
        VriResult VRI_CALL CreateFence(VriDevice* device, uint64_t initialValue, VriFence** out)
        {
            if (!device || !out)
                return VriResult_InvalidArgument;
            *out = ToHandle(new FenceGL {Dev(device), initialValue, initialValue, {}});
            return VriResult_Success;
        }
        void VRI_CALL DestroyFence(VriFence* fence)
        {
            if (!fence)
                return;
            FenceGL* f = Fen(fence);
            for (const auto& signal : f->pending)
                glDeleteSync(signal.sync);
            delete f;
        }
        uint64_t VRI_CALL GetFenceValue(VriFence* fence)
        {
            FenceGL* f = Fen(fence);
            while (!f->pending.empty())
            {
                const auto&  signal = f->pending.front();
                const GLenum result = glClientWaitSync(signal.sync, 0, 0);
                if (result == GL_WAIT_FAILED)
                {
                    f->failed = true;
                    break;
                }
                if (result != GL_ALREADY_SIGNALED && result != GL_CONDITION_SATISFIED)
                    break;
                f->value = signal.value;
                glDeleteSync(signal.sync);
                f->pending.pop_front();
            }
            return f->value;
        }
        VriResult VRI_CALL WaitForFence(VriFence* fence, uint64_t value, uint64_t timeoutNs)
        {
            if (!fence)
                return VriResult_InvalidArgument;
            FenceGL*           f = Fen(fence);
            core::WaitDeadline deadline(timeoutNs);
            while (GetFenceValue(fence) < value)
            {
                if (f->failed)
                    return VriResult_Failure;
                const uint64_t remaining = deadline.Remaining();
                if (!remaining)
                    return VriResult_Timeout;
                if (f->pending.empty())
                {
                    // One immediate GL context cannot enqueue a future signal while
                    // its owner is blocked here. Never manufacture completion.
                    if (timeoutNs == UINT64_MAX)
                        return VriResult_InvalidArgument;
                    std::this_thread::yield();
                    continue;
                }
#if defined(__EMSCRIPTEN__)
                // WebGL permits only a zero native timeout; yield for GPU progress.
                emscripten_sleep(1);
#else
                const GLenum status = glClientWaitSync(
                    f->pending.front().sync, GL_SYNC_FLUSH_COMMANDS_BIT, std::min<uint64_t>(remaining, 1000000));
                if (status == GL_WAIT_FAILED)
                    return VriResult_Failure;
#endif
            }
            return VriResult_Success;
        }

        void VRI_CALL Wait(VriFence* fence, uint64_t value)
        {
            if (WaitForFence(fence, value, UINT64_MAX) != VriResult_Success && fence)
            {
                Fen(fence)->device->ReportError(
                    "Wait: GL timeline value has not been submitted or synchronization failed");
            }
        }
