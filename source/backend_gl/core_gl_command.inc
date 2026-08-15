        // core_gl_command.inc - command recording (immediate-mode) and queue submission.
        // #included into core_gl.cpp (one TU); see core_gl.cpp for the split rationale.
        // ---- command recording (immediate) ---------------------------------
        void VRI_CALL CmdBeginRendering(VriCommandBuffer* cmd, const VriAttachmentsDesc* a)
        {
            CommandBufferGL* c = CB(cmd);

            // FBO identity = the attachment set. Reuse a cached, already-configured FBO
            // (no per-pass gen/delete); only a freshly-created one needs configuring.
            FboKey key {};
            key.colorCount = a->colorNum <= FboKey::kMaxColor ? a->colorNum : FboKey::kMaxColor;
            for (uint32_t i = 0; i < key.colorCount; ++i)
            {
                const DescriptorGL* v = Desc(a->colors[i].view);
                key.colorId[i]        = v->texture->id;
                key.colorMip[i]       = v->mip;
            }
            if (a->depth)
            {
                const DescriptorGL* dv = Desc(a->depth->view);
                key.depthId            = dv->texture->id;
                key.depthMip           = dv->mip;
            }

#if !defined(VRI_GL_ES_HEADERS) // GL 4.5: configure + clear the FBO by name (undeclared on GLES3)
            const bool dsa = c->device->Features().dsa;
#endif
            bool         isNew = false;
            const GLuint fbo   = c->device->AcquireFbo(key, isNew);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            c->fbo = fbo;

            if (isNew) // configure attachments + draw buffers once for this attachment set
            {
                std::vector<GLenum> drawBufs(a->colorNum);
                for (uint32_t i = 0; i < a->colorNum; ++i)
                {
                    const DescriptorGL* v   = Desc(a->colors[i].view);
                    const GLenum        att = GL_COLOR_ATTACHMENT0 + i;
                    if (v->texture->isRenderbuffer) // MSAA renderbuffer attachment
                    {
#if !defined(VRI_GL_ES_HEADERS)
                        if (dsa)
                            glNamedFramebufferRenderbuffer(fbo, att, GL_RENDERBUFFER, v->texture->id);
                        else
#endif
                            glFramebufferRenderbuffer(GL_FRAMEBUFFER, att, GL_RENDERBUFFER, v->texture->id);
                    }
                    else if (a->viewMask != 0 && g_FramebufferTextureMultiviewOVR)
                    {
                        // Multiview: bind all views of the array texture in one pass (OVR_multiview).
                        g_FramebufferTextureMultiviewOVR(GL_FRAMEBUFFER,
                                                         att,
                                                         v->texture->id,
                                                         static_cast<GLint>(v->mip),
                                                         static_cast<GLint>(v->layer),
                                                         static_cast<GLsizei>(std::popcount(a->viewMask)));
                    }
                    else
                    {
#if !defined(VRI_GL_ES_HEADERS)
                        if (dsa)
                            glNamedFramebufferTexture(fbo, att, v->texture->id, static_cast<GLint>(v->mip));
                        else
#endif
                            glFramebufferTexture2D(
                                GL_FRAMEBUFFER, att, v->texture->target, v->texture->id, static_cast<GLint>(v->mip));
                    }
                    drawBufs[i] = att;
                }
#if !defined(VRI_GL_ES_HEADERS)
                if (dsa)
                    glNamedFramebufferDrawBuffers(fbo, static_cast<GLsizei>(drawBufs.size()), drawBufs.data());
                else
#endif
                    glDrawBuffers(static_cast<GLsizei>(drawBufs.size()), drawBufs.data());
                if (a->depth)
                {
                    const DescriptorGL* dv = Desc(a->depth->view);
                    const GLenum        attach =
                        dv->texture->glFormat == GL_DEPTH_STENCIL ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT;
                    if (a->viewMask != 0 && g_FramebufferTextureMultiviewOVR)
                        g_FramebufferTextureMultiviewOVR(GL_FRAMEBUFFER,
                                                         attach,
                                                         dv->texture->id,
                                                         static_cast<GLint>(dv->mip),
                                                         static_cast<GLint>(dv->layer),
                                                         static_cast<GLsizei>(std::popcount(a->viewMask)));
                    else
#if !defined(VRI_GL_ES_HEADERS)
                        if (dsa)
                        glNamedFramebufferTexture(fbo, attach, dv->texture->id, static_cast<GLint>(dv->mip));
                    else
#endif
                        glFramebufferTexture2D(
                            GL_FRAMEBUFFER, attach, dv->texture->target, dv->texture->id, static_cast<GLint>(dv->mip));
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
#if !defined(VRI_GL_ES_HEADERS)
                    if (dsa)
                        glClearNamedFramebufferfv(fbo, GL_COLOR, static_cast<GLint>(i), col);
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
                    const GLfloat depth   = a->depth->clearValue.depthStencil.depth;
                    const GLint   stencil = static_cast<GLint>(a->depth->clearValue.depthStencil.stencil);
#if !defined(VRI_GL_ES_HEADERS)
                    if (dsa)
                        glClearNamedFramebufferfi(fbo, GL_DEPTH_STENCIL, 0, depth, stencil);
                    else
#endif
                        glClearBufferfi(GL_DEPTH_STENCIL, 0, depth, stencil);
                }
                else
                {
                    const GLfloat d = a->depth->clearValue.depthStencil.depth;
#if !defined(VRI_GL_ES_HEADERS)
                    if (dsa)
                        glClearNamedFramebufferfv(fbo, GL_DEPTH, 0, &d);
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
                    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,
                                           GL_COLOR_ATTACHMENT0,
                                           r.dst->target,
                                           r.dst->id,
                                           static_cast<GLint>(r.dstMip));
                    const GLenum drawBuf = GL_COLOR_ATTACHMENT0;
                    glDrawBuffers(1, &drawBuf);
                    glBlitFramebuffer(0,
                                      0,
                                      static_cast<GLint>(r.dst->width),
                                      static_cast<GLint>(r.dst->height),
                                      0,
                                      0,
                                      static_cast<GLint>(r.dst->width),
                                      static_cast<GLint>(r.dst->height),
                                      GL_COLOR_BUFFER_BIT,
                                      GL_NEAREST);
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
            if (!num)
                return;
            glViewport(static_cast<GLint>(vps[0].x),
                       static_cast<GLint>(vps[0].y),
                       static_cast<GLsizei>(vps[0].width),
                       static_cast<GLsizei>(vps[0].height));
            glDepthRangef(vps[0].minDepth, vps[0].maxDepth);
        }
        void VRI_CALL CmdSetScissors(VriCommandBuffer*, const VriRect* r, uint32_t num)
        {
            if (!num)
                return;
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
            PipelineGL*      p = Pipe(pipeline);
            c->boundPipeline   = p;
            glUseProgram(p->program);
            if (p->isCompute)
                return; // no graphics state to apply
            if (p->cullEnable)
            {
                glEnable(GL_CULL_FACE);
                glCullFace(p->cullFace);
            }
            else
                glDisable(GL_CULL_FACE);
            glFrontFace(p->frontFace);
            if (p->depthTest)
            {
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(p->depthFunc);
            }
            else
                glDisable(GL_DEPTH_TEST);
            glDepthMask(p->depthWrite ? GL_TRUE : GL_FALSE);
            if (p->blendEnable)
            {
                glEnable(GL_BLEND);
                glBlendFuncSeparate(p->srcRGB, p->dstRGB, p->srcA, p->dstA);
                glBlendEquationSeparate(p->rgbOp, p->aOp);
            }
            else
                glDisable(GL_BLEND);
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
            else
                glDisable(GL_STENCIL_TEST);
            c->topology = p->topology;
#if !defined(VRI_GL_ES_HEADERS) && defined(GL_PATCHES) && defined(GL_PATCH_VERTICES) // tessellation: desktop GL only
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
                                           lb->type == VriDescriptorType_StructuredBuffer) ?
                                              GL_SHADER_STORAGE_BUFFER :
                                              GL_UNIFORM_BUFFER;
                    glBindBufferRange(
                        target,
                        lb->glUnit,
                        view->buffer->id,
                        static_cast<GLintptr>(view->bufferOffset),
                        static_cast<GLsizeiptr>(view->bufferRange ? view->bufferRange : view->buffer->size));
                }
#if !defined(VRI_GL_ES_HEADERS) // image load/store is desktop GL 4.2+; WebGL2 has no compute
                else if (lb->type == VriDescriptorType_StorageTexture && view->texture)
                {
                    // storage image (compute UAV): bind to its image unit for image load/store.
                    glBindImageTexture(lb->glUnit,
                                       view->texture->id,
                                       static_cast<GLint>(view->mip),
                                       GL_FALSE,
                                       0,
                                       GL_READ_WRITE,
                                       view->texture->internalFormat);
                }
#endif
            }
            // Textures + samplers are bound per combined sampler the pipeline declared:
            // GLSL fuses (texture, sampler) into one sampler2D at a texture unit.
#if !defined(VRI_GL_ES_HEADERS) // GL 4.5: glBindTextureUnit + glTextureParameteri (undeclared on GLES3)
            const bool dsa = c->device->Features().dsa; // no active-unit churn on the modern path
#endif
            if (c->boundPipeline)
                for (const CombinedSamplerGL& cs : c->boundPipeline->combinedSamplers)
                {
#if !defined(VRI_GL_ES_HEADERS)
                    if (!dsa)
#endif
                        glActiveTexture(GL_TEXTURE0 + cs.unit);
                    if (cs.texSet == setIndex)
                    {
                        auto it = s->bound.find(cs.texBinding);
                        if (it != s->bound.end() && it->second && it->second->texture)
                        {
                            const DescriptorGL* tv = it->second;
                            if (tv->texture->isRenderbuffer)
                                continue; // renderbuffers aren't sampled
                                          // Honor the view's base mip so a mip-range view samples that level.
#if !defined(VRI_GL_ES_HEADERS)
                            if (dsa)
                            {
                                glBindTextureUnit(cs.unit, tv->texture->id);
                                glTextureParameteri(
                                    tv->texture->id, GL_TEXTURE_BASE_LEVEL, static_cast<GLint>(tv->mip));
                            }
                            else
#endif
                            {
                                glBindTexture(tv->texture->target, tv->texture->id);
                                glTexParameteri(
                                    tv->texture->target, GL_TEXTURE_BASE_LEVEL, static_cast<GLint>(tv->mip));
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
                const void*     p   = bytes + pm.offset;
                const GLint     loc = pm.location;
                const GLsizei   n   = static_cast<GLsizei>(pm.count);
                const GLfloat*  f   = static_cast<const GLfloat*>(p);
                const GLboolean tr  = pm.transpose ? GL_TRUE : GL_FALSE;
                if (pm.columns == 4)
                    glUniformMatrix4fv(loc, n, tr, f);
                else if (pm.columns == 3)
                    glUniformMatrix3fv(loc, n, tr, f);
                else if (pm.columns == 2)
                    glUniformMatrix2fv(loc, n, tr, f);
                else if (pm.basetype == 1) // int
                {
                    const GLint* v = static_cast<const GLint*>(p);
                    if (pm.vecsize == 1)
                        glUniform1iv(loc, n, v);
                    else if (pm.vecsize == 2)
                        glUniform2iv(loc, n, v);
                    else if (pm.vecsize == 3)
                        glUniform3iv(loc, n, v);
                    else
                        glUniform4iv(loc, n, v);
                }
                else if (pm.basetype == 2) // uint
                {
                    const GLuint* v = static_cast<const GLuint*>(p);
                    if (pm.vecsize == 1)
                        glUniform1uiv(loc, n, v);
                    else if (pm.vecsize == 2)
                        glUniform2uiv(loc, n, v);
                    else if (pm.vecsize == 3)
                        glUniform3uiv(loc, n, v);
                    else
                        glUniform4uiv(loc, n, v);
                }
                else // float
                {
                    if (pm.vecsize == 1)
                        glUniform1fv(loc, n, f);
                    else if (pm.vecsize == 2)
                        glUniform2fv(loc, n, f);
                    else if (pm.vecsize == 3)
                        glUniform3fv(loc, n, f);
                    else
                        glUniform4fv(loc, n, f);
                }
            }
        }
        void VRI_CALL CmdSetVertexBuffers(VriCommandBuffer*             cmd,
                                          uint32_t                      baseSlot,
                                          const VriVertexBufferBinding* bindings,
                                          uint32_t                      num)
        {
            CommandBufferGL* c = CB(cmd);
            for (uint32_t i = 0; i < num; ++i)
                c->vertexBuffers[baseSlot + i] =
                    VertexBufferBindingGL {Buf(bindings[i].buffer)->id, bindings[i].offset};
        }
        void VRI_CALL CmdSetIndexBuffer(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, VriIndexType type)
        {
            CommandBufferGL* c = CB(cmd);
            c->indexBuffer     = Buf(buffer)->id;
            c->indexOffset     = offset;
            c->indexType       = type == VriIndexType_UInt16 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT;
        }
        // Configure the default VAO's attribute pointers for the bound pipeline +
        // vertex buffers (classic glVertexAttribPointer path; GLES3/WebGL2 has no
        // separate attribute format). Disables any attribs the previous draw enabled.
        // baseVertex != 0 is only passed on the WebGL2/GLES tier (no glDrawElementsBaseVertex):
        // there it's emulated by shifting the per-vertex attribute pointers by baseVertex*stride.
        void SetupVertexAttribs(CommandBufferGL* c, int32_t baseVertex = 0)
        {
            // Separate-format path (GL 4.3+): bind the pipeline's pre-baked VAO and
            // only point each stream at its buffer (format is already in the VAO).
            // formatVao is always 0 on Emscripten (the 4.3+ path compiles out there).
#if !defined(VRI_GL_ES_HEADERS)
            if (c->boundPipeline && c->boundPipeline->formatVao)
            {
                const PipelineGL* p   = c->boundPipeline;
                const bool        dsa = c->device->Features().dsa;
                glBindVertexArray(p->formatVao);
                for (const VboBindingGL& vb : p->vboBindings)
                {
                    auto it = c->vertexBuffers.find(vb.slot);
                    if (it == c->vertexBuffers.end())
                        continue;
                    if (dsa)
                        glVertexArrayVertexBuffer(
                            p->formatVao, vb.slot, it->second.id, static_cast<GLintptr>(it->second.offset), vb.stride);
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
                    // Emulated base-vertex: shift per-vertex streams (divisor == 0) by
                    // baseVertex*stride; per-instance streams are indexed by instance, untouched.
                    const uintptr_t baseOff = (baseVertex && a.divisor == 0) ?
                                                  static_cast<uintptr_t>(static_cast<int64_t>(baseVertex) * a.stride) :
                                                  0u;
                    const void*     attrOff =
                        reinterpret_cast<const void*>(static_cast<uintptr_t>(it->second.offset + a.offset) + baseOff);
                    if (a.integer)
                        glVertexAttribIPointer(a.location, a.size, a.type, a.stride, attrOff);
                    else
                        glVertexAttribPointer(a.location, a.size, a.type, a.normalized, a.stride, attrOff);
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
            if (!p)
                return;
            if (p->baseVertexLoc >= 0)
                glUniform1i(p->baseVertexLoc, baseVertex);
            if (p->baseInstanceLoc >= 0)
                glUniform1i(p->baseInstanceLoc, static_cast<GLint>(baseInstance));
        }
        void VRI_CALL CmdDraw(VriCommandBuffer* cmd, const VriDrawDesc* d)
        {
            CommandBufferGL* c = CB(cmd);
            SetupVertexAttribs(c);
            // baseVertex maps to `first` (gl_VertexID already includes it, like VK).
#if !defined(VRI_GL_ES_HEADERS)
            if (d->baseInstance != 0 && c->device->Features().baseInstance) // GL 4.2: honor firstInstance
            {
                ApplyBaseOffsetUniforms(c, static_cast<int32_t>(d->baseVertex), d->baseInstance);
                glDrawArraysInstancedBaseInstance(c->topology,
                                                  static_cast<GLint>(d->baseVertex),
                                                  static_cast<GLsizei>(d->vertexNum),
                                                  static_cast<GLsizei>(d->instanceNum),
                                                  d->baseInstance);
                return;
            }
#endif
            if (d->baseInstance != 0) // requested but unsupported here: explicit, never silent
                c->device->ReportError(
                    "CmdDraw: baseInstance != 0 requires GL 4.2 (ARB_base_instance); unavailable on this context");
            glDrawArraysInstanced(c->topology,
                                  static_cast<GLint>(d->baseVertex),
                                  static_cast<GLsizei>(d->vertexNum),
                                  static_cast<GLsizei>(d->instanceNum));
        }
        void VRI_CALL CmdDrawIndexed(VriCommandBuffer* cmd, const VriDrawIndexedDesc* d)
        {
            CommandBufferGL* c = CB(cmd);
#if defined(VRI_GL_ES_HEADERS)
            // WebGL2/GLES has no glDrawElementsBaseVertex; bake vertexOffset into the per-vertex
            // attribute pointers so base-vertex indexed draws (e.g. an ImGui popup/combo whose
            // 2nd draw list has vtxOffset > 0) render correctly instead of being rejected.
            SetupVertexAttribs(c, d->vertexOffset);
#else
            SetupVertexAttribs(c);
#endif
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, c->indexBuffer);
            const uint32_t  indexSize = c->indexType == GL_UNSIGNED_SHORT ? 2u : 4u;
            const uintptr_t offset =
                static_cast<uintptr_t>(c->indexOffset) + static_cast<uintptr_t>(d->baseIndex) * indexSize;
            const void*   ip   = reinterpret_cast<const void*>(offset);
            const GLsizei inst = static_cast<GLsizei>(d->instanceNum ? d->instanceNum : 1u);
#if !defined(VRI_GL_ES_HEADERS)
            // Base offsets: vertexOffset is GL 3.2 (glDrawElements*BaseVertex, all desktop
            // incl. macOS 4.1); baseInstance is GL 4.2. gl_VertexID/gl_InstanceID parity is
            // handled by SPIRV-Cross (gl_Base*ARB or the fallback uniforms set below).
            if (d->vertexOffset != 0 || d->baseInstance != 0)
            {
                const bool canBaseInstance = c->device->Features().baseInstance;
                if (d->baseInstance != 0 && !canBaseInstance) // explicit, never silent
                    c->device->ReportError("CmdDrawIndexed: baseInstance != 0 requires GL 4.2 (ARB_base_instance); "
                                           "unavailable on this context");
                ApplyBaseOffsetUniforms(c, d->vertexOffset, d->baseInstance);
                if (d->baseInstance != 0 && canBaseInstance)
                    glDrawElementsInstancedBaseVertexBaseInstance(c->topology,
                                                                  static_cast<GLsizei>(d->indexNum),
                                                                  c->indexType,
                                                                  ip,
                                                                  inst,
                                                                  d->vertexOffset,
                                                                  d->baseInstance);
                else if (inst > 1)
                    glDrawElementsInstancedBaseVertex(
                        c->topology, static_cast<GLsizei>(d->indexNum), c->indexType, ip, inst, d->vertexOffset);
                else
                    glDrawElementsBaseVertex(
                        c->topology, static_cast<GLsizei>(d->indexNum), c->indexType, ip, d->vertexOffset);
                return;
            }
#else
            // vertexOffset is emulated via the attribute pointers (SetupVertexAttribs above);
            // base-instance has no WebGL2 equivalent, so it stays explicit (never silent).
            if (d->baseInstance != 0)
                c->device->ReportError(
                    "CmdDrawIndexed: baseInstance != 0 is unsupported on WebGL2 (no ARB_base_instance)");
#endif
            if (d->instanceNum <= 1)
                glDrawElements(c->topology, static_cast<GLsizei>(d->indexNum), c->indexType, ip);
            else
                glDrawElementsInstanced(c->topology,
                                        static_cast<GLsizei>(d->indexNum),
                                        c->indexType,
                                        ip,
                                        static_cast<GLsizei>(d->instanceNum));
        }
        void VRI_CALL
        CmdDrawIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint32_t drawNum, uint32_t stride)
        {
            CommandBufferGL* c = CB(cmd);
#if !defined(VRI_GL_ES_HEADERS)
            // GL indirect commands match VK/GL layout {count, instanceCount, first,
            // baseInstance}. glDrawArraysIndirect is GL 4.0 (all desktop); >1 draw uses
            // multi-draw-indirect (4.3) where available, else loops single draws.
            SetupVertexAttribs(c);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, Buf(buffer)->id);
            if (drawNum > 1 && c->device->Features().drawIndirect)
                glMultiDrawArraysIndirect(c->topology,
                                          reinterpret_cast<const void*>(static_cast<uintptr_t>(offset)),
                                          static_cast<GLsizei>(drawNum),
                                          static_cast<GLsizei>(stride));
            else
                for (uint32_t i = 0; i < (drawNum ? drawNum : 1u); ++i)
                    glDrawArraysIndirect(c->topology,
                                         reinterpret_cast<const void*>(
                                             static_cast<uintptr_t>(offset + static_cast<uint64_t>(i) * stride)));
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
#elif defined(VRI_GL_NATIVE_ES)
            // Native OpenGL ES 3.1: glDrawArraysIndirect is core (unlike WebGL2). Core ES has
            // no multi-draw-indirect, so issue one indirect draw per command from the buffer.
            SetupVertexAttribs(c);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, Buf(buffer)->id);
            for (uint32_t i = 0; i < (drawNum ? drawNum : 1u); ++i)
                glDrawArraysIndirect(
                    c->topology,
                    reinterpret_cast<const void*>(static_cast<uintptr_t>(offset + static_cast<uint64_t>(i) * stride)));
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
#else
            (void)buffer;
            (void)offset;
            (void)drawNum;
            (void)stride;
            c->device->ReportError("CmdDrawIndirect: indirect draw is unavailable on WebGL2"); // explicit, never silent
#endif
        }
        // Indexed-indirect (glMultiDrawElementsIndirect, GL 4.3) + the indirect-count variants
        // (glMultiDraw*IndirectCount, ARB_indirect_parameters / GL 4.6). The element-array + indirect
        // buffers are already bound (CmdSetIndexBuffer / below). Desktop GL only.
        void VRI_CALL CmdDrawIndexedIndirect(VriCommandBuffer* cmd,
                                             VriBuffer*        buffer,
                                             uint64_t          offset,
                                             uint32_t          drawNum,
                                             uint32_t          stride)
        {
            CommandBufferGL* c = CB(cmd);
#if !defined(VRI_GL_ES_HEADERS)
            SetupVertexAttribs(c);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, Buf(buffer)->id);
            if (drawNum > 1 && c->device->Features().drawIndirect)
                glMultiDrawElementsIndirect(c->topology,
                                            c->indexType,
                                            reinterpret_cast<const void*>(static_cast<uintptr_t>(offset)),
                                            static_cast<GLsizei>(drawNum),
                                            static_cast<GLsizei>(stride));
            else
                for (uint32_t i = 0; i < (drawNum ? drawNum : 1u); ++i)
                    glDrawElementsIndirect(c->topology,
                                           c->indexType,
                                           reinterpret_cast<const void*>(
                                               static_cast<uintptr_t>(offset + static_cast<uint64_t>(i) * stride)));
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
#else
            (void)buffer;
            (void)offset;
            (void)drawNum;
            (void)stride;
            c->device->ReportError("CmdDrawIndexedIndirect: indirect draw is unavailable on WebGL2");
#endif
        }
        void VRI_CALL CmdDrawIndirectCount(VriCommandBuffer* cmd,
                                           VriBuffer*        buffer,
                                           uint64_t          offset,
                                           VriBuffer*        countBuffer,
                                           uint64_t          countOffset,
                                           uint32_t          maxDrawNum,
                                           uint32_t          stride)
        {
            CommandBufferGL* c = CB(cmd);
#if !defined(VRI_GL_ES_HEADERS)
            if (!c->device->Features().drawIndirectCount)
            {
                c->device->ReportError("CmdDrawIndirectCount: needs OpenGL 4.6 (ARB_indirect_parameters)");
                return;
            }
            SetupVertexAttribs(c);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, Buf(buffer)->id);
            glBindBuffer(GL_PARAMETER_BUFFER, Buf(countBuffer)->id); // count source
            glMultiDrawArraysIndirectCount(c->topology,
                                           reinterpret_cast<const void*>(static_cast<uintptr_t>(offset)),
                                           static_cast<GLintptr>(countOffset),
                                           static_cast<GLsizei>(maxDrawNum),
                                           static_cast<GLsizei>(stride));
            glBindBuffer(GL_PARAMETER_BUFFER, 0);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
#else
            (void)buffer;
            (void)offset;
            (void)countBuffer;
            (void)countOffset;
            (void)maxDrawNum;
            (void)stride;
            c->device->ReportError("CmdDrawIndirectCount: unavailable on WebGL2");
#endif
        }
        void VRI_CALL CmdDrawIndexedIndirectCount(VriCommandBuffer* cmd,
                                                  VriBuffer*        buffer,
                                                  uint64_t          offset,
                                                  VriBuffer*        countBuffer,
                                                  uint64_t          countOffset,
                                                  uint32_t          maxDrawNum,
                                                  uint32_t          stride)
        {
            CommandBufferGL* c = CB(cmd);
#if !defined(VRI_GL_ES_HEADERS)
            if (!c->device->Features().drawIndirectCount)
            {
                c->device->ReportError("CmdDrawIndexedIndirectCount: needs OpenGL 4.6 (ARB_indirect_parameters)");
                return;
            }
            SetupVertexAttribs(c);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, Buf(buffer)->id);
            glBindBuffer(GL_PARAMETER_BUFFER, Buf(countBuffer)->id);
            glMultiDrawElementsIndirectCount(c->topology,
                                             c->indexType,
                                             reinterpret_cast<const void*>(static_cast<uintptr_t>(offset)),
                                             static_cast<GLintptr>(countOffset),
                                             static_cast<GLsizei>(maxDrawNum),
                                             static_cast<GLsizei>(stride));
            glBindBuffer(GL_PARAMETER_BUFFER, 0);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
#else
            (void)buffer;
            (void)offset;
            (void)countBuffer;
            (void)countOffset;
            (void)maxDrawNum;
            (void)stride;
            c->device->ReportError("CmdDrawIndexedIndirectCount: unavailable on WebGL2");
#endif
        }
        void VRI_CALL CmdDispatch(VriCommandBuffer*, const VriDispatchDesc* d)
        {
#if defined(VRI_GL_ES_HEADERS)
            (void)d; // compute is unavailable in WebGL2 (no pipeline can be created)
#else
            glDispatchCompute(d->x, d->y, d->z);
            // Make SSBO writes visible to the subsequent copy/readback (single context,
            // but the GPU pipeline still needs an explicit memory barrier).
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT |
                            GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
#endif
        }
        void VRI_CALL CmdDispatchIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset)
        {
            CommandBufferGL* c = CB(cmd);
#if !defined(VRI_GL_ES_HEADERS)
            // glDispatchComputeIndirect is GL 4.3 (compute already requires 4.3, so no
            // extra gate). Command layout {x, y, z} matches VK/WebGPU.
            glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, Buf(buffer)->id);
            glDispatchComputeIndirect(static_cast<GLintptr>(offset));
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT |
                            GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
            glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
#else
            (void)buffer;
            (void)offset;
            c->device->ReportError(
                "CmdDispatchIndirect: indirect dispatch is unavailable on WebGL2"); // explicit, never silent
#endif
        }
        void VRI_CALL CmdBarrier(VriCommandBuffer*, const VriBarrierGroupDesc*) {} // GL is ordered on one context
        void VRI_CALL
        CmdClearStorageBuffer(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint64_t size, uint32_t value)
        {
            CommandBufferGL* c = CB(cmd);
#if !defined(VRI_GL_ES_HEADERS)
            BufferGL*        b  = Buf(buffer);
            const GLsizeiptr sz = size ? static_cast<GLsizeiptr>(size) : static_cast<GLsizeiptr>(b->size - offset);
            glBindBuffer(GL_COPY_WRITE_BUFFER, b->id);
            // R32UI source replicates the uint32 across the range (glClearBufferSubData, GL 4.3).
            glClearBufferSubData(GL_COPY_WRITE_BUFFER,
                                 GL_R32UI,
                                 static_cast<GLintptr>(offset),
                                 sz,
                                 GL_RED_INTEGER,
                                 GL_UNSIGNED_INT,
                                 &value);
            glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
#else
            (void)buffer;
            (void)offset;
            (void)size;
            (void)value;
            c->device->ReportError("CmdClearStorageBuffer: glClearBufferSubData is unavailable on WebGL2/GLES");
#endif
        }
        void VRI_CALL CmdClearStorageTexture(VriCommandBuffer* cmd, VriTexture* texture, const VriClearColor* value)
        {
            CommandBufferGL* c = CB(cmd);
#if !defined(VRI_GL_ES_HEADERS)
            TextureGL* t = Tex(texture);
            // glClearTexImage (GL 4.4) clears the whole level (all layers). Use the texture's component
            // format but read the value as FLOAT (GL converts to the internal format) or, for integer
            // textures, as UINT - matching the portable "f32 for float/unorm, u32 for integer" contract.
            const bool isInt = t->glFormat == GL_RED_INTEGER || t->glFormat == GL_RG_INTEGER ||
                               t->glFormat == GL_RGB_INTEGER || t->glFormat == GL_RGBA_INTEGER;
            glClearTexImage(t->id, 0, t->glFormat, isInt ? GL_UNSIGNED_INT : GL_FLOAT, value);
#else
            (void)texture;
            (void)value;
            c->device->ReportError("CmdClearStorageTexture: glClearTexImage is unavailable on WebGL2/GLES");
#endif
        }
        void VRI_CALL CmdCopyBuffer(VriCommandBuffer*, VriBuffer* dst, VriBuffer* src, const VriBufferCopyDesc* r)
        {
            // Keep an element-class buffer on GL_ELEMENT_ARRAY_BUFFER so WebGL never
            // sees it bound to a non-element target (which would taint it).
            const BufferGL* s = Buf(src);
            const BufferGL* d = Buf(dst);
#if !defined(VRI_GL_ES_HEADERS)
            if (s->device->Features().dsa) // GL 4.5: copy without disturbing binding points
            {
                glCopyNamedBufferSubData(s->id,
                                         d->id,
                                         static_cast<GLintptr>(r->srcOffset),
                                         static_cast<GLintptr>(r->dstOffset),
                                         static_cast<GLsizeiptr>(r->size));
                return;
            }
#endif
            // Two cases need the CPU bounce (read back with glGetBufferSubData, write with
            // glBufferSubData; each buffer bound only to its own target so neither gets tainted):
            //  1. WebGL2/GLES reject glCopyBufferSubData between an element-array buffer and a
            //     non-element one ("cannot copy into an element buffer destination from a
            //     non-element buffer source"). The common case is a staging upload into an index buffer.
            //  2. Apple's desktop-GL driver leaves glCopyBufferSubData unimplemented (it logs
            //     "gldCopyBufferSubData: NEEDS IMPLEMENTATION" and silently does nothing), and DSA
            //     (glCopyNamedBufferSubData) isn't available either on GL 4.1, so always bounce.
            const bool srcEl = s->target == GL_ELEMENT_ARRAY_BUFFER;
            const bool dstEl = d->target == GL_ELEMENT_ARRAY_BUFFER;
#if defined(__APPLE__) && !defined(VRI_GL_ES_HEADERS)
            const bool cpuBounce = true;
#else
            const bool cpuBounce = s->device->IsES() && srcEl != dstEl;
#endif
            if (cpuBounce)
            {
                void* tmp = std::malloc(static_cast<size_t>(r->size));
                glBindBuffer(s->target, s->id);
                ReadBufferSubData(
                    s->target, static_cast<GLintptr>(r->srcOffset), static_cast<GLsizeiptr>(r->size), tmp);
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
            glCopyBufferSubData(rt,
                                wt,
                                static_cast<GLintptr>(r->srcOffset),
                                static_cast<GLintptr>(r->dstOffset),
                                static_cast<GLsizeiptr>(r->size));
            glBindBuffer(rt, 0);
            glBindBuffer(wt, 0);
        }
        void VRI_CALL CmdCopyTexture(VriCommandBuffer*, VriTexture* dst, VriTexture* src, const VriTextureCopyDesc*)
        {
            // glCopyImageSubData is GL 4.3 / GLES 3.2; use a portable framebuffer
            // blit so the path also works on GLES 3.0 / WebGL2.
            const TextureGL* s       = reinterpret_cast<const TextureGL*>(src);
            const TextureGL* d       = reinterpret_cast<const TextureGL*>(dst);
            GLuint           fbos[2] = {0, 0};
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
        void VRI_CALL CmdUploadBufferToTexture(VriCommandBuffer*,
                                               VriTexture*                     dst,
                                               VriBuffer*                      src,
                                               const VriBufferTextureCopyDesc* region)
        {
            TextureGL*     t     = Tex(dst);
            const uint32_t w     = region->texture.width ? region->texture.width : t->width;
            const uint32_t h     = region->texture.height ? region->texture.height : t->height;
            const void*    off   = reinterpret_cast<const void*>(static_cast<uintptr_t>(region->bufferOffset));
            const bool     array = t->target == GL_TEXTURE_2D_ARRAY; // 2D array texture -> 3D sub-image, z = layer
            const bool     cube  = t->target == GL_TEXTURE_CUBE_MAP; // cube -> per-face 2D sub-image (z = face)
            const GLint    mip   = static_cast<GLint>(region->texture.mip);
            const GLint    z     = static_cast<GLint>(region->texture.baseLayer);
            const GLsizei  d     = static_cast<GLsizei>(region->texture.layerNum ? region->texture.layerNum : 1u);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Buf(src)->id); // PBO source binding stays global even with DSA
#if !defined(VRI_GL_ES_HEADERS)
            if (t->device->Features()
                    .dsa) // GL 4.5: upload without binding the texture (cube uses the layered 3D path, z = face)
            {
                if (array || cube)
                    glTextureSubImage3D(t->id,
                                        mip,
                                        region->texture.x,
                                        region->texture.y,
                                        z,
                                        static_cast<GLsizei>(w),
                                        static_cast<GLsizei>(h),
                                        d,
                                        t->glFormat,
                                        t->glType,
                                        off);
                else
                    glTextureSubImage2D(t->id,
                                        mip,
                                        region->texture.x,
                                        region->texture.y,
                                        static_cast<GLsizei>(w),
                                        static_cast<GLsizei>(h),
                                        t->glFormat,
                                        t->glType,
                                        off);
            }
            else
#endif
            {
                glBindTexture(t->target, t->id);
                if (cube)
                    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + z,
                                    mip,
                                    region->texture.x,
                                    region->texture.y,
                                    static_cast<GLsizei>(w),
                                    static_cast<GLsizei>(h),
                                    t->glFormat,
                                    t->glType,
                                    off);
                else if (array)
                    glTexSubImage3D(t->target,
                                    mip,
                                    region->texture.x,
                                    region->texture.y,
                                    z,
                                    static_cast<GLsizei>(w),
                                    static_cast<GLsizei>(h),
                                    d,
                                    t->glFormat,
                                    t->glType,
                                    off);
                else
                    glTexSubImage2D(t->target,
                                    mip,
                                    region->texture.x,
                                    region->texture.y,
                                    static_cast<GLsizei>(w),
                                    static_cast<GLsizei>(h),
                                    t->glFormat,
                                    t->glType,
                                    off);
                glBindTexture(t->target, 0);
            }
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }
        void VRI_CALL CmdReadbackTextureToBuffer(VriCommandBuffer*,
                                                 VriBuffer*                      dst,
                                                 VriTexture*                     src,
                                                 const VriBufferTextureCopyDesc* region)
        {
            // No glGetTextureImage on GLES/WebGL: read via a temp FBO + glReadPixels
            // into a PBO. glReadPixels reads bottom-up; combined with the in-shader
            // flip_vert_y this yields VRI top-left row order.
            const TextureGL* t   = reinterpret_cast<const TextureGL*>(src);
            BufferGL*        b   = Buf(dst);
            const GLint      mip = static_cast<GLint>(region->texture.mip);
            GLsizei          w   = static_cast<GLsizei>(t->width >> region->texture.mip);
            if (w < 1)
                w = 1;
            GLsizei h = static_cast<GLsizei>(t->height >> region->texture.mip);
            if (h < 1)
                h = 1;

            GLuint fbo = 0;
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            // Array / cube textures must attach a single layer (glFramebufferTexture2D rejects a
            // 2D_ARRAY target); honor the region's baseLayer. Plain 2D uses the 2D attach.
            if (t->target == GL_TEXTURE_2D_ARRAY || t->target == GL_TEXTURE_CUBE_MAP || t->layerNum > 1)
                glFramebufferTextureLayer(
                    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, t->id, mip, static_cast<GLint>(region->texture.baseLayer));
            else
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
            if (b->device->IsES() && readFmt == GL_BGRA)
                readFmt = GL_RGBA;
            glReadPixels(
                0, 0, w, h, readFmt, t->glType, reinterpret_cast<void*>(static_cast<uintptr_t>(region->bufferOffset)));
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

