// core_gl_resource.inc - buffers, textures, explicit-memory stubs, views and samplers.
// #included into core_gl.cpp (one TU); see core_gl.cpp for the split rationale.
// ---- queries -------------------------------------------------------
const VriDeviceDesc* VRI_CALL GetDeviceDesc(const VriDevice* device) { return &Dev(device)->Desc(); }
// Answered from the same tables that decide what this backend actually does:
// ToGLFormat is what CreateTexture uses, ToGLVertexFormat what the vertex input
// uses. Both fall back to a substitute for an unmapped format (RGBA8 / float4),
// so a blanket "everything is supported" reply was actively misleading: it told
// callers that, say, D32_SFLOAT_S8_UINT was fine when CreateTexture would hand
// them an RGBA8 texture, and attaching that as depth fails framebuffer
// completeness. It also withheld DepthStencil from the two depth formats GL does
// implement, so the usual "probe D24S8, else fall back" pattern picked the wrong
// one here. Report per format instead.
// Whether this context samples the block-compressed `format`; false for every other format.
bool CompressedFormatAvailable(const GlFeatures& caps, VriFormat format)
{
    switch (format)
    {
        case VriFormat_BC1_UNORM:
        case VriFormat_BC2_UNORM:
        case VriFormat_BC3_UNORM:
            return caps.s3tc;
        case VriFormat_BC4_UNORM:
        case VriFormat_BC5_UNORM:
            return caps.rgtc;
        case VriFormat_BC6H_UFLOAT:
        case VriFormat_BC7_UNORM:
            return caps.bptc;
        case VriFormat_ETC2_RGBA8_UNORM:
            return caps.etc2;
        case VriFormat_ASTC_4x4_UNORM:
            return caps.astc;
        default:
            return false;
    }
}

VriFormatSupportFlags VRI_CALL GetFormatSupport(const VriDevice* device, VriFormat format)
{
    VriFormatSupportFlags r    = VriFormatSupport_None;
    const GlFeatures&     caps = Dev(device)->Features();
    // A compressed format is sampled only: it cannot be rendered to, stored to or read back on GL.
    if (ToGLCompressedFormat(format).internalFormat != 0)
        return CompressedFormatAvailable(caps, format) ? VriFormatSupport_Texture : VriFormatSupport_None;

    const GLFormat g = ToGLFormatOrNone(format);
    if (g.internalFormat != 0)
    {
        r |= VriFormatSupport_Texture; // sampleable, depth included (shadow maps)
        if (IsGLDepthFormat(g))
        {
            r |= VriFormatSupport_DepthStencil;
        }
        // A float texture is always sampleable, but rendering to one is core
        // only on desktop GL; on ES/WebGL2 it needs an extension, and without
        // it the framebuffer comes out incomplete. The two widths are gated
        // separately: an ES 3.0 context can carry
        // EXT_color_buffer_half_float alone and render RGBA16F but not RGBA32F.
        else if (!IsGLFloatFormat(g) || (g.type == GL_HALF_FLOAT ? caps.colorBufferHalfFloat : caps.colorBufferFloat))
        {
            r |= VriFormatSupport_ColorAttachment;
            // Renderable is not blendable. On ES, EXT_color_buffer_float makes a
            // 32-bit float target renderable but blending into one additionally
            // needs EXT_float_blend, and without it the draw is
            // GL_INVALID_OPERATION. (Half-float blending rides along with the
            // renderability extension; desktop GL has all of it from 3.0.)
            if (g.type != GL_FLOAT || caps.floatBlend)
                r |= VriFormatSupport_Blend;
            // Storage textures ride on image load/store, which is a property of
            // the context rather than the format: desktop GL 4.2+ only, and the
            // binding path does not exist in the ES build at all. On top of that
            // the unit accepts only the formats the spec requires (GL 4.2 table
            // 8.27), so gate on both.
            if (caps.imageLoadStore)
            {
                switch (format)
                {
                    case VriFormat_RGBA32_SFLOAT:
                    case VriFormat_RGBA16_SFLOAT:
                    case VriFormat_R32_SFLOAT:
                    case VriFormat_RGBA8_UNORM:
                        r |= VriFormatSupport_StorageTexture;
                        break;
                    default:
                        break;
                }
            }
        }
    }

    if (HasGLVertexFormat(format))
        r |= VriFormatSupport_VertexBuffer;

    return r;
}
VriResult VRI_CALL GetQueue(VriDevice* device, VriQueueType type, uint32_t, VriQueue** outQueue)
{
    if (type >= VriQueueType_Count)
        return VriResult_InvalidArgument;
    *outQueue = ToHandle(Dev(device)->GetQueue(type));
    return VriResult_Success;
}
// OpenGL has no portable VRAM budget query (only vendor extensions like NVX_gpu_memory_info).
VriResult VRI_CALL GetVideoMemoryInfo(const VriDevice*, VriMemoryLocation, VriVideoMemoryInfo*)
{
    return VriResult_Unsupported;
}

// ---- command allocation / lifecycle --------------------------------
VriResult VRI_CALL CreateCommandAllocator(VriDevice* device, VriQueueType, VriCommandAllocator** out)
{
    *out = ToHandle(new CommandAllocatorGL {Dev(device)});
    return VriResult_Success;
}
void VRI_CALL      ResetCommandAllocator(VriCommandAllocator*) {}
void VRI_CALL      DestroyCommandAllocator(VriCommandAllocator* a) { delete reinterpret_cast<CommandAllocatorGL*>(a); }
VriResult VRI_CALL CreateCommandBuffer(VriCommandAllocator* allocator, VriCommandBuffer** out)
{
    *out = ToHandle(new CommandBufferGL {reinterpret_cast<CommandAllocatorGL*>(allocator)->device, 0, GL_TRIANGLES});
    return VriResult_Success;
}
VriResult VRI_CALL BeginCommandBuffer(VriCommandBuffer* cmd)
{
    CB(cmd)->fbo        = 0;
    CB(cmd)->scissorSet = false;
    return VriResult_Success;
}
VriResult VRI_CALL EndCommandBuffer(VriCommandBuffer*) { return VriResult_Success; }

// ---- resources -----------------------------------------------------
VriResult VRI_CALL CreateBuffer(VriDevice* device, const VriBufferDesc* desc, VriBuffer** out)
{
    if (core::RejectDesc(*Dev(device), core::CheckBufferDesc("CreateBuffer", *desc)))
    {
        *out = nullptr;
        return VriResult_InvalidArgument;
    }
    ClearGLErrors();
    GLuint     id        = 0;
    GLenum     usage     = GL_STATIC_DRAW;
    GLbitfield mapAccess = 0;
    if (desc->memoryLocation == VriMemoryLocation_HostReadback)
    {
        usage     = GL_STREAM_READ;
        mapAccess = GL_MAP_READ_BIT;
    }
    else if (desc->memoryLocation == VriMemoryLocation_HostUpload)
    {
        usage     = GL_STREAM_DRAW;
        mapAccess = GL_MAP_WRITE_BIT;
    }
    // Index buffers must stay element-class on WebGL (see BufferGL::target);
    // everything else uses the neutral GL_COPY_WRITE_BUFFER binding point.
    const GLenum target = (desc->usage & VriBufferUsage_IndexBuffer) ? GL_ELEMENT_ARRAY_BUFFER : GL_COPY_WRITE_BUFFER;
    void*        persistentPtr = nullptr;
#if !defined(VRI_GL_ES_HEADERS)
    if (Dev(device)->Features().dsa) // GL 4.5 DSA (implies 4.4 immutable buffer storage)
    {
        glCreateBuffers(1, &id);
        // Immutable storage. Mappable buffers are persistently + coherently mapped
        // once here (MapBuffer just returns that pointer; UnmapBuffer is a no-op),
        // avoiding per-map driver overhead; device-local buffers get DYNAMIC_STORAGE
        // so glNamedBufferSubData stays legal (copies work on immutable storage too).
        GLbitfield storageFlags =
            mapAccess ? (mapAccess | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT) : GL_DYNAMIC_STORAGE_BIT;
        glNamedBufferStorage(id, static_cast<GLsizeiptr>(desc->size), nullptr, storageFlags);
        if (mapAccess)
            persistentPtr = glMapNamedBufferRange(
                id, 0, static_cast<GLsizeiptr>(desc->size), mapAccess | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
    }
    else
#endif
    {
        glGenBuffers(1, &id);
        glBindBuffer(target, id);
        glBufferData(target, static_cast<GLsizeiptr>(desc->size), nullptr, usage);
        glBindBuffer(target, 0);
    }
    if (CreationFailed(Dev(device), "CreateBuffer", "buffer"))
    {
        glDeleteBuffers(1, &id); // deleting also unmaps a persistent mapping
        *out = nullptr;
        return VriResult_Failure;
    }
    BufferGL* b      = new BufferGL {};
    b->device        = Dev(device);
    b->id            = id;
    b->size          = desc->size;
    b->target        = target;
    b->mapAccess     = mapAccess;
    b->persistentPtr = persistentPtr;
    {
        // OpenGL has no allocation-size query, so this is the requested size. It is the
        // honest figure available here, and the field's contract is what the object holds.
        VriObjectInfo info = MakeObjectInfo(VriObjectType_Buffer);
        info.memoryBytes   = desc->size;
        info.width         = static_cast<uint32_t>(desc->size);
        b->device->Objects().Track(ToHandle(b), info);
    }
    *out = ToHandle(b);
    return VriResult_Success;
}
void VRI_CALL DestroyBuffer(VriBuffer* buffer)
{
    if (!buffer)
        return;
    BufferGL* b = Buf(buffer);
    b->device->Objects().Untrack(buffer);
#if !defined(VRI_GL_ES_HEADERS)
    if (b->persistentPtr)
        glUnmapNamedBuffer(b->id); // release the persistent mapping (glDeleteBuffers would too)
#endif
    glDeleteBuffers(1, &b->id);
    delete b;
}
void* VRI_CALL MapBuffer(VriBuffer* buffer, uint64_t offset, uint64_t size)
{
    BufferGL*        b      = Buf(buffer);
    const GLbitfield access = b->mapAccess ? b->mapAccess : GL_MAP_READ_BIT;
    const uint64_t   len    = size ? size : (b->size - offset);
    if (b->device->IsES())
    {
        // WebGL2 cannot map buffers: stage through a CPU shadow.
        b->shadow    = std::malloc(static_cast<size_t>(len));
        b->mapOffset = offset;
        b->mapLen    = len;
        b->mapWrite  = (access & GL_MAP_WRITE_BIT) != 0;
        if (access & GL_MAP_READ_BIT)
        {
            glBindBuffer(b->target, b->id);
            ReadBufferSubData(b->target, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(len), b->shadow);
            glBindBuffer(b->target, 0);
        }
        return b->shadow;
    }
    if (b->persistentPtr) // GL 4.4: persistently mapped at creation, return that pointer
        return static_cast<char*>(b->persistentPtr) + offset;
#if !defined(VRI_GL_ES_HEADERS)
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
            glBufferSubData(
                b->target, static_cast<GLintptr>(b->mapOffset), static_cast<GLsizeiptr>(b->mapLen), b->shadow);
            glBindBuffer(b->target, 0);
        }
        std::free(b->shadow);
        b->shadow = nullptr;
        return;
    }
#if !defined(VRI_GL_ES_HEADERS)
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
VriResult VRI_CALL MapBufferAsync(VriBuffer*              buffer,
                                  uint64_t                offset,
                                  uint64_t                size,
                                  VriBufferMappedCallback callback,
                                  void*                   userData)
{
    return core::MapBufferNow(MapBuffer, buffer, offset, size, callback, userData);
}
uint64_t VRI_CALL GetBufferDeviceAddress(const VriBuffer*) { return 0; }

// GL 4.1 (macOS) lacks immutable storage (glTexStorage*, GL 4.2+; always present on
// GLES3/WebGL2). Emulate it with mutable glTexImage* over every mip level (NULL data
// just allocates). The texture must already be bound to `target`. is3D covers 2D-array
// (depth = layer count, which does not shrink with mips); isCube allocates all 6 faces.
// Uses only GLES3-safe entry points, so it compiles for the WebGL2 build too (where it
// is never reached - textureStorage is always available there).
void TexImageAllocate(GLenum          target,
                      const GLFormat& gf,
                      GLsizei         mips,
                      GLsizei         w,
                      GLsizei         h,
                      GLsizei         depth,
                      bool            is3D,
                      bool            isCube)
{
    for (GLsizei level = 0; level < mips; ++level)
    {
        const GLsizei lw = (w >> level) > 0 ? (w >> level) : 1;
        const GLsizei lh = (h >> level) > 0 ? (h >> level) : 1;
        GLsizei       ld = depth;    // an array keeps its layer count
        if (target == GL_TEXTURE_3D) // a volume's depth shrinks with each mip
            ld = (depth >> level) > 0 ? (depth >> level) : 1;
        if (is3D)
            glTexImage3D(
                target, level, static_cast<GLint>(gf.internalFormat), lw, lh, ld, 0, gf.format, gf.type, nullptr);
        else if (isCube)
            for (GLenum f = 0; f < 6; ++f)
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f,
                             level,
                             static_cast<GLint>(gf.internalFormat),
                             lw,
                             lh,
                             0,
                             gf.format,
                             gf.type,
                             nullptr);
        else
            glTexImage2D(target, level, static_cast<GLint>(gf.internalFormat), lw, lh, 0, gf.format, gf.type, nullptr);
    }
    glTexParameteri(target, GL_TEXTURE_MAX_LEVEL, mips - 1); // mip-complete (immutable does this implicitly)
}

VriResult VRI_CALL CreateTexture(VriDevice* device, const VriTextureDesc* desc, VriTexture** out)
{
    if (core::RejectDesc(*Dev(device), core::CheckTextureDesc("CreateTexture", *desc)))
    {
        *out = nullptr;
        return VriResult_InvalidArgument;
    }
    const GLCompressedFormat compressed = ToGLCompressedFormat(desc->format);
    if (compressed.internalFormat != 0 && !CompressedFormatAvailable(Dev(device)->Features(), desc->format))
    {
        Dev(device)->ReportError("CreateTexture: this GL context does not support the compressed format");
        *out = nullptr;
        return VriResult_Unsupported;
    }
    // Texture types this backend cannot create as described are refused rather than created as another type.
    // GLES and WebGL have no 1D textures, and desktop GL's are not wired here (texture1DMaxDim is 0).
    if (desc->type == VriTextureType_1D || desc->type == VriTextureType_1DArray)
    {
        Dev(device)->ReportError("CreateTexture: GL has no 1D textures here; use a 2D texture 1 texel high");
        *out = nullptr;
        return VriResult_Unsupported;
    }
    // A multisample array would need GL_TEXTURE_2D_MULTISAMPLE_ARRAY and per-layer attachments, which this backend
    // does not have; GLES and WebGL multisample through renderbuffers, which cannot be arrays.
    if (desc->type == VriTextureType_2DArray && desc->sampleNum > 1)
    {
        Dev(device)->ReportError("CreateTexture: GL has no multisample 2DArray textures here");
        *out = nullptr;
        return VriResult_Unsupported;
    }
    // See GlFeatures::cubeMapArray: desktop GL 4.0+ only; ES shaders are ESSL 300, which cannot sample one.
    if (desc->type == VriTextureType_CubeArray && !Dev(device)->Features().cubeMapArray)
    {
        Dev(device)->ReportError("CreateTexture: GL has cube map arrays on desktop GL 4.0+ only, not on GLES or "
                                 "WebGL");
        *out = nullptr;
        return VriResult_Unsupported;
    }
    // S3TC, RGTC and ETC2 have no 3D textures, and BPTC and ASTC have them only through further extensions.
    if (compressed.internalFormat != 0 && desc->type == VriTextureType_3D)
    {
        Dev(device)->ReportError("CreateTexture: a block-compressed format cannot be a 3D texture on GL");
        *out = nullptr;
        return VriResult_Unsupported;
    }
    // A compressed texture has no pixel format or type: glCompressedTexSubImage* fills it.
    const GLFormat gf =
        compressed.internalFormat != 0 ? GLFormat {compressed.internalFormat, 0, 0, 0} : ToGLFormat(desc->format);
    const GLsizei mips     = static_cast<GLsizei>(desc->mipNum);
    const GLsizei w        = static_cast<GLsizei>(desc->width);
    const GLsizei h        = static_cast<GLsizei>(desc->height);
    const GLsizei layers   = static_cast<GLsizei>(desc->layerNum);
    const GLsizei depth    = static_cast<GLsizei>(desc->depth);
    const bool    ms       = desc->sampleNum > 1;
    const bool    isCube   = desc->type == VriTextureType_Cube;
    const bool    isVolume = !ms && desc->type == VriTextureType_3D;
    // The type picks the target, not the layer count: a 2DArray of one layer is still sampled as an array. A cube
    // map array is stored like a 2D array of layer-faces (layer * 6 + face); glTexStorage3D takes their count.
    const bool cubeArray = desc->type == VriTextureType_CubeArray;
    const bool isArray   = desc->type == VriTextureType_2DArray || cubeArray;

    // Immutable storage: GL 4.2+ and always present on GLES3/WebGL2; false only on macOS GL 4.1.
    const bool texStorage = Dev(device)->Features().textureStorage;
#if !defined(VRI_GL_ES_HEADERS)                   // DSA (4.5) entry points are undeclared in the GLES3/WebGL2 headers
    const bool dsa = Dev(device)->Features().dsa; // GL 4.5: glCreateTextures + glTextureStorage, no bind-to-edit
#endif
    ClearGLErrors();
    GLuint id             = 0;
    GLenum target         = GL_TEXTURE_2D;
    bool   isRenderbuffer = false;
    if (ms)
    {
#if defined(VRI_GL_ES_HEADERS)
        // WebGL2 has no multisample textures - use a multisample renderbuffer
        // (render-only; resolved to a single-sample texture).
        glGenRenderbuffers(1, &id);
        glBindRenderbuffer(GL_RENDERBUFFER, id);
        glRenderbufferStorageMultisample(
            GL_RENDERBUFFER, static_cast<GLsizei>(desc->sampleNum), gf.internalFormat, w, h);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        target         = GL_RENDERBUFFER;
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
            if (texStorage)
                glTexStorage2DMultisample(
                    target, static_cast<GLsizei>(desc->sampleNum), gf.internalFormat, w, h, GL_TRUE);
            else // GL 4.1: glTexImage2DMultisample (3.2 core) instead of immutable storage
                glTexImage2DMultisample(
                    target, static_cast<GLsizei>(desc->sampleNum), gf.internalFormat, w, h, GL_TRUE);
            glBindTexture(target, 0);
        }
#endif
    }
    else if (isCube)
    {
        // Cubemap: glTexStorage2D on GL_TEXTURE_CUBE_MAP allocates all 6 square faces.
        target = GL_TEXTURE_CUBE_MAP;
#if !defined(VRI_GL_ES_HEADERS)
        if (dsa)
        {
            glCreateTextures(target, 1, &id);
            glTextureStorage2D(id, mips, gf.internalFormat, w, h);
        }
        else
#endif
        {
            glGenTextures(1, &id);
            glBindTexture(target, id);
            if (texStorage)
                glTexStorage2D(target, mips, gf.internalFormat, w, h);
            else
                TexImageAllocate(target, gf, mips, w, h, 0, /*is3D*/ false, /*isCube*/ true);
            glBindTexture(target, 0);
        }
    }
    else if (isVolume)
    {
        // 3D texture: glTexStorage3D on GL_TEXTURE_3D, depth = slice count.
        target = GL_TEXTURE_3D;
#if !defined(VRI_GL_ES_HEADERS)
        if (dsa)
        {
            glCreateTextures(target, 1, &id);
            glTextureStorage3D(id, mips, gf.internalFormat, w, h, depth);
        }
        else
#endif
        {
            glGenTextures(1, &id);
            glBindTexture(target, id);
            if (texStorage)
                glTexStorage3D(target, mips, gf.internalFormat, w, h, depth);
            else
                TexImageAllocate(target, gf, mips, w, h, depth, /*is3D*/ true, /*isCube*/ false);
            glBindTexture(target, 0);
        }
    }
    else if (isArray)
    {
        // 2D array texture (GLES3/WebGL2 + desktop) or cube map array: glTexStorage3D, depth = layer count.
        target = cubeArray ? GL_TEXTURE_CUBE_MAP_ARRAY : GL_TEXTURE_2D_ARRAY;
#if !defined(VRI_GL_ES_HEADERS)
        if (dsa)
        {
            glCreateTextures(target, 1, &id);
            glTextureStorage3D(id, mips, gf.internalFormat, w, h, layers);
        }
        else
#endif
        {
            glGenTextures(1, &id);
            glBindTexture(target, id);
            if (texStorage)
                glTexStorage3D(target, mips, gf.internalFormat, w, h, layers);
            else
                TexImageAllocate(target, gf, mips, w, h, layers, /*is3D*/ true, /*isCube*/ false);
            glBindTexture(target, 0);
        }
    }
#if !defined(VRI_GL_ES_HEADERS)
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
        if (texStorage)
            glTexStorage2D(GL_TEXTURE_2D, mips, gf.internalFormat, w, h);
        else
            TexImageAllocate(GL_TEXTURE_2D, gf, mips, w, h, 0, /*is3D*/ false, /*isCube*/ false);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    if (CreationFailed(Dev(device), "CreateTexture", "texture"))
    {
        if (isRenderbuffer)
            glDeleteRenderbuffers(1, &id);
        else
            glDeleteTextures(1, &id);
        *out = nullptr;
        return VriResult_Failure;
    }

    TextureGL* t      = new TextureGL {};
    t->device         = Dev(device);
    t->id             = id;
    t->target         = target;
    t->isRenderbuffer = isRenderbuffer;
    t->glFormat       = gf.format;
    t->glType         = gf.type;
    t->internalFormat = gf.internalFormat;
    t->width          = desc->width;
    t->height         = desc->height;
    t->depth          = isVolume ? static_cast<uint32_t>(depth) : 1u;
    t->mipNum         = static_cast<uint32_t>(mips);
    t->layerNum       = static_cast<uint32_t>(layers);
    t->texelSize      = gf.texelSize;
    t->blockBytes     = compressed.blockBytes;
    {
        // No allocation query in GL: this is the mip-0 footprint plus the 1/3 mip-tail
        // estimate, which is the same approximation the GL driver's own tools report.
        VriObjectInfo info = MakeObjectInfo(VriObjectType_Texture);
        uint64_t      size = static_cast<uint64_t>(t->width) * t->height * t->depth * t->layerNum * gf.texelSize;
        if (t->blockBytes != 0)
            size = (static_cast<uint64_t>(t->width) + 3) / 4 * ((t->height + 3) / 4) * t->depth * t->layerNum *
                   t->blockBytes;
        if (t->mipNum > 1)
            size += size / 3;
        info.memoryBytes = size;
        info.width       = t->width;
        info.height      = t->height;
        info.depth       = t->depth;
        info.mipNum      = t->mipNum;
        info.layerNum    = t->layerNum;
        info.format      = desc->format;
        t->device->Objects().Track(ToHandle(t), info);
    }
    *out = ToHandle(t);
    return VriResult_Success;
}
void VRI_CALL DestroyTexture(VriTexture* texture)
{
    if (!texture)
        return;
    TextureGL* t = Tex(texture);
    t->device->Objects().Untrack(texture);
    t->device->EvictFbosReferencing(t->id); // drop cached render-pass FBOs using it
    if (t->isRenderbuffer)
        glDeleteRenderbuffers(1, &t->id);
    else
        glDeleteTextures(1, &t->id);
    delete t;
}

// ---- explicit memory: unsupported on GL (no exposed memory objects) -
void VRI_CALL GetBufferMemoryDesc(const VriDevice*, const VriBufferDesc*, VriMemoryLocation, VriMemoryDesc* o)
{
    if (o)
        *o = VriMemoryDesc {};
}
void VRI_CALL GetTextureMemoryDesc(const VriDevice*, const VriTextureDesc*, VriMemoryLocation, VriMemoryDesc* o)
{
    if (o)
        *o = VriMemoryDesc {};
}
VriResult VRI_CALL AllocateMemory(VriDevice*, const VriMemoryDesc*, VriMemory**) { return VriResult_Unsupported; }
void VRI_CALL      FreeMemory(VriMemory*) {}
VriResult VRI_CALL BindBufferMemory(VriDevice*, VriBuffer*, VriMemory*, uint64_t) { return VriResult_Unsupported; }
VriResult VRI_CALL BindTextureMemory(VriDevice*, VriTexture*, VriMemory*, uint64_t) { return VriResult_Unsupported; }

// ---- views & samplers ----------------------------------------------
VriResult VRI_CALL CreateBufferView(VriDevice* device, const VriBufferViewDesc* desc, VriDescriptor** out)
{
    DescriptorGL* v = new DescriptorGL {};
    v->kind         = DescriptorGL::Kind::BufferView;
    v->device       = Dev(device);
    v->buffer       = Buf(desc->buffer);
    v->bufferOffset = desc->offset;
    v->bufferRange  = desc->size;
    *out            = ToHandle(v);
    return VriResult_Success;
}
VriResult VRI_CALL CreateTextureView(VriDevice* device, const VriTextureViewDesc* desc, VriDescriptor** out)
{
    DescriptorGL* v = new DescriptorGL {};
    v->kind         = DescriptorGL::Kind::TextureView;
    v->device       = Dev(device);
    v->texture      = reinterpret_cast<const TextureGL*>(desc->texture);
    v->mip          = desc->baseMip;
    v->layer        = desc->baseLayer;
    *out            = ToHandle(v);
    return VriResult_Success;
}
VriResult VRI_CALL CreateSampler(VriDevice* device, const VriSamplerDesc* desc, VriDescriptor** out)
{
    GLuint s = 0;
#if !defined(VRI_GL_ES_HEADERS)
    if (Dev(device)->Features().dsa)
        glCreateSamplers(1, &s); // GL 4.5: pre-initialized sampler object
    else
#endif
        glGenSamplers(1, &s);
    glSamplerParameteri(s, GL_TEXTURE_MAG_FILTER, desc->magFilter == VriFilter_Linear ? GL_LINEAR : GL_NEAREST);
    GLint minFilter;
    if (desc->minFilter == VriFilter_Linear)
    {
        minFilter = desc->mipmapMode == VriMipmapMode_Linear ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR_MIPMAP_NEAREST;
    }
    else
    {
        minFilter = desc->mipmapMode == VriMipmapMode_Linear ? GL_NEAREST_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_NEAREST;
    }
    glSamplerParameteri(s, GL_TEXTURE_MIN_FILTER, minFilter);
    glSamplerParameterf(s, GL_TEXTURE_MIN_LOD, desc->minLod);
    glSamplerParameterf(s, GL_TEXTURE_MAX_LOD, desc->maxLod);
    glSamplerParameteri(s, GL_TEXTURE_WRAP_S, ToGLAddress(desc->addressModeU));
    glSamplerParameteri(s, GL_TEXTURE_WRAP_T, ToGLAddress(desc->addressModeV));
    glSamplerParameteri(s, GL_TEXTURE_WRAP_R, ToGLAddress(desc->addressModeW));
    if (desc->compareEnable) // shadow PCF: sampler2DShadow reads compare(ref, depth)
    {
        glSamplerParameteri(s, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
        glSamplerParameteri(s, GL_TEXTURE_COMPARE_FUNC, ToGLCompareOp(desc->compareOp));
    }
    DescriptorGL* v = new DescriptorGL {};
    v->kind         = DescriptorGL::Kind::Sampler;
    v->device       = Dev(device);
    v->sampler      = s;
    *out            = ToHandle(v);
    return VriResult_Success;
}
void VRI_CALL DestroyDescriptor(VriDescriptor* descriptor)
{
    if (!descriptor)
        return;
    DescriptorGL* v = Desc(descriptor);
    if (v->sampler)
        glDeleteSamplers(1, &v->sampler);
    delete v;
}
