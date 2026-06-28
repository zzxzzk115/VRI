// pipeline_cache_gl.cpp - VriPipelineCacheInterface for OpenGL, emulated with program binaries.
// The cache is a key -> program-binary map (objects_gl.h PipelineCacheGL); core_gl.cpp populates and
// reads it during pipeline creation (glGetProgramBinary / glProgramBinary). This file only manages
// the object and (de)serializes the map. Serialized blobs carry a vendor/renderer/version tag so a
// foreign blob is rejected and the cache starts empty; a per-entry binary that the driver later
// can't ingest is caught by the link-status check in core_gl and quietly rebuilt.

#include "pipeline_cache_gl.h"

#include "device_gl.h"
#include "objects_gl.h"

#include <cstring>

namespace vri::gl
{
    namespace
    {
        inline DeviceGL* Dev(VriDevice* h) { return reinterpret_cast<DeviceGL*>(h); }

        constexpr uint32_t kMagic = 0x4c475256u; // "VRGL"

        // Tag the cache to this GL implementation; program binaries are only portable within it.
        uint64_t DeviceTag()
        {
            uint64_t     h        = 0xcbf29ce484222325ull;
            const GLenum names[3] = {GL_VENDOR, GL_RENDERER, GL_VERSION};
            for (GLenum e : names)
            {
                const GLubyte* s = glGetString(e);
                if (s)
                    h = Fnv1a(s, std::strlen(reinterpret_cast<const char*>(s)), h);
            }
            return h;
        }

        template<typename T>
        void Put(std::vector<uint8_t>& v, const T& x)
        {
            const auto* p = reinterpret_cast<const uint8_t*>(&x);
            v.insert(v.end(), p, p + sizeof(T));
        }
        template<typename T>
        bool Get(const uint8_t*& p, const uint8_t* end, T& out)
        {
            if (p + sizeof(T) > end)
                return false;
            std::memcpy(&out, p, sizeof(T));
            p += sizeof(T);
            return true;
        }

        VriResult VRI_CALL CreatePipelineCache(VriDevice*         device,
                                               const void*        initialData,
                                               size_t             initialSize,
                                               VriPipelineCache** out)
        {
            if (!out)
                return VriResult_InvalidArgument;
            PipelineCacheGL* c = new PipelineCacheGL {};
            c->device          = Dev(device);
            // Seed from a previous blob, but only if it came from this same GL implementation.
            if (initialData && initialSize >= sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t))
            {
                const uint8_t* p     = static_cast<const uint8_t*>(initialData);
                const uint8_t* end   = p + initialSize;
                uint32_t       magic = 0, count = 0;
                uint64_t       tag = 0;
                if (Get(p, end, magic) && magic == kMagic && Get(p, end, tag) && tag == DeviceTag() &&
                    Get(p, end, count))
                {
                    for (uint32_t i = 0; i < count; ++i)
                    {
                        uint64_t key = 0;
                        uint32_t fmt = 0, sz = 0;
                        if (!Get(p, end, key) || !Get(p, end, fmt) || !Get(p, end, sz) || p + sz > end)
                            break;
                        ProgramBinaryGL pb;
                        pb.format = fmt;
                        pb.data.assign(p, p + sz);
                        p += sz;
                        c->entries[key] = std::move(pb);
                    }
                }
                // A foreign / corrupt blob just yields an empty cache - not an error.
            }
            *out = ToHandle(c);
            return VriResult_Success;
        }

        void VRI_CALL DestroyPipelineCache(VriPipelineCache* h)
        {
            if (h)
                delete PipeCacheGL(h);
        }

        VriResult VRI_CALL GetPipelineCacheData(VriPipelineCache* h, void* data, size_t* size)
        {
            if (!h || !size)
                return VriResult_InvalidArgument;
            PipelineCacheGL*     c = PipeCacheGL(h);
            std::vector<uint8_t> blob;
            Put(blob, kMagic);
            Put(blob, DeviceTag());
            Put(blob, static_cast<uint32_t>(c->entries.size()));
            for (const auto& kv : c->entries)
            {
                Put(blob, kv.first);
                Put(blob, static_cast<uint32_t>(kv.second.format));
                Put(blob, static_cast<uint32_t>(kv.second.data.size()));
                blob.insert(blob.end(), kv.second.data.begin(), kv.second.data.end());
            }
            if (!data)
            {
                *size = blob.size();
                return VriResult_Success;
            }
            if (*size < blob.size())
                return VriResult_InvalidArgument;
            std::memcpy(data, blob.data(), blob.size());
            *size = blob.size();
            return VriResult_Success;
        }

        const VriPipelineCacheInterface g_pipelineCacheGL = {
            CreatePipelineCache,
            DestroyPipelineCache,
            GetPipelineCacheData,
        };
    } // namespace

    const VriPipelineCacheInterface* GetPipelineCacheInterfaceGL() { return &g_pipelineCacheGL; }
} // namespace vri::gl
