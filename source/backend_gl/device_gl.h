// device_gl.h - the OpenGL VriDevice implementation + factory.
//
// The GL "device" owns a GL context. For headless use it creates a hidden GLFW
// window+context; rendering targets FBOs. (Windowed presentation lands later.)
#pragma once

#include "gl_loader.h"

#include <vri/vri.h>

#include <cstdint>
#include <unordered_map>

#include "core/device_base.h"
#include "objects_gl.h"

struct GLFWwindow;

namespace vri::gl
{
    // Render-pass FBO identity: the set of color (+ depth) attachments. Two passes with
    // the same attachments share one fully-configured FBO (cached on the device), so we
    // stop gen/deleting an FBO every CmdBeginRendering. Configuration depends only on
    // which texture/renderbuffer + mip is attached where (load/clear ops are per-pass).
    struct FboKey
    {
        static constexpr int kMaxColor = 8;
        uint32_t colorId[kMaxColor] = {};
        uint32_t colorMip[kMaxColor] = {};
        uint32_t colorCount = 0;
        uint32_t depthId = 0;
        uint32_t depthMip = 0;
        bool operator==(const FboKey& o) const
        {
            if (colorCount != o.colorCount || depthId != o.depthId || depthMip != o.depthMip) return false;
            for (uint32_t i = 0; i < colorCount; ++i)
                if (colorId[i] != o.colorId[i] || colorMip[i] != o.colorMip[i]) return false;
            return true;
        }
    };
    struct FboKeyHash
    {
        size_t operator()(const FboKey& k) const
        {
            size_t h = 1469598103934665603ull;
            auto mix = [&](uint32_t v) { h = (h ^ v) * 1099511628211ull; };
            mix(k.colorCount); mix(k.depthId); mix(k.depthMip);
            for (uint32_t i = 0; i < k.colorCount; ++i) { mix(k.colorId[i]); mix(k.colorMip[i]); }
            return h;
        }
    };

    // Detected desktop-GL capability tiers. The backend uses the highest path each
    // context supports (never a blanket GLES/WebGL2 downgrade); ES/WebGL2 keep the
    // LCD paths (all flags false). See docs/gl-backend-audit.md.
    struct GlFeatures
    {
        uint32_t major = 0, minor = 0;
        bool     clipControl    = false; // GL 4.5: glClipControl -> native top-left + depth [0,1], no in-shader flip
        bool     dsa            = false; // GL 4.5: direct state access
        bool     spirvIngest    = false; // GL 4.6: ARB_gl_spirv (glShaderBinary + glSpecializeShader)
        bool     separateAttrib = false; // GL 4.3: glVertexAttribFormat / glBindVertexBuffer
        bool     baseInstance   = false; // GL 4.2: *BaseInstance draws
        bool     drawIndirect   = false; // GL 4.3: multi-draw-indirect
        bool     bufferStorage  = false; // GL 4.4: persistent-mapped buffers
        // Immutable texture storage (glTexStorage*) is GL 4.2+ AND core in GLES3/WebGL2;
        // the lone gap is desktop GL 4.1 (macOS), which falls back to glTexImage*.
        bool     textureStorage = true;  // default true: ES/WebGL2 baseline always has it
    };

    class DeviceGL final : public core::DeviceBase
    {
    public:
        ~DeviceGL() override;

        VriResult Init(const VriDeviceCreationDesc& desc);

        QueueGL*             GetQueue(VriQueueType /*type*/) { return &m_queue; }
        const VriDeviceDesc& Desc() const { return m_desc; }
        GLuint               DefaultVao() const { return m_vao; }
        // Shader transpile target: GLSL version + ES profile. ES (and WebGL) lack
        // glClipControl/DSA, so the same non-DSA command path serves both; the Y
        // flip is done in-shader via SPIRV-Cross flip_vert_y (see core_gl.cpp).
        bool                 IsES() const { return m_es; }
        uint32_t             ShaderVersion() const { return m_shaderVersion; }
        const GlFeatures&    Features() const { return m_features; }
        GLFWwindow*          Window() const { return m_window; } // hidden context-owning window (macOS present)
        void                 ReportError(const char* message) const;
        // Route a backend diagnostic (GL KHR_debug message) to the app's callback.
        void                 Diagnostic(VriMessageSeverity severity, const char* message) const;

        // Render-pass FBO cache. AcquireFbo returns a cached FBO for the attachment set
        // (isNew=true means the caller must configure its attachments once); textures
        // evict the FBOs that reference them on destroy.
        GLuint AcquireFbo(const FboKey& key, bool& isNew);
        void   EvictFbosReferencing(GLuint glId);

    private:
        void FillDeviceDesc();
        void DetectFeatures();
        void FillRegistry();

        GLFWwindow*          m_window = nullptr; // hidden context-owning window
        GLuint              m_vao = 0;           // default VAO (GL core requires one bound)
        bool                m_es = false;        // OpenGL ES / WebGL profile
        uint32_t            m_shaderVersion = 420; // GLSL #version for SPIRV-Cross
        GlFeatures          m_features = {};       // detected desktop-GL capability tiers
        std::unordered_map<FboKey, GLuint, FboKeyHash> m_fboCache; // render-pass FBOs by attachment set
        VriGraphicsAPI      m_api = VriGraphicsAPI_OpenGL;
        QueueGL             m_queue = {};
        VriDeviceDesc       m_desc = {};
        VriCallbackInterface m_callback = {};
    };

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult);
} // namespace vri::gl
