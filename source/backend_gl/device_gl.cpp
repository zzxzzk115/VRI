#include "device_gl.h"
#include "core_gl.h"
#include "swapchain_gl.h"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstring>

namespace vri::gl
{
    DeviceGL::~DeviceGL()
    {
        if (m_window)
        {
            for (auto& [key, fbo] : m_fboCache)
                glDeleteFramebuffers(1, &fbo);
            m_fboCache.clear();
            glfwMakeContextCurrent(nullptr);
            glfwDestroyWindow(m_window);
        }
        // Intentionally not calling glfwTerminate() (another device may exist).
    }

    GLuint DeviceGL::AcquireFbo(const FboKey& key, bool& isNew)
    {
        auto it = m_fboCache.find(key);
        if (it != m_fboCache.end())
        {
            isNew = false;
            return it->second;
        }
        GLuint fbo = 0;
#if !defined(__EMSCRIPTEN__)
        // DSA (4.5) requires glCreateFramebuffers so the object exists before any
        // glNamedFramebuffer* call; the classic path uses gen + bind-to-configure.
        // (glCreateFramebuffers isn't declared in the Emscripten GLES3 headers.)
        if (m_features.dsa) glCreateFramebuffers(1, &fbo);
        else
#endif
        glGenFramebuffers(1, &fbo);
        m_fboCache.emplace(key, fbo);
        isNew = true;
        return fbo;
    }

    void DeviceGL::EvictFbosReferencing(GLuint glId)
    {
        for (auto it = m_fboCache.begin(); it != m_fboCache.end();)
        {
            bool refs = it->first.depthId == glId;
            for (uint32_t i = 0; i < it->first.colorCount && !refs; ++i)
                refs = it->first.colorId[i] == glId;
            if (refs)
            {
                glDeleteFramebuffers(1, &it->second);
                it = m_fboCache.erase(it);
            }
            else
                ++it;
        }
    }

    void DeviceGL::ReportError(const char* message) const { Diagnostic(VriMessageSeverity_Error, message); }

    void DeviceGL::Diagnostic(VriMessageSeverity severity, const char* message) const
    {
        if (m_callback.MessageCallback)
            m_callback.MessageCallback(m_callback.userArg, severity, message);
        else
            std::fprintf(stderr, "[VRI/GL] %s\n", message);
    }

#if !defined(__EMSCRIPTEN__)
    // KHR_debug message pump (desktop GL 4.3+): forward the driver's diagnostics to the app
    // callback. WebGL2/ES has no glDebugMessageCallback, so this is desktop-only.
    namespace
    {
        void APIENTRY GLDebugCallback(GLenum /*source*/, GLenum type, GLuint /*id*/, GLenum severity,
                                      GLsizei /*length*/, const GLchar* message, const void* userParam)
        {
            if (severity == GL_DEBUG_SEVERITY_NOTIFICATION)
                return; // too chatty (buffer-mapped hints etc.)
            const auto* dev = static_cast<const DeviceGL*>(userParam);
            if (!dev)
                return;
            const VriMessageSeverity s = (severity == GL_DEBUG_SEVERITY_HIGH ||
                                          type == GL_DEBUG_TYPE_ERROR || type == GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR)
                                             ? VriMessageSeverity_Error
                                             : VriMessageSeverity_Warning;
            dev->Diagnostic(s, message ? message : "<null>");
        }
    } // namespace
#endif

    VriResult DeviceGL::Init(const VriDeviceCreationDesc& desc)
    {
        if (desc.callbackInterface)
            m_callback = *desc.callbackInterface;

        m_api = desc.graphicsAPI;
#if defined(__EMSCRIPTEN__)
        // In the browser the only GL is WebGL2 (== GLES3). Honor a VriGraphicsAPI_OpenGL
        // request (e.g. ?backend=webgl) as ES too, otherwise the desktop-GL paths run
        // against WebGL2 and abort: real buffer mapping ("glMapBufferRange access must
        // include INVALIDATE_*" -> "buffer was never mapped") and ESSL 430.
        m_es = true;
#else
        m_es = (desc.graphicsAPI == VriGraphicsAPI_OpenGLES);
#endif
        // The command path is the GLES3/WebGL2-compatible (non-DSA) subset, which
        // also runs on desktop GL. We still create a 4.x core context on desktop
        // so debug tooling and future fast-paths are available. ESSL 300 (ES 3.0)
        // is the WebGL2 baseline; ES 3.1 / ESSL 310 isn't available in WebGL2.
        // Desktop uses GLSL 430 so compute shaders (SSBO + local_size) transpile.
        m_shaderVersion = m_es ? 300u : 430u;

        if (!glfwInit())
        {
            ReportError("glfwInit failed");
            return VriResult_Failure;
        }

        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
#if defined(__EMSCRIPTEN__)
        // Opaque canvas: a swapchain backbuffer is opaque, but a WebGL2 context defaults to
        // an alpha canvas, so low fragment alpha would make the page show through (the cube
        // looks transparent / "empty"). 0 alpha bits => the browser composites it opaque.
        glfwWindowHint(GLFW_ALPHA_BITS, 0);
#endif
        if (m_es)
        {
            glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0); // WebGL2 == ES 3.0
        }
        else
        {
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        }

        m_window = glfwCreateWindow(1, 1, "vri-gl", nullptr, nullptr);
        if (!m_window)
        {
            ReportError(m_es ? "glfwCreateWindow (GLES 3.1) failed" : "glfwCreateWindow (GL 4.6 core) failed");
            return VriResult_Failure;
        }
        glfwMakeContextCurrent(m_window);

#if !defined(__EMSCRIPTEN__)
        // On Emscripten the GLES3/WebGL2 symbols are provided directly; only native
        // platforms need a runtime loader.
        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        {
            ReportError("gladLoadGLLoader failed");
            return VriResult_Failure;
        }
#endif

#if !defined(__EMSCRIPTEN__)
        // KHR_debug (GL 4.3+): forward driver diagnostics to the app callback. Synchronous so
        // the message arrives at the offending call. WebGL2/ES has no glDebugMessageCallback.
        if (desc.enableValidation && glDebugMessageCallback)
        {
            glEnable(GL_DEBUG_OUTPUT);
            glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            glDebugMessageCallback(GLDebugCallback, this);
        }
#endif

        // GL core (and GLES) require a VAO bound for any draw. Non-DSA: gen+bind.
        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);

        m_queue.device = this;
        FillDeviceDesc();   // queries the GL version
        DetectFeatures();   // derives the desktop-GL capability tiers from it

        // Coordinate system: on GL 4.5+ use the native VRI convention via glClipControl
        // (top-left origin + depth [0,1], matching Vulkan/WebGPU) - no per-vertex flip,
        // and depth is exact rather than the [-1,1]->[0,1] compression. ES/WebGL2 and
        // pre-4.5 desktop have no glClipControl, so they flip clip-space Y in-shader
        // (SPIRV-Cross flip_vert_y); both conventions read back top-left identically.
#if !defined(__EMSCRIPTEN__)
        if (m_features.clipControl)
            glClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);
#endif

        FillRegistry();
        return VriResult_Success;
    }

    void DeviceGL::FillDeviceDesc()
    {
        m_desc = {};
        m_desc.graphicsAPI = m_api;

        const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        if (renderer)
            std::strncpy(m_desc.adapter.name, renderer, sizeof(m_desc.adapter.name) - 1);
        m_desc.adapter.type = VriAdapterType_Unknown;

        // GL_MAJOR/MINOR_VERSION are core in GL 3.0+ and GLES 3.0+ (and avoid the
        // glad-specific GLVersion global, which is absent on Emscripten).
        GLint v = 0;
        glGetIntegerv(GL_MAJOR_VERSION, &v);
        m_desc.apiVersionMajor = static_cast<uint32_t>(v);
        glGetIntegerv(GL_MINOR_VERSION, &v);
        m_desc.apiVersionMinor = static_cast<uint32_t>(v);

        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &v);
        m_desc.texture2DMaxDim = static_cast<uint32_t>(v);
        glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &v);
        m_desc.texture3DMaxDim = static_cast<uint32_t>(v);
        glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &v);
        m_desc.textureArrayLayerMaxNum = static_cast<uint32_t>(v);
        glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &v);
        m_desc.attachmentColorMaxNum = static_cast<uint32_t>(v);
        glGetIntegerv(GL_MAX_VIEWPORT_DIMS, &v);
        m_desc.attachmentMaxDim = static_cast<uint32_t>(v);
        m_desc.viewportMaxNum = 1;
        for (int t = 0; t < VriQueueType_Count; ++t)
            m_desc.queueCount[t] = 1;
        // Compute needs desktop GL 4.3+ or GLES 3.1+. WebGL2 (== ES 3.0) has no
        // compute, so report it honestly (the validation layer + CreateComputePipeline
        // gate on this).
        const uint32_t major = m_desc.apiVersionMajor, minor = m_desc.apiVersionMinor;
        const bool computeOk = m_es ? (major > 3 || (major == 3 && minor >= 1))
                                     : (major > 4 || (major == 4 && minor >= 3));
        m_desc.hasComputeShader = computeOk ? VRI_TRUE : VRI_FALSE;
        m_desc.hasGeometryShader = m_es ? VRI_FALSE : VRI_TRUE; // GLES has no geometry shaders
        m_desc.hasTessellation = m_es ? VRI_FALSE : VRI_TRUE;   // desktop GL only (GLES3/WebGL2 has none)
    }

    void DeviceGL::DetectFeatures()
    {
        GlFeatures f{};
        f.major = m_desc.apiVersionMajor;
        f.minor = m_desc.apiVersionMinor;
        const auto atLeast = [&](uint32_t mj, uint32_t mn) { return f.major > mj || (f.major == mj && f.minor >= mn); };
        // Desktop GL only: pick the highest path each version unlocks. ES/WebGL2 keep
        // every flag false (the LCD paths). Version gating is sufficient on the core
        // context we request; extension scanning can refine this later if needed.
        if (!m_es)
        {
            f.baseInstance   = atLeast(4, 2);
            f.separateAttrib = atLeast(4, 3);
            f.drawIndirect   = atLeast(4, 3);
            f.bufferStorage  = atLeast(4, 4);
            f.clipControl    = atLeast(4, 5);
            f.dsa            = atLeast(4, 5);
            f.spirvIngest    = atLeast(4, 6); // ARB_gl_spirv is core in 4.6
        }
        m_features = f;
    }

    void DeviceGL::FillRegistry()
    {
        m_registry.Register(VRI_INTERFACE_CORE, GetCoreInterfaceGL(), sizeof(VriCoreInterface));
        // Presentation: Win32 WGL (desktop) + Web canvas (Emscripten). Other desktop
        // platforms register the interface but report Unsupported at CreateSwapChain.
        m_registry.Register(VRI_INTERFACE_SWAPCHAIN, GetSwapChainInterfaceGL(), sizeof(VriSwapChainInterface));
    }

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult)
    {
        DeviceGL* device = new DeviceGL();
        outResult = device->Init(desc);
        if (outResult != VriResult_Success)
        {
            delete device;
            return nullptr;
        }
        return device;
    }
} // namespace vri::gl
