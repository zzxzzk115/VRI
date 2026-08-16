#include "device_gl.h"
#include "core_gl.h"
#include "pipeline_cache_gl.h"
#include "query_gl.h"
#include "swapchain_gl.h"

#include "core/imgui_vri.h" // backend-agnostic built-in ImGui renderer

#if defined(VRI_GL_EGL)
#include <EGL/egl.h>
#include <EGL/eglext.h>
#else
#include <GLFW/glfw3.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vri::gl
{
    // OVR_multiview entry point (declared in gl_loader.h); loaded from the platform proc loader at
    // device init, null when the extension is absent.
    PFN_FramebufferTextureMultiviewOVR g_FramebufferTextureMultiviewOVR = nullptr;

    DeviceGL::~DeviceGL()
    {
#if defined(VRI_GL_EGL)
        if (m_eglDisplay)
        {
            for (auto& [key, fbo] : m_fboCache) // context still current here
                glDeleteFramebuffers(1, &fbo);
            m_fboCache.clear();
            EGLDisplay dpy = static_cast<EGLDisplay>(m_eglDisplay);
            eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            if (m_eglSurface)
                eglDestroySurface(dpy, static_cast<EGLSurface>(m_eglSurface));
            if (m_eglContext)
                eglDestroyContext(dpy, static_cast<EGLContext>(m_eglContext));
            eglTerminate(dpy);
        }
#else
        if (m_window)
        {
            for (auto& [key, fbo] : m_fboCache)
                glDeleteFramebuffers(1, &fbo);
            m_fboCache.clear();
            glfwMakeContextCurrent(nullptr);
            glfwDestroyWindow(m_window);
        }
        // Intentionally not calling glfwTerminate() (another device may exist).
#endif
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
#if !defined(VRI_GL_ES_HEADERS)
        // DSA (4.5) requires glCreateFramebuffers so the object exists before any
        // glNamedFramebuffer* call; the classic path uses gen + bind-to-configure.
        // (glCreateFramebuffers isn't declared in the Emscripten GLES3 headers.)
        if (m_features.dsa)
            glCreateFramebuffers(1, &fbo);
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
    void DeviceGL::ReportWarning(const char* message) const { Diagnostic(VriMessageSeverity_Warning, message); }

    void DeviceGL::Diagnostic(VriMessageSeverity severity, const char* message) const
    {
        if (m_callback.MessageCallback)
            m_callback.MessageCallback(m_callback.userArg, severity, message);
        else
            std::fprintf(stderr, "[VRI/GL] %s\n", message);
    }

#if !defined(VRI_GL_ES_HEADERS)
    // KHR_debug message pump (desktop GL 4.3+): forward the driver's diagnostics to the app
    // callback. WebGL2/ES has no glDebugMessageCallback, so this is desktop-only.
    namespace
    {
        void APIENTRY GLDebugCallback(GLenum /*source*/,
                                      GLenum type,
                                      GLuint /*id*/,
                                      GLenum severity,
                                      GLsizei /*length*/,
                                      const GLchar* message,
                                      const void*   userParam)
        {
            if (severity == GL_DEBUG_SEVERITY_NOTIFICATION)
                return; // too chatty (buffer-mapped hints etc.)
            // Performance / "other" messages are benign hints (e.g. a DYNAMIC_DRAW buffer moved
            // VIDEO<->HOST, or pixel-transfer sync) that fire every frame for normal patterns like
            // ImGui's per-frame buffers. Surface only actual correctness issues, not perf noise.
            if (type == GL_DEBUG_TYPE_PERFORMANCE || type == GL_DEBUG_TYPE_OTHER || type == GL_DEBUG_TYPE_MARKER ||
                type == GL_DEBUG_TYPE_PUSH_GROUP || type == GL_DEBUG_TYPE_POP_GROUP)
                return;
            const auto* dev = static_cast<const DeviceGL*>(userParam);
            if (!dev)
                return;
            const VriMessageSeverity s = (severity == GL_DEBUG_SEVERITY_HIGH || type == GL_DEBUG_TYPE_ERROR ||
                                          type == GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR) ?
                                             VriMessageSeverity_Error :
                                             VriMessageSeverity_Warning;
            dev->Diagnostic(s, message ? message : "<null>");
        }
    } // namespace
#endif

#if defined(VRI_GL_EGL)
    // EGL_MESA_platform_surfaceless lets us obtain a display with no X11/Wayland/GBM, so
    // the context comes up over a headless SSH session. These tokens live in recent
    // <EGL/eglext.h> / EGL 1.5; declare them if the SDK is older.
#if !defined(EGL_PLATFORM_SURFACELESS_MESA)
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif
#if !defined(EGL_PLATFORM_WAYLAND_EXT)
#define EGL_PLATFORM_WAYLAND_EXT 0x31D8
#endif
#if !defined(EGL_CONTEXT_MAJOR_VERSION)
#define EGL_CONTEXT_MAJOR_VERSION 0x3098
#endif
#if !defined(EGL_CONTEXT_MINOR_VERSION)
#define EGL_CONTEXT_MINOR_VERSION 0x30FB
#endif
#if !defined(EGL_CONTEXT_OPENGL_PROFILE_MASK)
#define EGL_CONTEXT_OPENGL_PROFILE_MASK 0x30FD
#endif
#if !defined(EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT)
#define EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT 0x00000001
#endif

    // On Linux the GL backend gets its context from EGL for BOTH desktop GL and native GLES
    // (no GLFW), so one path serves Wayland and X11 and headless. m_es selects the client API.
    bool DeviceGL::InitEGL(const void* nativeDisplay)
    {
        const bool es = m_es;
        // Display selection, in order of preference:
        //  1. Wayland: the app handed us its wl_display (nativeDisplay). EGL must use the
        //     SAME connection for the context and the window surface, so bind that exact
        //     display via the Wayland platform - this is what makes windowed present work.
        //  2. A running X11 server (DISPLAY): the default display can make window surfaces;
        //     X11 window IDs are valid across connections, so no shared handle is needed.
        //  3. Headless (no display server): the surfaceless platform, rendering into FBOs.
        auto getPlatformDisplay =
            reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
        const bool haveX11          = std::getenv("DISPLAY") != nullptr;
        const bool haveWindowSystem = nativeDisplay != nullptr || haveX11;
        EGLDisplay dpy              = EGL_NO_DISPLAY;
        if (nativeDisplay) // Wayland: bind the app's wl_display
        {
            if (getPlatformDisplay)
                dpy = getPlatformDisplay(EGL_PLATFORM_WAYLAND_EXT, const_cast<void*>(nativeDisplay), nullptr);
            if (dpy == EGL_NO_DISPLAY)
                dpy = eglGetDisplay(reinterpret_cast<EGLNativeDisplayType>(const_cast<void*>(nativeDisplay)));
        }
        if (dpy == EGL_NO_DISPLAY && haveX11)
            dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (dpy == EGL_NO_DISPLAY && getPlatformDisplay) // headless
            dpy = getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
        if (dpy == EGL_NO_DISPLAY)
            dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY); // last-resort implementation default
        if (dpy == EGL_NO_DISPLAY)
        {
            ReportError("EGL: no display available");
            return false;
        }

        EGLint eglMajor = 0, eglMinor = 0;
        if (!eglInitialize(dpy, &eglMajor, &eglMinor))
        {
            ReportError("EGL: eglInitialize failed");
            return false;
        }
        if (!eglBindAPI(es ? EGL_OPENGL_ES_API : EGL_OPENGL_API))
        {
            ReportError("EGL: eglBindAPI failed");
            eglTerminate(dpy);
            return false;
        }

        // A renderable RGBA8 config. We render into VRI's own FBO textures (depth is a separate
        // VRI texture), so the default framebuffer needs no depth/stencil. ES advertises via
        // EGL_OPENGL_ES2_BIT (covers ES3 on EGL <= 1.4); desktop GL via EGL_OPENGL_BIT. Require a
        // pbuffer surface first; if the (surfaceless) display has none, retry without it.
        const EGLint renderableType = es ? EGL_OPENGL_ES2_BIT : EGL_OPENGL_BIT;
        EGLConfig    cfg            = nullptr;
        EGLint       nCfg           = 0;
        auto         chooseConfig   = [&](EGLint surfaceType) {
            const EGLint a[] = {
                EGL_SURFACE_TYPE,
                surfaceType,
                EGL_RENDERABLE_TYPE,
                renderableType,
                EGL_RED_SIZE,
                8,
                EGL_GREEN_SIZE,
                8,
                EGL_BLUE_SIZE,
                8,
                EGL_ALPHA_SIZE,
                8,
                EGL_NONE,
            };
            return eglChooseConfig(dpy, a, &cfg, 1, &nCfg) && nCfg > 0;
        };
        // Window-capable config when presenting; pbuffer otherwise. Fall back to any config.
        const EGLint preferred = haveWindowSystem ? (EGL_WINDOW_BIT | EGL_PBUFFER_BIT) : EGL_PBUFFER_BIT;
        if (!chooseConfig(preferred) && !chooseConfig(EGL_PBUFFER_BIT) && !chooseConfig(EGL_DONT_CARE))
        {
            ReportError("EGL: no matching config");
            eglTerminate(dpy);
            return false;
        }

        EGLContext ctx = EGL_NO_CONTEXT;
        if (es)
        {
            const EGLint ctxAttr[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE}; // highest 3.x the driver grants
            ctx                    = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctxAttr);
        }
        else
        {
            // Desktop GL: request the highest core context the driver grants, descending. Core
            // profile is 3.2+, so the oldest fallback omits it. (Mirrors the desktop GLFW hints.)
            static const struct
            {
                EGLint major, minor;
                bool   core;
            } kVers[] = {
                {4, 6, true},
                {4, 5, true},
                {4, 3, true},
                {4, 1, true},
                {3, 3, true},
                {3, 1, false},
            };
            for (const auto& v : kVers)
            {
                EGLint a[9];
                int    i = 0;
                a[i++]   = EGL_CONTEXT_MAJOR_VERSION;
                a[i++]   = v.major;
                a[i++]   = EGL_CONTEXT_MINOR_VERSION;
                a[i++]   = v.minor;
                if (v.core)
                {
                    a[i++] = EGL_CONTEXT_OPENGL_PROFILE_MASK;
                    a[i++] = EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT;
                }
                a[i++] = EGL_NONE;
                ctx    = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, a);
                if (ctx != EGL_NO_CONTEXT)
                    break;
            }
        }
        if (ctx == EGL_NO_CONTEXT)
        {
            ReportError("EGL: eglCreateContext failed");
            eglTerminate(dpy);
            return false;
        }

        // Render-to-FBO only: try a surfaceless make-current first (EGL_KHR_surfaceless_context),
        // falling back to a 1x1 pbuffer where that extension is absent.
        EGLSurface surf = EGL_NO_SURFACE;
        if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx))
        {
            const EGLint pb[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
            surf              = eglCreatePbufferSurface(dpy, cfg, pb);
            if (surf == EGL_NO_SURFACE || !eglMakeCurrent(dpy, surf, surf, ctx))
            {
                ReportError("EGL: eglMakeCurrent failed (surfaceless and pbuffer)");
                if (surf != EGL_NO_SURFACE)
                    eglDestroySurface(dpy, surf);
                eglDestroyContext(dpy, ctx);
                eglTerminate(dpy);
                return false;
            }
        }

        m_eglDisplay = dpy;
        m_eglContext = ctx;
        m_eglSurface = surf;
        m_eglConfig  = cfg;

#if !defined(VRI_GL_NATIVE_ES)
        // Desktop GL via EGL: the command path uses real GL entry points, so load glad through
        // eglGetProcAddress (EGL_KHR_get_all_proc_addresses returns core functions too). The
        // native-ES build links libGLESv2 directly and needs no loader.
        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(eglGetProcAddress)))
        {
            ReportError("EGL: gladLoadGLLoader failed");
            return false;
        }
        g_FramebufferTextureMultiviewOVR = reinterpret_cast<PFN_FramebufferTextureMultiviewOVR>(
            eglGetProcAddress("glFramebufferTextureMultiviewOVR")); // OVR_multiview (null if absent)
#endif
        return true;
    }
#endif // VRI_GL_EGL

    VriResult DeviceGL::Init(const VriDeviceCreationDesc& desc)
    {
        if (desc.callbackInterface)
            m_callback = *desc.callbackInterface;

        m_api = desc.graphicsAPI;
#if defined(VRI_GL_ES_HEADERS)
        // The only GL here is the ES profile: WebGL2 (== GLES3) in the browser, or native
        // OpenGL ES on the embedded build. Honor a VriGraphicsAPI_OpenGL request as ES too,
        // otherwise the desktop-GL paths run against an ES context and abort (real buffer
        // mapping "glMapBufferRange access must include INVALIDATE_*", ESSL 430, ...).
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

#if defined(VRI_GL_EGL)
        // Linux: bring up the context via EGL (no GLFW) for both desktop GL and native GLES -
        // one path for Wayland, X11, and headless. InitEGL makes it current; the shared code
        // below (VAO, FillDeviceDesc, ...) then runs.
        if (!InitEGL(desc.nativeDisplay))
            return VriResult_Failure;
#else
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
#if defined(__APPLE__)
            // Apple froze OpenGL at 4.1 (core profile), and macOS only grants a core
            // profile >= 3.2 when GLFW_OPENGL_FORWARD_COMPAT is set. DetectFeatures then
            // derives the (lower) capability tiers from the actual 4.1 context.
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#else
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
#endif
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        }

        m_window = glfwCreateWindow(1, 1, "vri-gl", nullptr, nullptr);
        if (!m_window)
        {
            ReportError(m_es ? "glfwCreateWindow (GLES 3.1) failed" : "glfwCreateWindow (GL core) failed");
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
        g_FramebufferTextureMultiviewOVR = reinterpret_cast<PFN_FramebufferTextureMultiviewOVR>(
            glfwGetProcAddress("glFramebufferTextureMultiviewOVR")); // OVR_multiview (null if absent)
#endif
#endif // !VRI_GL_EGL (GLFW context bring-up)

#if !defined(VRI_GL_ES_HEADERS)
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
        FillDeviceDesc(); // queries the GL version
        DetectFeatures(); // derives the desktop-GL capability tiers from it

        // Coordinate system: on GL 4.5+ use the native VRI convention via glClipControl
        // (top-left origin + depth [0,1], matching Vulkan/WebGPU) - no per-vertex flip,
        // and depth is exact rather than the [-1,1]->[0,1] compression. ES/WebGL2 and
        // pre-4.5 desktop have no glClipControl, so they flip clip-space Y in-shader
        // (SPIRV-Cross flip_vert_y); both conventions read back top-left identically.
#if !defined(VRI_GL_ES_HEADERS)
        if (m_features.clipControl)
            glClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS); // filter across cube faces (always on in WebGL2/core ES3)
#endif

        FillRegistry();
        return VriResult_Success;
    }

    void DeviceGL::FillDeviceDesc()
    {
        m_desc = {};
        // Report the resolved profile, not just what was requested: an OpenGL request that
        // ran as ES (native-ES build / WebGL2 / an explicit ES context) reports OpenGLES, so
        // the app can label it honestly ("OpenGL ES" vs "OpenGL" / "WebGL") with the version.
        m_desc.graphicsAPI = m_es ? VriGraphicsAPI_OpenGLES : VriGraphicsAPI_OpenGL;

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

        // Match the GLSL version to the actual context: for GL >= 3.3 the GLSL #version
        // equals the GL version (4.1 -> 410, 4.6 -> 460). The 430 default assumed a >= 4.3
        // context; on macOS GL 4.1 (max GLSL 410) that #version is rejected. SPIR-V-Cross
        // emits binding-less GLSL here (bindings set via the GL API), so 410 is fine for
        // graphics; compute/SSBO (needs 430) stay gated off by hasComputeShader.
        if (!m_es && m_desc.apiVersionMajor >= 3)
            m_shaderVersion = m_desc.apiVersionMajor * 100u + m_desc.apiVersionMinor * 10u;

        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &v);
        m_desc.texture2DMaxDim = static_cast<uint32_t>(v);
        glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &v);
        m_desc.texture3DMaxDim = static_cast<uint32_t>(v);
        glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &v);
        m_desc.textureArrayLayerMaxNum = static_cast<uint32_t>(v);
        glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &v);
        m_desc.attachmentColorMaxNum = static_cast<uint32_t>(v);
        GLint viewportDims[2]        = {0, 0};             // GL_MAX_VIEWPORT_DIMS returns TWO ints (w, h) - a
        glGetIntegerv(GL_MAX_VIEWPORT_DIMS, viewportDims); // single &v would overflow the stack
        m_desc.attachmentMaxDim = static_cast<uint32_t>(viewportDims[0]);
        m_desc.viewportMaxNum   = 1;
        for (int t = 0; t < VriQueueType_Count; ++t)
            m_desc.queueCount[t] = 1;
        // Compute needs desktop GL 4.3+ or GLES 3.1+. WebGL2 (== ES 3.0) has no
        // compute, so report it honestly (the validation layer + CreateComputePipeline
        // gate on this).
        const uint32_t major = m_desc.apiVersionMajor, minor = m_desc.apiVersionMinor;
        bool computeOk = m_es ? (major > 3 || (major == 3 && minor >= 1)) : (major > 4 || (major == 4 && minor >= 3));
#if defined(VRI_GL_NATIVE_ES)
        // The native-ES build currently shares WebGL2's command path, where CmdDispatch is a
        // no-op. Report compute unavailable so it fails loudly rather than silently doing
        // nothing (ES 3.1 compute is a follow-up - flip this on with the dispatch path).
        computeOk = false;
#endif
        m_desc.hasComputeShader  = computeOk ? VRI_TRUE : VRI_FALSE;
        m_desc.hasGeometryShader = m_es ? VRI_FALSE : VRI_TRUE; // GLES has no geometry shaders
        m_desc.hasTessellation   = m_es ? VRI_FALSE : VRI_TRUE; // desktop GL only (GLES3/WebGL2 has none)
        // Desktop GL has native timer queries (ARB_timer_query, core 3.3; ticks are nanoseconds).
        // Require GL 4.4 so CmdCopyQueries can resolve into a buffer via a query buffer object
        // (ARB_query_buffer_object, core 4.4). GLES/WebGL lack a core timer query entirely.
#if !defined(VRI_GL_ES_HEADERS)
        const bool timerOk                = !m_es && (major > 4 || (major == 4 && minor >= 4));
        m_desc.hasTimestampQueries        = timerOk ? VRI_TRUE : VRI_FALSE;
        m_desc.timestampPeriodNanoseconds = timerOk ? 1.0f : 0.0f;
        // Pipeline-statistics queries via ARB_pipeline_statistics_query (core 4.6): 11 GL query objects.
        m_desc.hasPipelineStatistics = (!m_es && (major > 4 || (major == 4 && minor >= 6))) ? VRI_TRUE : VRI_FALSE;
        // GPU-driven draw count via glMultiDraw*IndirectCount (ARB_indirect_parameters, core 4.6).
        m_desc.hasDrawIndirectCount = (!m_es && (major > 4 || (major == 4 && minor >= 6))) ? VRI_TRUE : VRI_FALSE;
        // Storage-buffer clear via glClearBufferSubData (ARB_clear_buffer_object, core 4.3).
        m_desc.hasClearStorageBuffer = (!m_es && (major > 4 || (major == 4 && minor >= 3))) ? VRI_TRUE : VRI_FALSE;
        // Storage-texture clear via glClearTexImage (ARB_clear_texture, core 4.4).
        m_desc.hasClearStorageTexture = (!m_es && (major > 4 || (major == 4 && minor >= 4))) ? VRI_TRUE : VRI_FALSE;
#endif
        // Multiview (single-pass layered rendering) via GL_OVR_multiview2 - available on desktop GL
        // AND GLES (the key path for standalone VR). Needs the loaded entry point + the multiview2
        // extension (gl_ViewID_OVR with view-dependent control flow, which spirv-cross emits) + >=2
        // views. (WebGL2/Emscripten has no proc loader here, so the pointer stays null = unsupported.)
        bool  hasMv2   = false;
        GLint extCount = 0;
        glGetIntegerv(GL_NUM_EXTENSIONS, &extCount);
        for (GLint i = 0; i < extCount; ++i)
        {
            const char* e = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, static_cast<GLuint>(i)));
            if (e && std::strcmp(e, "GL_OVR_multiview2") == 0)
            {
                hasMv2 = true;
                break;
            }
        }
        GLint maxViews = 0;
        if (g_FramebufferTextureMultiviewOVR && hasMv2)
            glGetIntegerv(GL_MAX_VIEWS_OVR, &maxViews);
        const bool multiviewOk = g_FramebufferTextureMultiviewOVR != nullptr && hasMv2 && maxViews >= 2;
        m_desc.hasMultiview    = multiviewOk ? VRI_TRUE : VRI_FALSE;
        m_desc.maxViewCount    = multiviewOk ? static_cast<uint32_t>(maxViews) : 0u;
    }

    namespace
    {
        bool HasGLExtension(const char* name)
        {
            GLint count = 0;
            glGetIntegerv(GL_NUM_EXTENSIONS, &count);
            for (GLint i = 0; i < count; ++i)
            {
                const char* e = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, static_cast<GLuint>(i)));
                if (e && std::strcmp(e, name) == 0)
                    return true;
            }
            return false;
        }
    } // namespace

    void DeviceGL::DetectFeatures()
    {
        GlFeatures f {};
        f.major            = m_desc.apiVersionMajor;
        f.minor            = m_desc.apiVersionMinor;
        const auto atLeast = [&](uint32_t mj, uint32_t mn) { return f.major > mj || (f.major == mj && f.minor >= mn); };
        // Desktop GL only: pick the highest path each version unlocks. ES/WebGL2 keep
        // every flag false (the LCD paths). Version gating is sufficient on the core
        // context we request; extension scanning can refine this later if needed.
        if (!m_es)
        {
            f.baseInstance      = atLeast(4, 2);
            f.separateAttrib    = atLeast(4, 3);
            f.drawIndirect      = atLeast(4, 3);
            f.bufferStorage     = atLeast(4, 4);
            f.clipControl       = atLeast(4, 5);
            f.dsa               = atLeast(4, 5);
            f.spirvIngest       = atLeast(4, 6); // ARB_gl_spirv is core in 4.6
            f.drawIndirectCount = atLeast(4, 6); // ARB_indirect_parameters (glMultiDraw*IndirectCount)
            f.textureStorage    = atLeast(4, 2); // glTexStorage*; macOS GL 4.1 lacks it -> glTexImage*
#if !defined(VRI_GL_ES_HEADERS)
            // Pipeline cache via program binaries (ARB_get_program_binary, core 4.1). Require at
            // least one supported binary format - some drivers advertise the API but zero formats.
            GLint binFormats = 0;
            glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &binFormats);
            f.programBinary = atLeast(4, 1) && binFormats > 0;
#endif
            f.colorBufferFloat = atLeast(3, 0); // core since GL 3.0
            f.imageLoadStore   = atLeast(4, 2); // glBindImageTexture (macOS GL 4.1 misses it)
        }
        else
        {
            // ES/WebGL2: a float texture is sampleable but only renderable with the
            // extension, and the storage-texture binding path is compiled out here.
            f.colorBufferFloat = HasGLExtension("GL_EXT_color_buffer_float");
        }
#if defined(VRI_GL_ES_HEADERS)
        f.imageLoadStore = false; // the binding path does not exist in this build
#endif
        m_features = f;
    }

    void DeviceGL::FillRegistry()
    {
        m_registry.Register(VRI_INTERFACE_CORE, GetCoreInterfaceGL(), sizeof(VriCoreInterface));
        // Presentation: Win32 WGL (desktop) + Web canvas (Emscripten). Other desktop
        // platforms register the interface but report Unsupported at CreateSwapChain.
        m_registry.Register(VRI_INTERFACE_SWAPCHAIN, GetSwapChainInterfaceGL(), sizeof(VriSwapChainInterface));
        m_registry.Register(VRI_INTERFACE_QUERY, GetQueryInterfaceGL(), sizeof(VriQueryInterface));
        m_registry.Register(VRI_INTERFACE_IMGUI, core::GetImguiInterface(), sizeof(VriImguiInterface));
        // Pipeline cache emulated via program binaries; desktop GL only (GLES/WebGL2 expose no
        // program-binary API), so register it only where the cap was detected.
        if (m_features.programBinary)
            m_registry.Register(
                VRI_INTERFACE_PIPELINE_CACHE, GetPipelineCacheInterfaceGL(), sizeof(VriPipelineCacheInterface));
    }

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult)
    {
        DeviceGL* device = new DeviceGL();
        outResult        = device->Init(desc);
        if (outResult != VriResult_Success)
        {
            delete device;
            return nullptr;
        }
        return device;
    }
} // namespace vri::gl
