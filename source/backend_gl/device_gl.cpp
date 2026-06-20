#include "device_gl.h"
#include "core_gl.h"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstring>

namespace vri::gl
{
    DeviceGL::~DeviceGL()
    {
        if (m_window)
        {
            glfwMakeContextCurrent(nullptr);
            glfwDestroyWindow(m_window);
        }
        // Intentionally not calling glfwTerminate() (another device may exist).
    }

    void DeviceGL::ReportError(const char* message) const
    {
        if (m_callback.MessageCallback)
            m_callback.MessageCallback(m_callback.userArg, VriMessageSeverity_Error, message);
        else
            std::fprintf(stderr, "[VRI/GL] %s\n", message);
    }

    VriResult DeviceGL::Init(const VriDeviceCreationDesc& desc)
    {
        if (desc.callbackInterface)
            m_callback = *desc.callbackInterface;

        m_api = desc.graphicsAPI;
        m_es = (desc.graphicsAPI == VriGraphicsAPI_OpenGLES);
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
