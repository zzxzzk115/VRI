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

        // Coordinate system: GLES/WebGL have no glClipControl, so we do NOT use it
        // on any profile. The VRI Y-up convention is honored by flipping clip-space
        // Y in-shader (SPIRV-Cross flip_vert_y), which makes the bottom-left GL
        // framebuffer read back top-left like Vulkan/WebGPU. (Depth stays GL's
        // [-1,1] for now; remapping to [0,1] is a later, depth-only concern.)

        // GL core (and GLES) require a VAO bound for any draw. Non-DSA: gen+bind.
        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);

        m_queue.device = this;
        FillDeviceDesc();
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
        // Desktop GL has tessellation, but VRI's SPIR-V->GLSL transpile of the
        // tessellation-control stage isn't yet driver-robust (strict drivers reject the
        // SPIRV-Cross output), so report it unsupported rather than fail at pipeline
        // creation. Backend wiring (TCS/TES compile, GL_PATCHES, patch vertices) is in
        // place; flip this to !m_es once the transpile is verified across drivers.
        m_desc.hasTessellation = VRI_FALSE;
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
