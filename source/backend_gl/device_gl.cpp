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

        if (!glfwInit())
        {
            ReportError("glfwInit failed");
            return VriResult_Failure;
        }

        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        m_window = glfwCreateWindow(1, 1, "vri-gl", nullptr, nullptr);
        if (!m_window)
        {
            ReportError("glfwCreateWindow (GL 4.6 core) failed");
            return VriResult_Failure;
        }
        glfwMakeContextCurrent(m_window);

        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        {
            ReportError("gladLoadGLLoader failed");
            return VriResult_Failure;
        }

        // Align GL with the VRI standard coordinate system (Y-up clip, depth
        // [0,1], top-left framebuffer origin) -- matches Vulkan/D3D12/WebGPU.
        glClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);

        // GL core requires a VAO bound for any draw (even attribute-less ones).
        glCreateVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);

        m_queue.device = this;
        FillDeviceDesc();
        FillRegistry();
        return VriResult_Success;
    }

    void DeviceGL::FillDeviceDesc()
    {
        m_desc = {};
        m_desc.graphicsAPI = VriGraphicsAPI_OpenGL;

        const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        if (renderer)
            std::strncpy(m_desc.adapter.name, renderer, sizeof(m_desc.adapter.name) - 1);
        m_desc.adapter.type = VriAdapterType_Unknown;

        m_desc.apiVersionMajor = static_cast<uint32_t>(GLVersion.major);
        m_desc.apiVersionMinor = static_cast<uint32_t>(GLVersion.minor);

        GLint v = 0;
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
        m_desc.hasComputeShader = VRI_TRUE; // GL 4.3+
        m_desc.hasGeometryShader = VRI_TRUE;
        m_desc.hasTessellation = VRI_TRUE;
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
