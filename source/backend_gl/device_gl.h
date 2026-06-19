// device_gl.h - the OpenGL VriDevice implementation + factory.
//
// The GL "device" owns a GL context. For headless use it creates a hidden GLFW
// window+context; rendering targets FBOs. (Windowed presentation lands later.)
#pragma once

#include "gl_loader.h"

#include <vri/vri.h>

#include "core/device_base.h"
#include "objects_gl.h"

struct GLFWwindow;

namespace vri::gl
{
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
        void                 ReportError(const char* message) const;

    private:
        void FillDeviceDesc();
        void FillRegistry();

        GLFWwindow*          m_window = nullptr; // hidden context-owning window
        GLuint              m_vao = 0;           // default VAO (GL core requires one bound)
        bool                m_es = false;        // OpenGL ES / WebGL profile
        uint32_t            m_shaderVersion = 420; // GLSL #version for SPIRV-Cross
        VriGraphicsAPI      m_api = VriGraphicsAPI_OpenGL;
        QueueGL             m_queue = {};
        VriDeviceDesc       m_desc = {};
        VriCallbackInterface m_callback = {};
    };

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult);
} // namespace vri::gl
