// device_gl.h - the OpenGL VriDevice implementation + factory.
//
// The GL "device" owns a GL context. For headless use it creates a hidden GLFW
// window+context; rendering targets FBOs. (Windowed presentation lands later.)
#pragma once

#include <glad/glad.h>

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
        void                 ReportError(const char* message) const;

    private:
        void FillDeviceDesc();
        void FillRegistry();

        GLFWwindow*          m_window = nullptr; // hidden context-owning window
        GLuint              m_vao = 0;           // default VAO (GL core requires one bound)
        QueueGL             m_queue = {};
        VriDeviceDesc       m_desc = {};
        VriCallbackInterface m_callback = {};
    };

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult);
} // namespace vri::gl
