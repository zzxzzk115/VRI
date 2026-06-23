// swapchain_gl.h - OpenGL windowed presentation as the swapchain interface.
//
// GL has no swapchain object: the backbuffers are plain renderable color textures
// (the app renders into them through the normal FBO path), and Present blits the
// acquired one onto the target window's default framebuffer (FBO 0) + SwapBuffers.
// The device's single GL context is retargeted to the window's device context for
// the present (no second context / share lists). Win32 (WGL) is implemented first;
// other platforms report VriResult_Unsupported until their GL context glue lands.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gl_loader.h"

#include <vri/vri.h>
#include <vri/ext/vri_ext_swapchain.h>

#include "objects_gl.h"

namespace vri::gl
{
    class DeviceGL;

    struct SwapChainGL
    {
        DeviceGL*               device = nullptr;
        std::vector<TextureGL*> textures;           // backbuffer color textures (owned)
        uint32_t                width = 0;
        uint32_t                height = 0;
        VriFormat               format = VriFormat_Unknown; // backbuffer format (reused on Resize)
        uint32_t                currentIndex = 0;
        bool                    vsync = true;
        GLuint                  blitFbo = 0;        // read FBO holding the backbuffer for the present blit
#if defined(__EMSCRIPTEN__)
        std::string             canvasSelector = "#canvas"; // sized to the swapchain so the present blit fills it
#elif defined(_WIN32)
        void*                   hwnd = nullptr;     // HWND target window
        void*                   hdc = nullptr;      // HDC for that window (GetDC)
#elif defined(__linux__) && !defined(__EMSCRIPTEN__)
        void*                   xdisplay = nullptr; // X11 Display*
        unsigned long           xwindow = 0;        // X11 Window / GLXDrawable
#elif defined(__APPLE__)
        void*                   nsWindow = nullptr; // NSWindow* target (present retargets the NSOpenGL context to its view)
#endif
    };

    inline VriSwapChain* ToHandle(SwapChainGL* s) { return reinterpret_cast<VriSwapChain*>(s); }

    const VriSwapChainInterface* GetSwapChainInterfaceGL();
} // namespace vri::gl
