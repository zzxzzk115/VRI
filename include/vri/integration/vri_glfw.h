/*
 * vri_glfw.h - build a VriWindowHandle from a GLFWwindow* (GLFW 3.x).
 *
 * Optional helper (not pulled in by <vri/vri.h>; include it yourself where you
 * already use GLFW). Requires GLFW headers on the include path. On Linux it
 * supports both X11 and Wayland; on GLFW >= 3.4 it picks via glfwGetPlatform(),
 * otherwise it assumes X11 (define VRI_GLFW_WAYLAND to force Wayland).
 */
#ifndef VRI_INTEGRATION_GLFW_H
#define VRI_INTEGRATION_GLFW_H

#include "../ext/vri_ext_swapchain.h"

#if defined(_WIN32)
#    define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#    define GLFW_EXPOSE_NATIVE_COCOA
#elif defined(__linux__)
#    define GLFW_EXPOSE_NATIVE_X11
#    define GLFW_EXPOSE_NATIVE_WAYLAND
#endif

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#if defined(__cplusplus)
extern "C" {
#endif

static inline VriWindowHandle vriWindowHandleFromGLFW(GLFWwindow* window)
{
    VriWindowHandle h;
    h.type = VriWindowSystem_None;

#if defined(_WIN32)
    h.type = VriWindowSystem_Win32;
    h.handle.win32.hwnd = (void*)glfwGetWin32Window(window);
    h.handle.win32.hinstance = (void*)0; /* backend resolves via GetModuleHandle(NULL) */
#elif defined(__APPLE__)
    h.type = VriWindowSystem_Cocoa;
    h.handle.cocoa.layer = (void*)glfwGetCocoaWindow(window); /* NSWindow*; backend derives CAMetalLayer */
#elif defined(__linux__)
    {
#    if defined(GLFW_VERSION_MAJOR) && (GLFW_VERSION_MAJOR > 3 || (GLFW_VERSION_MAJOR == 3 && GLFW_VERSION_MINOR >= 4))
        const int useWayland = (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND);
#    elif defined(VRI_GLFW_WAYLAND)
        const int useWayland = 1;
#    else
        const int useWayland = 0;
#    endif
        if (useWayland)
        {
            h.type = VriWindowSystem_Wayland;
            h.handle.wayland.display = (void*)glfwGetWaylandDisplay();
            h.handle.wayland.surface = (void*)glfwGetWaylandWindow(window);
        }
        else
        {
            h.type = VriWindowSystem_Xlib;
            h.handle.xlib.display = (void*)glfwGetX11Display();
            h.handle.xlib.window = (uint64_t)glfwGetX11Window(window);
        }
    }
#else
    (void)window;
#endif

    return h;
}

#if defined(__cplusplus)
}
#endif

#endif /* VRI_INTEGRATION_GLFW_H */
