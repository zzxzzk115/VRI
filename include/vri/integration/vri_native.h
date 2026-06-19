/*
 * vri_native.h - construct a VriWindowHandle directly from raw native handles.
 *
 * Optional helper (not pulled in by <vri/vri.h>). Use when you manage windows
 * yourself; for GLFW/SDL see vri_glfw.h / vri_sdl2.h / vri_sdl3.h.
 */
#ifndef VRI_INTEGRATION_NATIVE_H
#define VRI_INTEGRATION_NATIVE_H

#include "../ext/vri_ext_swapchain.h"

#if defined(__cplusplus)
extern "C" {
#endif

static inline VriWindowHandle vriWindowHandleWin32(void* hwnd, void* hinstance)
{
    VriWindowHandle h;
    h.type = VriWindowSystem_Win32;
    h.handle.win32.hwnd = hwnd;
    h.handle.win32.hinstance = hinstance;
    return h;
}

static inline VriWindowHandle vriWindowHandleXlib(void* display, uint64_t window)
{
    VriWindowHandle h;
    h.type = VriWindowSystem_Xlib;
    h.handle.xlib.display = display;
    h.handle.xlib.window = window;
    return h;
}

static inline VriWindowHandle vriWindowHandleWayland(void* display, void* surface)
{
    VriWindowHandle h;
    h.type = VriWindowSystem_Wayland;
    h.handle.wayland.display = display;
    h.handle.wayland.surface = surface;
    return h;
}

static inline VriWindowHandle vriWindowHandleCocoa(void* layerOrWindow)
{
    VriWindowHandle h;
    h.type = VriWindowSystem_Cocoa;
    h.handle.cocoa.layer = layerOrWindow;
    return h;
}

static inline VriWindowHandle vriWindowHandleAndroid(void* nativeWindow)
{
    VriWindowHandle h;
    h.type = VriWindowSystem_Android;
    h.handle.android.window = nativeWindow;
    return h;
}

static inline VriWindowHandle vriWindowHandleWeb(const char* canvasSelector)
{
    VriWindowHandle h;
    h.type = VriWindowSystem_Web;
    h.handle.web.selector = canvasSelector;
    return h;
}

#if defined(__cplusplus)
}
#endif

#endif /* VRI_INTEGRATION_NATIVE_H */
