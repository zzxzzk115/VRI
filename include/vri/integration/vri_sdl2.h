/*
 * vri_sdl2.h - build a VriWindowHandle from an SDL_Window* (SDL 2.x).
 *
 * Optional helper (not pulled in by <vri/vri.h>). Requires SDL2 headers on the
 * include path. Uses SDL_GetWindowWMInfo / SDL_SysWMinfo.
 */
#ifndef VRI_INTEGRATION_SDL2_H
#define VRI_INTEGRATION_SDL2_H

#include "../ext/vri_ext_swapchain.h"

#include <SDL.h>
#include <SDL_syswm.h>

#if defined(__cplusplus)
extern "C" {
#endif

static inline VriWindowHandle vriWindowHandleFromSDL2(SDL_Window* window)
{
    VriWindowHandle h;
    h.type = VriWindowSystem_None;

    SDL_SysWMinfo info;
    SDL_VERSION(&info.version);
    if (SDL_GetWindowWMInfo(window, &info) != SDL_TRUE)
        return h;

    switch (info.subsystem)
    {
#if defined(SDL_VIDEO_DRIVER_WINDOWS)
        case SDL_SYSWM_WINDOWS:
            h.type = VriWindowSystem_Win32;
            h.handle.win32.hwnd = (void*)info.info.win.window;
            h.handle.win32.hinstance = (void*)info.info.win.hinstance;
            break;
#endif
#if defined(SDL_VIDEO_DRIVER_X11)
        case SDL_SYSWM_X11:
            h.type = VriWindowSystem_Xlib;
            h.handle.xlib.display = (void*)info.info.x11.display;
            h.handle.xlib.window = (uint64_t)info.info.x11.window;
            break;
#endif
#if defined(SDL_VIDEO_DRIVER_WAYLAND)
        case SDL_SYSWM_WAYLAND:
            h.type = VriWindowSystem_Wayland;
            h.handle.wayland.display = (void*)info.info.wl.display;
            h.handle.wayland.surface = (void*)info.info.wl.surface;
            break;
#endif
#if defined(SDL_VIDEO_DRIVER_COCOA)
        case SDL_SYSWM_COCOA:
            h.type = VriWindowSystem_Cocoa;
            h.handle.cocoa.layer = (void*)info.info.cocoa.window;
            break;
#endif
#if defined(SDL_VIDEO_DRIVER_ANDROID)
        case SDL_SYSWM_ANDROID:
            h.type = VriWindowSystem_Android;
            h.handle.android.window = (void*)info.info.android.window;
            break;
#endif
        default:
            break;
    }

    return h;
}

#if defined(__cplusplus)
}
#endif

#endif /* VRI_INTEGRATION_SDL2_H */
