/*
 * vri_sdl3.h - build a VriWindowHandle from an SDL_Window* (SDL 3.x).
 *
 * Optional helper (not pulled in by <vri/vri.h>). Requires SDL3 headers on the
 * include path. Uses the SDL3 window properties API.
 */
#ifndef VRI_INTEGRATION_SDL3_H
#define VRI_INTEGRATION_SDL3_H

#include "../ext/vri_ext_swapchain.h"

#include <SDL3/SDL.h>

#if defined(__cplusplus)
extern "C" {
#endif

static inline VriWindowHandle vriWindowHandleFromSDL3(SDL_Window* window)
{
    VriWindowHandle h;
    h.type = VriWindowSystem_None;

    const SDL_PropertiesID props = SDL_GetWindowProperties(window);

#if defined(SDL_PLATFORM_WIN32) || defined(_WIN32)
    h.type = VriWindowSystem_Win32;
    h.handle.win32.hwnd = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
    h.handle.win32.hinstance = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, NULL);
#elif defined(SDL_PLATFORM_MACOS) || defined(__APPLE__)
    /* The Vulkan (MoltenVK) backend needs a CAMetalLayer for vkCreateMetalSurfaceEXT.
     * Let SDL do the Objective-C: SDL_Metal_CreateView attaches a metal-backed view to
     * the window (even one not created with SDL_WINDOW_METAL) and SDL_Metal_GetLayer
     * returns its CAMetalLayer*. The view lives with the window. */
    h.type = VriWindowSystem_Cocoa;
    /* window (NSWindow*) is the generic handle (used by the GL/NSOpenGL backend); the
     * Vulkan/Metal backend wants a CAMetalLayer, which SDL builds for us via a metal view. */
    h.handle.cocoa.window = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, NULL);
    {
        SDL_MetalView view = SDL_Metal_CreateView(window);
        h.handle.cocoa.layer = view ? SDL_Metal_GetLayer(view) : NULL;
    }
#elif defined(SDL_PLATFORM_ANDROID) || defined(__ANDROID__)
    h.type = VriWindowSystem_Android;
    h.handle.android.window = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, NULL);
#elif defined(__linux__)
    {
        void* x11Display = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL);
        if (x11Display)
        {
            h.type = VriWindowSystem_Xlib;
            h.handle.xlib.display = x11Display;
            h.handle.xlib.window = (uint64_t)SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);
        }
        else
        {
            h.type = VriWindowSystem_Wayland;
            h.handle.wayland.display = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, NULL);
            h.handle.wayland.surface = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, NULL);
        }
    }
#else
    (void)props;
    (void)window;
#endif

    return h;
}

#if defined(__cplusplus)
}
#endif

#endif /* VRI_INTEGRATION_SDL3_H */
