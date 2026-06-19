/*
 * vri_ext_swapchain.h - presentation as a queryable extension interface.
 * Query via vriGetInterface(device, VRI_INTERFACE_SWAPCHAIN, ...).
 *
 * The swapchain takes a native window handle (VriWindowHandle). VRI ships
 * optional integration headers under include/vri/integration/ that fill this
 * union from GLFW3 / SDL2 / SDL3 handles; raw native handles can be used too.
 */
#ifndef VRI_EXT_SWAPCHAIN_H
#define VRI_EXT_SWAPCHAIN_H

#include "../vri_base.h"
#include "../vri_handles.h"
#include "../vri_format.h"

VRI_EXTERN_C_BEGIN

/* Which native windowing system `VriWindowHandle` carries. */
typedef enum VriWindowSystem
{
    VriWindowSystem_None = 0,
    VriWindowSystem_Win32,    /* HWND + HINSTANCE */
    VriWindowSystem_Xlib,     /* Display* + Window */
    VriWindowSystem_Wayland,  /* wl_display* + wl_surface* */
    VriWindowSystem_Cocoa,    /* CAMetalLayer* (or NSView*) */
    VriWindowSystem_Android,  /* ANativeWindow* */
    VriWindowSystem_Web,      /* Emscripten canvas selector, e.g. "#canvas" */
    VriWindowSystem_MaxEnum = 0x7fffffff
} VriWindowSystem;

typedef struct VriWindowHandle
{
    VriWindowSystem type;
    union
    {
        struct { void*       hwnd; void* hinstance; }      win32;
        struct { void*       display; uint64_t window; }   xlib;
        struct { void*       display; void* surface; }     wayland;
        struct { void*       layer; }                      cocoa;
        struct { void*       window; }                     android;
        struct { const char* selector; }                  web;
    } handle;
} VriWindowHandle;

typedef struct VriSwapChainDesc
{
    VriWindowHandle window;
    VriQueue*       queue;       /* presentation queue */
    VriFormat       format;      /* desired backbuffer format */
    uint32_t        width;
    uint32_t        height;
    uint32_t        textureNum;  /* desired image count (e.g. 2-3) */
    VriBool         vsync;
} VriSwapChainDesc;

typedef struct VriSwapChainInterface
{
    VriResult (VRI_CALL *CreateSwapChain)(VriDevice* device, const VriSwapChainDesc* desc, VriSwapChain** outSwapChain);
    void      (VRI_CALL *DestroySwapChain)(VriSwapChain* swapChain);

    /* Borrowed array of backbuffer textures (owned by the swapchain). */
    VriResult (VRI_CALL *GetSwapChainTextures)(VriSwapChain* swapChain, VriTexture** outTextures, uint32_t* ioCount);

    /* Acquire the next image; signals `acquireFence` to `signalValue` when ready. */
    VriResult (VRI_CALL *AcquireNextTexture)(VriSwapChain* swapChain, VriFence* acquireFence, uint64_t signalValue, uint32_t* outIndex);
    VriResult (VRI_CALL *Present)(VriSwapChain* swapChain, VriFence* waitFence, uint64_t waitValue);
    VriResult (VRI_CALL *Resize)(VriSwapChain* swapChain, uint32_t width, uint32_t height);
} VriSwapChainInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_SWAPCHAIN_H */
