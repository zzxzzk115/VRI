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
#include "../vri_format.h"
#include "../vri_handles.h"

VRI_EXTERN_C_BEGIN

/* Which native windowing system `VriWindowHandle` carries. */
typedef enum VriWindowSystem
{
    VriWindowSystem_None = 0,
    VriWindowSystem_Win32,   /* HWND + HINSTANCE */
    VriWindowSystem_Xlib,    /* Display* + Window */
    VriWindowSystem_Wayland, /* wl_display* + wl_surface* */
    VriWindowSystem_Cocoa,   /* CAMetalLayer* (or NSView*) */
    VriWindowSystem_Android, /* ANativeWindow* */
    VriWindowSystem_Web,     /* Emscripten canvas selector, e.g. "#canvas" */
    VriWindowSystem_MaxEnum = 0x7fffffff
} VriWindowSystem;

typedef struct VriWindowHandle
{
    VriWindowSystem type;
    union
    {
        struct
        {
            void* hwnd;
            void* hinstance;
        } win32;
        struct
        {
            void*    display;
            uint64_t window;
        } xlib;
        struct
        {
            void* display;
            void* surface;
        } wayland;
        struct
        {
            void* layer;
            void* window;
        } cocoa; /* layer = CAMetalLayer (Vulkan/Metal); window = NSWindow or NSView (OpenGL) */
        struct
        {
            void* window;
        } android;
        struct
        {
            const char* selector;
        } web;
    } handle;
} VriWindowHandle;

/* Presentation pacing. The backend maps each to a native present mode; if the requested mode is
 * not supported it falls back to a guaranteed mode (Fifo / vsync) and emits a diagnostic warning. */
typedef enum VriPresentMode
{
    VriPresentMode_Fifo = 0,    /* vsync, no tearing, capped to the display refresh (always supported) */
    VriPresentMode_FifoRelaxed, /* vsync, but tears instead of stalling when a frame is late (adaptive) */
    VriPresentMode_Mailbox,     /* no tearing, low latency; renders uncapped, latest frame shown each refresh */
    VriPresentMode_Immediate,   /* no vsync: tears, lowest latency, uncapped */
} VriPresentMode;

typedef struct VriSwapChainDesc
{
    VriWindowHandle window;
    VriQueue*       queue;  /* presentation queue */
    VriFormat       format; /* desired backbuffer format */
    uint32_t        width;
    uint32_t        height;
    uint32_t        textureNum;  /* desired image count (e.g. 2-3) */
    VriPresentMode  presentMode; /* default 0 == Fifo (vsync) */
} VriSwapChainDesc;

typedef struct VriSwapChainInterface
{
    VriResult(VRI_CALL* CreateSwapChain)(VriDevice* device, const VriSwapChainDesc* desc, VriSwapChain** outSwapChain);
    void(VRI_CALL* DestroySwapChain)(VriSwapChain* swapChain);

    /* Borrowed array of backbuffer textures (owned by the swapchain). */
    VriResult(VRI_CALL* GetSwapChainTextures)(VriSwapChain* swapChain, VriTexture** outTextures, uint32_t* ioCount);

    /* Acquire the next image; signals `acquireFence` to `signalValue` when ready. */
    VriResult(VRI_CALL* AcquireNextTexture)(VriSwapChain* swapChain,
                                            VriFence*     acquireFence,
                                            uint64_t      signalValue,
                                            uint32_t*     outIndex);
    VriResult(VRI_CALL* Present)(VriSwapChain* swapChain, VriFence* waitFence, uint64_t waitValue);
    VriResult(VRI_CALL* Resize)(VriSwapChain* swapChain, uint32_t width, uint32_t height);
} VriSwapChainInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_SWAPCHAIN_H */
