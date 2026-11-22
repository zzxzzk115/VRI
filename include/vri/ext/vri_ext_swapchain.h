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
    /* A `format` the backend cannot present returns VriResult_Unsupported, creates nothing and logs no
       error, so the caller can try formats in turn; no backend substitutes another format. Vulkan and
       WebGPU take the formats the surface lists (WebGPU compares the channel format and takes sRGB from
       the view). D3D12 takes the flip-model formats RGBA8, BGRA8, RGB10A2 and RGBA16F, sRGB included. GL
       takes RGBA8, RGBA8 sRGB and BGRA8, the formats of its 8-bit window framebuffer. Metal takes the
       CAMetalLayer formats BGRA8, BGRA8 sRGB, RGB10A2 and RGBA16F. */
    VriResult(VRI_CALL* CreateSwapChain)(VriDevice* device, const VriSwapChainDesc* desc, VriSwapChain** outSwapChain);
    /* DestroySwapChain and Resize may follow a Present at once: the backend keeps the swapchain's images
       alive until the GPU, the queued presents included, is done with them, so the caller need not wait for
       anything first. Vulkan and D3D12 wait on the host for the present queue to go idle; native WebGPU
       polls the device until it is idle; GL defers deleting a texture still in use; Metal command buffers
       retain the textures they use; the browser's WebGPU keeps released objects alive until the GPU is done. */
    void(VRI_CALL* DestroySwapChain)(VriSwapChain* swapChain);

    /* Borrowed array of backbuffer textures (owned by the swapchain). */
    VriResult(VRI_CALL* GetSwapChainTextures)(VriSwapChain* swapChain, VriTexture** outTextures, uint32_t* ioCount);

    /* Acquire the next image; signals `acquireFence` to `signalValue` when ready. Known limitation: Vulkan
       instead waits on the host until the image is available and never signals `acquireFence`. */
    VriResult(VRI_CALL* AcquireNextTexture)(VriSwapChain* swapChain,
                                            VriFence*     acquireFence,
                                            uint64_t      signalValue,
                                            uint32_t*     outIndex);
    /* Presents the image acquired last. With `waitFence`, the present waits on the GPU until the fence
       reaches `waitValue`, so the app does not wait for its render work on the host. Vulkan turns the
       wait into a binary semaphore for vkQueuePresentKHR, and Metal encodes an event wait. D3D12, GL and
       WebGPU ignore the fence: D3D12 presents after the work already submitted to the swapchain's queue,
       but not after work on another queue; GL and WebGPU have one queue and present after all work
       already submitted. With a NULL `waitFence`, Vulkan presents without waiting, so wait for the
       render work on the host first. */
    VriResult(VRI_CALL* Present)(VriSwapChain* swapChain, VriFence* waitFence, uint64_t waitValue);
    /* Rebuilds the images at the new size with the creation format. On Vulkan it returns
       VriResult_Unsupported, keeping the old images, if the surface no longer lists that format. */
    VriResult(VRI_CALL* Resize)(VriSwapChain* swapChain, uint32_t width, uint32_t height);

    /* The extent the swapchain was ACTUALLY created with. May differ from the size passed to
       CreateSwapChain/Resize: window systems (Vulkan surface currentExtent) dictate the real
       size, and during rapid resizes the caller's window metrics lag behind. Hosts must size
       render areas/viewports from THIS, not from the requested size - a render area larger
       than the backbuffer is undefined behavior (GPU faults, black window). May be null in
       backends that predate it; fall back to the requested size then. */
    VriResult(VRI_CALL* GetSwapChainExtent)(VriSwapChain* swapChain, uint32_t* outWidth, uint32_t* outHeight);
} VriSwapChainInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_SWAPCHAIN_H */
