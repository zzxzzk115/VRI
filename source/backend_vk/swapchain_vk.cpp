// swapchain_vk.cpp - Vulkan swapchain (presentation) implementation.
//
// Phase-1 swapchain: correct but serialized (the app CPU-waits its render fence
// before Present, so no binary-semaphore frame pipelining yet). A pipelined
// path with per-frame binary acquire/present semaphores is a later refinement.

#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__APPLE__)
#define VK_USE_PLATFORM_METAL_EXT
#endif

#include <vulkan/vulkan.h>

#include "conversions_vk.h"
#include "device_vk.h"
#include "swapchain_vk.h"

#include <cstdio>
#include <vector>

namespace vri::vk
{
    namespace
    {
        inline SwapChainVK* SC(VriSwapChain* h) { return reinterpret_cast<SwapChainVK*>(h); }
        inline DeviceVK*    Dev(VriDevice* h) { return reinterpret_cast<DeviceVK*>(h); }
        inline QueueVK*     Q(VriQueue* h) { return reinterpret_cast<QueueVK*>(h); }

#if !defined(_WIN32) && !defined(__APPLE__)
        // Android / Wayland / Xlib surface creation without pulling <android/native_window.h>,
        // <wayland-client.h> or <X11/Xlib.h> into the build: the create-info structs are declared
        // here with ABI-identical layouts (the real ones differ only in the *names* of the opaque
        // pointer types) and the entry points are resolved through the loader. That keeps the
        // Vulkan backend free of platform-SDK build dependencies, exactly as the Metal path above
        // avoids needing QuartzCore at build time.
        constexpr VkStructureType kXlibSurfaceCreateInfo    = static_cast<VkStructureType>(1000004000);
        constexpr VkStructureType kWaylandSurfaceCreateInfo = static_cast<VkStructureType>(1000006000);
        constexpr VkStructureType kAndroidSurfaceCreateInfo = static_cast<VkStructureType>(1000008000);

        struct XlibSurfaceCreateInfo
        {
            VkStructureType sType;
            const void*     pNext;
            VkFlags         flags;
            void*           dpy;    // Display*
            unsigned long   window; // Window (an XID)
        };
        struct WaylandSurfaceCreateInfo
        {
            VkStructureType sType;
            const void*     pNext;
            VkFlags         flags;
            void*           display; // struct wl_display*
            void*           surface; // struct wl_surface*
        };
        struct AndroidSurfaceCreateInfo
        {
            VkStructureType sType;
            const void*     pNext;
            VkFlags         flags;
            void*           window; // ANativeWindow*
        };

        // One helper for all three: they share the shape "resolve entry point, fill sType, call".
        template<typename CreateInfo>
        VkSurfaceKHR CreateSurfaceFrom(DeviceVK* d, const char* entryPoint, CreateInfo& ci, VkStructureType sType)
        {
            using Pfn = VkResult (*)(VkInstance, const CreateInfo*, const VkAllocationCallbacks*, VkSurfaceKHR*);
            auto fn   = reinterpret_cast<Pfn>(vkGetInstanceProcAddr(d->Instance(), entryPoint));
            if (!fn)
                return VK_NULL_HANDLE; // instance extension not enabled / not supported by the loader
            ci.sType             = sType;
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            if (fn(d->Instance(), &ci, nullptr, &surface) != VK_SUCCESS)
                return VK_NULL_HANDLE;
            return surface;
        }
#endif

        VkSurfaceKHR CreateSurface(DeviceVK* d, const VriWindowHandle& window)
        {
            VkSurfaceKHR surface = VK_NULL_HANDLE;
#if defined(_WIN32)
            if (window.type != VriWindowSystem_Win32)
                return VK_NULL_HANDLE;
            VkWin32SurfaceCreateInfoKHR ci = {VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
            ci.hinstance = window.handle.win32.hinstance ? static_cast<HINSTANCE>(window.handle.win32.hinstance) :
                                                           GetModuleHandleW(nullptr);
            ci.hwnd      = static_cast<HWND>(window.handle.win32.hwnd);
            if (vkCreateWin32SurfaceKHR(d->Instance(), &ci, nullptr, &surface) != VK_SUCCESS)
                return VK_NULL_HANDLE;
#elif defined(__APPLE__)
            // MoltenVK: cocoa.layer is a CAMetalLayer* (the SDL3/GLFW integration headers
            // derive it from the window). vkCreateMetalSurfaceEXT wraps it as a surface.
            if (window.type != VriWindowSystem_Cocoa || window.handle.cocoa.layer == nullptr)
                return VK_NULL_HANDLE;
            VkMetalSurfaceCreateInfoEXT ci = {VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT};
            ci.pLayer                      = static_cast<const CAMetalLayer*>(window.handle.cocoa.layer);
            if (vkCreateMetalSurfaceEXT(d->Instance(), &ci, nullptr, &surface) != VK_SUCCESS)
                return VK_NULL_HANDLE;
#elif defined(__ANDROID__)
            if (window.type != VriWindowSystem_Android || window.handle.android.window == nullptr)
                return VK_NULL_HANDLE;
            AndroidSurfaceCreateInfo aci {};
            aci.window = window.handle.android.window;
            surface    = CreateSurfaceFrom(d, "vkCreateAndroidSurfaceKHR", aci, kAndroidSurfaceCreateInfo);
#else
            // Linux/BSD: whichever window system the app handed us. Both surface extensions are
            // requested at instance creation when the loader reports them, so a Wayland-only or
            // X11-only system still gets the one it has.
            if (window.type == VriWindowSystem_Wayland)
            {
                if (!window.handle.wayland.display || !window.handle.wayland.surface)
                    return VK_NULL_HANDLE;
                WaylandSurfaceCreateInfo wci {};
                wci.display = window.handle.wayland.display;
                wci.surface = window.handle.wayland.surface;
                surface     = CreateSurfaceFrom(d, "vkCreateWaylandSurfaceKHR", wci, kWaylandSurfaceCreateInfo);
            }
            else if (window.type == VriWindowSystem_Xlib)
            {
                if (!window.handle.xlib.display || !window.handle.xlib.window)
                    return VK_NULL_HANDLE;
                XlibSurfaceCreateInfo xci {};
                xci.dpy    = window.handle.xlib.display;
                xci.window = static_cast<unsigned long>(window.handle.xlib.window);
                surface    = CreateSurfaceFrom(d, "vkCreateXlibSurfaceKHR", xci, kXlibSurfaceCreateInfo);
            }
#endif
            return surface;
        }

        void DestroyTextures(SwapChainVK* sc)
        {
            for (TextureVK* t : sc->textures)
                delete t; // images are owned by the swapchain, not freed here
            sc->textures.clear();
        }

        VriResult BuildSwapchain(SwapChainVK*   sc,
                                 uint32_t       width,
                                 uint32_t       height,
                                 VriFormat      format,
                                 uint32_t       textureNum,
                                 VriPresentMode presentMode)
        {
            DeviceVK*        d    = sc->device;
            VkPhysicalDevice phys = d->PhysicalDevice();

            VkSurfaceCapabilitiesKHR caps = {};
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, sc->surface, &caps);

            // format
            uint32_t formatCount = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(phys, sc->surface, &formatCount, nullptr);
            std::vector<VkSurfaceFormatKHR> formats(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(phys, sc->surface, &formatCount, formats.data());

            const VkFormat     wanted = ToVkFormat(format);
            VkSurfaceFormatKHR chosen =
                formats.empty() ? VkSurfaceFormatKHR {wanted, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR} : formats[0];
            for (const VkSurfaceFormatKHR& f : formats)
            {
                if (f.format == wanted)
                {
                    chosen = f;
                    break;
                }
            }

            // present mode: map the requested VriPresentMode -> VkPresentModeKHR, falling back to FIFO
            // (the only mode guaranteed present) with a warning if the surface lacks the requested one.
            uint32_t modeCount = 0;
            vkGetPhysicalDeviceSurfacePresentModesKHR(phys, sc->surface, &modeCount, nullptr);
            std::vector<VkPresentModeKHR> modes(modeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(phys, sc->surface, &modeCount, modes.data());
            const auto supported = [&](VkPresentModeKHR m) {
                for (VkPresentModeKHR x : modes)
                    if (x == m)
                        return true;
                return false;
            };
            VkPresentModeKHR want = VK_PRESENT_MODE_FIFO_KHR;
            const char*      name = "Fifo";
            switch (presentMode)
            {
                case VriPresentMode_FifoRelaxed:
                    want = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
                    name = "FifoRelaxed";
                    break;
                case VriPresentMode_Mailbox:
                    want = VK_PRESENT_MODE_MAILBOX_KHR;
                    name = "Mailbox";
                    break;
                case VriPresentMode_Immediate:
                    want = VK_PRESENT_MODE_IMMEDIATE_KHR;
                    name = "Immediate";
                    break;
                default:
                    want = VK_PRESENT_MODE_FIFO_KHR;
                    name = "Fifo";
                    break;
            }
            VkPresentModeKHR mode = want;
            if (want != VK_PRESENT_MODE_FIFO_KHR && !supported(want))
            {
                char msg[160];
                std::snprintf(msg,
                              sizeof(msg),
                              "present mode '%s' is unsupported on this surface; falling back to Fifo (vsync)",
                              name);
                sc->device->ReportWarning(msg);
                mode = VK_PRESENT_MODE_FIFO_KHR;
            }

            // extent
            VkExtent2D extent = caps.currentExtent;
            if (extent.width == 0xFFFFFFFFu)
            {
                extent.width  = width;
                extent.height = height;
            }
            if (extent.width == 0 || extent.height == 0)
                return VriResult_Failure; // minimized; caller retries on resize

            uint32_t imageCount = textureNum < caps.minImageCount ? caps.minImageCount : textureNum;
            if (caps.maxImageCount != 0 && imageCount > caps.maxImageCount)
                imageCount = caps.maxImageCount;

            VkSwapchainCreateInfoKHR ci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
            ci.surface                  = sc->surface;
            ci.minImageCount            = imageCount;
            ci.imageFormat              = chosen.format;
            ci.imageColorSpace          = chosen.colorSpace;
            ci.imageExtent              = extent;
            ci.imageArrayLayers         = 1;
            ci.imageUsage =
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ci.preTransform     = caps.currentTransform;
            ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
            ci.presentMode      = mode;
            ci.clipped          = VK_TRUE;
            ci.oldSwapchain     = sc->swapchain;

            VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
            if (vkCreateSwapchainKHR(d->Device(), &ci, nullptr, &newSwapchain) != VK_SUCCESS)
                return VriResult_Failure;

            if (sc->swapchain)
                vkDestroySwapchainKHR(d->Device(), sc->swapchain, nullptr);
            DestroyTextures(sc);

            sc->swapchain   = newSwapchain;
            sc->format      = chosen.format;
            sc->extent      = extent;
            sc->presentMode = mode;

            uint32_t count = 0;
            vkGetSwapchainImagesKHR(d->Device(), sc->swapchain, &count, nullptr);
            std::vector<VkImage> images(count);
            vkGetSwapchainImagesKHR(d->Device(), sc->swapchain, &count, images.data());

            sc->textures.reserve(count);
            for (VkImage image : images)
            {
                TextureVK* t  = new TextureVK {};
                t->device     = d;
                t->image      = image;
                t->allocation = VK_NULL_HANDLE;
                t->format     = sc->format;
                t->extent     = {extent.width, extent.height, 1};
                t->mipNum     = 1;
                t->layerNum   = 1;
                t->type       = VK_IMAGE_TYPE_2D;
                t->owned      = false;
                sc->textures.push_back(t);
            }
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateSwapChain(VriDevice* device, const VriSwapChainDesc* desc, VriSwapChain** out)
        {
            // A swapchain must present on a queue; reject a null queue instead of dereferencing it.
            if (!device || !desc || !desc->queue || !out)
                return VriResult_InvalidArgument;

            DeviceVK*    d           = Dev(device);
            SwapChainVK* sc          = new SwapChainVK {};
            sc->device               = d;
            sc->presentQueue         = Q(desc->queue)->queue;
            sc->swapchain            = VK_NULL_HANDLE;
            sc->requestedFormat      = desc->format;
            sc->requestedTextureNum  = desc->textureNum;
            sc->requestedPresentMode = desc->presentMode;

            sc->surface = CreateSurface(d, desc->window);
            if (!sc->surface)
            {
                delete sc;
                return VriResult_Unsupported;
            }

            VkBool32 supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(
                d->PhysicalDevice(), Q(desc->queue)->familyIndex, sc->surface, &supported);
            if (!supported)
            {
                vkDestroySurfaceKHR(d->Instance(), sc->surface, nullptr);
                delete sc;
                return VriResult_Unsupported;
            }

            VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            vkCreateFence(d->Device(), &fci, nullptr, &sc->acquireFence);

            VriResult r =
                BuildSwapchain(sc, desc->width, desc->height, desc->format, desc->textureNum, desc->presentMode);
            if (r != VriResult_Success)
            {
                vkDestroyFence(d->Device(), sc->acquireFence, nullptr);
                vkDestroySurfaceKHR(d->Instance(), sc->surface, nullptr);
                delete sc;
                return r;
            }

            *out = ToHandle(sc);
            return VriResult_Success;
        }

        void VRI_CALL DestroySwapChain(VriSwapChain* swapChain)
        {
            if (!swapChain)
                return;
            SwapChainVK* sc  = SC(swapChain);
            VkDevice     dev = sc->device->Device();
            vkDeviceWaitIdle(dev);
            DestroyTextures(sc);
            if (sc->swapchain)
                vkDestroySwapchainKHR(dev, sc->swapchain, nullptr);
            if (sc->acquireFence)
                vkDestroyFence(dev, sc->acquireFence, nullptr);
            if (sc->surface)
                vkDestroySurfaceKHR(sc->device->Instance(), sc->surface, nullptr);
            delete sc;
        }

        VriResult VRI_CALL GetSwapChainTextures(VriSwapChain* swapChain, VriTexture** outTextures, uint32_t* ioCount)
        {
            SwapChainVK*   sc = SC(swapChain);
            const uint32_t n  = static_cast<uint32_t>(sc->textures.size());
            if (!outTextures)
            {
                *ioCount = n;
                return VriResult_Success;
            }
            const uint32_t copyNum = *ioCount < n ? *ioCount : n;
            for (uint32_t i = 0; i < copyNum; ++i)
                outTextures[i] = ToHandle(sc->textures[i]);
            *ioCount = copyNum;
            return VriResult_Success;
        }

        VriResult VRI_CALL AcquireNextTexture(VriSwapChain* swapChain,
                                              VriFence* /*acquireFence*/,
                                              uint64_t /*signalValue*/,
                                              uint32_t* outIndex)
        {
            SwapChainVK* sc  = SC(swapChain);
            VkDevice     dev = sc->device->Device();
            vkResetFences(dev, 1, &sc->acquireFence);

            uint32_t index = 0;
            VkResult r =
                vkAcquireNextImageKHR(dev, sc->swapchain, UINT64_MAX, VK_NULL_HANDLE, sc->acquireFence, &index);
            if (r == VK_ERROR_OUT_OF_DATE_KHR)
                return VriResult_OutOfDate;
            if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR)
                return VriResult_Failure;

            vkWaitForFences(dev, 1, &sc->acquireFence, VK_TRUE, UINT64_MAX);
            sc->currentIndex = index;
            *outIndex        = index;
            return VriResult_Success;
        }

        VriResult VRI_CALL Present(VriSwapChain* swapChain, VriFence* /*waitFence*/, uint64_t /*waitValue*/)
        {
            SwapChainVK*     sc = SC(swapChain);
            VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
            pi.swapchainCount   = 1;
            pi.pSwapchains      = &sc->swapchain;
            pi.pImageIndices    = &sc->currentIndex;
            VkResult r          = vkQueuePresentKHR(sc->presentQueue, &pi);
            if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR)
                return VriResult_OutOfDate;
            return r == VK_SUCCESS ? VriResult_Success : VriResult_Failure;
        }

        VriResult VRI_CALL Resize(VriSwapChain* swapChain, uint32_t width, uint32_t height)
        {
            SwapChainVK* sc = SC(swapChain);
            vkDeviceWaitIdle(sc->device->Device());
            return BuildSwapchain(
                sc, width, height, sc->requestedFormat, sc->requestedTextureNum, sc->requestedPresentMode);
        }

        VriResult VRI_CALL GetSwapChainExtent(VriSwapChain* swapChain, uint32_t* outWidth, uint32_t* outHeight)
        {
            const SwapChainVK* sc = SC(swapChain);
            if (!outWidth || !outHeight)
                return VriResult_InvalidArgument;
            // The surface's currentExtent won over the requested size at build time (Vulkan
            // mandates it); this is what render areas/viewports must be sized from.
            *outWidth  = sc->extent.width;
            *outHeight = sc->extent.height;
            return VriResult_Success;
        }

        const VriSwapChainInterface g_swapChainVK = {
            CreateSwapChain,
            DestroySwapChain,
            GetSwapChainTextures,
            AcquireNextTexture,
            Present,
            Resize,
            GetSwapChainExtent,
        };
    } // namespace

    const VriSwapChainInterface* GetSwapChainInterfaceVK() { return &g_swapChainVK; }
} // namespace vri::vk
