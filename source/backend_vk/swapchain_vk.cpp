// swapchain_vk.cpp - Vulkan swapchain (presentation) implementation.
//
// AcquireNextTexture waits on the host until the image is available. Present waits on the GPU: given a fence, it
// submits a wait for the fence's value that signals the image's binary semaphore, and vkQueuePresentKHR waits on
// that semaphore, so the app does not wait for its render work before presenting.

#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__APPLE__)
#define VK_USE_PLATFORM_METAL_EXT
#endif

#include "vulkan_vk.h"

#include "conversions_vk.h"
#include "device_vk.h"
#include "swapchain_vk.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace vri::vk
{
    namespace
    {
        inline SwapChainVK* SC(VriSwapChain* h) { return reinterpret_cast<SwapChainVK*>(h); }
        inline DeviceVK*    Dev(VriDevice* h) { return reinterpret_cast<DeviceVK*>(h); }
        inline QueueVK*     Q(VriQueue* h) { return reinterpret_cast<QueueVK*>(h); }
        inline FenceVK*     Fen(VriFence* h) { return reinterpret_cast<FenceVK*>(h); }

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
            if (d->Vki().vkCreateWin32SurfaceKHR(d->Instance(), &ci, nullptr, &surface) != VK_SUCCESS)
                return VK_NULL_HANDLE;
#elif defined(__APPLE__)
            // MoltenVK: cocoa.layer is a CAMetalLayer* (the SDL3/GLFW integration headers
            // derive it from the window). vkCreateMetalSurfaceEXT wraps it as a surface.
            if (window.type != VriWindowSystem_Cocoa || window.handle.cocoa.layer == nullptr)
                return VK_NULL_HANDLE;
            VkMetalSurfaceCreateInfoEXT ci = {VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT};
            ci.pLayer                      = static_cast<const CAMetalLayer*>(window.handle.cocoa.layer);
            if (d->Vki().vkCreateMetalSurfaceEXT(d->Instance(), &ci, nullptr, &surface) != VK_SUCCESS)
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

        // The caller has waited for the device to go idle, so no present still waits on these semaphores.
        void DestroyTextures(SwapChainVK* sc)
        {
            for (TextureVK* t : sc->textures)
                delete t; // images are owned by the swapchain, not freed here
            sc->textures.clear();
            for (VkSemaphore semaphore : sc->presentSemaphores)
                sc->device->Vkd().vkDestroySemaphore(sc->device->Device(), semaphore, nullptr);
            sc->presentSemaphores.clear();
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
            d->Vki().vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, sc->surface, &caps);

            // format
            uint32_t formatCount = 0;
            d->Vki().vkGetPhysicalDeviceSurfaceFormatsKHR(phys, sc->surface, &formatCount, nullptr);
            std::vector<VkSurfaceFormatKHR> formats(formatCount);
            d->Vki().vkGetPhysicalDeviceSurfaceFormatsKHR(phys, sc->surface, &formatCount, formats.data());

            // The requested format, or Unsupported: the caller's texture views and pipelines are built for it, and
            // it may try another one. A surface always lists at least one format, so an empty list means the query
            // failed; the swapchain is then created with the requested format and vkCreateSwapchainKHR decides.
            const VkFormat     wanted = ToVkFormat(format);
            VkSurfaceFormatKHR chosen = {wanted, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
            bool               listed = formats.empty();
            for (const VkSurfaceFormatKHR& f : formats)
            {
                if (f.format == wanted)
                {
                    chosen = f;
                    listed = true;
                    break;
                }
            }
            if (!listed)
                return VriResult_Unsupported;

            // present mode: map the requested VriPresentMode -> VkPresentModeKHR, falling back to FIFO
            // (the only mode guaranteed present) with a warning if the surface lacks the requested one.
            uint32_t modeCount = 0;
            d->Vki().vkGetPhysicalDeviceSurfacePresentModesKHR(phys, sc->surface, &modeCount, nullptr);
            std::vector<VkPresentModeKHR> modes(modeCount);
            d->Vki().vkGetPhysicalDeviceSurfacePresentModesKHR(phys, sc->surface, &modeCount, modes.data());
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

            const VkExtent2D extent = ChooseExtent(caps, width, height);
            if (extent.width == 0 || extent.height == 0)
                return VriResult_Failure; // minimized; caller retries on resize

            const VkCompositeAlphaFlagBitsKHR compositeAlpha = ChooseCompositeAlpha(caps.supportedCompositeAlpha);
            if (compositeAlpha == 0)
            {
                char msg[128];
                std::snprintf(msg,
                              sizeof(msg),
                              "CreateSwapChain: the surface supports no composite alpha mode "
                              "(supportedCompositeAlpha 0x%x)",
                              static_cast<unsigned>(caps.supportedCompositeAlpha));
                d->ReportError(msg);
                return VriResult_Failure;
            }
            const VkImageUsageFlags usage = ChooseImageUsage(caps.supportedUsageFlags);
            if ((usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0 || (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0)
                d->ReportWarning("CreateSwapChain: the surface's images do not support both transfer usages; copies "
                                 "to or from the swapchain textures are invalid on it");

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
            ci.imageUsage               = usage;
            ci.imageSharingMode         = VK_SHARING_MODE_EXCLUSIVE;
            // The application renders upright. Declaring the identity transform leaves a rotated display, such as a
            // phone held sideways, to the compositor; passing the current transform would promise images that are
            // already rotated.
            ci.preTransform   = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) != 0 ?
                                    VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR :
                                    caps.currentTransform;
            ci.compositeAlpha = compositeAlpha;
            ci.presentMode    = mode;
            ci.clipped        = VK_TRUE;
            ci.oldSwapchain   = sc->swapchain;

            VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
            if (d->Vkd().vkCreateSwapchainKHR(d->Device(), &ci, nullptr, &newSwapchain) != VK_SUCCESS)
                return VriResult_Failure;

            if (sc->swapchain)
                d->Vkd().vkDestroySwapchainKHR(d->Device(), sc->swapchain, nullptr);
            DestroyTextures(sc);

            sc->swapchain = newSwapchain;
            DebugObjectsVK::Track(sc, d, VK_OBJECT_TYPE_SWAPCHAIN_KHR, newSwapchain);
            sc->format      = chosen.format;
            sc->extent      = extent;
            sc->presentMode = mode;

            uint32_t count = 0;
            d->Vkd().vkGetSwapchainImagesKHR(d->Device(), sc->swapchain, &count, nullptr);
            std::vector<VkImage> images(count);
            d->Vkd().vkGetSwapchainImagesKHR(d->Device(), sc->swapchain, &count, images.data());

            sc->textures.reserve(count);
            sc->presentSemaphores.reserve(count);
            for (VkImage image : images)
            {
                const VkSemaphoreCreateInfo sci       = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
                VkSemaphore                 semaphore = VK_NULL_HANDLE;
                if (d->Vkd().vkCreateSemaphore(d->Device(), &sci, nullptr, &semaphore) != VK_SUCCESS)
                {
                    // The old swapchain is already gone; leave none rather than one without semaphores.
                    DestroyTextures(sc);
                    d->Vkd().vkDestroySwapchainKHR(d->Device(), sc->swapchain, nullptr);
                    sc->swapchain = VK_NULL_HANDLE;
                    return VriResult_Failure;
                }
                sc->presentSemaphores.push_back(semaphore);

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
            d->Vki().vkGetPhysicalDeviceSurfaceSupportKHR(
                d->PhysicalDevice(), Q(desc->queue)->familyIndex, sc->surface, &supported);
            if (!supported)
            {
                d->Vki().vkDestroySurfaceKHR(d->Instance(), sc->surface, nullptr);
                delete sc;
                return VriResult_Unsupported;
            }

            VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            d->Vkd().vkCreateFence(d->Device(), &fci, nullptr, &sc->acquireFence);

            VriResult r =
                BuildSwapchain(sc, desc->width, desc->height, desc->format, desc->textureNum, desc->presentMode);
            if (r != VriResult_Success)
            {
                d->Vkd().vkDestroyFence(d->Device(), sc->acquireFence, nullptr);
                d->Vki().vkDestroySurfaceKHR(d->Instance(), sc->surface, nullptr);
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
            sc->device->Vkd().vkDeviceWaitIdle(dev);
            DestroyTextures(sc);
            if (sc->swapchain)
                sc->device->Vkd().vkDestroySwapchainKHR(dev, sc->swapchain, nullptr);
            if (sc->acquireFence)
                sc->device->Vkd().vkDestroyFence(dev, sc->acquireFence, nullptr);
            if (sc->surface)
                sc->device->Vki().vkDestroySurfaceKHR(sc->device->Instance(), sc->surface, nullptr);
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
            sc->device->Vkd().vkResetFences(dev, 1, &sc->acquireFence);

            uint32_t index = 0;
            VkResult r     = sc->device->Vkd().vkAcquireNextImageKHR(
                dev, sc->swapchain, UINT64_MAX, VK_NULL_HANDLE, sc->acquireFence, &index);
            if (r == VK_ERROR_OUT_OF_DATE_KHR)
                return VriResult_OutOfDate;
            if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR)
                return VriResult_Failure;

            sc->device->Vkd().vkWaitForFences(dev, 1, &sc->acquireFence, VK_TRUE, UINT64_MAX);
            sc->currentIndex = index;
            *outIndex        = index;
            return VriResult_Success;
        }

        VriResult VRI_CALL Present(VriSwapChain* swapChain, VriFence* waitFence, uint64_t waitValue)
        {
            SwapChainVK*     sc = SC(swapChain);
            VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
            if (waitFence)
            {
                if (Fen(waitFence)->device != sc->device)
                    return VriResult_InvalidArgument;
                // vkQueuePresentKHR waits only on binary semaphores: an empty submission turns the fence's value
                // into the image's binary semaphore.
                const VkSemaphore* ready = &sc->presentSemaphores[sc->currentIndex]; // read by vkQueuePresentKHR below
                VkSemaphoreSubmitInfo wait   = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
                wait.semaphore               = Fen(waitFence)->timeline;
                wait.value                   = waitValue;
                wait.stageMask               = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                VkSemaphoreSubmitInfo signal = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
                signal.semaphore             = *ready;
                signal.stageMask             = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                VkSubmitInfo2 si             = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
                si.waitSemaphoreInfoCount    = 1;
                si.pWaitSemaphoreInfos       = &wait;
                si.signalSemaphoreInfoCount  = 1;
                si.pSignalSemaphoreInfos     = &signal;
                const VkResult submitted     = sc->device->Ext().QueueSubmit2(sc->presentQueue, 1, &si, VK_NULL_HANDLE);
                if (submitted != VK_SUCCESS)
                    return submitted == VK_ERROR_DEVICE_LOST ? VriResult_DeviceLost : VriResult_Failure;
                pi.waitSemaphoreCount = 1;
                pi.pWaitSemaphores    = ready;
            }
            pi.swapchainCount = 1;
            pi.pSwapchains    = &sc->swapchain;
            pi.pImageIndices  = &sc->currentIndex;
            VkResult r        = sc->device->Vkd().vkQueuePresentKHR(sc->presentQueue, &pi);
            if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR)
                return VriResult_OutOfDate;
            return r == VK_SUCCESS ? VriResult_Success : VriResult_Failure;
        }

        VriResult VRI_CALL Resize(VriSwapChain* swapChain, uint32_t width, uint32_t height)
        {
            SwapChainVK* sc = SC(swapChain);
            sc->device->Vkd().vkDeviceWaitIdle(sc->device->Device());
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

    VkCompositeAlphaFlagBitsKHR ChooseCompositeAlpha(VkCompositeAlphaFlagsKHR supported)
    {
        // Some surfaces offer only the alpha behavior the platform's compositor decides (Android emulator surfaces
        // report inherit alone); the swapchain then has to take that instead of opaque.
        constexpr VkCompositeAlphaFlagBitsKHR kOrder[] = {
            VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        };
        for (VkCompositeAlphaFlagBitsKHR mode : kOrder)
            if ((supported & mode) != 0)
                return mode;
        return static_cast<VkCompositeAlphaFlagBitsKHR>(0);
    }

    VkImageUsageFlags ChooseImageUsage(VkImageUsageFlags supported)
    {
        return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               (supported & (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
    }

    VkExtent2D ChooseExtent(const VkSurfaceCapabilitiesKHR& caps, uint32_t width, uint32_t height)
    {
        if (caps.currentExtent.width != 0xFFFFFFFFu)
            return caps.currentExtent;
        return {std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width),
                std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height)};
    }
} // namespace vri::vk
