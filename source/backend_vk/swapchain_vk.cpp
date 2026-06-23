// swapchain_vk.cpp - Vulkan swapchain (presentation) implementation.
//
// Phase-1 swapchain: correct but serialized (the app CPU-waits its render fence
// before Present, so no binary-semaphore frame pipelining yet). A pipelined
// path with per-frame binary acquire/present semaphores is a later refinement.

#if defined(_WIN32)
#    define VK_USE_PLATFORM_WIN32_KHR
#    define WIN32_LEAN_AND_MEAN
#    define NOMINMAX
#    include <windows.h>
#elif defined(__APPLE__)
#    define VK_USE_PLATFORM_METAL_EXT
#endif

#include <vulkan/vulkan.h>

#include "swapchain_vk.h"
#include "conversions_vk.h"
#include "device_vk.h"

#include <vector>

namespace vri::vk
{
    namespace
    {
        inline SwapChainVK* SC(VriSwapChain* h) { return reinterpret_cast<SwapChainVK*>(h); }
        inline DeviceVK*    Dev(VriDevice* h)   { return reinterpret_cast<DeviceVK*>(h); }
        inline QueueVK*     Q(VriQueue* h)      { return reinterpret_cast<QueueVK*>(h); }

        VkSurfaceKHR CreateSurface(DeviceVK* d, const VriWindowHandle& window)
        {
            VkSurfaceKHR surface = VK_NULL_HANDLE;
#if defined(_WIN32)
            if (window.type != VriWindowSystem_Win32)
                return VK_NULL_HANDLE;
            VkWin32SurfaceCreateInfoKHR ci = {VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
            ci.hinstance = window.handle.win32.hinstance
                               ? static_cast<HINSTANCE>(window.handle.win32.hinstance)
                               : GetModuleHandleW(nullptr);
            ci.hwnd = static_cast<HWND>(window.handle.win32.hwnd);
            if (vkCreateWin32SurfaceKHR(d->Instance(), &ci, nullptr, &surface) != VK_SUCCESS)
                return VK_NULL_HANDLE;
#elif defined(__APPLE__)
            // MoltenVK: cocoa.layer is a CAMetalLayer* (the SDL3/GLFW integration headers
            // derive it from the window). vkCreateMetalSurfaceEXT wraps it as a surface.
            if (window.type != VriWindowSystem_Cocoa || window.handle.cocoa.layer == nullptr)
                return VK_NULL_HANDLE;
            VkMetalSurfaceCreateInfoEXT ci = {VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT};
            ci.pLayer = static_cast<const CAMetalLayer*>(window.handle.cocoa.layer);
            if (vkCreateMetalSurfaceEXT(d->Instance(), &ci, nullptr, &surface) != VK_SUCCESS)
                return VK_NULL_HANDLE;
#else
            (void)d;
            (void)window;
#endif
            return surface;
        }

        void DestroyTextures(SwapChainVK* sc)
        {
            for (TextureVK* t : sc->textures)
                delete t; // images are owned by the swapchain, not freed here
            sc->textures.clear();
        }

        VriResult BuildSwapchain(SwapChainVK* sc, uint32_t width, uint32_t height, VriFormat format, uint32_t textureNum, bool vsync)
        {
            DeviceVK* d = sc->device;
            VkPhysicalDevice phys = d->PhysicalDevice();

            VkSurfaceCapabilitiesKHR caps = {};
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, sc->surface, &caps);

            // format
            uint32_t formatCount = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(phys, sc->surface, &formatCount, nullptr);
            std::vector<VkSurfaceFormatKHR> formats(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(phys, sc->surface, &formatCount, formats.data());

            const VkFormat wanted = ToVkFormat(format);
            VkSurfaceFormatKHR chosen = formats.empty() ? VkSurfaceFormatKHR{wanted, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR} : formats[0];
            for (const VkSurfaceFormatKHR& f : formats)
            {
                if (f.format == wanted)
                {
                    chosen = f;
                    break;
                }
            }

            // present mode
            VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR; // always supported
            if (!vsync)
            {
                uint32_t modeCount = 0;
                vkGetPhysicalDeviceSurfacePresentModesKHR(phys, sc->surface, &modeCount, nullptr);
                std::vector<VkPresentModeKHR> modes(modeCount);
                vkGetPhysicalDeviceSurfacePresentModesKHR(phys, sc->surface, &modeCount, modes.data());
                for (VkPresentModeKHR m : modes)
                    if (m == VK_PRESENT_MODE_MAILBOX_KHR) { mode = m; break; }
            }

            // extent
            VkExtent2D extent = caps.currentExtent;
            if (extent.width == 0xFFFFFFFFu)
            {
                extent.width = width;
                extent.height = height;
            }
            if (extent.width == 0 || extent.height == 0)
                return VriResult_Failure; // minimized; caller retries on resize

            uint32_t imageCount = textureNum < caps.minImageCount ? caps.minImageCount : textureNum;
            if (caps.maxImageCount != 0 && imageCount > caps.maxImageCount)
                imageCount = caps.maxImageCount;

            VkSwapchainCreateInfoKHR ci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
            ci.surface = sc->surface;
            ci.minImageCount = imageCount;
            ci.imageFormat = chosen.format;
            ci.imageColorSpace = chosen.colorSpace;
            ci.imageExtent = extent;
            ci.imageArrayLayers = 1;
            ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ci.preTransform = caps.currentTransform;
            ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
            ci.presentMode = mode;
            ci.clipped = VK_TRUE;
            ci.oldSwapchain = sc->swapchain;

            VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
            if (vkCreateSwapchainKHR(d->Device(), &ci, nullptr, &newSwapchain) != VK_SUCCESS)
                return VriResult_Failure;

            if (sc->swapchain)
                vkDestroySwapchainKHR(d->Device(), sc->swapchain, nullptr);
            DestroyTextures(sc);

            sc->swapchain = newSwapchain;
            sc->format = chosen.format;
            sc->extent = extent;
            sc->presentMode = mode;

            uint32_t count = 0;
            vkGetSwapchainImagesKHR(d->Device(), sc->swapchain, &count, nullptr);
            std::vector<VkImage> images(count);
            vkGetSwapchainImagesKHR(d->Device(), sc->swapchain, &count, images.data());

            sc->textures.reserve(count);
            for (VkImage image : images)
            {
                TextureVK* t = new TextureVK{};
                t->device = d;
                t->image = image;
                t->allocation = VK_NULL_HANDLE;
                t->format = sc->format;
                t->extent = {extent.width, extent.height, 1};
                t->mipNum = 1;
                t->layerNum = 1;
                t->type = VK_IMAGE_TYPE_2D;
                t->owned = false;
                sc->textures.push_back(t);
            }
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateSwapChain(VriDevice* device, const VriSwapChainDesc* desc, VriSwapChain** out)
        {
            DeviceVK* d = Dev(device);
            SwapChainVK* sc = new SwapChainVK{};
            sc->device = d;
            sc->presentQueue = Q(desc->queue)->queue;
            sc->swapchain = VK_NULL_HANDLE;
            sc->requestedFormat = desc->format;
            sc->requestedTextureNum = desc->textureNum;
            sc->vsync = desc->vsync != VRI_FALSE;

            sc->surface = CreateSurface(d, desc->window);
            if (!sc->surface)
            {
                delete sc;
                return VriResult_Unsupported;
            }

            VkBool32 supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(d->PhysicalDevice(), Q(desc->queue)->familyIndex, sc->surface, &supported);
            if (!supported)
            {
                vkDestroySurfaceKHR(d->Instance(), sc->surface, nullptr);
                delete sc;
                return VriResult_Unsupported;
            }

            VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            vkCreateFence(d->Device(), &fci, nullptr, &sc->acquireFence);

            VriResult r = BuildSwapchain(sc, desc->width, desc->height, desc->format, desc->textureNum, desc->vsync != VRI_FALSE);
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
            if (!swapChain) return;
            SwapChainVK* sc = SC(swapChain);
            VkDevice dev = sc->device->Device();
            vkDeviceWaitIdle(dev);
            DestroyTextures(sc);
            if (sc->swapchain) vkDestroySwapchainKHR(dev, sc->swapchain, nullptr);
            if (sc->acquireFence) vkDestroyFence(dev, sc->acquireFence, nullptr);
            if (sc->surface) vkDestroySurfaceKHR(sc->device->Instance(), sc->surface, nullptr);
            delete sc;
        }

        VriResult VRI_CALL GetSwapChainTextures(VriSwapChain* swapChain, VriTexture** outTextures, uint32_t* ioCount)
        {
            SwapChainVK* sc = SC(swapChain);
            const uint32_t n = static_cast<uint32_t>(sc->textures.size());
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

        VriResult VRI_CALL AcquireNextTexture(VriSwapChain* swapChain, VriFence* /*acquireFence*/, uint64_t /*signalValue*/, uint32_t* outIndex)
        {
            SwapChainVK* sc = SC(swapChain);
            VkDevice dev = sc->device->Device();
            vkResetFences(dev, 1, &sc->acquireFence);

            uint32_t index = 0;
            VkResult r = vkAcquireNextImageKHR(dev, sc->swapchain, UINT64_MAX, VK_NULL_HANDLE, sc->acquireFence, &index);
            if (r == VK_ERROR_OUT_OF_DATE_KHR)
                return VriResult_OutOfDate;
            if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR)
                return VriResult_Failure;

            vkWaitForFences(dev, 1, &sc->acquireFence, VK_TRUE, UINT64_MAX);
            sc->currentIndex = index;
            *outIndex = index;
            return VriResult_Success;
        }

        VriResult VRI_CALL Present(VriSwapChain* swapChain, VriFence* /*waitFence*/, uint64_t /*waitValue*/)
        {
            SwapChainVK* sc = SC(swapChain);
            VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
            pi.swapchainCount = 1;
            pi.pSwapchains = &sc->swapchain;
            pi.pImageIndices = &sc->currentIndex;
            VkResult r = vkQueuePresentKHR(sc->presentQueue, &pi);
            if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR)
                return VriResult_OutOfDate;
            return r == VK_SUCCESS ? VriResult_Success : VriResult_Failure;
        }

        VriResult VRI_CALL Resize(VriSwapChain* swapChain, uint32_t width, uint32_t height)
        {
            SwapChainVK* sc = SC(swapChain);
            vkDeviceWaitIdle(sc->device->Device());
            return BuildSwapchain(sc, width, height, sc->requestedFormat, sc->requestedTextureNum, sc->vsync);
        }

        const VriSwapChainInterface g_swapChainVK = {
            CreateSwapChain,
            DestroySwapChain,
            GetSwapChainTextures,
            AcquireNextTexture,
            Present,
            Resize,
        };
    } // namespace

    const VriSwapChainInterface* GetSwapChainInterfaceVK() { return &g_swapChainVK; }
} // namespace vri::vk
