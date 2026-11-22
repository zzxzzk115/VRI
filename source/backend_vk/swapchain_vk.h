// swapchain_vk.h - Vulkan swapchain object + the swapchain interface table.
#pragma once

#include <vector>

#include "vulkan_vk.h"

#include <vri/ext/vri_ext_swapchain.h>
#include <vri/vri.h>

#include "objects_vk.h"

namespace vri::vk
{
    struct SwapChainVK
    {
        ~SwapChainVK() { DebugObjectsVK::Untrack(this); }
        DeviceVK*               device;
        VkSurfaceKHR            surface;
        VkSwapchainKHR          swapchain;
        VkQueue                 presentQueue;
        VkFormat                format;
        VkExtent2D              extent;
        VkPresentModeKHR        presentMode;
        std::vector<TextureVK*> textures; // wrap swapchain images (not owned)
        // Binary semaphores, one per image, that Present signals once its wait fence is reached and hands to
        // vkQueuePresentKHR. The one for an image is signalled again only after that image has been acquired again,
        // and so after the present that waited on it.
        std::vector<VkSemaphore> presentSemaphores;
        VkFence                  acquireFence;
        uint32_t                 currentIndex;
        // original creation request (reused on Resize)
        VriFormat      requestedFormat;
        uint32_t       requestedTextureNum;
        VriPresentMode requestedPresentMode;
    };

    inline VriSwapChain* ToHandle(SwapChainVK* s)
    {
        DebugObjectsVK::Track(s, s->device, VK_OBJECT_TYPE_SWAPCHAIN_KHR, s->swapchain);
        return reinterpret_cast<VriSwapChain*>(s);
    }

    const VriSwapChainInterface* GetSwapChainInterfaceVK();

    // Swapchain creation fields chosen from what the surface reports.
    // Opaque when the surface supports it, otherwise inherit, pre-multiplied, post-multiplied in that order; 0 when
    // the surface reports none of them.
    VkCompositeAlphaFlagBitsKHR ChooseCompositeAlpha(VkCompositeAlphaFlagsKHR supported);
    // Color attachment (every surface supports it) plus whichever of transfer source and destination the surface
    // supports.
    VkImageUsageFlags ChooseImageUsage(VkImageUsageFlags supported);
    // currentExtent, or the requested size clamped to the surface's limits when the surface leaves the size to the
    // swapchain.
    VkExtent2D ChooseExtent(const VkSurfaceCapabilitiesKHR& caps, uint32_t width, uint32_t height);
} // namespace vri::vk
