// swapchain_vk.h - Vulkan swapchain object + the swapchain interface table.
#pragma once

#include <vector>

#include <vulkan/vulkan.h>

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
        VkFence                 acquireFence;
        uint32_t                currentIndex;
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
} // namespace vri::vk
