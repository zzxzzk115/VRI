// swapchain_vk.h - Vulkan swapchain object + the swapchain interface table.
#pragma once

#include <vector>

#include <vulkan/vulkan.h>

#include <vri/vri.h>
#include <vri/ext/vri_ext_swapchain.h>

#include "objects_vk.h"

namespace vri::vk
{
    struct SwapChainVK
    {
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
        VriFormat               requestedFormat;
        uint32_t                requestedTextureNum;
        bool                    vsync;
    };

    inline VriSwapChain* ToHandle(SwapChainVK* s) { return reinterpret_cast<VriSwapChain*>(s); }

    const VriSwapChainInterface* GetSwapChainInterfaceVK();
} // namespace vri::vk
