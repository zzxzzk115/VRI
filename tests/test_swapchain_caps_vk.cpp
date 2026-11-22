// Vulkan swapchain creation fields chosen from the surface's capabilities, without a device or a window.
#include <doctest/doctest.h>

#include "backend_vk/swapchain_vk.h"

using namespace vri::vk;

TEST_CASE("Vulkan swapchain composite alpha prefers opaque and otherwise takes what the surface supports")
{
    CHECK(ChooseCompositeAlpha(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR | VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) ==
          VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);
    // The Android emulator's surfaces report inherit alone.
    CHECK(ChooseCompositeAlpha(VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) == VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR);
    CHECK(ChooseCompositeAlpha(VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR | VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) ==
          VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR);
    CHECK(
        ChooseCompositeAlpha(VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR | VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) ==
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR);
    CHECK(ChooseCompositeAlpha(VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) ==
          VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR);
    CHECK(ChooseCompositeAlpha(0) == 0);
}

TEST_CASE("Vulkan swapchain image usage requests transfers only where the surface supports them")
{
    constexpr VkImageUsageFlags kAll =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    CHECK(ChooseImageUsage(kAll | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT) == kAll);
    CHECK(ChooseImageUsage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    CHECK(ChooseImageUsage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT) ==
          (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
}

TEST_CASE("Vulkan swapchain extent follows currentExtent, else the requested size within the surface's limits")
{
    VkSurfaceCapabilitiesKHR caps = {};
    caps.minImageExtent           = {16, 8};
    caps.maxImageExtent           = {1024, 512};

    caps.currentExtent = {300, 200};
    VkExtent2D e       = ChooseExtent(caps, 640, 480);
    CHECK(e.width == 300);
    CHECK(e.height == 200);

    caps.currentExtent = {0xFFFFFFFFu, 0xFFFFFFFFu};
    e                  = ChooseExtent(caps, 640, 480);
    CHECK(e.width == 640);
    CHECK(e.height == 480);
    e = ChooseExtent(caps, 4096, 4);
    CHECK(e.width == 1024);
    CHECK(e.height == 8);
}
