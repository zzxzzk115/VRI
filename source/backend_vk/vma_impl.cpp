// Single translation unit that compiles the VulkanMemoryAllocator implementation.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100 4127 4189 4324 4505)
#endif

#include "vulkan_vk.h"

// VMA resolves its entry points from the vkGetInstanceProcAddr / vkGetDeviceProcAddr that
// DeviceVK::CreateAllocator passes in; static linking would reference the loader's symbols.
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
