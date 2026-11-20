// volk_impl.cpp - the single translation unit that defines volk's function pointers.
//
// The surface extensions of every platform the backend presents on are declared here, so their
// pointers exist; the files that create surfaces define the same VK_USE_PLATFORM_* macro first.

#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(__APPLE__)
#define VK_USE_PLATFORM_METAL_EXT
#endif

#define VOLK_IMPLEMENTATION
#include "vulkan_vk.h"
