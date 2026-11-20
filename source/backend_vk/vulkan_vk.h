// vulkan_vk.h - the Vulkan declarations every backend_vk file uses.
//
// The backend never links the Vulkan loader: volk opens it at run time and loads every entry
// point through vkGetInstanceProcAddr, so a binary built for Android API 26 has no undefined
// Vulkan 1.1+ symbols (the system libvulkan.so exports those only from API 28-33).
// VOLK_NAMESPACE puts volk's function pointers in volk:: (volk.h adds `using namespace volk`), so
// they cannot collide with a consumer that links the loader and calls Vulkan itself.
// Include this before any other Vulkan header (vk_mem_alloc.h included): it sets VK_NO_PROTOTYPES.
//
// Only the loader's global functions (vkCreateInstance, vkEnumerateInstance*, vkGetInstanceProcAddr)
// are called through volk's process-wide pointers. A pointer loaded from an instance or device is
// valid for that instance or device only, so every other call goes through its device's tables
// (DeviceVK::Vki/Vkd); VOLK_NO_DEVICE_PROTOTYPES makes a device-level call that skips them a
// compile error.
#pragma once

#if !defined(VOLK_NAMESPACE)
#define VOLK_NAMESPACE
#endif
// volk_impl.cpp (VOLK_IMPLEMENTATION) still needs the declarations to define the pointers.
#if !defined(VOLK_NO_DEVICE_PROTOTYPES) && !defined(VOLK_IMPLEMENTATION)
#define VOLK_NO_DEVICE_PROTOTYPES
#endif

#include <volk.h>
