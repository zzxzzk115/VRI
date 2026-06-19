// Single translation unit that compiles the VulkanMemoryAllocator implementation.
#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4100 4127 4189 4324 4505)
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
