// External memory/semaphore export (Vulkan, VK_KHR_external_memory_{win32,fd} +
// VK_KHR_external_semaphore_{win32,fd}) -- the "CUDA interop seam". VRI does not depend on
// CUDA, so this test verifies the EXPORT plumbing only: an exportable buffer/texture yields
// a valid OS handle (Win32 HANDLE / POSIX fd) with the right size, and an exportable fence
// yields a valid handle. A real CUDA round-trip lives in an optional, off-by-default example
// (it would pull the CUDA SDK into the build). Skips gracefully when the adapter lacks
// external memory (e.g. CI software rasterizers), keeping the suite green there.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#if defined(_WIN32)
#include <windows.h>
namespace
{
    bool                            HandleValid(void* h) { return h != nullptr && h != INVALID_HANDLE_VALUE; }
    bool                            CloseExportedHandle(void* h) { return CloseHandle(static_cast<HANDLE>(h)) != 0; }
    constexpr VriExternalHandleType kHandleType = VriExternalHandleType_OpaqueWin32;
} // namespace
#else
#include <unistd.h>
namespace
{
    bool HandleValid(void* h) { return static_cast<int>(reinterpret_cast<intptr_t>(h)) >= 0; }
    bool CloseExportedHandle(void* h) { return ::close(static_cast<int>(reinterpret_cast<intptr_t>(h))) == 0; }
    constexpr VriExternalHandleType kHandleType = VriExternalHandleType_OpaqueFd;
} // namespace
#endif

namespace
{
    struct Vk
    {
        VriDevice*       device = nullptr;
        VriCoreInterface core {};
        ~Vk()
        {
            if (device)
                vriDestroyDevice(device);
        }
    };

    bool InitVk(Vk& vk)
    {
        VriDeviceCreationDesc desc {};
        desc.graphicsAPI      = VriGraphicsAPI_Vulkan;
        desc.enableValidation = VRI_TRUE;
        desc.bestEffort       = VRI_TRUE;
        desc.enabledFeatures  = VriFeature_ExternalMemory;
        if (vriCreateDevice(&desc, &vk.device) != VriResult_Success)
            return false;
        return vriGetInterface(vk.device, VRI_INTERFACE_CORE, sizeof(vk.core), &vk.core) == VriResult_Success;
    }
} // namespace

TEST_CASE("Vulkan external memory: interface availability tracks the reported capability")
{
    Vk vk;
    if (!InitVk(vk))
    {
        MESSAGE("Vulkan unavailable - skipping external-memory test");
        return;
    }

    const VriDeviceDesc* d = vk.core.GetDeviceDesc(vk.device);
    VriExternalInterface ext {};
    const bool queryable = vriGetInterface(vk.device, VRI_INTERFACE_EXTERNAL, sizeof(ext), &ext) == VriResult_Success;
    CHECK(queryable == (d->hasExternalMemory != VRI_FALSE));
    if (queryable)
    {
        CHECK(ext.CreateExportableBuffer != nullptr);
        CHECK(ext.GetBufferMemoryHandle != nullptr);
        CHECK(ext.CreateExportableFence != nullptr);
        CHECK(ext.GetFenceHandle != nullptr);
    }
}

TEST_CASE("Vulkan external memory: exportable buffer yields a valid OS handle")
{
    Vk vk;
    if (!InitVk(vk))
    {
        MESSAGE("Vulkan unavailable - skipping external-memory test");
        return;
    }
    const VriCoreInterface& c   = vk.core;
    VriDevice*              dev = vk.device;
    if (c.GetDeviceDesc(dev)->hasExternalMemory == VRI_FALSE)
    {
        MESSAGE("adapter lacks external memory - skipping");
        return;
    }

    VriExternalInterface ext {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_EXTERNAL, sizeof(ext), &ext) == VriResult_Success);

    constexpr uint64_t kSize = 64 * 1024;
    VriBufferDesc      bd {};
    bd.size           = kSize;
    bd.usage          = VriBufferUsage_StorageBuffer | VriBufferUsage_TransferDst;
    bd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* buffer = nullptr;
    REQUIRE(ext.CreateExportableBuffer(dev, &bd, kHandleType, &buffer) == VriResult_Success);

    VriExternalMemoryInfo info {};
    REQUIRE(ext.GetBufferMemoryHandle(dev, buffer, kHandleType, &info) == VriResult_Success);
    CHECK(HandleValid(info.handle));
    CHECK(info.size >= kSize); // total allocation size (>= requested, may be aligned up)
    CHECK(info.dedicated == VRI_TRUE);
    CHECK(CloseExportedHandle(info.handle)); // caller owns the exported handle

    // Requesting the platform-mismatched handle type must be rejected, not crash.
#if defined(_WIN32)
    VriExternalMemoryInfo bad {};
    CHECK(ext.GetBufferMemoryHandle(dev, buffer, VriExternalHandleType_OpaqueFd, &bad) == VriResult_InvalidArgument);
#endif

    c.DestroyBuffer(buffer);
}

TEST_CASE("Vulkan external memory: exportable fence yields a valid OS handle")
{
    Vk vk;
    if (!InitVk(vk))
    {
        MESSAGE("Vulkan unavailable - skipping external-memory test");
        return;
    }
    const VriCoreInterface& c   = vk.core;
    VriDevice*              dev = vk.device;
    if (c.GetDeviceDesc(dev)->hasExternalMemory == VRI_FALSE)
    {
        MESSAGE("adapter lacks external memory - skipping");
        return;
    }

    VriExternalInterface ext {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_EXTERNAL, sizeof(ext), &ext) == VriResult_Success);

    VriFence* fence = nullptr;
    REQUIRE(ext.CreateExportableFence(dev, 0, kHandleType, &fence) == VriResult_Success);

    void* handle = nullptr;
    REQUIRE(ext.GetFenceHandle(dev, fence, kHandleType, &handle) == VriResult_Success);
    CHECK(HandleValid(handle));
    CHECK(CloseExportedHandle(handle));

    // The exportable fence is a normal timeline fence: it works through the core interface.
    CHECK(c.GetFenceValue(fence) == 0);
    c.DestroyFence(fence);
}
