// External memory/fence export (D3D12, shared committed resource + shared ID3D12Fence) --
// the "CUDA interop seam". VRI does not depend on CUDA, so this verifies the EXPORT plumbing
// only: an exportable buffer yields a valid NT HANDLE (D3D12Resource) with the right size, and
// an exportable fence yields a valid HANDLE (D3D12Fence). The real CUDA round-trip is the
// opt-in example-cuda-interop. Self-skips when D3D12 / external memory is unavailable (e.g. WARP
// or a runner without a D3D12 device), keeping the suite green there.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <windows.h>

#include <cstdint>

namespace
{
    bool HandleValid(void* h) { return h != nullptr && h != INVALID_HANDLE_VALUE; }

    struct D3D12App
    {
        VriDevice*       device = nullptr;
        VriCoreInterface core {};
        ~D3D12App()
        {
            if (device)
                vriDestroyDevice(device);
        }
    };

    bool Init(D3D12App& app)
    {
        VriDeviceCreationDesc desc {};
        desc.graphicsAPI      = VriGraphicsAPI_D3D12;
        desc.enableValidation = VRI_TRUE;
        desc.bestEffort       = VRI_TRUE;
        desc.enabledFeatures  = VriFeature_ExternalMemory;
        if (vriCreateDevice(&desc, &app.device) != VriResult_Success)
            return false;
        return vriGetInterface(app.device, VRI_INTERFACE_CORE, sizeof(app.core), &app.core) == VriResult_Success;
    }
} // namespace

TEST_CASE("D3D12 external memory: interface availability tracks the reported capability")
{
    D3D12App app;
    if (!Init(app))
    {
        MESSAGE("D3D12 unavailable - skipping external-memory test");
        return;
    }

    const VriDeviceDesc* d = app.core.GetDeviceDesc(app.device);
    VriExternalInterface ext {};
    const bool queryable = vriGetInterface(app.device, VRI_INTERFACE_EXTERNAL, sizeof(ext), &ext) == VriResult_Success;
    CHECK(queryable == (d->hasExternalMemory != VRI_FALSE));
    if (queryable)
    {
        CHECK(ext.CreateExportableBuffer != nullptr);
        CHECK(ext.GetBufferMemoryHandle != nullptr);
        CHECK(ext.CreateExportableFence != nullptr);
        CHECK(ext.GetFenceHandle != nullptr);
    }
}

TEST_CASE("D3D12 external memory: exportable buffer yields a valid shared handle")
{
    D3D12App app;
    if (!Init(app))
    {
        MESSAGE("D3D12 unavailable - skipping external-memory test");
        return;
    }
    const VriCoreInterface& c   = app.core;
    VriDevice*              dev = app.device;
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
    REQUIRE(ext.CreateExportableBuffer(dev, &bd, VriExternalHandleType_D3D12Resource, &buffer) == VriResult_Success);

    VriExternalMemoryInfo info {};
    REQUIRE(ext.GetBufferMemoryHandle(dev, buffer, VriExternalHandleType_D3D12Resource, &info) == VriResult_Success);
    CHECK(HandleValid(info.handle));
    CHECK(info.size >= kSize);
    CHECK(info.dedicated == VRI_TRUE);
    CHECK(CloseHandle(static_cast<HANDLE>(info.handle)) != 0); // caller owns the exported handle

    // A fence handle type for a memory call must be rejected, not crash.
    VriExternalMemoryInfo bad {};
    CHECK(ext.GetBufferMemoryHandle(dev, buffer, VriExternalHandleType_D3D12Fence, &bad) == VriResult_InvalidArgument);

    c.DestroyBuffer(buffer);
}

TEST_CASE("D3D12 external memory: exportable fence yields a valid shared handle")
{
    D3D12App app;
    if (!Init(app))
    {
        MESSAGE("D3D12 unavailable - skipping external-memory test");
        return;
    }
    const VriCoreInterface& c   = app.core;
    VriDevice*              dev = app.device;
    if (c.GetDeviceDesc(dev)->hasExternalMemory == VRI_FALSE)
    {
        MESSAGE("adapter lacks external memory - skipping");
        return;
    }

    VriExternalInterface ext {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_EXTERNAL, sizeof(ext), &ext) == VriResult_Success);

    VriFence* fence = nullptr;
    REQUIRE(ext.CreateExportableFence(dev, 0, VriExternalHandleType_D3D12Fence, &fence) == VriResult_Success);

    void* handle = nullptr;
    REQUIRE(ext.GetFenceHandle(dev, fence, VriExternalHandleType_D3D12Fence, &handle) == VriResult_Success);
    CHECK(HandleValid(handle));
    CHECK(CloseHandle(static_cast<HANDLE>(handle)) != 0);

    // The exportable fence is a normal timeline fence: it works through the core interface.
    CHECK(c.GetFenceValue(fence) == 0);
    c.DestroyFence(fence);
}
