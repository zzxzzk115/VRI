// Direct3D 12 device-creation smoke test (de-risks DXGI factory + adapter pick +
// D3D12CreateDevice + command queue). Falls back to the WARP software adapter, so it
// runs on machines without a GPU; skips gracefully only if even WARP is unavailable.
#include <doctest/doctest.h>

#include <vri/vri.h>

TEST_CASE("D3D12: device creates and reports an adapter")
{
    VriDeviceCreationDesc desc{};
    desc.graphicsAPI = VriGraphicsAPI_D3D12;
    desc.bestEffort = VRI_TRUE;

    VriDevice* device = nullptr;
    if (vriCreateDevice(&desc, &device) != VriResult_Success)
    {
        MESSAGE("D3D12 device unavailable - skipping");
        return;
    }

    VriCoreInterface core{};
    REQUIRE(vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(core), &core) == VriResult_Success);

    const VriDeviceDesc* dd = core.GetDeviceDesc(device);
    CHECK(dd->graphicsAPI == VriGraphicsAPI_D3D12);
    CHECK(dd->adapter.name[0] != '\0');
    CHECK(dd->apiVersionMajor == 12);
    CHECK(dd->attachmentColorMaxNum >= 8);
    MESSAGE("D3D12 adapter: ", dd->adapter.name);

    VriQueue* queue = nullptr;
    CHECK(core.GetQueue(device, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    vriDestroyDevice(device);
}
