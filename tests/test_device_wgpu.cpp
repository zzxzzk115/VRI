// WebGPU device-creation smoke test (de-risks the package, the async
// adapter/device request, and the wgpu_native runtime DLL). Skips gracefully if
// no WebGPU adapter is available.
#include <doctest/doctest.h>

#include <vri/vri.h>

TEST_CASE("WebGPU: device creates and reports an adapter")
{
    VriDeviceCreationDesc desc{};
    desc.graphicsAPI = VriGraphicsAPI_WebGPU;
    desc.bestEffort = VRI_TRUE;

    VriDevice* device = nullptr;
    if (vriCreateDevice(&desc, &device) != VriResult_Success)
    {
        MESSAGE("WebGPU device unavailable - skipping");
        return;
    }

    VriCoreInterface core{};
    REQUIRE(vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(core), &core) == VriResult_Success);

    const VriDeviceDesc* dd = core.GetDeviceDesc(device);
    CHECK(dd->graphicsAPI == VriGraphicsAPI_WebGPU);
    CHECK(dd->adapter.name[0] != '\0');
    MESSAGE("WebGPU adapter: ", dd->adapter.name);

    VriQueue* queue = nullptr;
    CHECK(core.GetQueue(device, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    core.DeviceWaitIdle(device);
    vriDestroyDevice(device);
}
