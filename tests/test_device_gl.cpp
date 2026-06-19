// OpenGL device-creation smoke test (de-risks the GLFW headless context + glad
// loader). Skips gracefully if no GL context can be created.
#include <doctest/doctest.h>

#include <vri/vri.h>

TEST_CASE("OpenGL: device creates and reports a renderer")
{
    VriDeviceCreationDesc desc{};
    desc.graphicsAPI = VriGraphicsAPI_OpenGL;
    desc.bestEffort = VRI_TRUE;

    VriDevice* device = nullptr;
    if (vriCreateDevice(&desc, &device) != VriResult_Success)
    {
        MESSAGE("OpenGL device unavailable - skipping");
        return;
    }

    VriCoreInterface core{};
    REQUIRE(vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(core), &core) == VriResult_Success);

    const VriDeviceDesc* dd = core.GetDeviceDesc(device);
    CHECK(dd->graphicsAPI == VriGraphicsAPI_OpenGL);
    CHECK(dd->adapter.name[0] != '\0');
    CHECK(dd->apiVersionMajor >= 4);
    MESSAGE("OpenGL renderer: ", dd->adapter.name);

    VriQueue* queue = nullptr;
    CHECK(core.GetQueue(device, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    core.DeviceWaitIdle(device);
    vriDestroyDevice(device);
}
