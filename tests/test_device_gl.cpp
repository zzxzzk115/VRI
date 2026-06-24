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
    // The GL backend reports the resolved profile: desktop GL -> OpenGL, the native ES build
    // (or an ES context) -> OpenGLES. Accept either (an OpenGL request can resolve to ES).
    CHECK((dd->graphicsAPI == VriGraphicsAPI_OpenGL || dd->graphicsAPI == VriGraphicsAPI_OpenGLES));
    CHECK(dd->adapter.name[0] != '\0');
    // Desktop GL requests a 4.x core context; the native OpenGL ES build (EGL) reports the
    // ES version (>= 3.0). Accept both (>= 3) so the one smoke test serves both variants.
    CHECK(dd->apiVersionMajor >= 3);
    MESSAGE("OpenGL renderer: ", dd->adapter.name);

    VriQueue* queue = nullptr;
    CHECK(core.GetQueue(device, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    core.DeviceWaitIdle(device);
    vriDestroyDevice(device);
}
