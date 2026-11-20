// MapBufferAsync off the browser: every backend maps on the calling thread, so the callback has run
// by the time the call returns, exactly once, and reads back what the GPU copied.
#include "tests/map_async_probe.h"

#include <doctest/doctest.h>

#include <vri/vri.h>

namespace
{
    void Check(VriGraphicsAPI api, const char* name)
    {
        VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI           = api;
        dc.enableValidation      = VRI_TRUE;
        dc.bestEffort            = VRI_TRUE;
        VriDevice* dev           = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << name << "] unavailable - skipped");
            return;
        }
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        INFO(name);
        CHECK(vri::test::CheckMapBufferAsync(dev, c, false) == 0);
        c.DeviceWaitIdle(dev);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("MapBufferAsync: readback round trip, callback before the call returns")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
    Check(VriGraphicsAPI_Metal, "Metal");
}
