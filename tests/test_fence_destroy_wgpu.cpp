// WebGPU: a fence destroyed while a submission that signals it is in flight. Its completion record outlives the
// fence; draining it must not write through the destroyed fence. Under AddressSanitizer a write through it is a
// heap-use-after-free.
#include "tests/fence_destroy_probe.h"

#include <doctest/doctest.h>

#include <vri/vri.h>

TEST_CASE("WebGPU: destroying a fence with a submission in flight leaves the queue usable")
{
    VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
    dc.graphicsAPI           = VriGraphicsAPI_WebGPU;
    dc.enableValidation      = VRI_TRUE;
    dc.bestEffort            = VRI_TRUE;
    VriDevice* dev           = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("WebGPU unavailable - skipping");
        return;
    }
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    for (int round = 0; round < 8; ++round)
        CHECK(vri::test::CheckFenceDestroyedInFlight(dev, c) == 0);
    c.DeviceWaitIdle(dev);
    vriDestroyDevice(dev);
}
