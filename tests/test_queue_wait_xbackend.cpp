// A queue submission that waits on a fence an earlier submission signals orders the work on the GPU. On every
// backend both submits succeed and the second sees the first's data. WebGPU and GL have one queue, so the wait
// is satisfied by submission order; a wait on a value no submission signals is refused there.
#include "tests/queue_wait_probe.h"

#include <doctest/doctest.h>

#include <vri/vri.h>

#include <string>

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
        INFO(std::string(name));
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        vri::test::OrderedWaitResult result;
        REQUIRE(vri::test::RunOrderedWait(dev, c, result) == 0);
        CHECK(result.submitA == VriResult_Success);
        CHECK(result.submitB == VriResult_Success);
        CHECK(result.dataOk);
        if (api == VriGraphicsAPI_WebGPU || api == VriGraphicsAPI_OpenGL)
            CHECK(vri::test::SubmitUnsignaledWait(dev, c) == VriResult_InvalidArgument);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("Queue waits: a submission waiting on an earlier submission's fence is ordered on the GPU")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
    Check(VriGraphicsAPI_Metal, "Metal");
}
