// Live video-memory budget query (VriCoreInterface::GetVideoMemoryInfo). On a real GPU the device
// budget is non-zero and usage never exceeds it; backends/adapters without the query (OpenGL,
// WebGPU, and software Vulkan without VK_EXT_memory_budget) report Unsupported and the test skips.
#include <doctest/doctest.h>

#include <vri/vri.h>

namespace
{
    void Check(VriGraphicsAPI api, const char* name)
    {
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        VriDevice* dev      = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << name << "] unavailable - skipped");
            return;
        }
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);

        VriVideoMemoryInfo info {};
        const VriResult    r = c.GetVideoMemoryInfo(dev, VriMemoryLocation_Device, &info);
        if (r == VriResult_Unsupported)
        {
            MESSAGE("[" << name << "] video-memory query unsupported - skipped");
        }
        else
        {
            REQUIRE(r == VriResult_Success);
            CHECK(info.budget > 0);           // a real device-local heap has a budget
            CHECK(info.usage <= info.budget); // usage never exceeds the budget
            MESSAGE("[" << name << "] VRAM budget " << (info.budget >> 20) << " MiB, usage " << (info.usage >> 20)
                        << " MiB");
        }
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("Video memory: live budget/usage query")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
}
