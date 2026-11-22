// CmdCopyTexture follows its region (offset, size, mip, layer; a zero size is the mip's full extent; no region is the
// whole of mip 0, layer 0) on every backend that comes up. See texture_copy_region_probe.h.
#include "tests/texture_copy_region_probe.h"

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
            MESSAGE("[" << std::string(name) << "] unavailable - skipped");
            return;
        }
        INFO(std::string(name));
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        CHECK(vri::test::CheckTextureCopyRegions(dev, c) == 0);
        // GLES (Android's GL device) cannot read depth back, so a depth copy has nothing to show there.
        if (c.GetDeviceDesc(dev)->graphicsAPI == VriGraphicsAPI_OpenGLES)
            MESSAGE("[" << std::string(name) << "] GLES cannot read depth back - depth copy not checked");
        else
            CHECK(vri::test::CheckDepthTextureCopy(dev, c) == 0);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("Texture copies follow their region: offset, size, mip and layer")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
    Check(VriGraphicsAPI_Metal, "Metal");
}
