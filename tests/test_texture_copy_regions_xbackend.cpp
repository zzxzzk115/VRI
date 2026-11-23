// CmdCopyTexture follows its region (offset, size, mip, layer; a zero size is the mip's full extent; no region is the
// whole of mip 0, layer 0) on every backend that comes up, and copies within one texture between different mips or
// layers while refusing a copy whose source and destination share a subresource. See texture_copy_region_probe.h.
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

TEST_CASE("Texture copies within one texture: different mips or layers copy, a shared subresource is refused")
{
    const auto check = [](VriGraphicsAPI api, const char* name) {
        vri::test::CopyErrors errors;
        VriCallbackInterface  callbacks = {};
        callbacks.MessageCallback       = vri::test::RecordCopyError;
        callbacks.userArg               = &errors;
        VriDeviceCreationDesc dc        = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI                  = api;
        dc.enableValidation             = VRI_TRUE;
        dc.bestEffort                   = VRI_TRUE;
        dc.callbackInterface            = &callbacks;
        VriDevice* dev                  = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << std::string(name) << "] unavailable - skipped");
            return;
        }
        INFO(std::string(name));
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        CHECK(vri::test::CheckCopyWithinTexture(dev, c, errors) == 0);
        vriDestroyDevice(dev);
    };
    check(VriGraphicsAPI_Vulkan, "Vulkan");
    check(VriGraphicsAPI_D3D12, "D3D12");
    check(VriGraphicsAPI_OpenGL, "OpenGL");
    check(VriGraphicsAPI_WebGPU, "WebGPU");
    check(VriGraphicsAPI_Metal, "Metal");
}
