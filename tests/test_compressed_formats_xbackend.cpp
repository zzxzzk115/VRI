// The sRGB and signed block-compressed formats: every backend answers GetFormatSupport for them
// from what it maps (D3D12 has every BC one; the GL backend has no compressed formats), and each
// supported one survives an upload/readback round trip byte for byte.
#include "tests/compressed_format_probe.h"

#include <doctest/doctest.h>

#include <vri/vri.h>

#include <string>

namespace
{
    bool IsBC(VriFormat f) { return f != VriFormat_ETC2_RGBA8_SRGB && f != VriFormat_ASTC_4x4_SRGB; }

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
        std::string supported;
        for (const vri::test::CompressedFormat& f : vri::test::kAddedCompressedFormats)
        {
            INFO(name << " " << f.name);
            const bool texture = (c.GetFormatSupport(dev, f.format) & VriFormatSupport_Texture) != 0;
            if (api == VriGraphicsAPI_D3D12)
                CHECK(texture == IsBC(f.format));
            if (api == VriGraphicsAPI_OpenGL)
                CHECK(!texture);
            if (!texture)
                continue;
            supported += std::string(" ") + f.name;
            CHECK(vri::test::CheckCompressedRoundTrip(dev, c, f) == 0);
        }
        MESSAGE("[" << name << "] supported:" << (supported.empty() ? " none" : supported));
        c.DeviceWaitIdle(dev);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("Compressed sRGB/signed formats: support query and upload/readback round trip")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
    Check(VriGraphicsAPI_Metal, "Metal");
}
