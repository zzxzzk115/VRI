// Buffer <-> texture copies honor bufferRowLength and bufferImageHeight: an odd-width R16F texture and a
// 3D texture go up from padded buffer rows and slices and come back into the same layout, byte for byte.
#include "tests/buffer_texture_copy_probe.h"

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
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        for (const vri::test::CopyCase& k : vri::test::kCopyCases)
        {
            INFO(std::string(name) << " format " << static_cast<int>(k.format) << " " << k.width << "x" << k.height
                                   << "x" << k.depth);
            if (!vri::test::CopyCaseSupported(dev, c, k))
            {
                MESSAGE(std::string(name)
                        << " format " << static_cast<int>(k.format) << " cannot be read back - skipped");
                continue;
            }
            CHECK(vri::test::CheckBufferTextureRoundTrip(dev, c, k) == 0);
        }
        c.DeviceWaitIdle(dev);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("Buffer-texture copies: padded rows and slices, odd width and 3D")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
    Check(VriGraphicsAPI_Metal, "Metal");
}
