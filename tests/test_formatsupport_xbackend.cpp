// GetFormatSupport contract: the reply has to describe what the backend actually does.
//
// The query exists so callers can probe rather than hard-code -- "is D24S8 usable here,
// or should I ask for D32S8?" is the standard depth+stencil dance. That only works if a
// backend answers per format. Two backends used to answer from a constant instead: the
// OpenGL one returned Texture|ColorAttachment|Blend|VertexBuffer for every format, and
// the WebGPU one Texture|VertexBuffer for every format plus ColorAttachment|Blend for
// anything it mapped. Neither ever set DepthStencil, so probing picked the wrong format
// on OpenGL and got nothing usable on WebGPU -- and both claimed support for formats
// their conversion tables would silently substitute (OpenGL hands out RGBA8 for an
// unmapped format, which then fails framebuffer completeness as a depth attachment).
//
// The checks below are the properties that make the query worth calling, so they hold on
// every backend rather than encoding any one backend's format list.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <string>

namespace
{
    bool Has(VriFormatSupportFlags f, VriFormatSupportBits bit) { return (f & bit) != 0; }

    void Check(VriGraphicsAPI api, const char* name)
    {
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        VriDevice* dev      = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << std::string(name) << "] unavailable - skipped");
            return;
        }
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);

        // 1. Depth formats are depth formats. Every backend implements at least plain
        //    D32_SFLOAT -- the depth tests all render through it -- so the query must say
        //    so, and must not also offer it as a color target.
        const VriFormatSupportFlags d32 = c.GetFormatSupport(dev, VriFormat_D32_SFLOAT);
        CHECK(Has(d32, VriFormatSupport_DepthStencil));
        CHECK_FALSE(Has(d32, VriFormatSupport_ColorAttachment));

        // 2. At least one combined depth+stencil format is reported, so the probe that
        //    picks between them has an answer. (Which one differs: D24S8 everywhere but
        //    Apple hardware, D32S8 there.)
        const bool d24s8 = Has(c.GetFormatSupport(dev, VriFormat_D24_UNORM_S8_UINT), VriFormatSupport_DepthStencil);
        const bool d32s8 = Has(c.GetFormatSupport(dev, VriFormat_D32_SFLOAT_S8_UINT), VriFormatSupport_DepthStencil);
        CHECK((d24s8 || d32s8));
        MESSAGE("[" << std::string(name) << "] depth+stencil: D24S8 " << d24s8 << ", D32S8 " << d32s8);

        // 3. Color formats are color formats. RGBA8_UNORM is the one every backend maps.
        const VriFormatSupportFlags rgba8 = c.GetFormatSupport(dev, VriFormat_RGBA8_UNORM);
        CHECK(Has(rgba8, VriFormatSupport_Texture));
        CHECK(Has(rgba8, VriFormatSupport_ColorAttachment));
        CHECK_FALSE(Has(rgba8, VriFormatSupport_DepthStencil));

        // 4. The reply varies with the format. A constant -- whatever its value -- cannot
        //    satisfy 1 and 3 at once, but state it directly so a future constant is caught
        //    at the point it is introduced rather than through one of the checks above.
        CHECK(d32 != rgba8);

        // 5. Unknown formats are not claimed. VriFormat_Unknown is nothing any backend can
        //    create, so anything other than None here means the reply ignores its argument.
        CHECK(c.GetFormatSupport(dev, VriFormat_Unknown) == VriFormatSupport_None);

        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("Format support: the query answers per format")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
    Check(VriGraphicsAPI_Metal, "Metal");
}
