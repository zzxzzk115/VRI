// Opacity micromaps (Metal): explicit "unsupported" contract. Metal has no opacity
// micromap feature, so the Metal backend never reports the capability nor registers
// the interface. This test pins that contract so a future change can't silently
// claim partial support. Mirrors the availability check in test_omm_d3d12.cpp.
// Self-skips when Metal isn't compiled in.
#include <doctest/doctest.h>

#include <vri/vri.h>

TEST_CASE("Metal OMM: unsupported - capability false and interface not queryable")
{
    VriDeviceCreationDesc dc {};
    dc.graphicsAPI      = VriGraphicsAPI_Metal;
    dc.enableValidation = VRI_TRUE;
    dc.bestEffort       = VRI_TRUE;
    dc.enabledFeatures  = VriFeature_RayQuery | VriFeature_OpacityMicromap;
    VriDevice* dev      = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("Metal unavailable - skipping");
        return;
    }
    struct Guard
    {
        VriDevice* d;
        ~Guard() { vriDestroyDevice(d); }
    } guard {dev};

    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    const VriDeviceDesc* d = c.GetDeviceDesc(dev);
    CHECK(d->hasOpacityMicromap == VRI_FALSE);

    VriOpacityMicromapInterface omm {};
    const bool queryable = vriGetInterface(dev, VRI_INTERFACE_OMM, sizeof(omm), &omm) == VriResult_Success;
    CHECK(queryable == false); // Metal: no OMM interface registered
}
