// Opacity micromaps on D3D12: NOT supported in this build. OMM on D3D12 is DXR 1.2,
// which needs the DirectX Agility SDK headers/runtime (the stock Windows SDK has no
// OMM symbols) and is unsupported by WARP. VRI's contract is "degradation always
// explicit": this test pins that contract so a future Agility-SDK wiring is a
// deliberate, tested change rather than a silent behavior shift.
//
// (Opacity micromaps ARE implemented on Vulkan - see test_omm_vk.cpp.)
#include <doctest/doctest.h>

#include <vri/vri.h>

namespace
{
    bool D3D12Available()
    {
        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = VriGraphicsAPI_D3D12; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success) return false;
        vriDestroyDevice(dev);
        return true;
    }
} // namespace

TEST_CASE("D3D12 OMM: explicitly unsupported - best-effort never grants it")
{
    if (!D3D12Available()) { MESSAGE("D3D12 unavailable - skipping"); return; }

    VriDeviceCreationDesc dc{};
    dc.graphicsAPI = VriGraphicsAPI_D3D12; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
    dc.enabledFeatures = VriFeature_OpacityMicromap;
    VriDevice* dev = nullptr;
    REQUIRE(vriCreateDevice(&dc, &dev) == VriResult_Success); // best-effort: device still creates

    VriCoreInterface core{};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(core), &core) == VriResult_Success);
    const VriDeviceDesc* d = core.GetDeviceDesc(dev);
    CHECK(d->hasOpacityMicromap == VRI_FALSE); // not granted

    VriOpacityMicromapInterface omm{};
    CHECK(vriGetInterface(dev, VRI_INTERFACE_OMM, sizeof(omm), &omm) != VriResult_Success); // not queryable

    vriDestroyDevice(dev);
}

TEST_CASE("D3D12 OMM: fail-fast (bestEffort=false) rejects it with Unsupported")
{
    if (!D3D12Available()) { MESSAGE("D3D12 unavailable - skipping"); return; }

    VriDeviceCreationDesc dc{};
    dc.graphicsAPI = VriGraphicsAPI_D3D12; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_FALSE;
    dc.enabledFeatures = VriFeature_OpacityMicromap;
    VriDevice* dev = nullptr;
    CHECK(vriCreateDevice(&dc, &dev) == VriResult_Unsupported);
    CHECK(dev == nullptr);
}
