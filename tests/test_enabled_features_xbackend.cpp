// VriDeviceDesc::enabledFeatures is what the device grants, on every backend: a subset of the
// request, matching the has* flag of each feature. Without bestEffort a backend either grants every
// requested feature or fails with VriResult_Unsupported.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <string>

namespace
{
    // Every VriFeatureBits bit.
    constexpr uint64_t kAllFeatures = (1ull << 10) - 1;

    struct FlagForFeature
    {
        VriFeatureBits feature;
        VriBool VriDeviceDesc::* flag;
    };
    const FlagForFeature kFlags[] = {
        {VriFeature_RayTracing, &VriDeviceDesc::hasRayTracing},
        {VriFeature_MeshShader, &VriDeviceDesc::hasMeshShader},
        {VriFeature_Bindless, &VriDeviceDesc::hasBindless},
        {VriFeature_VariableShadingRate, &VriDeviceDesc::hasVariableShadingRate},
        {VriFeature_OpacityMicromap, &VriDeviceDesc::hasOpacityMicromap},
        {VriFeature_RayQuery, &VriDeviceDesc::hasRayQuery},
        {VriFeature_ExternalMemory, &VriDeviceDesc::hasExternalMemory},
        {VriFeature_CooperativeVector, &VriDeviceDesc::hasCooperativeVector},
        {VriFeature_ShaderAtomicInt64, &VriDeviceDesc::hasShaderAtomicInt64},
    };

    VriResult Create(VriGraphicsAPI api, uint64_t features, bool bestEffort, VriDevice** dev)
    {
        VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI           = api;
        dc.enabledFeatures       = features;
        dc.bestEffort            = bestEffort ? VRI_TRUE : VRI_FALSE;
        *dev                     = nullptr;
        return vriCreateDevice(&dc, dev);
    }

    void Check(VriGraphicsAPI api, const char* name)
    {
        VriDevice* dev = nullptr;
        if (Create(api, kAllFeatures, true, &dev) != VriResult_Success)
        {
            MESSAGE("[" << name << "] unavailable - skipped");
            return;
        }
        INFO(std::string(name));
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        const VriDeviceDesc desc    = *c.GetDeviceDesc(dev);
        const uint64_t      granted = desc.enabledFeatures;
        vriDestroyDevice(dev);
        CHECK((granted & ~kAllFeatures) == 0);
        for (const FlagForFeature& f : kFlags)
        {
            INFO("feature bit " << static_cast<uint64_t>(f.feature));
            CHECK((desc.*f.flag == VRI_TRUE) == ((granted & f.feature) != 0));
        }

        // Without bestEffort: the granted set comes back whole; one more feature than that fails.
        REQUIRE(Create(api, granted, false, &dev) == VriResult_Success);
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        CHECK(c.GetDeviceDesc(dev)->enabledFeatures == granted);
        vriDestroyDevice(dev);
        const uint64_t missing = kAllFeatures & ~granted;
        if (missing)
        {
            CHECK(Create(api, missing & (~missing + 1), false, &dev) == VriResult_Unsupported);
            CHECK(dev == nullptr);
        }
    }
} // namespace

TEST_CASE("enabledFeatures: the grant, matching the has* flags; bestEffort=false fails on a missing feature")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
    Check(VriGraphicsAPI_Metal, "Metal");
}
