// Phase-0 smoke tests for the VRI entry points and C++ wrapper.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vri/vri.hpp>

TEST_CASE("vriEnumerateAdapters returns a count without crashing")
{
    VriAdapterDesc adapters[8];
    const uint32_t n = vriEnumerateAdapters(VriGraphicsAPI_Vulkan, adapters, 8);
    CHECK(n <= 8u);
}

TEST_CASE("vriCreateDevice rejects null arguments")
{
    CHECK(vriCreateDevice(nullptr, nullptr) == VriResult_InvalidArgument);
}

TEST_CASE("unsupported backend -> Unsupported, no leak via RAII")
{
    VriDeviceCreationDesc desc{};
    // Metal is never compiled in on this platform, so it must report Unsupported
    // regardless of which backends are enabled.
    desc.graphicsAPI = VriGraphicsAPI_Metal;
    desc.bestEffort  = VRI_TRUE;

    auto device = vri::Device::create(desc);
    REQUIRE_FALSE(device.has_value());
    CHECK(device.error() == VriResult_Unsupported);
}
