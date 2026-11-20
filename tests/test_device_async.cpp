// vriCreateDeviceAsync off the browser: every native backend creates its device before the call
// returns, so the callback has run by then, exactly once, with the same result vriCreateDevice gives.
#include <doctest/doctest.h>

#include <vri/vri.hpp>

namespace
{
    struct Outcome
    {
        int        calls  = 0;
        VriResult  result = VriResult_Failure;
        VriDevice* device = nullptr;
    };

    void VRI_CALL Record(void* userData, VriResult result, VriDevice* device)
    {
        auto* outcome = static_cast<Outcome*>(userData);
        ++outcome->calls;
        outcome->result = result;
        outcome->device = device;
    }
} // namespace

TEST_CASE("vriCreateDeviceAsync: a native backend answers before the call returns")
{
    VriDeviceCreationDesc desc = vriDefaultDeviceCreationDesc();
    desc.graphicsAPI           = VriGraphicsAPI_Auto;
    desc.bestEffort            = VRI_TRUE;

    Outcome outcome;
    REQUIRE(vriCreateDeviceAsync(&desc, Record, &outcome) == VriResult_Success);
    REQUIRE(outcome.calls == 1);
    if (outcome.result != VriResult_Success)
    {
        MESSAGE("no backend could create a device - skipping");
        return;
    }
    REQUIRE(outcome.device != nullptr);

    VriCoreInterface core {};
    REQUIRE(vriGetInterface(outcome.device, VRI_INTERFACE_CORE, sizeof(core), &core) == VriResult_Success);
    CHECK(core.GetDeviceDesc(outcome.device)->graphicsAPI != VriGraphicsAPI_Auto);
    vriDestroyDevice(outcome.device);
}

TEST_CASE("vriCreateDeviceAsync: WebGPU through the async path")
{
    VriDeviceCreationDesc desc  = vriDefaultDeviceCreationDesc();
    desc.graphicsAPI            = VriGraphicsAPI_WebGPU;
    desc.bestEffort             = VRI_TRUE;
    desc.coordinates.clipSpaceY = VriClipSpaceY_Up;

    Outcome outcome;
    REQUIRE(vriCreateDeviceAsync(&desc, Record, &outcome) == VriResult_Success);
    REQUIRE(outcome.calls == 1);
    if (outcome.result != VriResult_Success)
    {
        // Unsupported when the WebGPU backend is not built, Failure when no adapter exists.
        CHECK(outcome.device == nullptr);
        MESSAGE("WebGPU device unavailable - skipping");
        return;
    }
    VriCoreInterface core {};
    REQUIRE(vriGetInterface(outcome.device, VRI_INTERFACE_CORE, sizeof(core), &core) == VriResult_Success);
    const VriDeviceDesc* dd = core.GetDeviceDesc(outcome.device);
    CHECK(dd->graphicsAPI == VriGraphicsAPI_WebGPU);
    // The conventions are copied when the request starts, not when the device arrives.
    CHECK(dd->coordinates.clipSpaceY == VriClipSpaceY_Up);
    vriDestroyDevice(outcome.device);
}

TEST_CASE("vriCreateDeviceAsync: invalid requests")
{
    VriDeviceCreationDesc desc = vriDefaultDeviceCreationDesc();
    CHECK(vriCreateDeviceAsync(nullptr, Record, nullptr) == VriResult_InvalidArgument);
    CHECK(vriCreateDeviceAsync(&desc, nullptr, nullptr) == VriResult_InvalidArgument);

    // A desc the creation itself rejects is answered through the callback.
    desc.structSize = 0;
    Outcome outcome;
    REQUIRE(vriCreateDeviceAsync(&desc, Record, &outcome) == VriResult_Success);
    CHECK(outcome.calls == 1);
    CHECK(outcome.result == VriResult_InvalidArgument);
    CHECK(outcome.device == nullptr);
}

TEST_CASE("vri::Device::createAsync hands over an owning Device")
{
    VriDeviceCreationDesc desc = vriDefaultDeviceCreationDesc();
    desc.graphicsAPI           = VriGraphicsAPI_Auto;
    desc.bestEffort            = VRI_TRUE;

    int               calls = 0;
    vri::Device       device;
    const vri::Result r = vri::Device::createAsync(desc, [&](std::expected<vri::Device, vri::Result> created) {
        ++calls;
        if (created)
            device = std::move(*created);
    });
    CHECK(r == VriResult_Success);
    CHECK(calls == 1);
    if (!device)
        MESSAGE("no backend could create a device - skipping");
}
