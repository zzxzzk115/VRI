// VRI Validation layer: misuse is reported through the message callback and the
// offending call is suppressed (not forwarded), so it can't crash the backend.
// Here CmdDraw is issued outside a render pass; the layer must flag it. Correct
// usage stays silent (proven by every other test running with validation on).
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

namespace
{
    struct MsgState { int errors = 0; };
    void VRI_CALL OnMessage(void* userArg, VriMessageSeverity severity, const char* /*message*/)
    {
        if (severity == VriMessageSeverity_Error)
            static_cast<MsgState*>(userArg)->errors++;
    }

    // Create a validated device on the first available backend; returns nullptr if none.
    VriDevice* CreateAnyValidated(const VriCallbackInterface* cb)
    {
        const VriGraphicsAPI apis[] = {VriGraphicsAPI_Vulkan, VriGraphicsAPI_WebGPU, VriGraphicsAPI_OpenGL};
        for (VriGraphicsAPI api : apis)
        {
            VriDeviceCreationDesc dc{};
            dc.graphicsAPI = api; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE; dc.callbackInterface = cb;
            VriDevice* dev = nullptr;
            if (vriCreateDevice(&dc, &dev) == VriResult_Success)
                return dev;
        }
        return nullptr;
    }
}

TEST_CASE("VRI Validation flags a draw issued outside a render pass")
{
    MsgState state;
    VriCallbackInterface cb{}; cb.userArg = &state; cb.MessageCallback = OnMessage;

    VriDevice* dev = CreateAnyValidated(&cb);
    if (!dev) { MESSAGE("no backend available - skipped"); return; }

    VriCoreInterface c{};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);

    VriCommandAllocator* alloc = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
    const int before = state.errors;
    VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 1;
    c.CmdDraw(cmd, &draw); // ERROR: no active render pass -> reported + suppressed
    CHECK(state.errors > before);
    REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

    c.DestroyCommandAllocator(alloc);
    vriDestroyDevice(dev);
}
