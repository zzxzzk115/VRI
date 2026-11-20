// VriDeviceDesc::nativeValidation tells whether the native API's validation runs. Asked for and
// granted, it is VRI_TRUE; asked for but missing, the device still comes up, a warning says so and
// the flag is VRI_FALSE. Without the request only WebGPU (which always validates) reports it.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdlib>
#include <cstring>
#include <string>

namespace
{
    struct Messages
    {
        int validationWarnings = 0;
    };

    void VRI_CALL OnMessage(void* userArg, VriMessageSeverity severity, const char* message)
    {
        if (severity == VriMessageSeverity_Warning && message && std::strstr(message, "validation was requested"))
            ++static_cast<Messages*>(userArg)->validationWarnings;
    }

    // Creates `api` and returns whether it exists; fills the device's flag and the warnings seen.
    bool Create(VriGraphicsAPI api, bool validation, VriBool& nativeValidation, Messages& messages)
    {
        VriCallbackInterface callbacks = {};
        callbacks.MessageCallback      = OnMessage;
        callbacks.userArg              = &messages;
        VriDeviceCreationDesc dc       = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI                 = api;
        dc.enableValidation            = validation ? VRI_TRUE : VRI_FALSE;
        dc.bestEffort                  = VRI_TRUE;
        dc.callbackInterface           = &callbacks;
        VriDevice* dev                 = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        nativeValidation = c.GetDeviceDesc(dev)->nativeValidation;
        vriDestroyDevice(dev);
        return true;
    }

    void Check(VriGraphicsAPI api, const char* name)
    {
        VriBool  requested = VRI_FALSE;
        Messages requestedMessages;
        if (!Create(api, true, requested, requestedMessages))
        {
            MESSAGE("[" << name << "] unavailable - skipped");
            return;
        }
        INFO(std::string(name));
        // Either the native validation runs, or exactly one warning says it does not.
        CHECK((requested == VRI_TRUE) == (requestedMessages.validationWarnings == 0));
        CHECK(requestedMessages.validationWarnings <= 1);

        VriBool  plain = VRI_TRUE;
        Messages plainMessages;
        REQUIRE(Create(api, false, plain, plainMessages));
        CHECK(plainMessages.validationWarnings == 0);
        if (api == VriGraphicsAPI_WebGPU)
            CHECK(plain == VRI_TRUE);
        else if (api != VriGraphicsAPI_Metal) // Metal follows MTL_DEBUG_LAYER, set outside the program
            CHECK(plain == VRI_FALSE);
    }
} // namespace

TEST_CASE("nativeValidation: reports whether the native API validates")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
    Check(VriGraphicsAPI_Metal, "Metal");
}

#if defined(_WIN32)
TEST_CASE("nativeValidation: a missing Vulkan validation layer is reported")
{
    // The loader hides layers named in VK_LOADER_LAYERS_DISABLE (loader 1.3.234+).
    REQUIRE(_putenv("VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation") == 0);
    VriBool    flag = VRI_TRUE;
    Messages   messages;
    const bool created = Create(VriGraphicsAPI_Vulkan, true, flag, messages);
    REQUIRE(_putenv("VK_LOADER_LAYERS_DISABLE=") == 0);
    if (!created)
    {
        MESSAGE("Vulkan unavailable - skipped");
        return;
    }
    CHECK(flag == VRI_FALSE);
    CHECK(messages.validationWarnings == 1);
}
#endif
