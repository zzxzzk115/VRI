#include <vri/vri.h>

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>

extern "C" int vriTestTimelineC(VriDevice* device, size_t coreSize);

namespace
{
    void VRI_CALL Diagnostic(void* user, VriMessageSeverity severity, const char* message)
    {
        if (severity == VriMessageSeverity_Error)
        {
            ++*static_cast<uint32_t*>(user);
            std::fprintf(stderr, "[timeline test] %s\n", message);
        }
    }
} // namespace

TEST_CASE("timeline C ABI preserves timeout values and observes real completion")
{
    for (VriGraphicsAPI api : {VriGraphicsAPI_Vulkan,
                               VriGraphicsAPI_OpenGL,
                               VriGraphicsAPI_WebGPU,
                               VriGraphicsAPI_D3D12,
                               VriGraphicsAPI_Metal})
    {
        for (VriBool validation : {VRI_FALSE, VRI_TRUE})
        {
            CAPTURE(api);
            CAPTURE(validation);
            uint32_t             errors = 0;
            VriCallbackInterface callback {&errors, Diagnostic};
            auto                 creation = vriDefaultDeviceCreationDesc();
            creation.graphicsAPI          = api;
            creation.enableValidation     = validation;
            creation.callbackInterface    = &callback;
            VriDevice*      device        = nullptr;
            const VriResult result        = vriCreateDevice(&creation, &device);
            if (result == VriResult_Unsupported)
            {
                MESSAGE("Timeline backend unavailable: ", api);
                continue;
            }
            REQUIRE(result == VriResult_Success);
            CHECK(vriTestTimelineC(device, sizeof(VriCoreInterface)) == 0);
            vriDestroyDevice(device);
            CHECK(errors == 0);
        }
    }
}

TEST_CASE("submission rejects foreign fences before touching a native queue")
{
    for (VriGraphicsAPI api : {VriGraphicsAPI_Vulkan,
                               VriGraphicsAPI_OpenGL,
                               VriGraphicsAPI_WebGPU,
                               VriGraphicsAPI_D3D12,
                               VriGraphicsAPI_Metal})
    {
        for (VriBool validation : {VRI_FALSE, VRI_TRUE})
        {
            CAPTURE(api);
            CAPTURE(validation);
            auto creation             = vriDefaultDeviceCreationDesc();
            creation.graphicsAPI      = api;
            creation.enableValidation = validation;
            VriDevice*      first     = nullptr;
            const VriResult result    = vriCreateDevice(&creation, &first);
            if (result == VriResult_Unsupported)
            {
                MESSAGE("Submission backend unavailable: ", api);
                continue;
            }
            REQUIRE(result == VriResult_Success);
            VriDevice* second = nullptr;
            REQUIRE(vriCreateDevice(&creation, &second) == VriResult_Success);
            VriCoreInterface firstCore {};
            VriCoreInterface secondCore {};
            REQUIRE(vriGetInterface(first, VRI_INTERFACE_CORE, sizeof(firstCore), &firstCore) == VriResult_Success);
            REQUIRE(vriGetInterface(second, VRI_INTERFACE_CORE, sizeof(secondCore), &secondCore) == VriResult_Success);
            VriFence* foreign = nullptr;
            REQUIRE(secondCore.CreateFence(second, 0, &foreign) == VriResult_Success);
            VriQueue* queue = nullptr;
            REQUIRE(firstCore.GetQueue(first, VriQueueType_Graphics, 0, &queue) == VriResult_Success);
            VriFenceSubmitDesc signal {foreign, 1, VriPipelineStage_AllCommands};
            VriQueueSubmitDesc submit {};
            submit.signalFences   = &signal;
            submit.signalFenceNum = 1;
            CHECK(firstCore.QueueSubmit(queue, &submit) == VriResult_InvalidArgument);
            CHECK(secondCore.GetFenceValue(foreign) == 0);
            secondCore.DestroyFence(foreign);
            vriDestroyDevice(second);
            vriDestroyDevice(first);
        }
    }
}
