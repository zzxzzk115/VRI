// Swapchain formats on hidden Win32 windows (never shown): a backend that cannot present a requested format returns
// VriResult_Unsupported, creates nothing and logs no error, so a caller can try formats in turn; a format it can
// present is created as before. Vulkan is covered in test_swapchain_vk.cpp.
#include "swapchain_present_once.h"

#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <initializer_list>
#include <string>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace
{
    struct Messages
    {
        int errors = 0;
    };

    void VRI_CALL CountErrors(void* userArg, VriMessageSeverity severity, const char* message)
    {
        if (severity != VriMessageSeverity_Error)
            return;
        ++static_cast<Messages*>(userArg)->errors;
        MESSAGE(std::string(message));
    }

    // GL sets the window's pixel format, which a window takes only once, so each swapchain gets its own window.
    HWND CreateHiddenWindow()
    {
        return CreateWindowExA(
            0, "STATIC", "vri-test", WS_POPUP, 0, 0, 64, 32, nullptr, nullptr, GetModuleHandleA(nullptr), nullptr);
    }

    VriResult TryCreate(const VriSwapChainInterface& sw,
                        VriDevice*                   dev,
                        VriQueue*                    queue,
                        HWND                         hwnd,
                        VriFormat                    format,
                        VriSwapChain**               out)
    {
        VriSwapChainDesc sd {};
        sd.window.type                   = VriWindowSystem_Win32;
        sd.window.handle.win32.hwnd      = hwnd;
        sd.window.handle.win32.hinstance = GetModuleHandleA(nullptr);
        sd.queue                         = queue;
        sd.format                        = format;
        sd.width                         = 64;
        sd.height                        = 32;
        sd.textureNum                    = 2;
        return sw.CreateSwapChain(dev, &sd, out);
    }

    void
    Check(VriGraphicsAPI api, const std::string& name, std::initializer_list<VriFormat> refused, VriFormat accepted)
    {
        CAPTURE(name);
        Messages              messages;
        VriCallbackInterface  callbacks = {&messages, CountErrors};
        VriDeviceCreationDesc dc        = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI                  = api;
        dc.callbackInterface            = &callbacks;
        VriDevice* dev                  = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE(name << " unavailable - skipping");
            return;
        }
        VriCoreInterface      c {};
        VriSwapChainInterface sw {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_SWAPCHAIN, sizeof(sw), &sw) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        for (const VriFormat format : refused)
        {
            CAPTURE(format);
            HWND hwnd = CreateHiddenWindow();
            REQUIRE(hwnd != nullptr);
            VriSwapChain* swapChain = nullptr;
            CHECK(TryCreate(sw, dev, queue, hwnd, format, &swapChain) == VriResult_Unsupported);
            CHECK(swapChain == nullptr);
            if (swapChain)
                sw.DestroySwapChain(swapChain);
            DestroyWindow(hwnd);
        }
        CHECK(messages.errors == 0);

        HWND hwnd = CreateHiddenWindow();
        REQUIRE(hwnd != nullptr);
        VriSwapChain* swapChain = nullptr;
        CHECK(TryCreate(sw, dev, queue, hwnd, accepted, &swapChain) == VriResult_Success);
        if (swapChain)
        {
            vri::test::PresentOnce(dev, c, sw, swapChain); // see swapchain_present_once.h
            sw.DestroySwapChain(swapChain);
        }
        DestroyWindow(hwnd);
        vriDestroyDevice(dev);
        CHECK(messages.errors == 0);
    }
} // namespace

TEST_CASE("Swapchain formats: D3D12 refuses formats no flip-model swapchain takes")
{
    // R8 has a DXGI equivalent but no flip-model swapchain takes it.
    Check(VriGraphicsAPI_D3D12, "D3D12", {VriFormat_R8_UNORM}, VriFormat_BGRA8_UNORM);
}

TEST_CASE("Swapchain formats: GL refuses formats its window framebuffer does not hold")
{
    // GL presents through the window's 8-bit RGBA framebuffer and has no BGRA sRGB format.
    Check(VriGraphicsAPI_OpenGL, "OpenGL", {VriFormat_R8_UNORM, VriFormat_BGRA8_SRGB}, VriFormat_RGBA8_UNORM);
    // An sRGB swapchain presents through the encoding draw: its program builds and the present logs no error. What
    // reaches the screen is measured by hand (DETAIL.md, "Swapchain tests").
    Check(VriGraphicsAPI_OpenGL, "OpenGL sRGB", {}, VriFormat_RGBA8_SRGB);
}

TEST_CASE("Swapchain formats: WebGPU refuses formats its surface does not list")
{
    // A WebGPU surface lists no single-channel format.
    Check(VriGraphicsAPI_WebGPU, "WebGPU", {VriFormat_R8_UNORM}, VriFormat_BGRA8_UNORM);
}
