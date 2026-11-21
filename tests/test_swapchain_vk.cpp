// Vulkan swapchains on a hidden Win32 window (never shown).
// - Present(waitFence, waitValue) orders the present after the render work on the GPU: the frame is submitted and
//   presented with no host wait in between, under synchronization validation, which reports a present that no
//   semaphore orders after the image's last write.
// - A format the surface does not list is refused instead of replaced.
#include "swapchain_present_once.h"

#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace
{
    constexpr uint32_t kW      = 64;
    constexpr uint32_t kH      = 32;
    constexpr uint32_t kFrames = 6;

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

    HWND CreateHiddenWindow()
    {
        // A plain STATIC window needs no class registration; it is never shown. WS_POPUP has no frame, so the
        // client area, which the surface extent follows, is the whole window.
        return CreateWindowExA(
            0, "STATIC", "vri-test", WS_POPUP, 0, 0, kW, kH, nullptr, nullptr, GetModuleHandleA(nullptr), nullptr);
    }

    VriSwapChainDesc SwapChainDesc(HWND hwnd, VriQueue* queue, VriFormat format)
    {
        VriSwapChainDesc sd {};
        sd.window.type                   = VriWindowSystem_Win32;
        sd.window.handle.win32.hwnd      = hwnd;
        sd.window.handle.win32.hinstance = GetModuleHandleA(nullptr);
        sd.queue                         = queue;
        sd.format                        = format;
        sd.width                         = kW;
        sd.height                        = kH;
        sd.textureNum                    = 2;
        return sd;
    }

    // Validation (and, when `syncValidation`, synchronization validation) on; null when Vulkan is unavailable.
    VriDevice* CreateDevice(Messages& messages, VriCallbackInterface& callbacks, bool syncValidation)
    {
        callbacks                = {&messages, CountErrors};
        VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI           = VriGraphicsAPI_Vulkan;
        dc.enableValidation      = VRI_TRUE;
        dc.callbackInterface     = &callbacks;
        // The validation layer reads its settings when the instance is created.
        if (syncValidation)
            REQUIRE(_putenv("VK_KHRONOS_VALIDATION_VALIDATE_SYNC=1") == 0);
        VriDevice* device = nullptr;
        if (vriCreateDevice(&dc, &device) != VriResult_Success)
            device = nullptr;
        if (syncValidation)
            REQUIRE(_putenv("VK_KHRONOS_VALIDATION_VALIDATE_SYNC=") == 0);
        return device;
    }

    struct Frame
    {
        VriDescriptor*       view     = nullptr;
        VriBuffer*           readback = nullptr;
        VriCommandAllocator* alloc    = nullptr;
        VriCommandBuffer*    cmd      = nullptr;
    };

    // The red value frame `i` clears its image to.
    float FrameRed(uint32_t i) { return static_cast<float>(i + 1) / static_cast<float>(kFrames); }

    // Records: clear `backbuffer` to `red`, copy it to the readback buffer, leave it in the present layout.
    void RecordFrame(VriDevice* dev, VriCoreInterface& c, VriTexture* backbuffer, float red, Frame& f)
    {
        VriTextureViewDesc vd {};
        vd.texture  = backbuffer;
        vd.viewType = VriTextureViewType_2D;
        vd.aspect   = VriImageAspect_Color;
        vd.format   = VriFormat_BGRA8_UNORM;
        vd.mipNum   = 1;
        vd.layerNum = 1;
        REQUIRE(c.CreateTextureView(dev, &vd, &f.view) == VriResult_Success);
        VriBufferDesc bd {};
        bd.size           = static_cast<uint64_t>(kW) * kH * 4;
        bd.usage          = VriBufferUsage_TransferDst;
        bd.memoryLocation = VriMemoryLocation_HostReadback;
        REQUIRE(c.CreateBuffer(dev, &bd, &f.readback) == VriResult_Success);
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &f.alloc) == VriResult_Success);
        REQUIRE(c.CreateCommandBuffer(f.alloc, &f.cmd) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(f.cmd) == VriResult_Success);
        VriTextureBarrierDesc tb {};
        tb.texture       = backbuffer;
        tb.before.layout = VriLayout_Undefined;
        tb.after.layout  = VriLayout_ColorAttachment;
        tb.after.access  = VriAccess_ColorAttachmentWrite;
        tb.after.stages  = VriPipelineStage_ColorAttachmentOutput;
        tb.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc g {};
        g.textures   = &tb;
        g.textureNum = 1;
        c.CmdBarrier(f.cmd, &g);
        VriAttachmentDesc rt {};
        rt.view                    = f.view;
        rt.loadOp                  = VriAttachmentLoadOp_Clear;
        rt.storeOp                 = VriAttachmentStoreOp_Store;
        rt.clearValue.color.f32[0] = red;
        rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att {};
        att.colors            = &rt;
        att.colorNum          = 1;
        att.renderArea.width  = kW;
        att.renderArea.height = kH;
        att.layerNum          = 1;
        c.CmdBeginRendering(f.cmd, &att);
        c.CmdEndRendering(f.cmd);
        tb.before       = tb.after;
        tb.after.layout = VriLayout_CopySource;
        tb.after.access = VriAccess_CopySourceRead;
        tb.after.stages = VriPipelineStage_Transfer;
        c.CmdBarrier(f.cmd, &g);
        VriBufferTextureCopyDesc cp {};
        cp.texture.aspect   = VriImageAspect_Color;
        cp.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(f.cmd, f.readback, backbuffer, &cp);
        tb.before       = tb.after;
        tb.after.layout = VriLayout_Present;
        tb.after.access = VriAccess_None;
        tb.after.stages = VriPipelineStage_None;
        c.CmdBarrier(f.cmd, &g);
        REQUIRE(c.EndCommandBuffer(f.cmd) == VriResult_Success);
    }

    void DestroyFrame(VriCoreInterface& c, Frame& f)
    {
        c.DestroyCommandAllocator(f.alloc);
        c.DestroyDescriptor(f.view);
        c.DestroyBuffer(f.readback);
    }
} // namespace

TEST_CASE("Vulkan: Present waits on the GPU for its fence")
{
    Messages             messages;
    VriCallbackInterface callbacks {};
    VriDevice*           dev = CreateDevice(messages, callbacks, true);
    if (!dev)
    {
        MESSAGE("Vulkan unavailable - skipping");
        return;
    }
    VriCoreInterface      c {};
    VriSwapChainInterface sw {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_SWAPCHAIN, sizeof(sw), &sw) == VriResult_Success);
    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);
    HWND hwnd = CreateHiddenWindow();
    REQUIRE(hwnd != nullptr);
    VriSwapChainDesc sd        = SwapChainDesc(hwnd, queue, VriFormat_BGRA8_UNORM);
    VriSwapChain*    swapChain = nullptr;
    REQUIRE(sw.CreateSwapChain(dev, &sd, &swapChain) == VriResult_Success);
    uint32_t textureNum = 0;
    REQUIRE(sw.GetSwapChainTextures(swapChain, nullptr, &textureNum) == VriResult_Success);
    VriTexture* textures[8] = {};
    REQUIRE(textureNum <= 8);
    REQUIRE(sw.GetSwapChainTextures(swapChain, textures, &textureNum) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    // More frames than images, so every image is presented, acquired again and rendered again.
    Frame frames[kFrames];
    for (uint32_t i = 0; i < kFrames; ++i)
    {
        CAPTURE(i);
        uint32_t index = 0;
        REQUIRE(sw.AcquireNextTexture(swapChain, nullptr, 0, &index) == VriResult_Success);
        REQUIRE(index < textureNum);
        const float red = FrameRed(i);
        RecordFrame(dev, c, textures[index], red, frames[i]);
        VriFenceSubmitDesc signal {};
        signal.fence = fence;
        signal.value = i + 1;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &frames[i].cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        REQUIRE(c.QueueSubmit(queue, &submit) == VriResult_Success);
        // No host wait between the submission and the present.
        REQUIRE(sw.Present(swapChain, fence, i + 1) == VriResult_Success);
    }
    c.Wait(fence, kFrames);
    REQUIRE(c.DeviceWaitIdle(dev) == VriResult_Success);
    for (uint32_t i = 0; i < kFrames; ++i)
    {
        CAPTURE(i);
        // The image each present showed holds that frame's clear (BGRA: red is byte 2).
        const auto* px = static_cast<const uint8_t*>(c.MapBuffer(frames[i].readback, 0, 4));
        REQUIRE(px != nullptr);
        const auto expected = static_cast<int>(std::lround(FrameRed(i) * 255.0f));
        CHECK(std::abs(px[2] - expected) <= 1);
        c.UnmapBuffer(frames[i].readback);
        DestroyFrame(c, frames[i]);
    }
    c.DestroyFence(fence);
    sw.DestroySwapChain(swapChain);
    DestroyWindow(hwnd);
    vriDestroyDevice(dev);
    CHECK(messages.errors == 0);
}

TEST_CASE("Vulkan: a swapchain format the surface does not list is refused")
{
    Messages             messages;
    VriCallbackInterface callbacks {};
    VriDevice*           dev = CreateDevice(messages, callbacks, false);
    if (!dev)
    {
        MESSAGE("Vulkan unavailable - skipping");
        return;
    }
    VriCoreInterface      c {};
    VriSwapChainInterface sw {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_SWAPCHAIN, sizeof(sw), &sw) == VriResult_Success);
    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);
    HWND hwnd = CreateHiddenWindow();
    REQUIRE(hwnd != nullptr);

    // No Windows surface lists a single-channel format.
    VriSwapChainDesc sd        = SwapChainDesc(hwnd, queue, VriFormat_R8_UNORM);
    VriSwapChain*    swapChain = nullptr;
    CHECK(sw.CreateSwapChain(dev, &sd, &swapChain) == VriResult_Unsupported);
    CHECK(swapChain == nullptr);
    if (swapChain)
        sw.DestroySwapChain(swapChain);

    // Every Windows surface lists BGRA8_UNORM.
    swapChain = nullptr;
    sd        = SwapChainDesc(hwnd, queue, VriFormat_BGRA8_UNORM);
    REQUIRE(sw.CreateSwapChain(dev, &sd, &swapChain) == VriResult_Success);
    // Each swapchain presents before it is destroyed, Resize destroying the first: see swapchain_present_once.h.
    vri::test::PresentOnce(dev, c, sw, swapChain);
    CHECK(sw.Resize(swapChain, kW, kH) == VriResult_Success);
    vri::test::PresentOnce(dev, c, sw, swapChain);
    sw.DestroySwapChain(swapChain);

    DestroyWindow(hwnd);
    vriDestroyDevice(dev);
    CHECK(messages.errors == 0);
}
