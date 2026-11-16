// D3D12 sRGB swapchains: flip-model swapchains reject sRGB buffer formats, so the backend creates
// UNORM buffers and renders through sRGB render-target views. The swapchain lives on a hidden
// Win32 window (never shown); a backbuffer cleared to linear 0.5 must read back sRGB-encoded,
// before and after a resize.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <initializer_list>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace
{
    constexpr uint32_t kW = 64; // 256-byte rows: the readback row pitch D3D12 requires
    constexpr uint32_t kH = 32;

    // Clears backbuffer 0 to linear 0.5 and returns its first pixel's red and alpha bytes.
    void ClearAndRead(VriDevice* dev, VriCoreInterface& c, VriTexture* backbuffer, uint32_t width, uint8_t out[2])
    {
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);
        VriTextureViewDesc vd {};
        vd.texture          = backbuffer;
        vd.viewType         = VriTextureViewType_2D;
        vd.aspect           = VriImageAspect_Color;
        VriDescriptor* view = nullptr;
        REQUIRE(c.CreateTextureView(dev, &vd, &view) == VriResult_Success);
        VriBufferDesc bd {};
        bd.size             = uint64_t(width) * kH * 4;
        bd.usage            = VriBufferUsage_TransferDst;
        bd.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &readback) == VriResult_Success);
        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        VriTextureBarrierDesc tb {};
        tb.texture       = backbuffer;
        tb.before.layout = VriLayout_Present;
        tb.after.layout  = VriLayout_ColorAttachment;
        tb.after.access  = VriAccess_ColorAttachmentWrite;
        tb.after.stages  = VriPipelineStage_ColorAttachmentOutput;
        tb.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc g {};
        g.textures   = &tb;
        g.textureNum = 1;
        c.CmdBarrier(cmd, &g);
        VriAttachmentDesc rt {};
        rt.view    = view;
        rt.loadOp  = VriAttachmentLoadOp_Clear;
        rt.storeOp = VriAttachmentStoreOp_Store;
        for (float& v : rt.clearValue.color.f32)
            v = 0.5f;
        VriAttachmentsDesc att {};
        att.colors            = &rt;
        att.colorNum          = 1;
        att.renderArea.width  = width;
        att.renderArea.height = kH;
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        c.CmdEndRendering(cmd);
        tb.before       = tb.after;
        tb.after.layout = VriLayout_CopySource;
        tb.after.access = VriAccess_CopySourceRead;
        tb.after.stages = VriPipelineStage_Transfer;
        c.CmdBarrier(cmd, &g);
        VriBufferTextureCopyDesc cp {};
        cp.texture.aspect   = VriImageAspect_Color;
        cp.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, backbuffer, &cp);
        tb.before       = tb.after;
        tb.after.layout = VriLayout_Present;
        tb.after.access = VriAccess_None;
        c.CmdBarrier(cmd, &g);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
        VriFenceSubmitDesc signal {};
        signal.fence = fence;
        signal.value = 1;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        REQUIRE(c.QueueSubmit(queue, &submit) == VriResult_Success);
        c.Wait(fence, 1);
        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bd.size));
        REQUIRE(px != nullptr);
        out[0] = px[0];
        out[1] = px[3];
        c.UnmapBuffer(readback);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyDescriptor(view);
        c.DestroyBuffer(readback);
    }
} // namespace

TEST_CASE("D3D12: sRGB swapchains render through sRGB views on UNORM buffers")
{
    VriDeviceCreationDesc dc  = vriDefaultDeviceCreationDesc();
    dc.coordinates.clipSpaceY = VriClipSpaceY_Up;
    dc.graphicsAPI            = VriGraphicsAPI_D3D12;
    dc.enableValidation       = VRI_TRUE;
    dc.bestEffort             = VRI_TRUE;
    VriDevice* dev            = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    VriCoreInterface      c {};
    VriSwapChainInterface sw {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_SWAPCHAIN, sizeof(sw), &sw) == VriResult_Success);
    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    // A plain STATIC window needs no class registration; it is never shown.
    HWND hwnd = CreateWindowExA(0,
                                "STATIC",
                                "vri-test",
                                WS_OVERLAPPEDWINDOW,
                                0,
                                0,
                                kW,
                                kH,
                                nullptr,
                                nullptr,
                                GetModuleHandleA(nullptr),
                                nullptr);
    REQUIRE(hwnd != nullptr);

    for (const VriFormat format : {VriFormat_BGRA8_SRGB, VriFormat_RGBA8_SRGB})
    {
        CAPTURE(format);
        VriSwapChainDesc sd {};
        sd.window.type                   = VriWindowSystem_Win32;
        sd.window.handle.win32.hwnd      = hwnd;
        sd.window.handle.win32.hinstance = GetModuleHandleA(nullptr);
        sd.queue                         = queue;
        sd.format                        = format;
        sd.width                         = kW;
        sd.height                        = kH;
        sd.textureNum                    = 2;
        VriSwapChain* swapchain          = nullptr;
        REQUIRE(sw.CreateSwapChain(dev, &sd, &swapchain) == VriResult_Success);

        for (const uint32_t width : {kW, 2 * kW})
        {
            if (width != kW)
                REQUIRE(sw.Resize(swapchain, width, kH) == VriResult_Success);
            VriTexture* textures[2] {};
            uint32_t    count = 2;
            REQUIRE(sw.GetSwapChainTextures(swapchain, textures, &count) == VriResult_Success);
            REQUIRE(count >= 1);
            uint8_t px[2] {};
            ClearAndRead(dev, c, textures[0], width, px);
            // Linear 0.5 encodes to sRGB 188 (0.7354); alpha is stored linearly (128).
            CHECK(px[0] >= 187);
            CHECK(px[0] <= 189);
            CHECK(px[1] >= 127);
            CHECK(px[1] <= 128);
        }
        sw.DestroySwapChain(swapchain);
    }
    DestroyWindow(hwnd);
    vriDestroyDevice(dev);
}
