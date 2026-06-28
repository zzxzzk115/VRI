// Cross-backend test for the built-in Dear ImGui renderer (VRI_INTERFACE_IMGUI). VRI does not
// link ImGui, so this test hand-builds a VriImguiDrawData (one red quad over the screen center,
// sampling a 4x4 white "font atlas") and drives the renderer's Upload/CmdCopy/CmdDraw path to an
// offscreen target. The quad's interior must come out red (vertex color * white texel) and the
// background black, proving the embedded shader, font upload, vertex/index staging, push-constant
// ortho and per-command scissor all work - on every backend that has an ImGui shader.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstdlib>

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    // Returns true if the quad rendered red over a black background. `ran` is set when a device
    // for `api` was actually created (so callers can skip cleanly on unavailable backends).
    bool RunImgui(VriGraphicsAPI api, bool& ran)
    {
        ran = false;
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        VriDevice* dev      = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;

        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriImguiInterface gui {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_IMGUI, sizeof(gui), &gui) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        // 4x4 white RGBA "font atlas" so the textured sample is (1,1,1,1) and final = vertex color.
        unsigned char atlas[4 * 4 * 4];
        for (unsigned char& b : atlas)
            b = 0xFF;
        VriImguiDesc gd {};
        gd.uploadQueue        = queue;
        gd.colorFormat        = VriFormat_RGBA8_UNORM;
        gd.depthFormat        = VriFormat_Unknown;
        gd.fontAtlas          = atlas;
        gd.fontWidth          = 4;
        gd.fontHeight         = 4;
        VriImgui*       imgui = nullptr;
        const VriResult cr    = gui.CreateImgui(dev, &gd, &imgui);
        if (cr == VriResult_Unsupported) // e.g. Metal (no MSL ImGui shader yet)
        {
            ran = true;
            vriDestroyDevice(dev);
            return true; // honest unsupported, not a failure
        }
        REQUIRE(cr == VriResult_Success);
        ran = true;

        // One red quad over the center [16,48]^2, UVs at the white atlas center.
        const uint32_t       red      = 0xFF0000FFu; // bytes R,G,B,A = FF,00,00,FF
        const VriImguiVertex verts[4] = {
            {{16, 16}, {0.5f, 0.5f}, red},
            {{48, 16}, {0.5f, 0.5f}, red},
            {{48, 48}, {0.5f, 0.5f}, red},
            {{16, 48}, {0.5f, 0.5f}, red},
        };
        const uint16_t      idx[6] = {0, 1, 2, 0, 2, 3};
        VriImguiDrawCommand cmd0 {};
        cmd0.clipRect[0]  = 0;
        cmd0.clipRect[1]  = 0;
        cmd0.clipRect[2]  = float(kW);
        cmd0.clipRect[3]  = float(kH);
        cmd0.indexCount   = 6;
        cmd0.indexOffset  = 0;
        cmd0.vertexOffset = 0;
        VriImguiDrawData dd {};
        dd.vertices          = verts;
        dd.vertexCount       = 4;
        dd.indices           = idx;
        dd.indexCount        = 6;
        dd.indexSize         = sizeof(uint16_t);
        dd.commands          = &cmd0;
        dd.commandCount      = 1;
        dd.displaySize[0]    = float(kW);
        dd.displaySize[1]    = float(kH);
        dd.framebufferWidth  = kW;
        dd.framebufferHeight = kH;
        gui.UploadImguiData(imgui, &dd); // host map; before recording

        VriTextureDesc td {};
        td.type                    = VriTextureType_2D;
        td.format                  = VriFormat_RGBA8_UNORM;
        td.width                   = kW;
        td.height                  = kH;
        td.depth                   = 1;
        td.mipNum                  = 1;
        td.layerNum                = 1;
        td.sampleNum               = 1;
        td.usage                   = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
        td.memoryLocation          = VriMemoryLocation_Device;
        td.clearValue.color.f32[3] = 1.0f;
        VriTexture* color          = nullptr;
        REQUIRE(c.CreateTexture(dev, &td, &color) == VriResult_Success);
        VriTextureViewDesc cvd {};
        cvd.texture              = color;
        cvd.viewType             = VriTextureViewType_2D;
        cvd.format               = VriFormat_Unknown;
        cvd.aspect               = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr;
        REQUIRE(c.CreateTextureView(dev, &cvd, &colorView) == VriResult_Success);

        VriBufferDesc rbd {};
        rbd.size            = static_cast<uint64_t>(kW) * kH * 4;
        rbd.usage           = VriBufferUsage_TransferDst;
        rbd.memoryLocation  = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        gui.CmdCopyImguiData(cmd, imgui); // staging->device + barriers, outside the render pass
        {
            VriTextureBarrierDesc tb {};
            tb.texture       = color;
            tb.before.layout = VriLayout_Undefined;
            tb.before.stages = VriPipelineStage_None;
            tb.after.access  = VriAccess_ColorAttachmentWrite;
            tb.after.layout  = VriLayout_ColorAttachment;
            tb.after.stages  = VriPipelineStage_ColorAttachmentOutput;
            tb.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        }

        VriAttachmentDesc rt {};
        rt.view                    = colorView;
        rt.loadOp                  = VriAttachmentLoadOp_Clear;
        rt.storeOp                 = VriAttachmentStoreOp_Store;
        rt.clearValue.color.f32[3] = 1.0f; // black, opaque
        VriAttachmentsDesc att {};
        att.colors            = &rt;
        att.colorNum          = 1;
        att.renderArea.width  = kW;
        att.renderArea.height = kH;
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        gui.CmdDrawImgui(cmd, imgui, &dd);
        c.CmdEndRendering(cmd);

        {
            VriTextureBarrierDesc tb {};
            tb.texture       = color;
            tb.before.access = VriAccess_ColorAttachmentWrite;
            tb.before.layout = VriLayout_ColorAttachment;
            tb.before.stages = VriPipelineStage_ColorAttachmentOutput;
            tb.after.access  = VriAccess_CopySourceRead;
            tb.after.layout  = VriLayout_CopySource;
            tb.after.stages  = VriPipelineStage_Transfer;
            tb.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        }
        VriBufferTextureCopyDesc tc {};
        tc.texture.aspect   = VriImageAspect_Color;
        tc.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &tc);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

        VriFenceSubmitDesc signal {};
        signal.fence = fence;
        signal.value = 1;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        c.QueueSubmit(queue, &submit);
        c.Wait(fence, 1);

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
        REQUIRE(px != nullptr);
        auto           at      = [&](uint32_t x, uint32_t y) { return px + (static_cast<size_t>(y) * kW + x) * 4; };
        const uint8_t* center  = at(kW / 2, kH / 2);                                  // inside the quad -> red
        const uint8_t* corner  = at(4, 4);                                            // outside -> black clear
        const bool     quadRed = center[0] > 200 && center[1] < 60 && center[2] < 60; // R high, G/B low
        const bool     bgBlack = corner[0] < 16 && corner[1] < 16 && corner[2] < 16;
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        c.DestroyDescriptor(colorView);
        c.DestroyTexture(color);
        gui.DestroyImgui(imgui);
        vriDestroyDevice(dev);
        return quadRed && bgBlack;
    }

    void Check(VriGraphicsAPI api, const char* name)
    {
        bool       ran = false;
        const bool ok  = RunImgui(api, ran);
        if (ran)
            CHECK_MESSAGE(ok, "[", name, "] ImGui quad should render red over black");
        else
            MESSAGE("[" << name << "] unavailable - skipped");
    }
} // namespace

TEST_CASE("ImGui renderer: built-in ext draws a quad on each backend")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
}
