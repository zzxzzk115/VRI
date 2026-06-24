// D3D12 clear + readback (phase 1): render-target texture, RTV, command list,
// resource-state barriers, a render pass that clears to red, texture->buffer readback,
// fence sync, and host-readback map. No shaders/PSO yet - verifies the D3D12
// command/resource/submission plumbing. Runs on hardware or WARP; skips if neither.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;
}

TEST_CASE("D3D12: clear-to-red render pass + readback (phase 1, no shaders)")
{
    VriDeviceCreationDesc dc{};
    dc.graphicsAPI = VriGraphicsAPI_D3D12; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
    VriDevice* dev = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success) { MESSAGE("D3D12 unavailable - skipping"); return; }

    VriCoreInterface c{};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    VriTextureDesc td{};
    td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM;
    td.width = kW; td.height = kH; td.depth = 1; td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1;
    td.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc; td.memoryLocation = VriMemoryLocation_Device;
    td.clearValue.color.f32[0] = 1.0f; td.clearValue.color.f32[3] = 1.0f; // match the render-pass clear
    VriTexture* color = nullptr;
    REQUIRE(c.CreateTexture(dev, &td, &color) == VriResult_Success);
    VriTextureViewDesc cvd{};
    cvd.texture = color; cvd.viewType = VriTextureViewType_2D; cvd.format = VriFormat_Unknown; cvd.aspect = VriImageAspect_Color;
    VriDescriptor* colorView = nullptr;
    REQUIRE(c.CreateTextureView(dev, &cvd, &colorView) == VriResult_Success);

    VriBufferDesc rbd{};
    rbd.size = static_cast<uint64_t>(kW) * kH * 4; rbd.usage = VriBufferUsage_TransferDst; rbd.memoryLocation = VriMemoryLocation_HostReadback;
    VriBuffer* readback = nullptr;
    REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

    VriCommandAllocator* alloc = nullptr; REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr; REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
    VriFence* fence = nullptr; REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
    {
        VriTextureBarrierDesc tb{};
        tb.texture = color; tb.before.layout = VriLayout_Undefined; tb.before.stages = VriPipelineStage_None;
        tb.after.access = VriAccess_ColorAttachmentWrite; tb.after.layout = VriLayout_ColorAttachment; tb.after.stages = VriPipelineStage_ColorAttachmentOutput;
        tb.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc g{}; g.textures = &tb; g.textureNum = 1; c.CmdBarrier(cmd, &g);
    }
    VriAttachmentDesc rt{}; rt.view = colorView; rt.loadOp = VriAttachmentLoadOp_Clear; rt.storeOp = VriAttachmentStoreOp_Store;
    rt.clearValue.color.f32[0] = 1.0f; rt.clearValue.color.f32[3] = 1.0f; // red
    VriAttachmentsDesc att{}; att.colors = &rt; att.colorNum = 1; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
    c.CmdBeginRendering(cmd, &att);
    c.CmdEndRendering(cmd);
    {
        VriTextureBarrierDesc tb{};
        tb.texture = color; tb.before.access = VriAccess_ColorAttachmentWrite; tb.before.layout = VriLayout_ColorAttachment; tb.before.stages = VriPipelineStage_ColorAttachmentOutput;
        tb.after.access = VriAccess_CopySourceRead; tb.after.layout = VriLayout_CopySource; tb.after.stages = VriPipelineStage_Transfer;
        tb.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc g{}; g.textures = &tb; g.textureNum = 1; c.CmdBarrier(cmd, &g);
    }
    VriBufferTextureCopyDesc tc{}; tc.texture.aspect = VriImageAspect_Color; tc.texture.layerNum = 1;
    c.CmdReadbackTextureToBuffer(cmd, readback, color, &tc);
    REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

    VriFenceSubmitDesc signal{}; signal.fence = fence; signal.value = 1;
    VriQueueSubmitDesc submit{}; submit.commandBuffers = &cmd; submit.commandBufferNum = 1; submit.signalFences = &signal; submit.signalFenceNum = 1;
    c.QueueSubmit(queue, &submit);
    c.Wait(fence, 1);

    const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
    REQUIRE(px != nullptr);
    const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
    const bool red = px[o + 0] == 255 && px[o + 1] == 0 && px[o + 2] == 0 && px[o + 3] == 255;
    c.UnmapBuffer(readback);
    CHECK(red);

    c.DestroyFence(fence); c.DestroyCommandAllocator(alloc);
    c.DestroyBuffer(readback); c.DestroyDescriptor(colorView); c.DestroyTexture(color);
    vriDestroyDevice(dev);
}
