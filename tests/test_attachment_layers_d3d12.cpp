// D3D12 render-target views of single array layers: each layer of a 2D array texture is cleared
// through its own view to a different color and read back. A Texture2D RTV addresses slice 0 only,
// so without array views every clear lands in layer 0.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

TEST_CASE("D3D12: a render-target view of one array layer renders to that layer")
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
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    constexpr uint32_t kLayers = 3;
    constexpr uint32_t kSize   = 64; // 256-byte rows: the readback row pitch D3D12 requires
    VriTextureDesc     td {};
    td.type            = VriTextureType_2DArray;
    td.format          = VriFormat_RGBA8_UNORM;
    td.width           = kSize;
    td.height          = kSize;
    td.depth           = 1;
    td.mipNum          = 1;
    td.layerNum        = kLayers;
    td.sampleNum       = 1;
    td.usage           = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
    td.memoryLocation  = VriMemoryLocation_Device;
    VriTexture* target = nullptr;
    REQUIRE(c.CreateTexture(dev, &td, &target) == VriResult_Success);
    VriDescriptor* views[kLayers] {};
    VriBuffer*     readbacks[kLayers] {};
    for (uint32_t layer = 0; layer < kLayers; ++layer)
    {
        VriTextureViewDesc vd {};
        vd.texture   = target;
        vd.viewType  = VriTextureViewType_2D;
        vd.aspect    = VriImageAspect_Color;
        vd.mipNum    = 1;
        vd.baseLayer = layer;
        vd.layerNum  = 1;
        REQUIRE(c.CreateTextureView(dev, &vd, &views[layer]) == VriResult_Success);
        VriBufferDesc bd {};
        bd.size           = uint64_t(kSize) * kSize * 4;
        bd.usage          = VriBufferUsage_TransferDst;
        bd.memoryLocation = VriMemoryLocation_HostReadback;
        REQUIRE(c.CreateBuffer(dev, &bd, &readbacks[layer]) == VriResult_Success);
    }
    VriCommandAllocator* alloc = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
    VriTextureBarrierDesc tb {};
    tb.texture      = target;
    tb.after.layout = VriLayout_ColorAttachment;
    tb.after.access = VriAccess_ColorAttachmentWrite;
    tb.after.stages = VriPipelineStage_ColorAttachmentOutput;
    tb.aspect       = VriImageAspect_Color;
    VriBarrierGroupDesc g {};
    g.textures   = &tb;
    g.textureNum = 1;
    c.CmdBarrier(cmd, &g);
    for (uint32_t layer = 0; layer < kLayers; ++layer)
    {
        VriAttachmentDesc rt {};
        rt.view                        = views[layer];
        rt.loadOp                      = VriAttachmentLoadOp_Clear;
        rt.storeOp                     = VriAttachmentStoreOp_Store;
        rt.clearValue.color.f32[layer] = 1.0f; // layer 0 red, 1 green, 2 blue
        rt.clearValue.color.f32[3]     = 1.0f;
        VriAttachmentsDesc att {};
        att.colors            = &rt;
        att.colorNum          = 1;
        att.renderArea.width  = kSize;
        att.renderArea.height = kSize;
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        c.CmdEndRendering(cmd);
    }
    tb.before       = tb.after;
    tb.after.layout = VriLayout_CopySource;
    tb.after.access = VriAccess_CopySourceRead;
    tb.after.stages = VriPipelineStage_Transfer;
    c.CmdBarrier(cmd, &g);
    for (uint32_t layer = 0; layer < kLayers; ++layer)
    {
        VriBufferTextureCopyDesc cp {};
        cp.texture.aspect    = VriImageAspect_Color;
        cp.texture.baseLayer = layer;
        cp.texture.layerNum  = 1;
        c.CmdReadbackTextureToBuffer(cmd, readbacks[layer], target, &cp);
    }
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

    for (uint32_t layer = 0; layer < kLayers; ++layer)
    {
        CAPTURE(layer);
        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readbacks[layer], 0, 4));
        REQUIRE(px != nullptr);
        for (uint32_t channel = 0; channel < 3; ++channel)
            CHECK(px[channel] == (channel == layer ? 255 : 0));
        c.UnmapBuffer(readbacks[layer]);
    }

    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    for (uint32_t layer = 0; layer < kLayers; ++layer)
    {
        c.DestroyDescriptor(views[layer]);
        c.DestroyBuffer(readbacks[layer]);
    }
    c.DestroyTexture(target);
    vriDestroyDevice(dev);
}
