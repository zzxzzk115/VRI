// D3D12 attachment-view descriptors: every live render-target and depth-stencil view holds its own
// CPU descriptor. Each layer of an array texture is cleared through its own view, with more views
// alive than VRI once held in its wrapping RTV (256) and DSV (64) rings, and every layer is read
// back. Running out of descriptors is an error, and destroying a view frees its descriptor.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>
#include <vector>

// Creates one attachment view per layer of a `layers`-layer texture, keeps them all alive, clears
// each layer through its own view to a value that names the layer and reads every layer back.
static void ClearThroughLiveViews(bool depth, uint32_t layers)
{
    VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
    dc.graphicsAPI           = VriGraphicsAPI_D3D12;
    dc.enableValidation      = VRI_TRUE;
    dc.bestEffort            = VRI_TRUE;
    VriDevice* dev           = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    constexpr uint32_t kSize       = 64; // 256-byte rows and 16 KiB layers: D3D12's copy alignments
    constexpr uint64_t kLayerBytes = uint64_t(kSize) * kSize * 4;
    VriTextureDesc     td {};
    td.type      = VriTextureType_2DArray;
    td.format    = depth ? VriFormat_D32_SFLOAT : VriFormat_RGBA8_UNORM;
    td.width     = kSize;
    td.height    = kSize;
    td.depth     = 1;
    td.mipNum    = 1;
    td.layerNum  = layers;
    td.sampleNum = 1;
    td.usage     = (depth ? VriTextureUsage_DepthStencilAttachment : VriTextureUsage_ColorAttachment) |
               VriTextureUsage_TransferSrc;
    td.memoryLocation  = VriMemoryLocation_Device;
    VriTexture* target = nullptr;
    REQUIRE(c.CreateTexture(dev, &td, &target) == VriResult_Success);
    const VriImageAspectFlags   aspect = depth ? VriImageAspect_Depth : VriImageAspect_Color;
    std::vector<VriDescriptor*> views(layers);
    for (uint32_t layer = 0; layer < layers; ++layer)
    {
        VriTextureViewDesc vd {};
        vd.texture   = target;
        vd.viewType  = VriTextureViewType_2D;
        vd.format    = td.format;
        vd.aspect    = aspect;
        vd.mipNum    = 1;
        vd.baseLayer = layer;
        vd.layerNum  = 1;
        REQUIRE(c.CreateTextureView(dev, &vd, &views[layer]) == VriResult_Success);
    }
    VriBufferDesc bd {};
    bd.size             = kLayerBytes * layers;
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
    tb.texture      = target;
    tb.after.layout = depth ? VriLayout_DepthStencilAttachment : VriLayout_ColorAttachment;
    tb.after.access = depth ? VriAccess_DepthStencilAttachmentWrite : VriAccess_ColorAttachmentWrite;
    tb.after.stages = depth ? VriPipelineStage_EarlyFragmentTests : VriPipelineStage_ColorAttachmentOutput;
    tb.aspect       = aspect;
    VriBarrierGroupDesc g {};
    g.textures   = &tb;
    g.textureNum = 1;
    c.CmdBarrier(cmd, &g);
    for (uint32_t layer = 0; layer < layers; ++layer)
    {
        VriAttachmentDesc a {};
        a.view    = views[layer];
        a.loadOp  = VriAttachmentLoadOp_Clear;
        a.storeOp = VriAttachmentStoreOp_Store;
        VriAttachmentsDesc att {};
        if (depth)
        {
            a.clearValue.depthStencil.depth = float(layer + 1) / 1024.0f; // exact in a 32-bit float
            att.depth                       = &a;
        }
        else
        {
            a.clearValue.color.f32[0] = float(layer & 0xff) / 255.0f;
            a.clearValue.color.f32[1] = float(layer >> 8) / 255.0f;
            a.clearValue.color.f32[3] = 1.0f;
            att.colors                = &a;
            att.colorNum              = 1;
        }
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
    for (uint32_t layer = 0; layer < layers; ++layer)
    {
        VriBufferTextureCopyDesc cp {};
        cp.bufferOffset      = kLayerBytes * layer;
        cp.texture.aspect    = aspect;
        cp.texture.baseLayer = layer;
        cp.texture.layerNum  = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, target, &cp);
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

    const auto* bytes = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bd.size));
    REQUIRE(bytes != nullptr);
    uint32_t wrong = 0;
    for (uint32_t layer = 0; layer < layers; ++layer)
    {
        const uint8_t* px = bytes + kLayerBytes * layer;
        bool           ok = false;
        if (depth)
        {
            float value = 0;
            std::memcpy(&value, px, sizeof(value));
            ok = value == float(layer + 1) / 1024.0f;
        }
        else
            ok = px[0] == (layer & 0xff) && px[1] == (layer >> 8);
        if (!ok && wrong++ == 0)
            MESSAGE("first wrong layer: " << layer);
    }
    CHECK(wrong == 0);
    c.UnmapBuffer(readback);

    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    for (VriDescriptor* view : views)
        c.DestroyDescriptor(view);
    c.DestroyBuffer(readback);
    c.DestroyTexture(target);
    vriDestroyDevice(dev);
}

TEST_CASE("D3D12: every live render-target view keeps its own descriptor") { ClearThroughLiveViews(false, 300); }

TEST_CASE("D3D12: every live depth-stencil view keeps its own descriptor") { ClearThroughLiveViews(true, 80); }

static void VRI_CALL CountError(void* userArg, VriMessageSeverity severity, const char*)
{
    if (severity == VriMessageSeverity_Error)
        ++*static_cast<int*>(userArg);
}

TEST_CASE("D3D12: running out of render-target views is an error, and destroying a view frees its descriptor")
{
    int                   errors = 0;
    VriCallbackInterface  callback {&errors, CountError};
    VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
    dc.graphicsAPI           = VriGraphicsAPI_D3D12;
    dc.bestEffort            = VRI_TRUE;
    dc.callbackInterface     = &callback;
    VriDevice* dev           = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    VriTextureDesc td {};
    td.type            = VriTextureType_2D;
    td.format          = VriFormat_RGBA8_UNORM;
    td.width           = 4;
    td.height          = 4;
    td.depth           = 1;
    td.mipNum          = 1;
    td.layerNum        = 1;
    td.sampleNum       = 1;
    td.usage           = VriTextureUsage_ColorAttachment;
    td.memoryLocation  = VriMemoryLocation_Device;
    VriTexture* target = nullptr;
    REQUIRE(c.CreateTexture(dev, &td, &target) == VriResult_Success);
    VriTextureViewDesc vd {};
    vd.texture  = target;
    vd.viewType = VriTextureViewType_2D;
    vd.aspect   = VriImageAspect_Color;
    vd.mipNum   = 1;
    vd.layerNum = 1;
    // Bounded so that a backend without a limit ends the loop and fails the checks below.
    std::vector<VriDescriptor*> views;
    VriResult                   result = VriResult_Success;
    while (views.size() < 100000)
    {
        VriDescriptor* view = nullptr;
        result              = c.CreateTextureView(dev, &vd, &view);
        if (result != VriResult_Success)
        {
            CHECK(view == nullptr);
            break;
        }
        views.push_back(view);
    }
    CHECK(result == VriResult_OutOfMemory);
    CHECK(errors == 1);
    REQUIRE(!views.empty());
    c.DestroyDescriptor(views.back());
    views.pop_back();
    VriDescriptor* again = nullptr;
    CHECK(c.CreateTextureView(dev, &vd, &again) == VriResult_Success);
    if (again)
        views.push_back(again);
    for (VriDescriptor* view : views)
        c.DestroyDescriptor(view);
    c.DestroyTexture(target);
    vriDestroyDevice(dev);
}
