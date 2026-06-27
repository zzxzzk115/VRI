// Variable Rate Shading (D3D12, RSSetShadingRate): request the feature, confirm the
// interface is queryable iff the capability is reported, and render a triangle with a
// 2x2 coarse shading rate. A coarse rate changes shading frequency, not color, so the
// solid-red triangle center must still read back red. Skips gracefully when the
// adapter lacks VRS (e.g. older WARP). Mirrors test_vrs_vk.cpp on D3D12.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/triangle_dxbc.h" // g_triangleDxbcVS + g_triangleDxbcPS

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    struct Dx
    {
        VriDevice*       device = nullptr;
        VriCoreInterface core {};
        ~Dx()
        {
            if (device)
                vriDestroyDevice(device);
        }
    };

    bool Init(Dx& dx)
    {
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = VriGraphicsAPI_D3D12;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        dc.enabledFeatures  = VriFeature_VariableShadingRate;
        if (vriCreateDevice(&dc, &dx.device) != VriResult_Success)
            return false;
        return vriGetInterface(dx.device, VRI_INTERFACE_CORE, sizeof(dx.core), &dx.core) == VriResult_Success;
    }
} // namespace

TEST_CASE("D3D12 VRS: interface availability tracks the reported capability")
{
    Dx dx;
    if (!Init(dx))
    {
        MESSAGE("D3D12 unavailable - skipping VRS test");
        return;
    }
    const VriDeviceDesc*    d = dx.core.GetDeviceDesc(dx.device);
    VriShadingRateInterface vrs {};
    const bool queryable = vriGetInterface(dx.device, VRI_INTERFACE_VRS, sizeof(vrs), &vrs) == VriResult_Success;
    CHECK(queryable == (d->hasVariableShadingRate != VRI_FALSE));
    if (queryable)
        CHECK(vrs.CmdSetShadingRate != nullptr);
}

TEST_CASE("D3D12 VRS: render a triangle with a 2x2 coarse shading rate")
{
    Dx dx;
    if (!Init(dx))
    {
        MESSAGE("D3D12 unavailable - skipping VRS test");
        return;
    }
    const VriCoreInterface& c   = dx.core;
    VriDevice*              dev = dx.device;
    if (c.GetDeviceDesc(dev)->hasVariableShadingRate == VRI_FALSE)
    {
        MESSAGE("adapter lacks VRS - skipping");
        return;
    }

    VriShadingRateInterface vrs {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_VRS, sizeof(vrs), &vrs) == VriResult_Success);

    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

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
    td.clearValue.color.f32[3] = 1.0f; // match the render-pass clear
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

    VriPipelineLayoutDesc ld {};
    VriPipelineLayout*    layout = nullptr;
    REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

    VriShaderDesc sh[2] {};
    sh[0].stage          = VriShaderStage_Vertex;
    sh[0].bytecode       = g_triangleDxbcVS;
    sh[0].bytecodeSize   = sizeof(g_triangleDxbcVS);
    sh[0].entryPointName = "vertexMain";
    sh[1].stage          = VriShaderStage_Fragment;
    sh[1].bytecode       = g_triangleDxbcPS;
    sh[1].bytecodeSize   = sizeof(g_triangleDxbcPS);
    sh[1].entryPointName = "fragmentMain";
    VriColorAttachmentDesc ca {};
    ca.format         = VriFormat_RGBA8_UNORM;
    ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd {};
    pd.pipelineLayout          = layout;
    pd.shaders                 = sh;
    pd.shaderNum               = 2;
    pd.inputAssembly.topology  = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode  = VriCullMode_None;
    pd.rasterization.lineWidth = 1.0f;
    pd.multisample.sampleNum   = 1;
    pd.outputMerger.colors     = &ca;
    pd.outputMerger.colorNum   = 1;
    VriPipeline* pipeline      = nullptr;
    REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

    VriCommandAllocator* alloc = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
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
    rt.clearValue.color.f32[3] = 1.0f;
    VriAttachmentsDesc att {};
    att.colors            = &rt;
    att.colorNum          = 1;
    att.renderArea.width  = kW;
    att.renderArea.height = kH;
    att.layerNum          = 1;
    c.CmdBeginRendering(cmd, &att);
    VriViewport vp {0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1};
    c.CmdSetViewports(cmd, &vp, 1);
    VriRect sc {0, 0, kW, kH};
    c.CmdSetScissors(cmd, &sc, 1);
    c.CmdSetPipeline(cmd, pipeline);

    VriShadingRateDesc rate {};
    rate.shadingRate        = VriShadingRate_2x2;
    rate.primitiveCombiner  = VriShadingRateCombiner_Keep;
    rate.attachmentCombiner = VriShadingRateCombiner_Keep;
    vrs.CmdSetShadingRate(cmd, &rate);

    VriDrawDesc draw {};
    draw.vertexNum   = 3;
    draw.instanceNum = 1;
    c.CmdDraw(cmd, &draw);
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
    const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
    CHECK(px[o + 0] == 255); // coarse shading preserves the solid color
    CHECK(px[o + 1] == 0);
    CHECK(px[o + 2] == 0);
    c.UnmapBuffer(readback);

    c.DeviceWaitIdle(dev);
    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyBuffer(readback);
    c.DestroyDescriptor(colorView);
    c.DestroyTexture(color);
}
