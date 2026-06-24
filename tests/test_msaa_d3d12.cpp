// D3D12 MSAA (4x multisample render target + resolve). The red triangle is rendered to
// a 4-sample color target whose attachment carries a single-sample resolveView; at
// CmdEndRendering the backend ResolveSubresource's it down. The resolved image is read
// back and an edge pixel must show PARTIAL red coverage (0 < R < 255) - antialiasing a
// non-multisampled (hard-edged) render can't produce. Exercises TEXTURE2DMS RTVs +
// ResolveSubresource. Runs on hardware or WARP.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/triangle_dxbc.h" // g_triangleDxbcVS + g_triangleDxbcPS (solid red)

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;
    constexpr uint32_t kSamples = 4;
}

TEST_CASE("D3D12: 4x MSAA + resolve produces antialiased edges")
{
    VriDeviceCreationDesc dc{};
    dc.graphicsAPI = VriGraphicsAPI_D3D12; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
    VriDevice* dev = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success) { MESSAGE("D3D12 unavailable - skipping"); return; }

    VriCoreInterface c{};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    VriQueue* queue = nullptr; REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    VriTextureDesc mt{}; mt.type = VriTextureType_2D; mt.format = VriFormat_RGBA8_UNORM;
    mt.width = kW; mt.height = kH; mt.depth = 1; mt.mipNum = 1; mt.layerNum = 1; mt.sampleNum = kSamples;
    mt.usage = VriTextureUsage_ColorAttachment; mt.memoryLocation = VriMemoryLocation_Device;
    mt.clearValue.color.f32[3] = 1.0f; // match the render-pass clear
    VriTexture* msaa = nullptr; REQUIRE(c.CreateTexture(dev, &mt, &msaa) == VriResult_Success);
    VriTextureViewDesc mvd{}; mvd.texture = msaa; mvd.viewType = VriTextureViewType_2D; mvd.format = VriFormat_Unknown; mvd.aspect = VriImageAspect_Color;
    VriDescriptor* msaaView = nullptr; REQUIRE(c.CreateTextureView(dev, &mvd, &msaaView) == VriResult_Success);

    VriTextureDesc rt{}; rt.type = VriTextureType_2D; rt.format = VriFormat_RGBA8_UNORM;
    rt.width = kW; rt.height = kH; rt.depth = 1; rt.mipNum = 1; rt.layerNum = 1; rt.sampleNum = 1;
    rt.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc; rt.memoryLocation = VriMemoryLocation_Device;
    VriTexture* resolve = nullptr; REQUIRE(c.CreateTexture(dev, &rt, &resolve) == VriResult_Success);
    VriTextureViewDesc rvd{}; rvd.texture = resolve; rvd.viewType = VriTextureViewType_2D; rvd.format = VriFormat_Unknown; rvd.aspect = VriImageAspect_Color;
    VriDescriptor* resolveView = nullptr; REQUIRE(c.CreateTextureView(dev, &rvd, &resolveView) == VriResult_Success);

    VriBufferDesc rbd{}; rbd.size = static_cast<uint64_t>(kW) * kH * 4; rbd.usage = VriBufferUsage_TransferDst; rbd.memoryLocation = VriMemoryLocation_HostReadback;
    VriBuffer* readback = nullptr; REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

    VriPipelineLayoutDesc ld{}; VriPipelineLayout* layout = nullptr; REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);
    VriShaderDesc sh[2]{};
    sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = g_triangleDxbcVS; sh[0].bytecodeSize = sizeof(g_triangleDxbcVS); sh[0].entryPointName = "vertexMain";
    sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = g_triangleDxbcPS; sh[1].bytecodeSize = sizeof(g_triangleDxbcPS); sh[1].entryPointName = "fragmentMain";
    VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd{};
    pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
    pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
    pd.multisample.sampleNum = kSamples; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
    VriPipeline* pipeline = nullptr; REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

    VriCommandAllocator* alloc = nullptr; REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr; REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
    VriFence* fence = nullptr; REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
    {
        VriTextureBarrierDesc tb[2]{};
        tb[0].texture = msaa; tb[0].before.layout = VriLayout_Undefined; tb[0].after.access = VriAccess_ColorAttachmentWrite; tb[0].after.layout = VriLayout_ColorAttachment; tb[0].after.stages = VriPipelineStage_ColorAttachmentOutput; tb[0].aspect = VriImageAspect_Color;
        tb[1].texture = resolve; tb[1].before.layout = VriLayout_Undefined; tb[1].after.access = VriAccess_ColorAttachmentWrite; tb[1].after.layout = VriLayout_ColorAttachment; tb[1].after.stages = VriPipelineStage_ColorAttachmentOutput; tb[1].aspect = VriImageAspect_Color;
        VriBarrierGroupDesc g{}; g.textures = tb; g.textureNum = 2; c.CmdBarrier(cmd, &g);
    }
    VriAttachmentDesc att{}; att.view = msaaView; att.loadOp = VriAttachmentLoadOp_Clear; att.storeOp = VriAttachmentStoreOp_Store; att.clearValue.color.f32[3] = 1.0f; att.resolveView = resolveView;
    VriAttachmentsDesc rp{}; rp.colors = &att; rp.colorNum = 1; rp.renderArea.width = kW; rp.renderArea.height = kH; rp.layerNum = 1;
    c.CmdBeginRendering(cmd, &rp);
    VriViewport vp{0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
    VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
    c.CmdSetPipeline(cmd, pipeline);
    VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 1; c.CmdDraw(cmd, &draw);
    c.CmdEndRendering(cmd);
    {
        VriTextureBarrierDesc tb{}; tb.texture = resolve; tb.before.access = VriAccess_ColorAttachmentWrite; tb.before.layout = VriLayout_ColorAttachment; tb.before.stages = VriPipelineStage_ColorAttachmentOutput;
        tb.after.access = VriAccess_CopySourceRead; tb.after.layout = VriLayout_CopySource; tb.after.stages = VriPipelineStage_Transfer; tb.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc g{}; g.textures = &tb; g.textureNum = 1; c.CmdBarrier(cmd, &g);
    }
    VriBufferTextureCopyDesc tc{}; tc.texture.aspect = VriImageAspect_Color; tc.texture.layerNum = 1;
    c.CmdReadbackTextureToBuffer(cmd, readback, resolve, &tc);
    REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

    VriFenceSubmitDesc signal{}; signal.fence = fence; signal.value = 1;
    VriQueueSubmitDesc submit{}; submit.commandBuffers = &cmd; submit.commandBufferNum = 1; submit.signalFences = &signal; submit.signalFenceNum = 1;
    c.QueueSubmit(queue, &submit);
    c.Wait(fence, 1);

    const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
    REQUIRE(px != nullptr);
    bool edgeBlend = false, anyRed = false;
    for (uint32_t i = 0; i < kW * kH; ++i)
    {
        const uint8_t r = px[i * 4 + 0], g = px[i * 4 + 1], b = px[i * 4 + 2];
        if (r == 255 && g == 0 && b == 0) anyRed = true;
        if (r > 0 && r < 255 && g == 0 && b == 0) edgeBlend = true; // partial coverage (MSAA)
    }
    c.UnmapBuffer(readback);
    CHECK(anyRed);
    CHECK(edgeBlend);

    c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
    c.DestroyBuffer(readback); c.DestroyDescriptor(msaaView); c.DestroyDescriptor(resolveView); c.DestroyTexture(msaa); c.DestroyTexture(resolve);
    vriDestroyDevice(dev);
}
