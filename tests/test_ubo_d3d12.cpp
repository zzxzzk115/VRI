// D3D12 constant buffer / descriptor set (phase 3a): the triangle's color comes from a
// constant buffer at set=0, binding=0, bound as a ROOT CBV. Exercises CreatePipelineLayout
// (root signature with a root CBV), CreateDescriptorPool/AllocateDescriptorSets/
// CreateBufferView/UpdateDescriptorRanges, CmdCopyBuffer (staging->device upload), and
// CmdSetDescriptorSet. Center pixel must be the UBO color (green) - parity with the others.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>

#include "shaders/triangle_ubo_dxbc.h" // g_triangleUboDxbcVS + g_triangleUboDxbcPS

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;
} // namespace

TEST_CASE("D3D12: constant-buffer descriptor -> green triangle (phase 3a)")
{
    VriDeviceCreationDesc dc {};
    dc.graphicsAPI      = VriGraphicsAPI_D3D12;
    dc.enableValidation = VRI_TRUE;
    dc.bestEffort       = VRI_TRUE;
    VriDevice* dev      = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }

    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
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

    // green tint -> staging (UPLOAD) -> device constant buffer (copied on the queue)
    const float   green[4] = {0.0f, 1.0f, 0.0f, 1.0f};
    VriBufferDesc sb {};
    sb.size            = 16;
    sb.usage           = VriBufferUsage_TransferSrc;
    sb.memoryLocation  = VriMemoryLocation_HostUpload;
    VriBuffer* staging = nullptr;
    REQUIRE(c.CreateBuffer(dev, &sb, &staging) == VriResult_Success);
    {
        void* m = c.MapBuffer(staging, 0, 16);
        REQUIRE(m != nullptr);
        std::memcpy(m, green, 16);
        c.UnmapBuffer(staging);
    }
    VriBufferDesc ub {};
    ub.size           = 16;
    ub.usage          = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst;
    ub.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* ubo    = nullptr;
    REQUIRE(c.CreateBuffer(dev, &ub, &ubo) == VriResult_Success);

    VriDescriptorRangeDesc range {};
    range.baseRegister   = 0;
    range.descriptorNum  = 1;
    range.descriptorType = VriDescriptorType_ConstantBuffer;
    range.shaderStages   = VriShaderStage_Fragment;
    VriDescriptorSetDesc setDesc {};
    setDesc.registerSpace = 0;
    setDesc.ranges        = &range;
    setDesc.rangeNum      = 1;
    VriPipelineLayoutDesc ld {};
    ld.descriptorSets         = &setDesc;
    ld.descriptorSetNum       = 1;
    VriPipelineLayout* layout = nullptr;
    REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

    VriShaderDesc sh[2] {};
    sh[0].stage          = VriShaderStage_Vertex;
    sh[0].bytecode       = g_triangleUboDxbcVS;
    sh[0].bytecodeSize   = sizeof(g_triangleUboDxbcVS);
    sh[0].entryPointName = "vertexMain";
    sh[1].stage          = VriShaderStage_Fragment;
    sh[1].bytecode       = g_triangleUboDxbcPS;
    sh[1].bytecodeSize   = sizeof(g_triangleUboDxbcPS);
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

    VriDescriptorPoolDesc pdsc {};
    pdsc.descriptorSetMaxNum  = 1;
    pdsc.constantBufferMaxNum = 1;
    VriDescriptorPool* pool   = nullptr;
    REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
    VriDescriptorSet* set = nullptr;
    REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);
    VriBufferViewDesc uboView {};
    uboView.buffer               = ubo;
    uboView.offset               = 0;
    uboView.size                 = 16;
    VriDescriptor* uboDescriptor = nullptr;
    REQUIRE(c.CreateBufferView(dev, &uboView, &uboDescriptor) == VriResult_Success);
    VriDescriptorRangeUpdateDesc upd {};
    const VriDescriptor*         descs[1] = {uboDescriptor};
    upd.descriptors                       = descs;
    upd.descriptorNum                     = 1;
    upd.baseDescriptor                    = 0;
    c.UpdateDescriptorRanges(set, 0, 1, &upd);

    VriCommandAllocator* alloc = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
    VriBufferCopyDesc copy {};
    copy.size = 16;
    c.CmdCopyBuffer(cmd, ubo, staging, &copy);
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
    c.CmdSetPipelineLayout(cmd, layout);
    c.CmdSetDescriptorSet(cmd, 0, set);
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
    VriBufferTextureCopyDesc cp {};
    cp.texture.aspect   = VriImageAspect_Color;
    cp.texture.layerNum = 1;
    c.CmdReadbackTextureToBuffer(cmd, readback, color, &cp);
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
    const uint32_t o           = ((kH / 2) * kW + (kW / 2)) * 4;
    const bool     centerGreen = px[o + 0] == 0 && px[o + 1] == 255 && px[o + 2] == 0;
    c.UnmapBuffer(readback);
    CHECK(centerGreen);

    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    c.DestroyDescriptor(uboDescriptor);
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyBuffer(ubo);
    c.DestroyBuffer(staging);
    c.DestroyBuffer(readback);
    c.DestroyDescriptor(colorView);
    c.DestroyTexture(color);
    vriDestroyDevice(dev);
}
