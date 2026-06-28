// D3D12 sampled texture (phase 3b): separate SRV + sampler descriptor tables. Pass 1
// clears texA to blue; texA is transitioned to a shader resource; pass 2 samples texA
// (Texture2D + SamplerState) into texB; texB is read back. Exercises shader-visible
// CBV/SRV/UAV + sampler heaps, per-set descriptor tables, CreateShaderResourceView and
// CreateSampler. Center of texB must be blue. Render-to-texture avoids a texture upload.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/tests/triangle_tex_dxbc.h" // g_triangleTexDxbcVS + g_triangleTexDxbcPS

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;
} // namespace

TEST_CASE("D3D12: sampled texture (SRV + sampler tables) -> blue (phase 3b)")
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

    auto makeTex = [&](VriTextureUsageFlags usage, VriClearValue clear) {
        VriTextureDesc td {};
        td.type           = VriTextureType_2D;
        td.format         = VriFormat_RGBA8_UNORM;
        td.width          = kW;
        td.height         = kH;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = 1;
        td.sampleNum      = 1;
        td.usage          = usage;
        td.memoryLocation = VriMemoryLocation_Device;
        td.clearValue     = clear; // match the render-pass clear (fast-clear, no D3D12 mismatch warning)
        VriTexture* t     = nullptr;
        REQUIRE(c.CreateTexture(dev, &td, &t) == VriResult_Success);
        return t;
    };
    VriClearValue blueClear {};
    blueClear.color.f32[2] = 1.0f;
    blueClear.color.f32[3] = 1.0f; // texA cleared blue
    VriClearValue blackClear {};
    blackClear.color.f32[3] = 1.0f; // texB cleared opaque black
    VriTexture* texA =
        makeTex(VriTextureUsage_ColorAttachment | VriTextureUsage_ShaderResource, blueClear); // rendered then sampled
    VriTexture* texB =
        makeTex(VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc, blackClear); // final target
    VriTextureViewDesc va {};
    va.texture           = texA;
    va.viewType          = VriTextureViewType_2D;
    va.format            = VriFormat_Unknown;
    va.aspect            = VriImageAspect_Color;
    VriDescriptor* viewA = nullptr;
    REQUIRE(c.CreateTextureView(dev, &va, &viewA) == VriResult_Success);
    VriTextureViewDesc vb {};
    vb.texture           = texB;
    vb.viewType          = VriTextureViewType_2D;
    vb.format            = VriFormat_Unknown;
    vb.aspect            = VriImageAspect_Color;
    VriDescriptor* viewB = nullptr;
    REQUIRE(c.CreateTextureView(dev, &vb, &viewB) == VriResult_Success);

    VriSamplerDesc smpDesc {};
    smpDesc.magFilter      = VriFilter_Linear;
    smpDesc.minFilter      = VriFilter_Linear;
    smpDesc.addressModeU   = VriAddressMode_ClampToEdge;
    smpDesc.addressModeV   = VriAddressMode_ClampToEdge;
    smpDesc.addressModeW   = VriAddressMode_ClampToEdge;
    VriDescriptor* sampler = nullptr;
    REQUIRE(c.CreateSampler(dev, &smpDesc, &sampler) == VriResult_Success);

    VriBufferDesc rbd {};
    rbd.size            = static_cast<uint64_t>(kW) * kH * 4;
    rbd.usage           = VriBufferUsage_TransferDst;
    rbd.memoryLocation  = VriMemoryLocation_HostReadback;
    VriBuffer* readback = nullptr;
    REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

    VriDescriptorRangeDesc ranges[2] {};
    ranges[0].baseRegister   = 0;
    ranges[0].descriptorNum  = 1;
    ranges[0].descriptorType = VriDescriptorType_Texture;
    ranges[0].shaderStages   = VriShaderStage_Fragment;
    ranges[1].baseRegister   = 1;
    ranges[1].descriptorNum  = 1;
    ranges[1].descriptorType = VriDescriptorType_Sampler;
    ranges[1].shaderStages   = VriShaderStage_Fragment;
    VriDescriptorSetDesc setDesc {};
    setDesc.registerSpace = 0;
    setDesc.ranges        = ranges;
    setDesc.rangeNum      = 2;
    VriPipelineLayoutDesc ld {};
    ld.descriptorSets         = &setDesc;
    ld.descriptorSetNum       = 1;
    VriPipelineLayout* layout = nullptr;
    REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

    VriShaderDesc sh[2] {};
    sh[0].stage          = VriShaderStage_Vertex;
    sh[0].bytecode       = g_triangleTexDxbcVS;
    sh[0].bytecodeSize   = sizeof(g_triangleTexDxbcVS);
    sh[0].entryPointName = "vertexMain";
    sh[1].stage          = VriShaderStage_Fragment;
    sh[1].bytecode       = g_triangleTexDxbcPS;
    sh[1].bytecodeSize   = sizeof(g_triangleTexDxbcPS);
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
    pdsc.descriptorSetMaxNum = 1;
    pdsc.textureMaxNum       = 1;
    pdsc.samplerMaxNum       = 1;
    VriDescriptorPool* pool  = nullptr;
    REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
    VriDescriptorSet* set = nullptr;
    REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);
    VriDescriptorRangeUpdateDesc upd[2] {};
    const VriDescriptor*         texDescs[1] = {viewA};
    upd[0].descriptors                       = texDescs;
    upd[0].descriptorNum                     = 1;
    upd[0].baseDescriptor                    = 0;
    const VriDescriptor* smpDescs[1]         = {sampler};
    upd[1].descriptors                       = smpDescs;
    upd[1].descriptorNum                     = 1;
    upd[1].baseDescriptor                    = 0;
    c.UpdateDescriptorRanges(set, 0, 2, upd);

    VriCommandAllocator* alloc = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    auto barrier = [&](VriTexture*           t,
                       VriLayout             before,
                       VriLayout             after,
                       VriAccessFlags        afterAccess,
                       VriPipelineStageFlags afterStage) {
        VriTextureBarrierDesc tb {};
        tb.texture       = t;
        tb.before.layout = before;
        tb.after.layout  = after;
        tb.after.access  = afterAccess;
        tb.after.stages  = afterStage;
        tb.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc g {};
        g.textures   = &tb;
        g.textureNum = 1;
        c.CmdBarrier(cmd, &g);
    };

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
    // pass 1: clear texA to blue
    barrier(texA,
            VriLayout_Undefined,
            VriLayout_ColorAttachment,
            VriAccess_ColorAttachmentWrite,
            VriPipelineStage_ColorAttachmentOutput);
    {
        VriAttachmentDesc rt {};
        rt.view                    = viewA;
        rt.loadOp                  = VriAttachmentLoadOp_Clear;
        rt.storeOp                 = VriAttachmentStoreOp_Store;
        rt.clearValue.color.f32[2] = 1.0f;
        rt.clearValue.color.f32[3] = 1.0f; // blue
        VriAttachmentsDesc att {};
        att.colors            = &rt;
        att.colorNum          = 1;
        att.renderArea.width  = kW;
        att.renderArea.height = kH;
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        c.CmdEndRendering(cmd);
    }
    barrier(texA,
            VriLayout_ColorAttachment,
            VriLayout_ShaderResource,
            VriAccess_ShaderResourceRead,
            VriPipelineStage_FragmentShader);
    // pass 2: sample texA into texB
    barrier(texB,
            VriLayout_Undefined,
            VriLayout_ColorAttachment,
            VriAccess_ColorAttachmentWrite,
            VriPipelineStage_ColorAttachmentOutput);
    {
        VriAttachmentDesc rt {};
        rt.view                    = viewB;
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
    }
    barrier(texB, VriLayout_ColorAttachment, VriLayout_CopySource, VriAccess_CopySourceRead, VriPipelineStage_Transfer);
    VriBufferTextureCopyDesc cp {};
    cp.texture.aspect   = VriImageAspect_Color;
    cp.texture.layerNum = 1;
    c.CmdReadbackTextureToBuffer(cmd, readback, texB, &cp);
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
    const uint32_t o    = ((kH / 2) * kW + (kW / 2)) * 4;
    const bool     blue = px[o + 0] == 0 && px[o + 1] == 0 && px[o + 2] == 255;
    c.UnmapBuffer(readback);
    CHECK(blue);

    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    c.DestroyDescriptor(sampler);
    c.DestroyDescriptor(viewA);
    c.DestroyDescriptor(viewB);
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyBuffer(readback);
    c.DestroyTexture(texA);
    c.DestroyTexture(texB);
    vriDestroyDevice(dev);
}
