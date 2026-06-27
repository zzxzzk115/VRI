// D3D12 depth test (DSV attachment + depth-test PSO). Two overlapping triangles via a
// vertex buffer: a GREEN one at z=0.3 then a RED one at z=0.7, drawn in that order with
// depthCompareOp=Less + depth write. The red (farther) must fail the depth test, so the
// center stays GREEN regardless of draw order - proving depth testing works. Exercises
// the DSV heap, depth attachment in CmdBeginRendering, depth clear, and depth PSO state.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>

#include "shaders/triangle_vbuf_dxbc.h" // g_triangleVbufDxbcVS + g_triangleVbufDxbcPS

namespace
{
    constexpr uint32_t kW          = 64;
    constexpr uint32_t kH          = 64;
    const float        kVertices[] = {
        // green triangle, z = 0.3 (near)
        0.0f,
        -0.5f,
        0.3f,
        0.0f,
        1.0f,
        0.0f,
        0.5f,
        0.5f,
        0.3f,
        0.0f,
        1.0f,
        0.0f,
        -0.5f,
        0.5f,
        0.3f,
        0.0f,
        1.0f,
        0.0f,
        // red triangle, z = 0.7 (far)
        0.0f,
        -0.5f,
        0.7f,
        1.0f,
        0.0f,
        0.0f,
        0.5f,
        0.5f,
        0.7f,
        1.0f,
        0.0f,
        0.0f,
        -0.5f,
        0.5f,
        0.7f,
        1.0f,
        0.0f,
        0.0f,
    };
} // namespace

TEST_CASE("D3D12: depth test keeps the nearer triangle (green over red)")
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

    VriTextureDesc dtd {};
    dtd.type                          = VriTextureType_2D;
    dtd.format                        = VriFormat_D32_SFLOAT;
    dtd.width                         = kW;
    dtd.height                        = kH;
    dtd.depth                         = 1;
    dtd.mipNum                        = 1;
    dtd.layerNum                      = 1;
    dtd.sampleNum                     = 1;
    dtd.usage                         = VriTextureUsage_DepthStencilAttachment;
    dtd.memoryLocation                = VriMemoryLocation_Device;
    dtd.clearValue.depthStencil.depth = 1.0f; // match the render-pass clear
    VriTexture* depth                 = nullptr;
    REQUIRE(c.CreateTexture(dev, &dtd, &depth) == VriResult_Success);
    VriTextureViewDesc dvd {};
    dvd.texture              = depth;
    dvd.viewType             = VriTextureViewType_2D;
    dvd.format               = VriFormat_D32_SFLOAT;
    dvd.aspect               = VriImageAspect_Depth;
    VriDescriptor* depthView = nullptr;
    REQUIRE(c.CreateTextureView(dev, &dvd, &depthView) == VriResult_Success);

    VriBufferDesc rbd {};
    rbd.size            = static_cast<uint64_t>(kW) * kH * 4;
    rbd.usage           = VriBufferUsage_TransferDst;
    rbd.memoryLocation  = VriMemoryLocation_HostReadback;
    VriBuffer* readback = nullptr;
    REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);
    VriBufferDesc vbd {};
    vbd.size           = sizeof(kVertices);
    vbd.usage          = VriBufferUsage_VertexBuffer;
    vbd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* vbuf    = nullptr;
    REQUIRE(c.CreateBuffer(dev, &vbd, &vbuf) == VriResult_Success);
    {
        void* m = c.MapBuffer(vbuf, 0, sizeof(kVertices));
        REQUIRE(m != nullptr);
        std::memcpy(m, kVertices, sizeof(kVertices));
        c.UnmapBuffer(vbuf);
    }

    VriPipelineLayoutDesc ld {};
    VriPipelineLayout*    layout = nullptr;
    REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);
    VriShaderDesc sh[2] {};
    sh[0].stage          = VriShaderStage_Vertex;
    sh[0].bytecode       = g_triangleVbufDxbcVS;
    sh[0].bytecodeSize   = sizeof(g_triangleVbufDxbcVS);
    sh[0].entryPointName = "vertexMain";
    sh[1].stage          = VriShaderStage_Fragment;
    sh[1].bytecode       = g_triangleVbufDxbcPS;
    sh[1].bytecodeSize   = sizeof(g_triangleVbufDxbcPS);
    sh[1].entryPointName = "fragmentMain";
    VriVertexStreamDesc stream {};
    stream.stride      = 6 * sizeof(float);
    stream.bindingSlot = 0;
    stream.stepRate    = VriVertexStepRate_PerVertex;
    VriVertexAttributeDesc attrs[2] {};
    attrs[0].format      = VriFormat_RGB32_SFLOAT;
    attrs[0].offset      = 0;
    attrs[0].streamIndex = 0;
    attrs[1].format      = VriFormat_RGB32_SFLOAT;
    attrs[1].offset      = 3 * sizeof(float);
    attrs[1].streamIndex = 0;
    VriColorAttachmentDesc ca {};
    ca.format         = VriFormat_RGBA8_UNORM;
    ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd {};
    pd.pipelineLayout                  = layout;
    pd.shaders                         = sh;
    pd.shaderNum                       = 2;
    pd.vertexInput.streams             = &stream;
    pd.vertexInput.streamNum           = 1;
    pd.vertexInput.attributes          = attrs;
    pd.vertexInput.attributeNum        = 2;
    pd.inputAssembly.topology          = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode          = VriCullMode_None;
    pd.rasterization.lineWidth         = 1.0f;
    pd.depthStencil.depthTest          = VRI_TRUE;
    pd.depthStencil.depthWrite         = VRI_TRUE;
    pd.depthStencil.depthCompareOp     = VriCompareOp_Less;
    pd.multisample.sampleNum           = 1;
    pd.outputMerger.colors             = &ca;
    pd.outputMerger.colorNum           = 1;
    pd.outputMerger.depthStencilFormat = VriFormat_D32_SFLOAT;
    VriPipeline* pipeline              = nullptr;
    REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

    VriCommandAllocator* alloc = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
    {
        VriTextureBarrierDesc tb[2] {};
        tb[0].texture       = color;
        tb[0].before.layout = VriLayout_Undefined;
        tb[0].after.access  = VriAccess_ColorAttachmentWrite;
        tb[0].after.layout  = VriLayout_ColorAttachment;
        tb[0].after.stages  = VriPipelineStage_ColorAttachmentOutput;
        tb[0].aspect        = VriImageAspect_Color;
        tb[1].texture       = depth;
        tb[1].before.layout = VriLayout_Undefined;
        tb[1].after.access  = VriAccess_DepthStencilAttachmentWrite;
        tb[1].after.layout  = VriLayout_DepthStencilAttachment;
        tb[1].after.stages  = VriPipelineStage_EarlyFragmentTests;
        tb[1].aspect        = VriImageAspect_Depth;
        VriBarrierGroupDesc g {};
        g.textures   = tb;
        g.textureNum = 2;
        c.CmdBarrier(cmd, &g);
    }
    VriAttachmentDesc rt {};
    rt.view                    = colorView;
    rt.loadOp                  = VriAttachmentLoadOp_Clear;
    rt.storeOp                 = VriAttachmentStoreOp_Store;
    rt.clearValue.color.f32[3] = 1.0f;
    VriAttachmentDesc dat {};
    dat.view                          = depthView;
    dat.loadOp                        = VriAttachmentLoadOp_Clear;
    dat.storeOp                       = VriAttachmentStoreOp_Store;
    dat.clearValue.depthStencil.depth = 1.0f;
    VriAttachmentsDesc att {};
    att.colors            = &rt;
    att.colorNum          = 1;
    att.depth             = &dat;
    att.renderArea.width  = kW;
    att.renderArea.height = kH;
    att.layerNum          = 1;
    c.CmdBeginRendering(cmd, &att);
    VriViewport vp {0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1};
    c.CmdSetViewports(cmd, &vp, 1);
    VriRect sc {0, 0, kW, kH};
    c.CmdSetScissors(cmd, &sc, 1);
    c.CmdSetPipeline(cmd, pipeline);
    VriVertexBufferBinding vbb {};
    vbb.buffer = vbuf;
    vbb.offset = 0;
    c.CmdSetVertexBuffers(cmd, 0, &vbb, 1);
    VriDrawDesc green {};
    green.vertexNum   = 3;
    green.instanceNum = 1;
    green.baseVertex  = 0;
    c.CmdDraw(cmd, &green); // near
    VriDrawDesc red {};
    red.vertexNum   = 3;
    red.instanceNum = 1;
    red.baseVertex  = 3;
    c.CmdDraw(cmd, &red); // far -> depth-fails
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
    const uint32_t o      = ((kH / 2) * kW + (kW / 2)) * 4;
    const bool     green2 = px[o + 0] == 0 && px[o + 1] == 255 && px[o + 2] == 0;
    c.UnmapBuffer(readback);
    CHECK(green2);

    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyBuffer(vbuf);
    c.DestroyBuffer(readback);
    c.DestroyDescriptor(colorView);
    c.DestroyDescriptor(depthView);
    c.DestroyTexture(color);
    c.DestroyTexture(depth);
    vriDestroyDevice(dev);
}
