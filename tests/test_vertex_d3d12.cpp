// D3D12 vertex-buffer + indexed draw (phase 2): a vertex-colored (yellow) triangle
// from a vertex buffer + index buffer, exercising the reflection-built input layout
// (POSITION/COLOR0 semantics from the Slang DXBC) and DrawIndexedInstanced. Vertex/
// index data lives in UPLOAD-heap buffers (usable directly as VB/IB), so no staging
// copy is needed. Center pixel must be yellow - parity with the other backends.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>

#include "shaders/triangle_vbuf_dxbc.h" // g_triangleVbufDxbcVS + g_triangleVbufDxbcPS

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;
    const float kVertices[] = {
        0.0f, -0.5f, 0.0f, 1.0f, 1.0f, 0.0f,
        0.5f,  0.5f, 0.0f, 1.0f, 1.0f, 0.0f,
       -0.5f,  0.5f, 0.0f, 1.0f, 1.0f, 0.0f,
    };
    const uint32_t kIndices[] = {0, 1, 2};
}

TEST_CASE("D3D12: vertex-buffer + indexed draw -> yellow triangle (phase 2)")
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

    auto upload = [&](const void* data, uint64_t size, VriBufferUsageFlags usage) {
        VriBufferDesc bd{}; bd.size = size; bd.usage = usage; bd.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* b = nullptr; REQUIRE(c.CreateBuffer(dev, &bd, &b) == VriResult_Success);
        void* m = c.MapBuffer(b, 0, size); REQUIRE(m != nullptr); std::memcpy(m, data, static_cast<size_t>(size)); c.UnmapBuffer(b);
        return b;
    };
    VriBuffer* vbuf = upload(kVertices, sizeof(kVertices), VriBufferUsage_VertexBuffer);
    VriBuffer* ibuf = upload(kIndices, sizeof(kIndices), VriBufferUsage_IndexBuffer);

    VriPipelineLayoutDesc ld{};
    VriPipelineLayout* layout = nullptr; REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

    VriShaderDesc sh[2]{};
    sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = g_triangleVbufDxbcVS; sh[0].bytecodeSize = sizeof(g_triangleVbufDxbcVS); sh[0].entryPointName = "vertexMain";
    sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = g_triangleVbufDxbcPS; sh[1].bytecodeSize = sizeof(g_triangleVbufDxbcPS); sh[1].entryPointName = "fragmentMain";
    VriVertexStreamDesc stream{}; stream.stride = 6 * sizeof(float); stream.bindingSlot = 0; stream.stepRate = VriVertexStepRate_PerVertex;
    VriVertexAttributeDesc attrs[2]{};
    attrs[0].format = VriFormat_RGB32_SFLOAT; attrs[0].offset = 0;                attrs[0].streamIndex = 0;
    attrs[1].format = VriFormat_RGB32_SFLOAT; attrs[1].offset = 3 * sizeof(float); attrs[1].streamIndex = 0;
    VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd{};
    pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
    pd.vertexInput.streams = &stream; pd.vertexInput.streamNum = 1; pd.vertexInput.attributes = attrs; pd.vertexInput.attributeNum = 2;
    pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
    pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
    VriPipeline* pipeline = nullptr; REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

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
    VriAttachmentDesc rt{}; rt.view = colorView; rt.loadOp = VriAttachmentLoadOp_Clear; rt.storeOp = VriAttachmentStoreOp_Store; rt.clearValue.color.f32[3] = 1.0f;
    VriAttachmentsDesc att{}; att.colors = &rt; att.colorNum = 1; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
    c.CmdBeginRendering(cmd, &att);
    VriViewport vp{0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
    VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
    c.CmdSetPipeline(cmd, pipeline);
    VriVertexBufferBinding vbb{}; vbb.buffer = vbuf; vbb.offset = 0; c.CmdSetVertexBuffers(cmd, 0, &vbb, 1);
    c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt32);
    VriDrawIndexedDesc di{}; di.indexNum = 3; di.instanceNum = 1; c.CmdDrawIndexed(cmd, &di);
    c.CmdEndRendering(cmd);
    {
        VriTextureBarrierDesc tb{};
        tb.texture = color; tb.before.access = VriAccess_ColorAttachmentWrite; tb.before.layout = VriLayout_ColorAttachment; tb.before.stages = VriPipelineStage_ColorAttachmentOutput;
        tb.after.access = VriAccess_CopySourceRead; tb.after.layout = VriLayout_CopySource; tb.after.stages = VriPipelineStage_Transfer;
        tb.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc g{}; g.textures = &tb; g.textureNum = 1; c.CmdBarrier(cmd, &g);
    }
    VriBufferTextureCopyDesc cp{}; cp.texture.aspect = VriImageAspect_Color; cp.texture.layerNum = 1;
    c.CmdReadbackTextureToBuffer(cmd, readback, color, &cp);
    REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

    VriFenceSubmitDesc signal{}; signal.fence = fence; signal.value = 1;
    VriQueueSubmitDesc submit{}; submit.commandBuffers = &cmd; submit.commandBufferNum = 1; submit.signalFences = &signal; submit.signalFenceNum = 1;
    c.QueueSubmit(queue, &submit);
    c.Wait(fence, 1);

    const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
    REQUIRE(px != nullptr);
    const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
    const bool yellow = px[o + 0] == 255 && px[o + 1] == 255 && px[o + 2] == 0;
    c.UnmapBuffer(readback);
    CHECK(yellow);

    c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
    c.DestroyBuffer(vbuf); c.DestroyBuffer(ibuf); c.DestroyBuffer(readback); c.DestroyDescriptor(colorView); c.DestroyTexture(color);
    vriDestroyDevice(dev);
}
