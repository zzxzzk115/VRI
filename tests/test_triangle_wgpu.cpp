// WebGPU triangle parity test: render the (Slang-authored, WGSL-compiled)
// triangle offscreen and read back the center pixel. Mirrors the Vulkan test;
// WebGPU needs no explicit barriers. Skips gracefully without a device.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "common/shaders/triangle_wgsl.h" // g_triangleWgsl (const char[], vertexMain + fragmentMain)

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;
} // namespace

TEST_CASE("WebGPU: render a triangle offscreen and read back the center pixel")
{
    VriDeviceCreationDesc dc {};
    dc.graphicsAPI = VriGraphicsAPI_WebGPU;
    dc.bestEffort  = VRI_TRUE;

    VriDevice* dev = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("WebGPU device unavailable - skipping");
        return;
    }
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);

    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    VriTextureDesc td {};
    td.type           = VriTextureType_2D;
    td.format         = VriFormat_RGBA8_UNORM;
    td.width          = kW;
    td.height         = kH;
    td.depth          = 1;
    td.mipNum         = 1;
    td.layerNum       = 1;
    td.sampleNum      = 1;
    td.usage          = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
    td.memoryLocation = VriMemoryLocation_Device;
    VriTexture* color = nullptr;
    REQUIRE(c.CreateTexture(dev, &td, &color) == VriResult_Success);

    VriTextureViewDesc vd {};
    vd.texture          = color;
    vd.viewType         = VriTextureViewType_2D;
    vd.format           = VriFormat_Unknown;
    vd.aspect           = VriImageAspect_Color;
    VriDescriptor* view = nullptr;
    REQUIRE(c.CreateTextureView(dev, &vd, &view) == VriResult_Success);

    VriBufferDesc bd {};
    bd.size             = static_cast<uint64_t>(kW) * kH * 4; // 64*4 = 256 bytes/row (WebGPU-aligned)
    bd.usage            = VriBufferUsage_TransferDst;
    bd.memoryLocation   = VriMemoryLocation_HostReadback;
    VriBuffer* readback = nullptr;
    REQUIRE(c.CreateBuffer(dev, &bd, &readback) == VriResult_Success);

    VriPipelineLayoutDesc ld {};
    VriPipelineLayout*    layout = nullptr;
    REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

    VriShaderDesc shaders[2] {};
    shaders[0].stage          = VriShaderStage_Vertex;
    shaders[0].bytecode       = g_triangleWgsl;
    shaders[0].bytecodeSize   = sizeof(g_triangleWgsl);
    shaders[0].entryPointName = "vertexMain";
    shaders[1].stage          = VriShaderStage_Fragment;
    shaders[1].bytecode       = g_triangleWgsl;
    shaders[1].bytecodeSize   = sizeof(g_triangleWgsl);
    shaders[1].entryPointName = "fragmentMain";

    VriColorAttachmentDesc ca {};
    ca.format         = VriFormat_RGBA8_UNORM;
    ca.colorWriteMask = VriColorWrite_RGBA;

    VriGraphicsPipelineDesc pd {};
    pd.pipelineLayout          = layout;
    pd.shaders                 = shaders;
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

    VriAttachmentDesc rt {};
    rt.view                    = view;
    rt.loadOp                  = VriAttachmentLoadOp_Clear;
    rt.storeOp                 = VriAttachmentStoreOp_Store;
    rt.clearValue.color.f32[3] = 1.0f; // clear to opaque black
    VriAttachmentsDesc att {};
    att.colors            = &rt;
    att.colorNum          = 1;
    att.renderArea.width  = kW;
    att.renderArea.height = kH;
    att.layerNum          = 1;

    c.CmdBeginRendering(cmd, &att);
    VriViewport vp {0.0f, 0.0f, static_cast<float>(kW), static_cast<float>(kH), 0.0f, 1.0f};
    c.CmdSetViewports(cmd, &vp, 1);
    VriRect scissor {0, 0, kW, kH};
    c.CmdSetScissors(cmd, &scissor, 1);
    c.CmdSetPipeline(cmd, pipeline);
    VriDrawDesc draw {};
    draw.vertexNum   = 3;
    draw.instanceNum = 1;
    c.CmdDraw(cmd, &draw);
    c.CmdEndRendering(cmd);

    VriBufferTextureCopyDesc copy {};
    copy.texture.aspect   = VriImageAspect_Color;
    copy.texture.layerNum = 1;
    c.CmdReadbackTextureToBuffer(cmd, readback, color, &copy);

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

    const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bd.size));
    REQUIRE(px != nullptr);
    const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
    CHECK(px[o + 0] == 255); // R
    CHECK(px[o + 1] == 0);   // G
    CHECK(px[o + 2] == 0);   // B
    c.UnmapBuffer(readback);

    c.DeviceWaitIdle(dev);
    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyBuffer(readback);
    c.DestroyDescriptor(view);
    c.DestroyTexture(color);
    vriDestroyDevice(dev);
}
