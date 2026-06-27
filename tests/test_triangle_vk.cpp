// Phase 1 verification: render a triangle to an offscreen RGBA8 texture with
// the Vulkan backend, read it back, and check the center pixel. Skips
// gracefully (no failure) when no Vulkan device is available.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>

#include "shaders/triangle_spv.h"

namespace
{
    constexpr uint32_t kWidth  = 64;
    constexpr uint32_t kHeight = 64;

    struct Vk
    {
        VriDevice*       device = nullptr;
        VriCoreInterface core {};
        bool             ok = false;

        ~Vk()
        {
            if (device)
                vriDestroyDevice(device);
        }
    };

    bool InitVk(Vk& vk)
    {
        VriDeviceCreationDesc desc {};
        desc.graphicsAPI      = VriGraphicsAPI_Vulkan;
        desc.enableValidation = VRI_TRUE;
        desc.bestEffort       = VRI_TRUE;

        if (vriCreateDevice(&desc, &vk.device) != VriResult_Success)
            return false;
        if (vriGetInterface(vk.device, VRI_INTERFACE_CORE, sizeof(vk.core), &vk.core) != VriResult_Success)
            return false;
        vk.ok = true;
        return true;
    }
} // namespace

TEST_CASE("Vulkan: render a triangle offscreen and read back the center pixel")
{
    Vk vk;
    if (!InitVk(vk))
    {
        MESSAGE("Vulkan device unavailable - skipping triangle render test");
        return;
    }
    const VriCoreInterface& c   = vk.core;
    VriDevice*              dev = vk.device;

    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    // color target
    VriTextureDesc texDesc {};
    texDesc.type           = VriTextureType_2D;
    texDesc.format         = VriFormat_RGBA8_UNORM;
    texDesc.width          = kWidth;
    texDesc.height         = kHeight;
    texDesc.depth          = 1;
    texDesc.mipNum         = 1;
    texDesc.layerNum       = 1;
    texDesc.sampleNum      = 1;
    texDesc.usage          = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
    texDesc.memoryLocation = VriMemoryLocation_Device;
    VriTexture* color      = nullptr;
    REQUIRE(c.CreateTexture(dev, &texDesc, &color) == VriResult_Success);

    VriTextureViewDesc viewDesc {};
    viewDesc.texture         = color;
    viewDesc.viewType        = VriTextureViewType_2D;
    viewDesc.format          = VriFormat_Unknown;
    viewDesc.aspect          = VriImageAspect_Color;
    VriDescriptor* colorView = nullptr;
    REQUIRE(c.CreateTextureView(dev, &viewDesc, &colorView) == VriResult_Success);

    // readback buffer
    VriBufferDesc bufDesc {};
    bufDesc.size           = static_cast<uint64_t>(kWidth) * kHeight * 4;
    bufDesc.usage          = VriBufferUsage_TransferDst;
    bufDesc.memoryLocation = VriMemoryLocation_HostReadback;
    VriBuffer* readback    = nullptr;
    REQUIRE(c.CreateBuffer(dev, &bufDesc, &readback) == VriResult_Success);

    // pipeline (empty layout, no vertex input)
    VriPipelineLayoutDesc layoutDesc {};
    VriPipelineLayout*    layout = nullptr;
    REQUIRE(c.CreatePipelineLayout(dev, &layoutDesc, &layout) == VriResult_Success);

    // One Slang module, two entry points referenced by name.
    VriShaderDesc shaders[2] {};
    shaders[0].stage          = VriShaderStage_Vertex;
    shaders[0].bytecode       = g_triangleSpv;
    shaders[0].bytecodeSize   = sizeof(g_triangleSpv);
    shaders[0].entryPointName = "vertexMain";
    shaders[1].stage          = VriShaderStage_Fragment;
    shaders[1].bytecode       = g_triangleSpv;
    shaders[1].bytecodeSize   = sizeof(g_triangleSpv);
    shaders[1].entryPointName = "fragmentMain";

    VriColorAttachmentDesc colorAttach {};
    colorAttach.format         = VriFormat_RGBA8_UNORM;
    colorAttach.colorWriteMask = VriColorWrite_RGBA;

    VriGraphicsPipelineDesc pipeDesc {};
    pipeDesc.pipelineLayout                  = layout;
    pipeDesc.shaders                         = shaders;
    pipeDesc.shaderNum                       = 2;
    pipeDesc.inputAssembly.topology          = VriPrimitiveTopology_TriangleList;
    pipeDesc.rasterization.polygonMode       = VriPolygonMode_Fill;
    pipeDesc.rasterization.cullMode          = VriCullMode_None;
    pipeDesc.rasterization.frontFace         = VriFrontFace_CounterClockwise;
    pipeDesc.rasterization.lineWidth         = 1.0f;
    pipeDesc.multisample.sampleNum           = 1;
    pipeDesc.outputMerger.colors             = &colorAttach;
    pipeDesc.outputMerger.colorNum           = 1;
    pipeDesc.outputMerger.depthStencilFormat = VriFormat_Unknown;
    VriPipeline* pipeline                    = nullptr;
    REQUIRE(c.CreateGraphicsPipeline(dev, &pipeDesc, &pipeline) == VriResult_Success);

    // command recording
    VriCommandAllocator* allocator = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &allocator) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(allocator, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);

    // Undefined -> ColorAttachment
    {
        VriTextureBarrierDesc b {};
        b.texture       = color;
        b.before.layout = VriLayout_Undefined;
        b.before.stages = VriPipelineStage_None;
        b.after.access  = VriAccess_ColorAttachmentWrite;
        b.after.layout  = VriLayout_ColorAttachment;
        b.after.stages  = VriPipelineStage_ColorAttachmentOutput;
        b.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc g {};
        g.textures   = &b;
        g.textureNum = 1;
        c.CmdBarrier(cmd, &g);
    }

    VriAttachmentDesc colorRT {};
    colorRT.view                    = colorView;
    colorRT.loadOp                  = VriAttachmentLoadOp_Clear;
    colorRT.storeOp                 = VriAttachmentStoreOp_Store;
    colorRT.clearValue.color.f32[0] = 0.0f;
    colorRT.clearValue.color.f32[1] = 0.0f;
    colorRT.clearValue.color.f32[2] = 0.0f;
    colorRT.clearValue.color.f32[3] = 1.0f;

    VriAttachmentsDesc attachments {};
    attachments.colors            = &colorRT;
    attachments.colorNum          = 1;
    attachments.renderArea.width  = kWidth;
    attachments.renderArea.height = kHeight;
    attachments.layerNum          = 1;

    c.CmdBeginRendering(cmd, &attachments);
    VriViewport vp {0.0f, 0.0f, static_cast<float>(kWidth), static_cast<float>(kHeight), 0.0f, 1.0f};
    c.CmdSetViewports(cmd, &vp, 1);
    VriRect scissor {0, 0, kWidth, kHeight};
    c.CmdSetScissors(cmd, &scissor, 1);
    c.CmdSetPipeline(cmd, pipeline);
    VriDrawDesc draw {};
    draw.vertexNum   = 3;
    draw.instanceNum = 1;
    c.CmdDraw(cmd, &draw);
    c.CmdEndRendering(cmd);

    // ColorAttachment -> CopySource
    {
        VriTextureBarrierDesc b {};
        b.texture       = color;
        b.before.access = VriAccess_ColorAttachmentWrite;
        b.before.layout = VriLayout_ColorAttachment;
        b.before.stages = VriPipelineStage_ColorAttachmentOutput;
        b.after.access  = VriAccess_CopySourceRead;
        b.after.layout  = VriLayout_CopySource;
        b.after.stages  = VriPipelineStage_Transfer;
        b.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc g {};
        g.textures   = &b;
        g.textureNum = 1;
        c.CmdBarrier(cmd, &g);
    }

    VriBufferTextureCopyDesc copy {};
    copy.texture.aspect   = VriImageAspect_Color;
    copy.texture.layerNum = 1;
    c.CmdReadbackTextureToBuffer(cmd, readback, color, &copy);

    REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

    VriFenceSubmitDesc signal {};
    signal.fence  = fence;
    signal.value  = 1;
    signal.stages = VriPipelineStage_AllCommands;
    VriQueueSubmitDesc submit {};
    submit.commandBuffers   = &cmd;
    submit.commandBufferNum = 1;
    submit.signalFences     = &signal;
    submit.signalFenceNum   = 1;
    c.QueueSubmit(queue, &submit);
    c.Wait(fence, 1);

    // verify center pixel == red
    const uint8_t* pixels = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bufDesc.size));
    REQUIRE(pixels != nullptr);
    const uint32_t centerOffset = ((kHeight / 2) * kWidth + (kWidth / 2)) * 4;
    CHECK(pixels[centerOffset + 0] == 255); // R
    CHECK(pixels[centerOffset + 1] == 0);   // G
    CHECK(pixels[centerOffset + 2] == 0);   // B
    CHECK(pixels[centerOffset + 3] == 255); // A
    c.UnmapBuffer(readback);

    // cleanup
    c.DeviceWaitIdle(dev);
    c.DestroyFence(fence);
    c.DestroyCommandAllocator(allocator);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyBuffer(readback);
    c.DestroyDescriptor(colorView);
    c.DestroyTexture(color);
}
