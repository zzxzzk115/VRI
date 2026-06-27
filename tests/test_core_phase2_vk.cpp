// Phase 2 core verification (Vulkan): the explicit memory path
// (GetMemoryDesc -> AllocateMemory -> Bind) and the descriptor-set path
// (pool -> allocate -> update -> bind). Skips gracefully without a device.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/triangle_spv.h"
#include "shaders/triangle_ubo_spv.h"

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    struct Ctx
    {
        VriDevice*       device = nullptr;
        VriCoreInterface core {};
        ~Ctx()
        {
            if (device)
                vriDestroyDevice(device);
        }
    };

    bool Init(Ctx& ctx)
    {
        VriDeviceCreationDesc desc {};
        desc.graphicsAPI      = VriGraphicsAPI_Vulkan;
        desc.enableValidation = VRI_TRUE;
        desc.bestEffort       = VRI_TRUE;
        if (vriCreateDevice(&desc, &ctx.device) != VriResult_Success)
            return false;
        return vriGetInterface(ctx.device, VRI_INTERFACE_CORE, sizeof(ctx.core), &ctx.core) == VriResult_Success;
    }

    VriTextureViewDesc ColorViewDesc(VriTexture* t)
    {
        VriTextureViewDesc v {};
        v.texture  = t;
        v.viewType = VriTextureViewType_2D;
        v.format   = VriFormat_Unknown;
        v.aspect   = VriImageAspect_Color;
        return v;
    }

    void TransitionToColor(const VriCoreInterface& c, VriCommandBuffer* cmd, VriTexture* tex)
    {
        VriTextureBarrierDesc b {};
        b.texture       = tex;
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

    void TransitionToCopySrc(const VriCoreInterface& c, VriCommandBuffer* cmd, VriTexture* tex)
    {
        VriTextureBarrierDesc b {};
        b.texture       = tex;
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
} // namespace

TEST_CASE("Vulkan core: explicit memory (allocate + bind) backs a render target")
{
    Ctx ctx;
    if (!Init(ctx))
    {
        MESSAGE("Vulkan unavailable - skipping");
        return;
    }
    const VriCoreInterface& c   = ctx.core;
    VriDevice*              dev = ctx.device;

    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    // color target created UNBOUND, then backed by explicitly allocated memory.
    VriTextureDesc texDesc {};
    texDesc.type           = VriTextureType_2D;
    texDesc.format         = VriFormat_RGBA8_UNORM;
    texDesc.width          = kW;
    texDesc.height         = kH;
    texDesc.depth          = 1;
    texDesc.mipNum         = 1;
    texDesc.layerNum       = 1;
    texDesc.sampleNum      = 1;
    texDesc.usage          = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
    texDesc.memoryLocation = VriMemoryLocation_Undefined;

    VriMemoryDesc memDesc {};
    c.GetTextureMemoryDesc(dev, &texDesc, VriMemoryLocation_Device, &memDesc);
    CHECK(memDesc.size >= static_cast<uint64_t>(kW) * kH * 4);
    CHECK(memDesc.memoryTypeMask != 0);

    VriMemory* memory = nullptr;
    REQUIRE(c.AllocateMemory(dev, &memDesc, &memory) == VriResult_Success);

    VriTexture* color = nullptr;
    REQUIRE(c.CreateTexture(dev, &texDesc, &color) == VriResult_Success);
    REQUIRE(c.BindTextureMemory(dev, color, memory, 0) == VriResult_Success);

    VriTextureViewDesc viewDesc = ColorViewDesc(color);
    VriDescriptor*     view     = nullptr;
    REQUIRE(c.CreateTextureView(dev, &viewDesc, &view) == VriResult_Success);

    VriBufferDesc rbDesc {};
    rbDesc.size           = static_cast<uint64_t>(kW) * kH * 4;
    rbDesc.usage          = VriBufferUsage_TransferDst;
    rbDesc.memoryLocation = VriMemoryLocation_HostReadback;
    VriBuffer* readback   = nullptr;
    REQUIRE(c.CreateBuffer(dev, &rbDesc, &readback) == VriResult_Success);

    VriPipelineLayoutDesc layoutDesc {};
    VriPipelineLayout*    layout = nullptr;
    REQUIRE(c.CreatePipelineLayout(dev, &layoutDesc, &layout) == VriResult_Success);

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

    VriGraphicsPipelineDesc pd {};
    pd.pipelineLayout          = layout;
    pd.shaders                 = shaders;
    pd.shaderNum               = 2;
    pd.inputAssembly.topology  = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode  = VriCullMode_None;
    pd.rasterization.lineWidth = 1.0f;
    pd.multisample.sampleNum   = 1;
    pd.outputMerger.colors     = &colorAttach;
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
    TransitionToColor(c, cmd, color);

    VriAttachmentDesc rt {};
    rt.view                    = view;
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
    VriViewport vp {0, 0, (float)kW, (float)kH, 0, 1};
    c.CmdSetViewports(cmd, &vp, 1);
    VriRect sc {0, 0, kW, kH};
    c.CmdSetScissors(cmd, &sc, 1);
    c.CmdSetPipeline(cmd, pipeline);
    VriDrawDesc draw {};
    draw.vertexNum   = 3;
    draw.instanceNum = 1;
    c.CmdDraw(cmd, &draw);
    c.CmdEndRendering(cmd);
    TransitionToCopySrc(c, cmd, color);

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

    const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbDesc.size));
    REQUIRE(px != nullptr);
    const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
    CHECK(px[o + 0] == 255);
    CHECK(px[o + 1] == 0);
    CHECK(px[o + 2] == 0);
    c.UnmapBuffer(readback);

    c.DeviceWaitIdle(dev);
    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyBuffer(readback);
    c.DestroyDescriptor(view);
    c.DestroyTexture(color);
    c.FreeMemory(memory);
}

TEST_CASE("Vulkan core: descriptor set with a uniform buffer tints the triangle")
{
    Ctx ctx;
    if (!Init(ctx))
    {
        MESSAGE("Vulkan unavailable - skipping");
        return;
    }
    const VriCoreInterface& c   = ctx.core;
    VriDevice*              dev = ctx.device;

    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    // color target (VMA-backed)
    VriTextureDesc texDesc {};
    texDesc.type           = VriTextureType_2D;
    texDesc.format         = VriFormat_RGBA8_UNORM;
    texDesc.width          = kW;
    texDesc.height         = kH;
    texDesc.depth          = 1;
    texDesc.mipNum         = 1;
    texDesc.layerNum       = 1;
    texDesc.sampleNum      = 1;
    texDesc.usage          = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
    texDesc.memoryLocation = VriMemoryLocation_Device;
    VriTexture* color      = nullptr;
    REQUIRE(c.CreateTexture(dev, &texDesc, &color) == VriResult_Success);
    VriTextureViewDesc viewDesc  = ColorViewDesc(color);
    VriDescriptor*     colorView = nullptr;
    REQUIRE(c.CreateTextureView(dev, &viewDesc, &colorView) == VriResult_Success);

    VriBufferDesc rbDesc {};
    rbDesc.size           = static_cast<uint64_t>(kW) * kH * 4;
    rbDesc.usage          = VriBufferUsage_TransferDst;
    rbDesc.memoryLocation = VriMemoryLocation_HostReadback;
    VriBuffer* readback   = nullptr;
    REQUIRE(c.CreateBuffer(dev, &rbDesc, &readback) == VriResult_Success);

    // uniform buffer holding the tint color (green)
    VriBufferDesc uboDesc {};
    uboDesc.size           = 16;
    uboDesc.usage          = VriBufferUsage_ConstantBuffer;
    uboDesc.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* ubo         = nullptr;
    REQUIRE(c.CreateBuffer(dev, &uboDesc, &ubo) == VriResult_Success);
    {
        float* m = static_cast<float*>(c.MapBuffer(ubo, 0, 16));
        REQUIRE(m != nullptr);
        m[0] = 0.0f;
        m[1] = 1.0f;
        m[2] = 0.0f;
        m[3] = 1.0f; // green
        c.UnmapBuffer(ubo);
    }

    // pipeline layout: set 0, binding 0 = constant buffer (fragment)
    VriDescriptorRangeDesc range {};
    range.baseRegister   = 0;
    range.descriptorNum  = 1;
    range.descriptorType = VriDescriptorType_ConstantBuffer;
    range.shaderStages   = VriShaderStage_Fragment;
    VriDescriptorSetDesc setDesc {};
    setDesc.registerSpace = 0;
    setDesc.ranges        = &range;
    setDesc.rangeNum      = 1;
    VriPipelineLayoutDesc layoutDesc {};
    layoutDesc.descriptorSets   = &setDesc;
    layoutDesc.descriptorSetNum = 1;
    VriPipelineLayout* layout   = nullptr;
    REQUIRE(c.CreatePipelineLayout(dev, &layoutDesc, &layout) == VriResult_Success);

    VriShaderDesc shaders[2] {};
    shaders[0].stage          = VriShaderStage_Vertex;
    shaders[0].bytecode       = g_triangleUboSpv;
    shaders[0].bytecodeSize   = sizeof(g_triangleUboSpv);
    shaders[0].entryPointName = "vertexMain";
    shaders[1].stage          = VriShaderStage_Fragment;
    shaders[1].bytecode       = g_triangleUboSpv;
    shaders[1].bytecodeSize   = sizeof(g_triangleUboSpv);
    shaders[1].entryPointName = "fragmentMain";

    VriColorAttachmentDesc colorAttach {};
    colorAttach.format         = VriFormat_RGBA8_UNORM;
    colorAttach.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd {};
    pd.pipelineLayout          = layout;
    pd.shaders                 = shaders;
    pd.shaderNum               = 2;
    pd.inputAssembly.topology  = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode  = VriCullMode_None;
    pd.rasterization.lineWidth = 1.0f;
    pd.multisample.sampleNum   = 1;
    pd.outputMerger.colors     = &colorAttach;
    pd.outputMerger.colorNum   = 1;
    VriPipeline* pipeline      = nullptr;
    REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

    // descriptor pool + set + update
    VriDescriptorPoolDesc poolDesc {};
    poolDesc.descriptorSetMaxNum  = 1;
    poolDesc.constantBufferMaxNum = 1;
    VriDescriptorPool* pool       = nullptr;
    REQUIRE(c.CreateDescriptorPool(dev, &poolDesc, &pool) == VriResult_Success);

    VriDescriptorSet* set = nullptr;
    REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);

    VriBufferViewDesc uboView {};
    uboView.buffer               = ubo;
    uboView.viewType             = VriDescriptorType_ConstantBuffer;
    uboView.offset               = 0;
    uboView.size                 = 16;
    VriDescriptor* uboDescriptor = nullptr;
    REQUIRE(c.CreateBufferView(dev, &uboView, &uboDescriptor) == VriResult_Success);

    VriDescriptorRangeUpdateDesc update {};
    const VriDescriptor*         descriptors[1] = {uboDescriptor};
    update.descriptors                          = descriptors;
    update.descriptorNum                        = 1;
    update.baseDescriptor                       = 0;
    c.UpdateDescriptorRanges(set, 0, 1, &update);

    // record
    VriCommandAllocator* alloc = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
    TransitionToColor(c, cmd, color);
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
    VriViewport vp {0, 0, (float)kW, (float)kH, 0, 1};
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
    TransitionToCopySrc(c, cmd, color);
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

    const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbDesc.size));
    REQUIRE(px != nullptr);
    const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
    CHECK(px[o + 0] == 0);   // R
    CHECK(px[o + 1] == 255); // G  (tint from UBO)
    CHECK(px[o + 2] == 0);   // B
    c.UnmapBuffer(readback);

    c.DeviceWaitIdle(dev);
    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    c.DestroyDescriptor(uboDescriptor);
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyBuffer(ubo);
    c.DestroyBuffer(readback);
    c.DestroyDescriptor(colorView);
    c.DestroyTexture(color);
}
