// Bindless textures (Vulkan, descriptor indexing / VkFeature descriptorIndexing):
// a runtime-sized Texture2D[] array (last binding, PARTIALLY_BOUND +
// VARIABLE_DESCRIPTOR_COUNT + UPDATE_AFTER_BIND) holding red/green/blue textures;
// the fragment shader samples uTex[index] where index comes from a UBO. With index=1
// the triangle must read back GREEN - proving dynamic indexing into a partially-bound
// bindless array. Skips when the adapter lacks VriFeature_Bindless.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/tests/bindless_tex_spv.h" // g_bindlessTexSpv

namespace
{
    constexpr uint32_t kW = 64, kH = 64, kT = 16;
    constexpr uint32_t kArraySize = 64; // bindless array capacity (only 3 populated)

    struct Vk
    {
        VriDevice*       device = nullptr;
        VriCoreInterface core {};
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
        desc.enabledFeatures  = VriFeature_Bindless;
        if (vriCreateDevice(&desc, &vk.device) != VriResult_Success)
            return false;
        return vriGetInterface(vk.device, VRI_INTERFACE_CORE, sizeof(vk.core), &vk.core) == VriResult_Success;
    }
} // namespace

TEST_CASE("Vulkan bindless: capability is reported honestly")
{
    Vk vk;
    if (!InitVk(vk))
    {
        MESSAGE("Vulkan unavailable - skipping bindless test");
        return;
    }
    const VriDeviceDesc* d = vk.core.GetDeviceDesc(vk.device);
    // best-effort: granted only if the adapter supports descriptor indexing.
    CHECK(((d->enabledFeatures & VriFeature_Bindless) != 0) == (d->hasBindless != VRI_FALSE));
}

TEST_CASE("Vulkan bindless: sample a dynamic index into a runtime texture array")
{
    Vk vk;
    if (!InitVk(vk))
    {
        MESSAGE("Vulkan unavailable - skipping bindless test");
        return;
    }
    const VriCoreInterface& c   = vk.core;
    VriDevice*              dev = vk.device;
    if (c.GetDeviceDesc(dev)->hasBindless == VRI_FALSE)
    {
        MESSAGE("adapter lacks bindless - skipping");
        return;
    }

    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    // color target + view + readback
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
    VriTextureViewDesc cvd {};
    cvd.texture              = color;
    cvd.viewType             = VriTextureViewType_2D;
    cvd.format               = VriFormat_Unknown;
    cvd.aspect               = VriImageAspect_Color;
    VriDescriptor* colorView = nullptr;
    REQUIRE(c.CreateTextureView(dev, &cvd, &colorView) == VriResult_Success);
    VriBufferDesc rb {};
    rb.size             = static_cast<uint64_t>(kW) * kH * 4;
    rb.usage            = VriBufferUsage_TransferDst;
    rb.memoryLocation   = VriMemoryLocation_HostReadback;
    VriBuffer* readback = nullptr;
    REQUIRE(c.CreateBuffer(dev, &rb, &readback) == VriResult_Success);

    // three solid textures (red, green, blue) + their views + staging uploads
    const uint8_t  colors[3][4] = {{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}};
    VriTexture*    tex[3]       = {};
    VriDescriptor* texView[3]   = {};
    VriBuffer*     staging[3]   = {};
    const uint64_t texBytes     = static_cast<uint64_t>(kT) * kT * 4;
    for (int i = 0; i < 3; ++i)
    {
        VriTextureDesc std_ {};
        std_.type           = VriTextureType_2D;
        std_.format         = VriFormat_RGBA8_UNORM;
        std_.width          = kT;
        std_.height         = kT;
        std_.depth          = 1;
        std_.mipNum         = 1;
        std_.layerNum       = 1;
        std_.sampleNum      = 1;
        std_.usage          = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst;
        std_.memoryLocation = VriMemoryLocation_Device;
        REQUIRE(c.CreateTexture(dev, &std_, &tex[i]) == VriResult_Success);
        VriTextureViewDesc svd {};
        svd.texture  = tex[i];
        svd.viewType = VriTextureViewType_2D;
        svd.format   = VriFormat_Unknown;
        svd.aspect   = VriImageAspect_Color;
        REQUIRE(c.CreateTextureView(dev, &svd, &texView[i]) == VriResult_Success);
        VriBufferDesc sb {};
        sb.size           = texBytes;
        sb.usage          = VriBufferUsage_TransferSrc;
        sb.memoryLocation = VriMemoryLocation_HostUpload;
        REQUIRE(c.CreateBuffer(dev, &sb, &staging[i]) == VriResult_Success);
        uint8_t* m = static_cast<uint8_t*>(c.MapBuffer(staging[i], 0, texBytes));
        for (uint64_t b = 0; b < texBytes; b += 4)
        {
            m[b]     = colors[i][0];
            m[b + 1] = colors[i][1];
            m[b + 2] = colors[i][2];
            m[b + 3] = colors[i][3];
        }
        c.UnmapBuffer(staging[i]);
    }

    VriSamplerDesc smpDesc {};
    smpDesc.magFilter      = VriFilter_Nearest;
    smpDesc.minFilter      = VriFilter_Nearest;
    smpDesc.addressModeU   = VriAddressMode_ClampToEdge;
    smpDesc.addressModeV   = VriAddressMode_ClampToEdge;
    smpDesc.addressModeW   = VriAddressMode_ClampToEdge;
    smpDesc.maxLod         = 1.0f;
    VriDescriptor* sampler = nullptr;
    REQUIRE(c.CreateSampler(dev, &smpDesc, &sampler) == VriResult_Success);

    // UBO holding the index to sample (1 -> green)
    VriBufferDesc ub {};
    ub.size           = 16;
    ub.usage          = VriBufferUsage_ConstantBuffer;
    ub.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* ubo    = nullptr;
    REQUIRE(c.CreateBuffer(dev, &ub, &ubo) == VriResult_Success);
    {
        uint32_t* m = static_cast<uint32_t*>(c.MapBuffer(ubo, 0, 16));
        m[0]        = 1;
        c.UnmapBuffer(ubo);
    }
    VriBufferViewDesc uboView {};
    uboView.buffer               = ubo;
    uboView.viewType             = VriDescriptorType_ConstantBuffer;
    uboView.offset               = 0;
    uboView.size                 = 16;
    VriDescriptor* uboDescriptor = nullptr;
    REQUIRE(c.CreateBufferView(dev, &uboView, &uboDescriptor) == VriResult_Success);

    // pipeline layout: set 0 = { UBO@0, Sampler@1, Texture[kArraySize]@2 (bindless, last) }
    VriDescriptorRangeDesc ranges[3] {};
    ranges[0].baseRegister   = 0;
    ranges[0].descriptorNum  = 1;
    ranges[0].descriptorType = VriDescriptorType_ConstantBuffer;
    ranges[0].shaderStages   = VriShaderStage_Fragment;
    ranges[1].baseRegister   = 1;
    ranges[1].descriptorNum  = 1;
    ranges[1].descriptorType = VriDescriptorType_Sampler;
    ranges[1].shaderStages   = VriShaderStage_Fragment;
    ranges[2].baseRegister   = 2;
    ranges[2].descriptorNum  = kArraySize;
    ranges[2].descriptorType = VriDescriptorType_Texture;
    ranges[2].shaderStages   = VriShaderStage_Fragment;
    ranges[2].flags          = VriDescriptorRange_PartiallyBound | VriDescriptorRange_VariableSized;
    VriDescriptorSetDesc setDesc {};
    setDesc.registerSpace = 0;
    setDesc.ranges        = ranges;
    setDesc.rangeNum      = 3;
    VriPipelineLayoutDesc ld {};
    ld.descriptorSets         = &setDesc;
    ld.descriptorSetNum       = 1;
    VriPipelineLayout* layout = nullptr;
    REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

    VriShaderDesc sh[2] {};
    sh[0].stage          = VriShaderStage_Vertex;
    sh[0].bytecode       = g_bindlessTexSpv;
    sh[0].bytecodeSize   = sizeof(g_bindlessTexSpv);
    sh[0].entryPointName = "vertexMain";
    sh[1].stage          = VriShaderStage_Fragment;
    sh[1].bytecode       = g_bindlessTexSpv;
    sh[1].bytecodeSize   = sizeof(g_bindlessTexSpv);
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
    pdsc.samplerMaxNum        = 1;
    pdsc.textureMaxNum        = kArraySize;
    VriDescriptorPool* pool   = nullptr;
    REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
    VriDescriptorSet* set = nullptr;
    REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);

    const VriDescriptor*         uboArr[1] = {uboDescriptor};
    const VriDescriptor*         smpArr[1] = {sampler};
    const VriDescriptor*         texArr[3] = {texView[0], texView[1], texView[2]}; // array elems 0,1,2 (rest unbound)
    VriDescriptorRangeUpdateDesc updates[3] {};
    updates[0].descriptors    = uboArr;
    updates[0].descriptorNum  = 1;
    updates[0].baseDescriptor = 0;
    updates[1].descriptors    = smpArr;
    updates[1].descriptorNum  = 1;
    updates[1].baseDescriptor = 0;
    updates[2].descriptors    = texArr;
    updates[2].descriptorNum  = 3;
    updates[2].baseDescriptor = 0;
    c.UpdateDescriptorRanges(set, 0, 3, updates);

    VriCommandAllocator* alloc = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
    for (int i = 0; i < 3; ++i)
    {
        {
            VriTextureBarrierDesc tb {};
            tb.texture       = tex[i];
            tb.before.layout = VriLayout_Undefined;
            tb.before.stages = VriPipelineStage_None;
            tb.after.access  = VriAccess_CopyDestinationWrite;
            tb.after.layout  = VriLayout_CopyDestination;
            tb.after.stages  = VriPipelineStage_Transfer;
            tb.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        }
        VriBufferTextureCopyDesc up {};
        up.texture.aspect   = VriImageAspect_Color;
        up.texture.layerNum = 1;
        up.texture.width    = kT;
        up.texture.height   = kT;
        c.CmdUploadBufferToTexture(cmd, tex[i], staging[i], &up);
        {
            VriTextureBarrierDesc tb {};
            tb.texture       = tex[i];
            tb.before.access = VriAccess_CopyDestinationWrite;
            tb.before.layout = VriLayout_CopyDestination;
            tb.before.stages = VriPipelineStage_Transfer;
            tb.after.access  = VriAccess_ShaderResourceRead;
            tb.after.layout  = VriLayout_ShaderResource;
            tb.after.stages  = VriPipelineStage_FragmentShader;
            tb.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        }
    }
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

    const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rb.size));
    REQUIRE(px != nullptr);
    const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
    CHECK(px[o + 0] == 0); // index 1 -> green texture
    CHECK(px[o + 1] == 255);
    CHECK(px[o + 2] == 0);
    c.UnmapBuffer(readback);

    c.DeviceWaitIdle(dev);
    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyDescriptor(uboDescriptor);
    c.DestroyBuffer(ubo);
    c.DestroyDescriptor(sampler);
    for (int i = 0; i < 3; ++i)
    {
        c.DestroyBuffer(staging[i]);
        c.DestroyDescriptor(texView[i]);
        c.DestroyTexture(tex[i]);
    }
    c.DestroyBuffer(readback);
    c.DestroyDescriptor(colorView);
    c.DestroyTexture(color);
}
