// Bindless textures (Metal, Tier 2 argument buffers): the same runtime-sized Texture2D[]
// array the Vulkan bindless test uses, sampled at an index supplied by a UBO.
//
// What is specific to Metal is the ceiling. Metal binds at most 128 textures per stage to
// the direct argument table, so a descriptor set larger than that cannot be bound the way
// the backend binds every other set - it is promoted to an argument buffer instead. These
// cases pin that down from both directions:
//
//   * the reported capacity is a device property, not the 128-texture direct-binding limit;
//   * a 512-entry PARTIALLY_BOUND array samples correctly at index 200 (past the limit);
//   * a plain fixed-size 160-texture array - no bindless flags at all, the shape a material
//     table has - also samples correctly at index 150, because it is promoted on size alone.
//
// Skips gracefully when the adapter lacks bindless (pre-Metal-3 / Tier 1). Mirrors
// test_bindless_vk.cpp.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <vector>

#include "shaders/tests/bindless_tex_spv.h" // g_bindlessTexSpv

namespace
{
    constexpr uint32_t kW = 64, kH = 64, kT = 16;
    // Metal's direct argument table holds 128 textures per stage. Anything above it can only
    // be reached through an argument buffer, so the sampled indices below straddle it.
    constexpr uint32_t kDirectTextureLimit = 128;

    struct Mtl
    {
        VriDevice*       device = nullptr;
        VriCoreInterface core {};
        ~Mtl()
        {
            if (device)
                vriDestroyDevice(device);
        }
    };

    bool InitMtl(Mtl& mtl)
    {
        VriDeviceCreationDesc desc {};
        desc.graphicsAPI      = VriGraphicsAPI_Metal;
        desc.enableValidation = VRI_TRUE;
        desc.bestEffort       = VRI_TRUE;
        desc.enabledFeatures  = VriFeature_Bindless;
        if (vriCreateDevice(&desc, &mtl.device) != VriResult_Success)
            return false;
        return vriGetInterface(mtl.device, VRI_INTERFACE_CORE, sizeof(mtl.core), &mtl.core) == VriResult_Success;
    }

    struct Rgb
    {
        uint8_t r, g, b;
        bool    operator==(const Rgb& o) const { return r == o.r && g == o.g && b == o.b; }
    };

    // Render one triangle that samples uTex[sampleIndex] and return the center pixel.
    //
    // `arraySize` is the array size the descriptor range declares and `flags` its bindless
    // flags; `populated` is how many leading elements are filled (cycling red/green/blue), so
    // a PARTIALLY_BOUND range can leave the tail unbound while a plain fixed-size range is
    // filled end to end. Returns {0,0,0} and fails the enclosing case on any API error.
    Rgb RenderSampledIndex(const VriCoreInterface& c,
                           VriDevice*              dev,
                           uint32_t                arraySize,
                           VriDescriptorRangeFlags flags,
                           uint32_t                populated,
                           uint32_t                sampleIndex)
    {
        const Rgb kPalette[3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}};
        Rgb       result {0, 0, 0};

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

        // three solid textures (red, green, blue) + views + staging uploads; the array cycles them
        VriTexture*    tex[3]     = {};
        VriDescriptor* texView[3] = {};
        VriBuffer*     staging[3] = {};
        const uint64_t texBytes   = static_cast<uint64_t>(kT) * kT * 4;
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
                m[b]     = kPalette[i].r;
                m[b + 1] = kPalette[i].g;
                m[b + 2] = kPalette[i].b;
                m[b + 3] = 255;
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

        // UBO holding the index to sample
        VriBufferDesc ub {};
        ub.size           = 16;
        ub.usage          = VriBufferUsage_ConstantBuffer;
        ub.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* ubo    = nullptr;
        REQUIRE(c.CreateBuffer(dev, &ub, &ubo) == VriResult_Success);
        {
            uint32_t* m = static_cast<uint32_t*>(c.MapBuffer(ubo, 0, 16));
            m[0]        = sampleIndex;
            c.UnmapBuffer(ubo);
        }
        VriBufferViewDesc uboView {};
        uboView.buffer               = ubo;
        uboView.viewType             = VriDescriptorType_ConstantBuffer;
        uboView.offset               = 0;
        uboView.size                 = 16;
        VriDescriptor* uboDescriptor = nullptr;
        REQUIRE(c.CreateBufferView(dev, &uboView, &uboDescriptor) == VriResult_Success);

        // set 0 = { UBO@0, Sampler@1, Texture[arraySize]@2 }
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
        ranges[2].descriptorNum  = arraySize;
        ranges[2].descriptorType = VriDescriptorType_Texture;
        ranges[2].shaderStages   = VriShaderStage_Fragment;
        ranges[2].flags          = flags;
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
        pdsc.textureMaxNum        = arraySize;
        VriDescriptorPool* pool   = nullptr;
        REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
        VriDescriptorSet* set = nullptr;
        REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);

        std::vector<const VriDescriptor*> texArr(populated);
        for (uint32_t i = 0; i < populated; ++i)
            texArr[i] = texView[i % 3];
        const VriDescriptor*         uboArr[1] = {uboDescriptor};
        const VriDescriptor*         smpArr[1] = {sampler};
        VriDescriptorRangeUpdateDesc updates[3] {};
        updates[0].descriptors    = uboArr;
        updates[0].descriptorNum  = 1;
        updates[0].baseDescriptor = 0;
        updates[1].descriptors    = smpArr;
        updates[1].descriptorNum  = 1;
        updates[1].baseDescriptor = 0;
        updates[2].descriptors    = texArr.data();
        updates[2].descriptorNum  = populated;
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
        result           = {px[o + 0], px[o + 1], px[o + 2]};
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
        return result;
    }

    const Rgb kRed {255, 0, 0}, kGreen {0, 255, 0}, kBlue {0, 0, 255};
} // namespace

TEST_CASE("Metal bindless: capability is reported honestly")
{
    Mtl mtl;
    if (!InitMtl(mtl))
    {
        MESSAGE("Metal unavailable - skipping bindless test");
        return;
    }
    const VriDeviceDesc* d = mtl.core.GetDeviceDesc(mtl.device);
    // best-effort: granted only where Tier 2 argument buffers exist (Metal 3).
    CHECK(((d->enabledFeatures & VriFeature_Bindless) != 0) == (d->hasBindless != VRI_FALSE));
    if (d->hasBindless == VRI_FALSE)
    {
        CHECK(d->bindlessTextureMaxNum == 0); // no capacity claimed without the capability
        return;
    }
    // The point of the argument-buffer path: capacity is a device property, and it is not
    // the direct argument table's 128 textures per stage.
    CHECK(d->bindlessTextureMaxNum > kDirectTextureLimit);
    CHECK(d->bindlessSamplerMaxNum > 0);
}

TEST_CASE("Metal bindless: sample past the direct argument-table limit")
{
    Mtl mtl;
    if (!InitMtl(mtl))
    {
        MESSAGE("Metal unavailable - skipping bindless test");
        return;
    }
    const VriCoreInterface& c   = mtl.core;
    VriDevice*              dev = mtl.device;
    if (c.GetDeviceDesc(dev)->hasBindless == VRI_FALSE)
    {
        MESSAGE("adapter lacks Tier 2 argument buffers - skipping");
        return;
    }

    // 512-entry PARTIALLY_BOUND array, 201 populated (cycling red/green/blue), sampled at
    // 200 -> 200 % 3 == 2 -> blue. Index 200 is unreachable through the direct table.
    CHECK(RenderSampledIndex(c, dev, 512, VriDescriptorRange_PartiallyBound | VriDescriptorRange_VariableSized,
                             201, 200) == kBlue);

    // Same layout sampled below the limit still works - promotion must not break the easy case.
    CHECK(RenderSampledIndex(c, dev, 512, VriDescriptorRange_PartiallyBound | VriDescriptorRange_VariableSized,
                             201, 1) == kGreen);
}

TEST_CASE("Metal bindless: a plain oversized texture array is promoted, not truncated")
{
    Mtl mtl;
    if (!InitMtl(mtl))
    {
        MESSAGE("Metal unavailable - skipping bindless test");
        return;
    }
    const VriCoreInterface& c   = mtl.core;
    VriDevice*              dev = mtl.device;
    if (c.GetDeviceDesc(dev)->hasBindless == VRI_FALSE)
    {
        MESSAGE("adapter lacks Tier 2 argument buffers - skipping");
        return;
    }

    // No bindless flags at all - just a fixed 160-texture array, the shape a scene's material
    // table has. It exceeds the direct table, so it must be promoted rather than overflow it.
    // 150 % 3 == 0 -> red.
    CHECK(RenderSampledIndex(c, dev, 160, VriDescriptorRange_None, 160, 150) == kRed);
    // ... and an index inside the direct table's range still reads its own element, not a
    // neighbour: 127 % 3 == 1 -> green.
    CHECK(RenderSampledIndex(c, dev, 160, VriDescriptorRange_None, 160, 127) == kGreen);
}
