// Cross-backend mipmap parity: a 2-level texture is uploaded with distinct data per
// level (mip0 = red 128x128, mip1 = green 64x64), then sampled through a view scoped
// to mip 1 (baseMip=1). The center pixel must be green - proving per-mip upload and
// mip-range view selection. (Mip rows stay 256-byte aligned for WebGPU: 128*4=512,
// 64*4=256.) Runs on Vulkan, WebGPU, and OpenGL.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <vector>

#include "shaders/triangle_tex_spv.h"  // g_triangleTexSpv
#include "shaders/triangle_tex_wgsl.h" // g_triangleTexWgsl

namespace
{
    constexpr uint32_t kW    = 64;
    constexpr uint32_t kH    = 64;
    constexpr uint32_t kMip0 = 128; // base level
    constexpr uint32_t kMip1 = 64;  // level 1

    bool RunMipmap(VriGraphicsAPI api, const void* shader, size_t shaderSize, bool& ran)
    {
        ran = false;
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        VriDevice* dev      = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;
        ran = true;

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

        // 2-mip sampled texture: mip0 red (128x128), mip1 green (64x64).
        VriTextureDesc std_ {};
        std_.type           = VriTextureType_2D;
        std_.format         = VriFormat_RGBA8_UNORM;
        std_.width          = kMip0;
        std_.height         = kMip0;
        std_.depth          = 1;
        std_.mipNum         = 2;
        std_.layerNum       = 1;
        std_.sampleNum      = 1;
        std_.usage          = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst;
        std_.memoryLocation = VriMemoryLocation_Device;
        VriTexture* sampled = nullptr;
        REQUIRE(c.CreateTexture(dev, &std_, &sampled) == VriResult_Success);
        // View scoped to mip level 1 (the green level).
        VriTextureViewDesc svd {};
        svd.texture            = sampled;
        svd.viewType           = VriTextureViewType_2D;
        svd.format             = VriFormat_Unknown;
        svd.aspect             = VriImageAspect_Color;
        svd.baseMip            = 1;
        svd.mipNum             = 1;
        VriDescriptor* texView = nullptr;
        REQUIRE(c.CreateTextureView(dev, &svd, &texView) == VriResult_Success);
        VriSamplerDesc smpDesc {};
        smpDesc.magFilter      = VriFilter_Nearest;
        smpDesc.minFilter      = VriFilter_Nearest;
        smpDesc.addressModeU   = VriAddressMode_ClampToEdge;
        smpDesc.addressModeV   = VriAddressMode_ClampToEdge;
        smpDesc.addressModeW   = VriAddressMode_ClampToEdge;
        smpDesc.maxLod         = 1.0f;
        VriDescriptor* sampler = nullptr;
        REQUIRE(c.CreateSampler(dev, &smpDesc, &sampler) == VriResult_Success);

        auto makeStaging = [&](uint32_t dim, uint8_t r, uint8_t g, uint8_t b) {
            const uint64_t bytes = static_cast<uint64_t>(dim) * dim * 4;
            VriBufferDesc  sb {};
            sb.size           = bytes;
            sb.usage          = VriBufferUsage_TransferSrc;
            sb.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* s      = nullptr;
            REQUIRE(c.CreateBuffer(dev, &sb, &s) == VriResult_Success);
            uint8_t* m = static_cast<uint8_t*>(c.MapBuffer(s, 0, bytes));
            REQUIRE(m != nullptr);
            for (uint64_t i = 0; i < bytes; i += 4)
            {
                m[i]     = r;
                m[i + 1] = g;
                m[i + 2] = b;
                m[i + 3] = 255;
            }
            c.UnmapBuffer(s);
            return s;
        };
        VriBuffer* stage0 = makeStaging(kMip0, 255, 0, 0); // mip0 red
        VriBuffer* stage1 = makeStaging(kMip1, 0, 255, 0); // mip1 green

        VriBufferDesc rb {};
        rb.size             = static_cast<uint64_t>(kW) * kH * 4;
        rb.usage            = VriBufferUsage_TransferDst;
        rb.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rb, &readback) == VriResult_Success);

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
        sh[0].bytecode       = shader;
        sh[0].bytecodeSize   = shaderSize;
        sh[0].entryPointName = "vertexMain";
        sh[1].stage          = VriShaderStage_Fragment;
        sh[1].bytecode       = shader;
        sh[1].bytecodeSize   = shaderSize;
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
        const VriDescriptor*         texDescs[1] = {texView};
        const VriDescriptor*         smpDescs[1] = {sampler};
        VriDescriptorRangeUpdateDesc updates[2] {};
        updates[0].descriptors    = texDescs;
        updates[0].descriptorNum  = 1;
        updates[0].baseDescriptor = 0;
        updates[1].descriptors    = smpDescs;
        updates[1].descriptorNum  = 1;
        updates[1].baseDescriptor = 0;
        c.UpdateDescriptorRanges(set, 0, 2, updates);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        {
            VriTextureBarrierDesc tb {};
            tb.texture       = sampled;
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
        VriBufferTextureCopyDesc up0 {};
        up0.texture.aspect   = VriImageAspect_Color;
        up0.texture.layerNum = 1;
        up0.texture.mip      = 0;
        up0.texture.width    = kMip0;
        up0.texture.height   = kMip0;
        c.CmdUploadBufferToTexture(cmd, sampled, stage0, &up0);
        VriBufferTextureCopyDesc up1 {};
        up1.texture.aspect   = VriImageAspect_Color;
        up1.texture.layerNum = 1;
        up1.texture.mip      = 1;
        up1.texture.width    = kMip1;
        up1.texture.height   = kMip1;
        c.CmdUploadBufferToTexture(cmd, sampled, stage1, &up1);
        {
            VriTextureBarrierDesc tb {};
            tb.texture       = sampled;
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
        VriBufferTextureCopyDesc tc {};
        tc.texture.aspect   = VriImageAspect_Color;
        tc.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &tc);
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

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rb.size));
        REQUIRE(px != nullptr);
        const uint32_t o     = ((kH / 2) * kW + (kW / 2)) * 4;
        const bool     green = (px[o + 0] == 0 && px[o + 1] == 255 && px[o + 2] == 0); // mip1, not mip0 (red)
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyDescriptorPool(pool);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyDescriptor(sampler);
        c.DestroyDescriptor(texView);
        c.DestroyTexture(sampled);
        c.DestroyBuffer(stage0);
        c.DestroyBuffer(stage1);
        c.DestroyBuffer(readback);
        c.DestroyDescriptor(colorView);
        c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return green;
    }
} // namespace

TEST_CASE("mipmap parity: per-mip upload + a mip-1 view samples the level-1 data")
{
    bool       ran = false;
    const bool vk  = RunMipmap(VriGraphicsAPI_Vulkan, g_triangleTexSpv, sizeof(g_triangleTexSpv), ran);
    if (ran)
    {
        CHECK(vk);
    }
    else
    {
        MESSAGE("Vulkan unavailable - skipped");
    }

    const bool wgpu = RunMipmap(VriGraphicsAPI_WebGPU, g_triangleTexWgsl, sizeof(g_triangleTexWgsl), ran);
    if (ran)
    {
        CHECK(wgpu);
    }
    else
    {
        MESSAGE("WebGPU unavailable - skipped");
    }

    const bool gl = RunMipmap(VriGraphicsAPI_OpenGL, g_triangleTexSpv, sizeof(g_triangleTexSpv), ran);
    if (ran)
    {
        CHECK(gl);
    }
    else
    {
        MESSAGE("OpenGL unavailable - skipped");
    }
}
