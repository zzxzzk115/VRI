// Custom sampler border color (Vulkan VK_EXT_custom_border_color + D3D12 native float4
// border) via VriSamplerDesc::useCustomBorderColor. A blue texture is sampled far
// outside [0,1] with ClampToBorder and a custom GREEN border, so the triangle reads
// back green (not the texture's blue). Self-skips where unsupported.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/bordercolor_dxbc.h" // g_bordercolorDxbcVS / PS  (D3D12)
#include "shaders/bordercolor_spv.h"  // g_bordercolorSpv          (Vulkan)

namespace
{
    constexpr uint32_t kW = 64, kH = 64, kT = 64;

    void RunBorder(VriGraphicsAPI api)
    {
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        VriDevice* dev      = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("device unavailable - skipping");
            return;
        }
        struct Guard
        {
            VriDevice* d;
            ~Guard() { vriDestroyDevice(d); }
        } guard {dev};

        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        if (c.GetDeviceDesc(dev)->hasCustomBorderColor == VRI_FALSE)
        {
            MESSAGE("no custom border color - skipping");
            return;
        }

        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        VriShaderDesc sh[2] {};
        sh[0].stage          = VriShaderStage_Vertex;
        sh[0].entryPointName = "vertexMain";
        sh[1].stage          = VriShaderStage_Fragment;
        sh[1].entryPointName = "fragmentMain";
        if (api == VriGraphicsAPI_D3D12)
        {
            sh[0].bytecode     = g_bordercolorDxbcVS;
            sh[0].bytecodeSize = sizeof(g_bordercolorDxbcVS);
            sh[1].bytecode     = g_bordercolorDxbcPS;
            sh[1].bytecodeSize = sizeof(g_bordercolorDxbcPS);
        }
        else
        {
            sh[0].bytecode     = g_bordercolorSpv;
            sh[0].bytecodeSize = sizeof(g_bordercolorSpv);
            sh[1].bytecode     = g_bordercolorSpv;
            sh[1].bytecodeSize = sizeof(g_bordercolorSpv);
        }

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
        VriBufferDesc rb {};
        rb.size             = static_cast<uint64_t>(kW) * kH * 4;
        rb.usage            = VriBufferUsage_TransferDst;
        rb.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rb, &readback) == VriResult_Success);

        // blue sampled texture (only the border is read, but fill it to prove that)
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
        VriTexture* sampled = nullptr;
        REQUIRE(c.CreateTexture(dev, &std_, &sampled) == VriResult_Success);
        VriTextureViewDesc svd {};
        svd.texture            = sampled;
        svd.viewType           = VriTextureViewType_2D;
        svd.format             = VriFormat_Unknown;
        svd.aspect             = VriImageAspect_Color;
        VriDescriptor* texView = nullptr;
        REQUIRE(c.CreateTextureView(dev, &svd, &texView) == VriResult_Success);
        const uint64_t texBytes = static_cast<uint64_t>(kT) * kT * 4;
        VriBufferDesc  sb {};
        sb.size            = texBytes;
        sb.usage           = VriBufferUsage_TransferSrc;
        sb.memoryLocation  = VriMemoryLocation_HostUpload;
        VriBuffer* staging = nullptr;
        REQUIRE(c.CreateBuffer(dev, &sb, &staging) == VriResult_Success);
        {
            uint8_t* m = static_cast<uint8_t*>(c.MapBuffer(staging, 0, texBytes));
            for (uint64_t i = 0; i < texBytes; i += 4)
            {
                m[i]     = 0;
                m[i + 1] = 0;
                m[i + 2] = 255;
                m[i + 3] = 255;
            }
            c.UnmapBuffer(staging);
        }

        // ClampToBorder + custom GREEN border
        VriSamplerDesc smpDesc {};
        smpDesc.magFilter            = VriFilter_Nearest;
        smpDesc.minFilter            = VriFilter_Nearest;
        smpDesc.addressModeU         = VriAddressMode_ClampToBorder;
        smpDesc.addressModeV         = VriAddressMode_ClampToBorder;
        smpDesc.addressModeW         = VriAddressMode_ClampToBorder;
        smpDesc.maxLod               = 1.0f;
        smpDesc.useCustomBorderColor = VRI_TRUE;
        smpDesc.customBorderColor[0] = 0.0f;
        smpDesc.customBorderColor[1] = 1.0f;
        smpDesc.customBorderColor[2] = 0.0f;
        smpDesc.customBorderColor[3] = 1.0f;
        VriDescriptor* sampler       = nullptr;
        REQUIRE(c.CreateSampler(dev, &smpDesc, &sampler) == VriResult_Success);

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
        VriBufferTextureCopyDesc up {};
        up.texture.aspect   = VriImageAspect_Color;
        up.texture.layerNum = 1;
        up.texture.width    = kT;
        up.texture.height   = kT;
        c.CmdUploadBufferToTexture(cmd, sampled, staging, &up);
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
        VriRect scs {0, 0, kW, kH};
        c.CmdSetScissors(cmd, &scs, 1);
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
        const uint32_t o  = ((kH / 2) * kW + (kW / 2)) * 4;
        CHECK(px[o + 0] == 0); // custom green border (not the texture's blue)
        CHECK(px[o + 1] == 255);
        CHECK(px[o + 2] == 0);
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyDescriptorPool(pool);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyDescriptor(sampler);
        c.DestroyBuffer(staging);
        c.DestroyDescriptor(texView);
        c.DestroyTexture(sampled);
        c.DestroyBuffer(readback);
        c.DestroyDescriptor(colorView);
        c.DestroyTexture(color);
    }
} // namespace

TEST_CASE("Vulkan: custom sampler border color") { RunBorder(VriGraphicsAPI_Vulkan); }
TEST_CASE("D3D12: custom sampler border color") { RunBorder(VriGraphicsAPI_D3D12); }
