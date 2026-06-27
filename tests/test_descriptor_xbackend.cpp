// Cross-backend descriptor-set parity: the same VRI calls drive a UBO-tinted
// triangle on Vulkan AND WebGPU, asserting the center pixel is the UBO color
// (green). Uses a portable staging-buffer upload (a uniform buffer can't be
// host-mapped on WebGPU), exercising CmdCopyBuffer + CmdBarrier too.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/triangle_ubo_spv.h"  // g_triangleUboSpv  (Vulkan + OpenGL via SPIRV-Cross)
#include "shaders/triangle_ubo_wgsl.h" // g_triangleUboWgsl (WebGPU)

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    // Renders the UBO-tinted triangle on `api` and returns whether the center
    // pixel is green. Returns true with `ran=false` when the backend is absent.
    bool RunUboTint(VriGraphicsAPI api, const void* shader, size_t shaderSize, bool& ran)
    {
        ran = false;
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;

        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;
        ran = true;

        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);

        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        // color target + view
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
        VriTextureViewDesc vd {};
        vd.texture               = color;
        vd.viewType              = VriTextureViewType_2D;
        vd.format                = VriFormat_Unknown;
        vd.aspect                = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr;
        REQUIRE(c.CreateTextureView(dev, &vd, &colorView) == VriResult_Success);

        // readback buffer
        VriBufferDesc rb {};
        rb.size             = static_cast<uint64_t>(kW) * kH * 4;
        rb.usage            = VriBufferUsage_TransferDst;
        rb.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rb, &readback) == VriResult_Success);

        // staging (host-writable) -> device uniform buffer (portable upload)
        VriBufferDesc sb {};
        sb.size            = 16;
        sb.usage           = VriBufferUsage_TransferSrc;
        sb.memoryLocation  = VriMemoryLocation_HostUpload;
        VriBuffer* staging = nullptr;
        REQUIRE(c.CreateBuffer(dev, &sb, &staging) == VriResult_Success);
        {
            float* m = static_cast<float*>(c.MapBuffer(staging, 0, 16));
            REQUIRE(m != nullptr);
            m[0] = 0.0f;
            m[1] = 1.0f;
            m[2] = 0.0f;
            m[3] = 1.0f; // green
            c.UnmapBuffer(staging);
        }

        VriBufferDesc ub {};
        ub.size           = 16;
        ub.usage          = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst;
        ub.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* ubo    = nullptr;
        REQUIRE(c.CreateBuffer(dev, &ub, &ubo) == VriResult_Success);

        // pipeline layout: set 0 binding 0 = constant buffer (fragment)
        VriDescriptorRangeDesc range {};
        range.baseRegister   = 0;
        range.descriptorNum  = 1;
        range.descriptorType = VriDescriptorType_ConstantBuffer;
        range.shaderStages   = VriShaderStage_Fragment;
        VriDescriptorSetDesc setDesc {};
        setDesc.registerSpace = 0;
        setDesc.ranges        = &range;
        setDesc.rangeNum      = 1;
        VriPipelineLayoutDesc ld {};
        ld.descriptorSets         = &setDesc;
        ld.descriptorSetNum       = 1;
        VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriShaderDesc shaders[2] {};
        shaders[0].stage          = VriShaderStage_Vertex;
        shaders[0].bytecode       = shader;
        shaders[0].bytecodeSize   = shaderSize;
        shaders[0].entryPointName = "vertexMain";
        shaders[1].stage          = VriShaderStage_Fragment;
        shaders[1].bytecode       = shader;
        shaders[1].bytecodeSize   = shaderSize;
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

        // descriptor pool + set + update (bind buffer view of the ubo)
        VriDescriptorPoolDesc pdsc {};
        pdsc.descriptorSetMaxNum  = 1;
        pdsc.constantBufferMaxNum = 1;
        VriDescriptorPool* pool   = nullptr;
        REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
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
        const VriDescriptor*         descs[1] = {uboDescriptor};
        update.descriptors                    = descs;
        update.descriptorNum                  = 1;
        update.baseDescriptor                 = 0;
        c.UpdateDescriptorRanges(set, 0, 1, &update);

        // record
        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);

        // upload: staging -> ubo, then make it visible to the shader
        VriBufferCopyDesc copy {};
        copy.size = 16;
        c.CmdCopyBuffer(cmd, ubo, staging, &copy);
        {
            VriBufferBarrierDesc bb {};
            bb.buffer        = ubo;
            bb.before.access = VriAccess_CopyDestinationWrite;
            bb.before.stages = VriPipelineStage_Transfer;
            bb.after.access  = VriAccess_ConstantBufferRead;
            bb.after.stages  = VriPipelineStage_FragmentShader;
            VriBarrierGroupDesc g {};
            g.buffers   = &bb;
            g.bufferNum = 1;
            c.CmdBarrier(cmd, &g);
        }

        // color target: Undefined -> ColorAttachment (Vulkan; WebGPU ignores)
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

        // color -> copy source (Vulkan; WebGPU ignores)
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
        const bool     green = (px[o + 0] == 0 && px[o + 1] == 255 && px[o + 2] == 0);
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyDescriptor(uboDescriptor);
        c.DestroyDescriptorPool(pool);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(ubo);
        c.DestroyBuffer(staging);
        c.DestroyBuffer(readback);
        c.DestroyDescriptor(colorView);
        c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return green;
    }
} // namespace

TEST_CASE("descriptor set parity: UBO tints the triangle green on each backend")
{
    bool ran = false;

    const bool vk = RunUboTint(VriGraphicsAPI_Vulkan, g_triangleUboSpv, sizeof(g_triangleUboSpv), ran);
    if (ran)
    {
        CHECK(vk);
    }
    else
    {
        MESSAGE("Vulkan unavailable - skipped");
    }

    const bool wgpu = RunUboTint(VriGraphicsAPI_WebGPU, g_triangleUboWgsl, sizeof(g_triangleUboWgsl), ran);
    if (ran)
    {
        CHECK(wgpu);
    }
    else
    {
        MESSAGE("WebGPU unavailable - skipped");
    }

    // OpenGL consumes the SPIR-V (transpiled to GLSL by SPIRV-Cross); the UBO is
    // bound through the flattened (set,binding) -> uniform-block-binding map.
    const bool gl = RunUboTint(VriGraphicsAPI_OpenGL, g_triangleUboSpv, sizeof(g_triangleUboSpv), ran);
    if (ran)
    {
        CHECK(gl);
    }
    else
    {
        MESSAGE("OpenGL unavailable - skipped");
    }
}
