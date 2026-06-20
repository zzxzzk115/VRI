// Cross-backend sampled-texture parity: the same VRI calls (separate Texture +
// Sampler descriptors) sample a solid-blue 1x1 texture onto the triangle on each
// backend, asserting the center pixel is blue. On GL/WebGL the separate
// texture+sampler are fused into one combined sampler by SPIRV-Cross.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/triangle_tex_spv.h"  // g_triangleTexSpv  (Vulkan + OpenGL via SPIRV-Cross)
#include "shaders/triangle_tex_wgsl.h" // g_triangleTexWgsl (WebGPU)

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    bool RunTexSample(VriGraphicsAPI api, const void* shader, size_t shaderSize, bool& ran)
    {
        ran = false;
        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;
        ran = true;

        VriCoreInterface c{};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        // color target + view
        VriTextureDesc td{};
        td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM;
        td.width = kW; td.height = kH; td.depth = 1; td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1;
        td.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* color = nullptr;
        REQUIRE(c.CreateTexture(dev, &td, &color) == VriResult_Success);
        VriTextureViewDesc cvd{};
        cvd.texture = color; cvd.viewType = VriTextureViewType_2D; cvd.format = VriFormat_Unknown; cvd.aspect = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr;
        REQUIRE(c.CreateTextureView(dev, &cvd, &colorView) == VriResult_Success);

        // sampled solid-blue texture + its view + a sampler. 64 wide so each row is
        // 64*4 = 256 bytes, satisfying WebGPU's COPY_BYTES_PER_ROW_ALIGNMENT (256).
        constexpr uint32_t kTexW = 64, kTexH = 64;
        VriTextureDesc std_{};
        std_.type = VriTextureType_2D; std_.format = VriFormat_RGBA8_UNORM;
        std_.width = kTexW; std_.height = kTexH; std_.depth = 1; std_.mipNum = 1; std_.layerNum = 1; std_.sampleNum = 1;
        std_.usage = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst;
        std_.memoryLocation = VriMemoryLocation_Device;
        VriTexture* sampled = nullptr;
        REQUIRE(c.CreateTexture(dev, &std_, &sampled) == VriResult_Success);
        VriTextureViewDesc svd{};
        svd.texture = sampled; svd.viewType = VriTextureViewType_2D; svd.format = VriFormat_Unknown; svd.aspect = VriImageAspect_Color;
        VriDescriptor* texView = nullptr;
        REQUIRE(c.CreateTextureView(dev, &svd, &texView) == VriResult_Success);
        VriSamplerDesc smpDesc{};
        smpDesc.magFilter = VriFilter_Nearest; smpDesc.minFilter = VriFilter_Nearest;
        smpDesc.addressModeU = VriAddressMode_ClampToEdge; smpDesc.addressModeV = VriAddressMode_ClampToEdge; smpDesc.addressModeW = VriAddressMode_ClampToEdge;
        smpDesc.maxLod = 1.0f;
        VriDescriptor* sampler = nullptr;
        REQUIRE(c.CreateSampler(dev, &smpDesc, &sampler) == VriResult_Success);

        // staging (host) -> sampled texture (every texel blue)
        const uint64_t texBytes = static_cast<uint64_t>(kTexW) * kTexH * 4;
        VriBufferDesc sb{};
        sb.size = texBytes; sb.usage = VriBufferUsage_TransferSrc; sb.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* staging = nullptr;
        REQUIRE(c.CreateBuffer(dev, &sb, &staging) == VriResult_Success);
        {
            uint8_t* m = static_cast<uint8_t*>(c.MapBuffer(staging, 0, texBytes));
            REQUIRE(m != nullptr);
            for (uint64_t i = 0; i < texBytes; i += 4) { m[i + 0] = 0; m[i + 1] = 0; m[i + 2] = 255; m[i + 3] = 255; }
            c.UnmapBuffer(staging);
        }

        // readback buffer
        VriBufferDesc rb{};
        rb.size = static_cast<uint64_t>(kW) * kH * 4; rb.usage = VriBufferUsage_TransferDst; rb.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rb, &readback) == VriResult_Success);

        // pipeline layout: set 0 = {binding 0: Texture, binding 1: Sampler} (fragment)
        VriDescriptorRangeDesc ranges[2]{};
        ranges[0].baseRegister = 0; ranges[0].descriptorNum = 1; ranges[0].descriptorType = VriDescriptorType_Texture; ranges[0].shaderStages = VriShaderStage_Fragment;
        ranges[1].baseRegister = 1; ranges[1].descriptorNum = 1; ranges[1].descriptorType = VriDescriptorType_Sampler; ranges[1].shaderStages = VriShaderStage_Fragment;
        VriDescriptorSetDesc setDesc{};
        setDesc.registerSpace = 0; setDesc.ranges = ranges; setDesc.rangeNum = 2;
        VriPipelineLayoutDesc ld{};
        ld.descriptorSets = &setDesc; ld.descriptorSetNum = 1;
        VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = shader; sh[0].bytecodeSize = shaderSize; sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = shader; sh[1].bytecodeSize = shaderSize; sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
        VriPipeline* pipeline = nullptr;
        REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

        // descriptor pool + set + update (texture at binding 0, sampler at binding 1)
        VriDescriptorPoolDesc pdsc{};
        pdsc.descriptorSetMaxNum = 1; pdsc.textureMaxNum = 1; pdsc.samplerMaxNum = 1;
        VriDescriptorPool* pool = nullptr;
        REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
        VriDescriptorSet* set = nullptr;
        REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);
        const VriDescriptor* texDescs[1] = {texView};
        const VriDescriptor* smpDescs[1] = {sampler};
        VriDescriptorRangeUpdateDesc updates[2]{};
        updates[0].descriptors = texDescs; updates[0].descriptorNum = 1; updates[0].baseDescriptor = 0;
        updates[1].descriptors = smpDescs; updates[1].descriptorNum = 1; updates[1].baseDescriptor = 0;
        c.UpdateDescriptorRanges(set, 0, 2, updates);

        // record
        VriCommandAllocator* alloc = nullptr; REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr; REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr; REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);

        // sampled texture: Undefined -> CopyDestination, upload, -> ShaderResource
        {
            VriTextureBarrierDesc tb{};
            tb.texture = sampled; tb.before.layout = VriLayout_Undefined; tb.before.stages = VriPipelineStage_None;
            tb.after.access = VriAccess_CopyDestinationWrite; tb.after.layout = VriLayout_CopyDestination; tb.after.stages = VriPipelineStage_Transfer;
            tb.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc g{}; g.textures = &tb; g.textureNum = 1; c.CmdBarrier(cmd, &g);
        }
        VriBufferTextureCopyDesc up{}; up.bufferOffset = 0; up.texture.aspect = VriImageAspect_Color; up.texture.layerNum = 1; up.texture.width = kTexW; up.texture.height = kTexH;
        c.CmdUploadBufferToTexture(cmd, sampled, staging, &up);
        {
            VriTextureBarrierDesc tb{};
            tb.texture = sampled; tb.before.access = VriAccess_CopyDestinationWrite; tb.before.layout = VriLayout_CopyDestination; tb.before.stages = VriPipelineStage_Transfer;
            tb.after.access = VriAccess_ShaderResourceRead; tb.after.layout = VriLayout_ShaderResource; tb.after.stages = VriPipelineStage_FragmentShader;
            tb.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc g{}; g.textures = &tb; g.textureNum = 1; c.CmdBarrier(cmd, &g);
        }
        // color: Undefined -> ColorAttachment
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
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, set);
        VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 1; c.CmdDraw(cmd, &draw);
        c.CmdEndRendering(cmd);

        {
            VriTextureBarrierDesc tb{};
            tb.texture = color; tb.before.access = VriAccess_ColorAttachmentWrite; tb.before.layout = VriLayout_ColorAttachment; tb.before.stages = VriPipelineStage_ColorAttachmentOutput;
            tb.after.access = VriAccess_CopySourceRead; tb.after.layout = VriLayout_CopySource; tb.after.stages = VriPipelineStage_Transfer;
            tb.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc g{}; g.textures = &tb; g.textureNum = 1; c.CmdBarrier(cmd, &g);
        }
        VriBufferTextureCopyDesc tc{}; tc.texture.aspect = VriImageAspect_Color; tc.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &tc);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

        VriFenceSubmitDesc signal{}; signal.fence = fence; signal.value = 1;
        VriQueueSubmitDesc submit{}; submit.commandBuffers = &cmd; submit.commandBufferNum = 1; submit.signalFences = &signal; submit.signalFenceNum = 1;
        c.QueueSubmit(queue, &submit);
        c.Wait(fence, 1);

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rb.size));
        REQUIRE(px != nullptr);
        const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
        const bool blue = (px[o + 0] == 0 && px[o + 1] == 0 && px[o + 2] == 255);
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyDescriptorPool(pool); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyDescriptor(sampler); c.DestroyDescriptor(texView); c.DestroyTexture(sampled);
        c.DestroyBuffer(staging); c.DestroyBuffer(readback); c.DestroyDescriptor(colorView); c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return blue;
    }
} // namespace

TEST_CASE("sampled-texture parity: a solid texture tints the triangle on each backend")
{
    bool ran = false;
    const bool vk = RunTexSample(VriGraphicsAPI_Vulkan, g_triangleTexSpv, sizeof(g_triangleTexSpv), ran);
    if (ran) { CHECK(vk); } else { MESSAGE("Vulkan unavailable - skipped"); }

    const bool wgpu = RunTexSample(VriGraphicsAPI_WebGPU, g_triangleTexWgsl, sizeof(g_triangleTexWgsl), ran);
    if (ran) { CHECK(wgpu); } else { MESSAGE("WebGPU unavailable - skipped"); }

    const bool gl = RunTexSample(VriGraphicsAPI_OpenGL, g_triangleTexSpv, sizeof(g_triangleTexSpv), ran);
    if (ran) { CHECK(gl); } else { MESSAGE("OpenGL unavailable - skipped"); }
}
