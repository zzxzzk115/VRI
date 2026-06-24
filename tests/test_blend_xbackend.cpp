// Cross-backend alpha-blending parity: an opaque blue triangle, then a red
// triangle at alpha 0.5 with src-alpha blending, both covering the center. The
// blended result must be purple (~128,0,128) on every backend, proving the blend
// state is applied (without blending the center would be solid red).
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstdlib>
#include <utility>

#include "shaders/triangle_ubo_spv.h"  // g_triangleUboSpv  (color from a UBO)
#include "shaders/triangle_ubo_wgsl.h" // g_triangleUboWgsl

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    bool RunBlend(VriGraphicsAPI api, const void* shader, size_t shaderSize, bool& ran)
    {
        ran = false;
        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = api; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;
        ran = true;

        VriCoreInterface c{};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        VriTextureDesc td{};
        td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM;
        td.width = kW; td.height = kH; td.depth = 1; td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1;
        td.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
        td.memoryLocation = VriMemoryLocation_Device;
        td.clearValue.color.f32[3] = 1.0f; // match the render-pass clear
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

        // two UBO colors: opaque blue, then red at alpha 0.5
        auto makeUbo = [&](float r, float g, float b, float a) {
            VriBufferDesc ud{}; ud.size = 16; ud.usage = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst; ud.memoryLocation = VriMemoryLocation_Device;
            VriBuffer* u = nullptr; REQUIRE(c.CreateBuffer(dev, &ud, &u) == VriResult_Success);
            VriBufferDesc sd{}; sd.size = 16; sd.usage = VriBufferUsage_TransferSrc; sd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* s = nullptr; REQUIRE(c.CreateBuffer(dev, &sd, &s) == VriResult_Success);
            float* m = static_cast<float*>(c.MapBuffer(s, 0, 16)); REQUIRE(m != nullptr);
            m[0] = r; m[1] = g; m[2] = b; m[3] = a; c.UnmapBuffer(s);
            return std::pair<VriBuffer*, VriBuffer*>{u, s};
        };
        auto [uboBlue, stBlue] = makeUbo(0.0f, 0.0f, 1.0f, 1.0f);
        auto [uboRed, stRed] = makeUbo(1.0f, 0.0f, 0.0f, 0.5f);

        VriDescriptorRangeDesc range{}; range.baseRegister = 0; range.descriptorNum = 1; range.descriptorType = VriDescriptorType_ConstantBuffer; range.shaderStages = VriShaderStage_Fragment;
        VriDescriptorSetDesc setDesc{}; setDesc.registerSpace = 0; setDesc.ranges = &range; setDesc.rangeNum = 1;
        VriPipelineLayoutDesc ld{}; ld.descriptorSets = &setDesc; ld.descriptorSetNum = 1;
        VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = shader; sh[0].bytecodeSize = shaderSize; sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = shader; sh[1].bytecodeSize = shaderSize; sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
        ca.blend.enable = VRI_TRUE;
        ca.blend.srcColor = VriBlendFactor_SrcAlpha; ca.blend.dstColor = VriBlendFactor_OneMinusSrcAlpha; ca.blend.colorOp = VriBlendOp_Add;
        ca.blend.srcAlpha = VriBlendFactor_One; ca.blend.dstAlpha = VriBlendFactor_OneMinusSrcAlpha; ca.blend.alphaOp = VriBlendOp_Add;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
        VriPipeline* pipeline = nullptr;
        REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

        VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 2; pdsc.constantBufferMaxNum = 2;
        VriDescriptorPool* pool = nullptr;
        REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
        VriDescriptorSet* setBlue = nullptr; REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &setBlue, 1) == VriResult_Success);
        VriDescriptorSet* setRed = nullptr;  REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &setRed, 1) == VriResult_Success);
        auto bindUbo = [&](VriDescriptorSet* set, VriBuffer* ubo) {
            VriBufferViewDesc bv{}; bv.buffer = ubo; bv.viewType = VriDescriptorType_ConstantBuffer; bv.offset = 0; bv.size = 16;
            VriDescriptor* d = nullptr; REQUIRE(c.CreateBufferView(dev, &bv, &d) == VriResult_Success);
            const VriDescriptor* arr[1] = {d};
            VriDescriptorRangeUpdateDesc u{}; u.descriptors = arr; u.descriptorNum = 1; u.baseDescriptor = 0;
            c.UpdateDescriptorRanges(set, 0, 1, &u);
            return d;
        };
        VriDescriptor* dBlue = bindUbo(setBlue, uboBlue);
        VriDescriptor* dRed = bindUbo(setRed, uboRed);

        VriCommandAllocator* alloc = nullptr; REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr; REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr; REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        VriBufferCopyDesc cp{}; cp.size = 16;
        c.CmdCopyBuffer(cmd, uboBlue, stBlue, &cp);
        c.CmdCopyBuffer(cmd, uboRed, stRed, &cp);
        {
            VriBufferBarrierDesc bb[2]{};
            for (int i = 0; i < 2; ++i) { bb[i].before.access = VriAccess_CopyDestinationWrite; bb[i].before.stages = VriPipelineStage_Transfer; bb[i].after.access = VriAccess_ConstantBufferRead; bb[i].after.stages = VriPipelineStage_FragmentShader; }
            bb[0].buffer = uboBlue; bb[1].buffer = uboRed;
            VriBarrierGroupDesc g{}; g.buffers = bb; g.bufferNum = 2; c.CmdBarrier(cmd, &g);
        }
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
        VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 1;
        c.CmdSetDescriptorSet(cmd, 0, setBlue); c.CmdDraw(cmd, &draw); // opaque blue
        c.CmdSetDescriptorSet(cmd, 0, setRed);  c.CmdDraw(cmd, &draw); // red @ 0.5 over blue
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

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
        REQUIRE(px != nullptr);
        const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
        // red 0.5 over blue: ~ (128, 0, 128). Allow rounding slack.
        const bool purple = std::abs(int(px[o + 0]) - 128) <= 4 && px[o + 1] == 0 && std::abs(int(px[o + 2]) - 128) <= 4;
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyDescriptor(dBlue); c.DestroyDescriptor(dRed); c.DestroyDescriptorPool(pool);
        c.DestroyBuffer(uboBlue); c.DestroyBuffer(uboRed); c.DestroyBuffer(stBlue); c.DestroyBuffer(stRed);
        c.DestroyBuffer(readback); c.DestroyDescriptor(colorView); c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return purple;
    }
} // namespace

TEST_CASE("blend parity: src-alpha blending yields purple on each backend")
{
    bool ran = false;
    const bool vk = RunBlend(VriGraphicsAPI_Vulkan, g_triangleUboSpv, sizeof(g_triangleUboSpv), ran);
    if (ran) { CHECK(vk); } else { MESSAGE("Vulkan unavailable - skipped"); }

    const bool wgpu = RunBlend(VriGraphicsAPI_WebGPU, g_triangleUboWgsl, sizeof(g_triangleUboWgsl), ran);
    if (ran) { CHECK(wgpu); } else { MESSAGE("WebGPU unavailable - skipped"); }

    const bool gl = RunBlend(VriGraphicsAPI_OpenGL, g_triangleUboSpv, sizeof(g_triangleUboSpv), ran);
    if (ran) { CHECK(gl); } else { MESSAGE("OpenGL unavailable - skipped"); }
}
