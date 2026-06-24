// Cross-backend multiple-render-target parity: one triangle writes red to color
// attachment 0 and blue to attachment 1 in a single draw. Both targets are read
// back and must hold their respective color on every backend.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/triangle_mrt_spv.h"  // g_triangleMrtSpv  (SV_Target0 red, SV_Target1 blue)
#include "shaders/triangle_mrt_wgsl.h" // g_triangleMrtWgsl

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    bool RunMrt(VriGraphicsAPI api, const void* shader, size_t shaderSize, bool& ran)
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

        auto makeTarget = [&](VriTexture*& tex, VriDescriptor*& view) {
            VriTextureDesc td{}; td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM;
            td.width = kW; td.height = kH; td.depth = 1; td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1;
            td.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc; td.memoryLocation = VriMemoryLocation_Device;
            td.clearValue.color.f32[3] = 1.0f; // match the render-pass clear
            REQUIRE(c.CreateTexture(dev, &td, &tex) == VriResult_Success);
            VriTextureViewDesc vd{}; vd.texture = tex; vd.viewType = VriTextureViewType_2D; vd.format = VriFormat_Unknown; vd.aspect = VriImageAspect_Color;
            REQUIRE(c.CreateTextureView(dev, &vd, &view) == VriResult_Success);
        };
        VriTexture* tex0 = nullptr; VriDescriptor* view0 = nullptr; makeTarget(tex0, view0);
        VriTexture* tex1 = nullptr; VriDescriptor* view1 = nullptr; makeTarget(tex1, view1);

        auto makeReadback = [&](VriBuffer*& b) {
            VriBufferDesc rb{}; rb.size = static_cast<uint64_t>(kW) * kH * 4; rb.usage = VriBufferUsage_TransferDst; rb.memoryLocation = VriMemoryLocation_HostReadback;
            REQUIRE(c.CreateBuffer(dev, &rb, &b) == VriResult_Success);
        };
        VriBuffer* rb0 = nullptr; makeReadback(rb0);
        VriBuffer* rb1 = nullptr; makeReadback(rb1);

        VriPipelineLayoutDesc ld{};
        VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = shader; sh[0].bytecodeSize = shaderSize; sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = shader; sh[1].bytecodeSize = shaderSize; sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca[2]{};
        ca[0].format = VriFormat_RGBA8_UNORM; ca[0].colorWriteMask = VriColorWrite_RGBA;
        ca[1].format = VriFormat_RGBA8_UNORM; ca[1].colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = ca; pd.outputMerger.colorNum = 2;
        VriPipeline* pipeline = nullptr;
        REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr; REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr; REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr; REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        {
            VriTextureBarrierDesc tb[2]{};
            for (int i = 0; i < 2; ++i) {
                tb[i].texture = i == 0 ? tex0 : tex1;
                tb[i].before.layout = VriLayout_Undefined; tb[i].before.stages = VriPipelineStage_None;
                tb[i].after.access = VriAccess_ColorAttachmentWrite; tb[i].after.layout = VriLayout_ColorAttachment; tb[i].after.stages = VriPipelineStage_ColorAttachmentOutput;
                tb[i].aspect = VriImageAspect_Color;
            }
            VriBarrierGroupDesc g{}; g.textures = tb; g.textureNum = 2; c.CmdBarrier(cmd, &g);
        }

        VriAttachmentDesc rt[2]{};
        rt[0].view = view0; rt[0].loadOp = VriAttachmentLoadOp_Clear; rt[0].storeOp = VriAttachmentStoreOp_Store; rt[0].clearValue.color.f32[3] = 1.0f;
        rt[1].view = view1; rt[1].loadOp = VriAttachmentLoadOp_Clear; rt[1].storeOp = VriAttachmentStoreOp_Store; rt[1].clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att{}; att.colors = rt; att.colorNum = 2; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 1; c.CmdDraw(cmd, &draw);
        c.CmdEndRendering(cmd);

        {
            VriTextureBarrierDesc tb[2]{};
            for (int i = 0; i < 2; ++i) {
                tb[i].texture = i == 0 ? tex0 : tex1;
                tb[i].before.access = VriAccess_ColorAttachmentWrite; tb[i].before.layout = VriLayout_ColorAttachment; tb[i].before.stages = VriPipelineStage_ColorAttachmentOutput;
                tb[i].after.access = VriAccess_CopySourceRead; tb[i].after.layout = VriLayout_CopySource; tb[i].after.stages = VriPipelineStage_Transfer;
                tb[i].aspect = VriImageAspect_Color;
            }
            VriBarrierGroupDesc g{}; g.textures = tb; g.textureNum = 2; c.CmdBarrier(cmd, &g);
        }
        VriBufferTextureCopyDesc tc{}; tc.texture.aspect = VriImageAspect_Color; tc.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, rb0, tex0, &tc);
        c.CmdReadbackTextureToBuffer(cmd, rb1, tex1, &tc);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

        VriFenceSubmitDesc signal{}; signal.fence = fence; signal.value = 1;
        VriQueueSubmitDesc submit{}; submit.commandBuffers = &cmd; submit.commandBufferNum = 1; submit.signalFences = &signal; submit.signalFenceNum = 1;
        c.QueueSubmit(queue, &submit);
        c.Wait(fence, 1);

        const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
        const uint8_t* p0 = static_cast<const uint8_t*>(c.MapBuffer(rb0, 0, static_cast<uint64_t>(kW) * kH * 4));
        const bool red = p0 && p0[o + 0] == 255 && p0[o + 1] == 0 && p0[o + 2] == 0;
        if (p0) c.UnmapBuffer(rb0);
        const uint8_t* p1 = static_cast<const uint8_t*>(c.MapBuffer(rb1, 0, static_cast<uint64_t>(kW) * kH * 4));
        const bool blue = p1 && p1[o + 0] == 0 && p1[o + 1] == 0 && p1[o + 2] == 255;
        if (p1) c.UnmapBuffer(rb1);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(rb0); c.DestroyBuffer(rb1);
        c.DestroyDescriptor(view0); c.DestroyTexture(tex0); c.DestroyDescriptor(view1); c.DestroyTexture(tex1);
        vriDestroyDevice(dev);
        return red && blue;
    }
} // namespace

TEST_CASE("MRT parity: one draw writes red to target 0 and blue to target 1")
{
    bool ran = false;
    const bool vk = RunMrt(VriGraphicsAPI_Vulkan, g_triangleMrtSpv, sizeof(g_triangleMrtSpv), ran);
    if (ran) { CHECK(vk); } else { MESSAGE("Vulkan unavailable - skipped"); }

    const bool wgpu = RunMrt(VriGraphicsAPI_WebGPU, g_triangleMrtWgsl, sizeof(g_triangleMrtWgsl), ran);
    if (ran) { CHECK(wgpu); } else { MESSAGE("WebGPU unavailable - skipped"); }

    const bool gl = RunMrt(VriGraphicsAPI_OpenGL, g_triangleMrtSpv, sizeof(g_triangleMrtSpv), ran);
    if (ran) { CHECK(gl); } else { MESSAGE("OpenGL unavailable - skipped"); }
}
