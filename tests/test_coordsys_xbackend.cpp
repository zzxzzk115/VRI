// Cross-backend coordinate-system parity: VRI standardizes on Y-up clip space.
// The solid-red triangle is apex-down (apex at NDC y=-0.5 = bottom, wide base at
// y=+0.5 = top). We sample an off-center pixel pair that flips meaning under a
// vertical flip, asserting BOTH Vulkan and WebGPU agree (catches the Vulkan
// clip-Y-down mismatch fixed via negative-height viewport).
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/triangle_spv.h"  // g_triangleSpv  (Vulkan, solid red)
#include "shaders/triangle_wgsl.h" // g_triangleWgsl (WebGPU, solid red)

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    // Render the solid triangle and report whether two probe pixels are red.
    // topProbe (col44,row20): inside the WIDE base near the top  -> red iff Y-up.
    // botProbe (col44,row44): near the NARROW apex at the bottom -> red iff flipped.
    void RunSolidTriangle(VriGraphicsAPI api, const void* shader, size_t shaderSize,
                          bool& topRed, bool& botRed, bool& ran)
    {
        ran = false;
        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return;
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
        VriTextureViewDesc vdsc{};
        vdsc.texture = color; vdsc.viewType = VriTextureViewType_2D; vdsc.format = VriFormat_Unknown; vdsc.aspect = VriImageAspect_Color;
        VriDescriptor* view = nullptr;
        REQUIRE(c.CreateTextureView(dev, &vdsc, &view) == VriResult_Success);

        VriBufferDesc bd{};
        bd.size = static_cast<uint64_t>(kW) * kH * 4; bd.usage = VriBufferUsage_TransferDst; bd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &readback) == VriResult_Success);

        VriPipelineLayoutDesc ld{};
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

        VriCommandAllocator* alloc = nullptr; REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr; REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr; REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        {
            VriTextureBarrierDesc b{};
            b.texture = color; b.before.layout = VriLayout_Undefined; b.before.stages = VriPipelineStage_None;
            b.after.access = VriAccess_ColorAttachmentWrite; b.after.layout = VriLayout_ColorAttachment; b.after.stages = VriPipelineStage_ColorAttachmentOutput;
            b.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc g{}; g.textures = &b; g.textureNum = 1; c.CmdBarrier(cmd, &g);
        }
        VriAttachmentDesc rt{}; rt.view = view; rt.loadOp = VriAttachmentLoadOp_Clear; rt.storeOp = VriAttachmentStoreOp_Store; rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att{}; att.colors = &rt; att.colorNum = 1; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, (float)kW, (float)kH, 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 1; c.CmdDraw(cmd, &draw);
        c.CmdEndRendering(cmd);
        {
            VriTextureBarrierDesc b{};
            b.texture = color; b.before.access = VriAccess_ColorAttachmentWrite; b.before.layout = VriLayout_ColorAttachment; b.before.stages = VriPipelineStage_ColorAttachmentOutput;
            b.after.access = VriAccess_CopySourceRead; b.after.layout = VriLayout_CopySource; b.after.stages = VriPipelineStage_Transfer;
            b.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc g{}; g.textures = &b; g.textureNum = 1; c.CmdBarrier(cmd, &g);
        }
        VriBufferTextureCopyDesc copy{}; copy.texture.aspect = VriImageAspect_Color; copy.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &copy);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

        VriFenceSubmitDesc sig{}; sig.fence = fence; sig.value = 1;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(queue, &sub);
        c.Wait(fence, 1);

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bd.size));
        REQUIRE(px != nullptr);
        auto isRed = [&](uint32_t col, uint32_t row) {
            const uint32_t o = (row * kW + col) * 4;
            return px[o + 0] == 255 && px[o + 1] == 0 && px[o + 2] == 0;
        };
        topRed = isRed(44, 20);
        botRed = isRed(44, 44);
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(readback); c.DestroyDescriptor(view); c.DestroyTexture(color);
        vriDestroyDevice(dev);
    }

    void CheckYUp(VriGraphicsAPI api, const void* shader, size_t size, const char* name)
    {
        bool topRed = false, botRed = false, ran = false;
        RunSolidTriangle(api, shader, size, topRed, botRed, ran);
        if (!ran) { MESSAGE(name, " unavailable - skipped"); return; }
        CHECK(topRed);  // wide base near the top  -> Y-up
        CHECK(!botRed); // narrow apex at the bottom (not flipped)
    }
}

TEST_CASE("coordinate-system parity: both backends render Y-up (no vertical flip)")
{
    CheckYUp(VriGraphicsAPI_Vulkan, g_triangleSpv, sizeof(g_triangleSpv), "Vulkan");
    CheckYUp(VriGraphicsAPI_WebGPU, g_triangleWgsl, sizeof(g_triangleWgsl), "WebGPU");
    // OpenGL consumes the SPIR-V (transpiled to GLSL by SPIRV-Cross).
    CheckYUp(VriGraphicsAPI_OpenGL, g_triangleSpv, sizeof(g_triangleSpv), "OpenGL");
}
