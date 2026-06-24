// Legacy geometry-shader parity. A vertex->geometry->fragment pipeline where the
// geometry stage re-emits the triangle and assigns green; the center pixel must be
// green on backends with a geometry stage (Vulkan, desktop OpenGL). Backends
// without one (WebGPU, WebGL2) report hasGeometryShader=false and must reject the
// pipeline with VriResult_Unsupported - the explicit-capability contract.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/triangle_gs_spv.h" // g_triangleGsSpv (vertex/geometry/fragment; SPIR-V only)

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    void MakeGsShaders(VriShaderDesc sh[3])
    {
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = g_triangleGsSpv; sh[0].bytecodeSize = sizeof(g_triangleGsSpv); sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Geometry; sh[1].bytecode = g_triangleGsSpv; sh[1].bytecodeSize = sizeof(g_triangleGsSpv); sh[1].entryPointName = "geometryMain";
        sh[2].stage = VriShaderStage_Fragment; sh[2].bytecode = g_triangleGsSpv; sh[2].bytecodeSize = sizeof(g_triangleGsSpv); sh[2].entryPointName = "fragmentMain";
    }

    bool RunGeometry(VriGraphicsAPI api, bool& ran, bool& hasGeom)
    {
        ran = false; hasGeom = false;
        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = api; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;
        ran = true;

        VriCoreInterface c{};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        hasGeom = c.GetDeviceDesc(dev)->hasGeometryShader != VRI_FALSE;

        VriShaderDesc sh[3]{}; MakeGsShaders(sh);
        VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
        VriPipelineLayoutDesc ld{}; VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 3;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;

        if (!hasGeom)
        {
            // Contract: a geometry stage on a backend without one -> Unsupported.
            VriPipeline* p0 = nullptr;
            const bool unsupported = c.CreateGraphicsPipeline(dev, &pd, &p0) == VriResult_Unsupported;
            c.DestroyPipelineLayout(layout);
            vriDestroyDevice(dev);
            return unsupported;
        }

        VriQueue* queue = nullptr; REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        VriTextureDesc td{};
        td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM;
        td.width = kW; td.height = kH; td.depth = 1; td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1;
        td.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc; td.memoryLocation = VriMemoryLocation_Device;
        td.clearValue.color.f32[3] = 1.0f; // match the render-pass clear
        VriTexture* color = nullptr; REQUIRE(c.CreateTexture(dev, &td, &color) == VriResult_Success);
        VriTextureViewDesc cvd{}; cvd.texture = color; cvd.viewType = VriTextureViewType_2D; cvd.format = VriFormat_Unknown; cvd.aspect = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr; REQUIRE(c.CreateTextureView(dev, &cvd, &colorView) == VriResult_Success);
        VriBufferDesc rbd{}; rbd.size = static_cast<uint64_t>(kW) * kH * 4; rbd.usage = VriBufferUsage_TransferDst; rbd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr; REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

        VriPipeline* pipeline = nullptr;
        REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr; REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr; REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr; REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
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

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
        REQUIRE(px != nullptr);
        const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
        const bool green = (px[o + 0] == 0 && px[o + 1] == 255 && px[o + 2] == 0);
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(readback); c.DestroyDescriptor(colorView); c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return green;
    }
} // namespace

TEST_CASE("geometry-shader parity: GS re-emits the triangle green (or Unsupported where absent)")
{
    bool ran = false, hasGeom = false;
    const bool vk = RunGeometry(VriGraphicsAPI_Vulkan, ran, hasGeom);
    if (ran) { CHECK(vk); MESSAGE("Vulkan geometry=", hasGeom); } else { MESSAGE("Vulkan unavailable - skipped"); }

    const bool wgpu = RunGeometry(VriGraphicsAPI_WebGPU, ran, hasGeom);
    if (ran) { CHECK(wgpu); MESSAGE("WebGPU geometry=", hasGeom); } else { MESSAGE("WebGPU unavailable - skipped"); }

    const bool gl = RunGeometry(VriGraphicsAPI_OpenGL, ran, hasGeom);
    if (ran) { CHECK(gl); MESSAGE("OpenGL geometry=", hasGeom); } else { MESSAGE("OpenGL unavailable - skipped"); }
}
