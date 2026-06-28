// Legacy tessellation parity. A vertex->hull(TCS)->domain(TES)->fragment pipeline
// over a 3-control-point patch (PatchList topology) tessellates a triangle and
// paints it green; the center pixel must be green on backends with tessellation
// (Vulkan + desktop OpenGL). Backends without it (WebGPU, WebGL2) report
// hasTessellation=false and must reject the pipeline with VriResult_Unsupported.
//
// Authored in Slang (vertex/hull/domain/fragment in one module); requires Slang
// >= 2026.11, whose SPIR-V backend emits tessellation hull shaders (2025.11 crashes).
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/tests/triangle_tess_spv.h" // g_triangleTessSpv (vertex/hull/domain/fragment; SPIR-V only)

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    void MakeTessShaders(VriShaderDesc sh[4])
    {
        sh[0].stage          = VriShaderStage_Vertex;
        sh[0].bytecode       = g_triangleTessSpv;
        sh[0].bytecodeSize   = sizeof(g_triangleTessSpv);
        sh[0].entryPointName = "vertexMain";
        sh[1].stage          = VriShaderStage_TessControl;
        sh[1].bytecode       = g_triangleTessSpv;
        sh[1].bytecodeSize   = sizeof(g_triangleTessSpv);
        sh[1].entryPointName = "hullMain";
        sh[2].stage          = VriShaderStage_TessEval;
        sh[2].bytecode       = g_triangleTessSpv;
        sh[2].bytecodeSize   = sizeof(g_triangleTessSpv);
        sh[2].entryPointName = "domainMain";
        sh[3].stage          = VriShaderStage_Fragment;
        sh[3].bytecode       = g_triangleTessSpv;
        sh[3].bytecodeSize   = sizeof(g_triangleTessSpv);
        sh[3].entryPointName = "fragmentMain";
    }

    bool RunTess(VriGraphicsAPI api, bool& ran, bool& hasTess)
    {
        ran     = false;
        hasTess = false;
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
        hasTess = c.GetDeviceDesc(dev)->hasTessellation != VRI_FALSE;

        VriShaderDesc sh[4] {};
        MakeTessShaders(sh);
        VriColorAttachmentDesc ca {};
        ca.format         = VriFormat_RGBA8_UNORM;
        ca.colorWriteMask = VriColorWrite_RGBA;
        VriPipelineLayoutDesc ld {};
        VriPipelineLayout*    layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriGraphicsPipelineDesc pd {};
        pd.pipelineLayout                  = layout;
        pd.shaders                         = sh;
        pd.shaderNum                       = 4;
        pd.inputAssembly.topology          = VriPrimitiveTopology_PatchList;
        pd.tessellation.patchControlPoints = 3;
        pd.rasterization.cullMode          = VriCullMode_None;
        pd.rasterization.lineWidth         = 1.0f;
        pd.multisample.sampleNum           = 1;
        pd.outputMerger.colors             = &ca;
        pd.outputMerger.colorNum           = 1;

        if (!hasTess)
        {
            // Contract: tessellation stages on a backend without them -> Unsupported.
            VriPipeline* p0          = nullptr;
            const bool   unsupported = c.CreateGraphicsPipeline(dev, &pd, &p0) == VriResult_Unsupported;
            c.DestroyPipelineLayout(layout);
            vriDestroyDevice(dev);
            return unsupported;
        }

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
        VriBufferDesc rbd {};
        rbd.size            = static_cast<uint64_t>(kW) * kH * 4;
        rbd.usage           = VriBufferUsage_TransferDst;
        rbd.memoryLocation  = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

        VriPipeline* pipeline = nullptr;
        REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
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
        VriDrawDesc draw {};
        draw.vertexNum   = 3;
        draw.instanceNum = 1;
        c.CmdDraw(cmd, &draw); // 3 control points = one patch
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

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
        REQUIRE(px != nullptr);
        const uint32_t o     = ((kH / 2) * kW + (kW / 2)) * 4;
        const bool     green = (px[o + 0] == 0 && px[o + 1] == 255 && px[o + 2] == 0);
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(readback);
        c.DestroyDescriptor(colorView);
        c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return green;
    }
} // namespace

TEST_CASE("tessellation parity: hull/domain tessellate a triangle green (or Unsupported where absent)")
{
    bool       ran = false, hasTess = false;
    const bool vk = RunTess(VriGraphicsAPI_Vulkan, ran, hasTess);
    if (ran)
    {
        CHECK(vk);
        MESSAGE("Vulkan tessellation=", hasTess);
    }
    else
    {
        MESSAGE("Vulkan unavailable - skipped");
    }

    const bool wgpu = RunTess(VriGraphicsAPI_WebGPU, ran, hasTess);
    if (ran)
    {
        CHECK(wgpu);
        MESSAGE("WebGPU tessellation=", hasTess);
    }
    else
    {
        MESSAGE("WebGPU unavailable - skipped");
    }

    const bool gl = RunTess(VriGraphicsAPI_OpenGL, ran, hasTess);
    if (ran)
    {
        CHECK(gl);
        MESSAGE("OpenGL tessellation=", hasTess);
    }
    else
    {
        MESSAGE("OpenGL unavailable - skipped");
    }
}
