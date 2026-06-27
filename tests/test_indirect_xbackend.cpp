// Cross-backend indirect-draw parity: CmdDrawIndirect reads the draw arguments
// {vertexCount, instanceCount, firstVertex, firstInstance} from a GPU buffer instead
// of the CPU. The buffer holds {3,1,0,0}, so the solid-red procedural triangle is
// drawn and the center pixel must be red on every backend. The GL backend previously
// no-op'd CmdDrawIndirect (a silent gap); it now issues glDrawArraysIndirect.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/triangle_spv.h"  // g_triangleSpv  (solid red, procedural)
#include "shaders/triangle_wgsl.h" // g_triangleWgsl

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    // GL/VK/WebGPU share this non-indexed indirect layout.
    const uint32_t kIndirect[4] = {3u, 1u, 0u, 0u}; // vertexCount, instanceCount, firstVertex, firstInstance

    bool RunIndirect(VriGraphicsAPI api, const void* shader, size_t shaderSize, bool& ran)
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

        VriBufferDesc rbd {};
        rbd.size            = static_cast<uint64_t>(kW) * kH * 4;
        rbd.usage           = VriBufferUsage_TransferDst;
        rbd.memoryLocation  = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

        // device-local indirect buffer filled from a host staging buffer
        VriBufferDesc ibd {};
        ibd.size            = sizeof(kIndirect);
        ibd.usage           = VriBufferUsage_IndirectBuffer | VriBufferUsage_TransferDst;
        ibd.memoryLocation  = VriMemoryLocation_Device;
        VriBuffer* indirect = nullptr;
        REQUIRE(c.CreateBuffer(dev, &ibd, &indirect) == VriResult_Success);
        VriBufferDesc sbd {};
        sbd.size           = sizeof(kIndirect);
        sbd.usage          = VriBufferUsage_TransferSrc;
        sbd.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* staging = nullptr;
        REQUIRE(c.CreateBuffer(dev, &sbd, &staging) == VriResult_Success);
        {
            void* m = c.MapBuffer(staging, 0, sizeof(kIndirect));
            REQUIRE(m != nullptr);
            for (uint64_t i = 0; i < sizeof(kIndirect); ++i)
                static_cast<uint8_t*>(m)[i] = reinterpret_cast<const uint8_t*>(kIndirect)[i];
            c.UnmapBuffer(staging);
        }

        VriPipelineLayoutDesc ld {};
        VriPipelineLayout*    layout = nullptr;
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

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        VriBufferCopyDesc cp {};
        cp.size = sizeof(kIndirect);
        c.CmdCopyBuffer(cmd, indirect, staging, &cp);
        {
            VriBufferBarrierDesc bb {};
            bb.buffer        = indirect;
            bb.before.access = VriAccess_CopyDestinationWrite;
            bb.before.stages = VriPipelineStage_Transfer;
            bb.after.access  = VriAccess_IndirectBufferRead;
            bb.after.stages  = VriPipelineStage_DrawIndirect;
            VriBarrierGroupDesc g {};
            g.buffers   = &bb;
            g.bufferNum = 1;
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
        c.CmdDrawIndirect(cmd, indirect, 0, 1, sizeof(kIndirect));
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
        const uint32_t o   = ((kH / 2) * kW + (kW / 2)) * 4;
        const bool     red = (px[o + 0] == 255 && px[o + 1] == 0 && px[o + 2] == 0);
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(indirect);
        c.DestroyBuffer(staging);
        c.DestroyBuffer(readback);
        c.DestroyDescriptor(colorView);
        c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return red;
    }
} // namespace

TEST_CASE("indirect-draw parity: CmdDrawIndirect reads {3,1,0,0} from a buffer and draws the red triangle")
{
    bool       ran = false;
    const bool vk  = RunIndirect(VriGraphicsAPI_Vulkan, g_triangleSpv, sizeof(g_triangleSpv), ran);
    if (ran)
    {
        CHECK(vk);
    }
    else
    {
        MESSAGE("Vulkan unavailable - skipped");
    }

    const bool wgpu = RunIndirect(VriGraphicsAPI_WebGPU, g_triangleWgsl, sizeof(g_triangleWgsl), ran);
    if (ran)
    {
        CHECK(wgpu);
    }
    else
    {
        MESSAGE("WebGPU unavailable - skipped");
    }

    const bool gl = RunIndirect(VriGraphicsAPI_OpenGL, g_triangleSpv, sizeof(g_triangleSpv), ran);
    if (ran)
    {
        CHECK(gl);
    }
    else
    {
        MESSAGE("OpenGL unavailable - skipped");
    }
}
