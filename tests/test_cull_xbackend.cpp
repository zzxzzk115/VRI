// Cross-backend back-face culling parity. With cullMode=Back and a CCW front
// face (the VRI standard), a CCW-wound green triangle is front-facing (drawn) and
// the same triangle wound CW is back-facing (culled). Drawing green-front then
// red-back, the center must stay green on every backend - which also proves the
// winding/handedness compensation (VK negative-height viewport, GL flip_vert_y +
// front-face swap, WebGPU) is consistent, not just the Y orientation.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <utility>

#include "shaders/triangle_vbuf_spv.h"  // g_triangleVbufSpv  (position.xyz + color)
#include "shaders/triangle_vbuf_wgsl.h" // g_triangleVbufWgsl

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    bool RunCull(VriGraphicsAPI api, const void* shader, size_t shaderSize, bool& ran)
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

        // CCW (front, green): bottom -> top-right -> top-left is CCW in Y-up space.
        // CW (back, red): same triangle with the last two vertices swapped.
        const float frontGreen[] = {
            0.0f,
            -0.5f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.5f,
            0.5f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            -0.5f,
            0.5f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
        };
        const float backRed[] = {
            0.0f,
            -0.5f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            -0.5f,
            0.5f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.5f,
            0.5f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
        };
        auto makeVbuf = [&](const float* data) {
            const uint64_t sz = 18 * sizeof(float);
            VriBufferDesc  vd {};
            vd.size           = sz;
            vd.usage          = VriBufferUsage_VertexBuffer | VriBufferUsage_TransferDst;
            vd.memoryLocation = VriMemoryLocation_Device;
            VriBuffer* vb     = nullptr;
            REQUIRE(c.CreateBuffer(dev, &vd, &vb) == VriResult_Success);
            VriBufferDesc sd {};
            sd.size           = sz;
            sd.usage          = VriBufferUsage_TransferSrc;
            sd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* st     = nullptr;
            REQUIRE(c.CreateBuffer(dev, &sd, &st) == VriResult_Success);
            void* m = c.MapBuffer(st, 0, sz);
            REQUIRE(m != nullptr);
            for (uint64_t i = 0; i < sz; ++i)
                static_cast<uint8_t*>(m)[i] = reinterpret_cast<const uint8_t*>(data)[i];
            c.UnmapBuffer(st);
            return std::pair<VriBuffer*, VriBuffer*> {vb, st};
        };
        auto [vbFront, stFront] = makeVbuf(frontGreen);
        auto [vbBack, stBack]   = makeVbuf(backRed);

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
        VriVertexStreamDesc stream {};
        stream.stride      = 6 * sizeof(float);
        stream.bindingSlot = 0;
        stream.stepRate    = VriVertexStepRate_PerVertex;
        VriVertexAttributeDesc attrs[2] {};
        attrs[0].format      = VriFormat_RGB32_SFLOAT;
        attrs[0].offset      = 0;
        attrs[0].streamIndex = 0;
        attrs[1].format      = VriFormat_RGB32_SFLOAT;
        attrs[1].offset      = 3 * sizeof(float);
        attrs[1].streamIndex = 0;
        VriColorAttachmentDesc ca {};
        ca.format         = VriFormat_RGBA8_UNORM;
        ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd {};
        pd.pipelineLayout           = layout;
        pd.shaders                  = sh;
        pd.shaderNum                = 2;
        pd.vertexInput.streams      = &stream;
        pd.vertexInput.streamNum    = 1;
        pd.vertexInput.attributes   = attrs;
        pd.vertexInput.attributeNum = 2;
        pd.inputAssembly.topology   = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode   = VriCullMode_Back;
        pd.rasterization.frontFace  = VriFrontFace_CounterClockwise;
        pd.rasterization.lineWidth  = 1.0f;
        pd.multisample.sampleNum    = 1;
        pd.outputMerger.colors      = &ca;
        pd.outputMerger.colorNum    = 1;
        VriPipeline* pipeline       = nullptr;
        REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        VriBufferCopyDesc cp {};
        cp.size = 18 * sizeof(float);
        c.CmdCopyBuffer(cmd, vbFront, stFront, &cp);
        c.CmdCopyBuffer(cmd, vbBack, stBack, &cp);
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
        {
            VriVertexBufferBinding vbb {};
            vbb.buffer = vbFront;
            vbb.offset = 0;
            c.CmdSetVertexBuffers(cmd, 0, &vbb, 1);
            c.CmdDraw(cmd, &draw);
        } // front: drawn
        {
            VriVertexBufferBinding vbb {};
            vbb.buffer = vbBack;
            vbb.offset = 0;
            c.CmdSetVertexBuffers(cmd, 0, &vbb, 1);
            c.CmdDraw(cmd, &draw);
        } // back: culled
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
        c.DestroyBuffer(vbFront);
        c.DestroyBuffer(vbBack);
        c.DestroyBuffer(stFront);
        c.DestroyBuffer(stBack);
        c.DestroyBuffer(readback);
        c.DestroyDescriptor(colorView);
        c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return green;
    }
} // namespace

TEST_CASE("cull parity: CCW front drawn, CW back culled (center stays green)")
{
    bool       ran = false;
    const bool vk  = RunCull(VriGraphicsAPI_Vulkan, g_triangleVbufSpv, sizeof(g_triangleVbufSpv), ran);
    if (ran)
    {
        CHECK(vk);
    }
    else
    {
        MESSAGE("Vulkan unavailable - skipped");
    }

    const bool wgpu = RunCull(VriGraphicsAPI_WebGPU, g_triangleVbufWgsl, sizeof(g_triangleVbufWgsl), ran);
    if (ran)
    {
        CHECK(wgpu);
    }
    else
    {
        MESSAGE("WebGPU unavailable - skipped");
    }

    const bool gl = RunCull(VriGraphicsAPI_OpenGL, g_triangleVbufSpv, sizeof(g_triangleVbufSpv), ran);
    if (ran)
    {
        CHECK(gl);
    }
    else
    {
        MESSAGE("OpenGL unavailable - skipped");
    }
}
