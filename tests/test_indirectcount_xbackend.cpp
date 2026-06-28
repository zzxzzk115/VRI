// Cross-backend draw-indirect-count: the draw count itself comes from a GPU buffer. The arg buffer
// holds one VriDrawDesc {3,1,0,0} (a procedural red triangle) and maxDrawNum is 4; the count buffer
// holds the real count. With count=1 the triangle is drawn (center red); with count=0 nothing is
// drawn (center stays the clear color) - proving the count buffer, not maxDrawNum, decides how many
// draws issue. Gated on VriDeviceDesc::hasDrawIndirectCount (Vulkan now; D3D12 + desktop GL follow).
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>

#include "shaders/common/triangle_spv.h" // g_triangleSpv (solid red, procedural)
#if defined(_WIN32)
#include "shaders/common/triangle_dxbc.h" // g_triangleDxbcVS / g_triangleDxbcPS (D3D12)
#endif

namespace
{
    struct Shaders
    {
        const void* vs;
        size_t      vsSize;
        const void* ps;
        size_t      psSize;
    };

    constexpr uint32_t kW       = 64;
    constexpr uint32_t kH       = 64;
    constexpr uint32_t kMaxDraw = 4;
    constexpr uint32_t kStride  = 16; // sizeof(VriDrawDesc)
    // The arg buffer must hold maxDrawNum records (validation sizes against maxDrawNum, not the GPU
    // count). Four identical procedural-triangle draws {vertexNum, instanceNum, baseVertex, baseInstance}.
    const uint32_t kArgs[4 * 4] = {3u, 1u, 0u, 0u, 3u, 1u, 0u, 0u, 3u, 1u, 0u, 0u, 3u, 1u, 0u, 0u};

    // Upload `bytes` into a new IndirectBuffer; records the copy + barrier into `cmd`. `staging` is
    // returned so the caller can free it after the submit.
    VriBuffer* MakeIndirect(const VriCoreInterface& c,
                            VriDevice*              dev,
                            VriCommandBuffer*       cmd,
                            const void*             data,
                            uint64_t                bytes,
                            VriBuffer**             staging)
    {
        VriBufferDesc bd {};
        bd.size           = bytes;
        bd.usage          = VriBufferUsage_IndirectBuffer | VriBufferUsage_TransferDst;
        bd.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* buf    = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &buf) == VriResult_Success);
        VriBufferDesc sd {};
        sd.size           = bytes;
        sd.usage          = VriBufferUsage_TransferSrc;
        sd.memoryLocation = VriMemoryLocation_HostUpload;
        REQUIRE(c.CreateBuffer(dev, &sd, staging) == VriResult_Success);
        std::memcpy(c.MapBuffer(*staging, 0, bytes), data, static_cast<size_t>(bytes));
        c.UnmapBuffer(*staging);
        VriBufferCopyDesc cp {};
        cp.size = bytes;
        c.CmdCopyBuffer(cmd, buf, *staging, &cp);
        VriBufferBarrierDesc bb {};
        bb.buffer        = buf;
        bb.before.access = VriAccess_CopyDestinationWrite;
        bb.before.stages = VriPipelineStage_Transfer;
        bb.after.access  = VriAccess_IndirectBufferRead;
        bb.after.stages  = VriPipelineStage_DrawIndirect;
        VriBarrierGroupDesc g {};
        g.buffers   = &bb;
        g.bufferNum = 1;
        c.CmdBarrier(cmd, &g);
        return buf;
    }

    // Render the indirect-count triangle with the given GPU draw count; return the center pixel's red.
    uint8_t RenderWithCount(const VriCoreInterface& c,
                            VriDevice*              dev,
                            VriQueue*               queue,
                            uint32_t                drawCount,
                            const Shaders&          shaders)
    {
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
        td.clearValue.color.f32[3] = 1.0f;
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

        VriPipelineLayoutDesc ld {};
        VriPipelineLayout*    layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);
        VriShaderDesc sh[2] {};
        sh[0].stage          = VriShaderStage_Vertex;
        sh[0].bytecode       = shaders.vs;
        sh[0].bytecodeSize   = shaders.vsSize;
        sh[0].entryPointName = "vertexMain";
        sh[1].stage          = VriShaderStage_Fragment;
        sh[1].bytecode       = shaders.ps;
        sh[1].bytecodeSize   = shaders.psSize;
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
        VriBuffer* argStg = nullptr;
        VriBuffer* cntStg = nullptr;
        VriBuffer* args   = MakeIndirect(c, dev, cmd, kArgs, sizeof(kArgs), &argStg);
        VriBuffer* count  = MakeIndirect(c, dev, cmd, &drawCount, sizeof(uint32_t), &cntStg);
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
        rt.clearValue.color.f32[3] = 1.0f; // black, opaque
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
        c.CmdSetPipelineLayout(cmd, layout);
        // maxDrawNum is kMaxDraw, but the GPU count buffer holds the real count.
        c.CmdDrawIndirectCount(cmd, args, 0, count, 0, kMaxDraw, kStride);
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

        const uint8_t* px  = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
        const uint8_t  red = px[((kH / 2) * kW + (kW / 2)) * 4 + 0];
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(args);
        c.DestroyBuffer(count);
        c.DestroyBuffer(argStg);
        c.DestroyBuffer(cntStg);
        c.DestroyBuffer(readback);
        c.DestroyDescriptor(colorView);
        c.DestroyTexture(color);
        return red;
    }

    void Check(VriGraphicsAPI api, const char* name, const Shaders& shaders)
    {
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        VriDevice* dev      = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << name << "] unavailable - skipped");
            return;
        }
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        if (c.GetDeviceDesc(dev)->hasDrawIndirectCount == VRI_FALSE)
        {
            MESSAGE("[" << name << "] no draw-indirect-count - skipped");
            vriDestroyDevice(dev);
            return;
        }
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        const uint8_t red1 = RenderWithCount(c, dev, queue, 1, shaders); // GPU count 1 -> triangle drawn
        const uint8_t red0 = RenderWithCount(c, dev, queue, 0, shaders); // GPU count 0 -> nothing drawn
        CHECK(red1 > 200);
        CHECK(red0 < 60);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("Draw indirect count: the GPU count buffer drives the draw count")
{
    const Shaders spv {g_triangleSpv, sizeof(g_triangleSpv), g_triangleSpv, sizeof(g_triangleSpv)};
    Check(VriGraphicsAPI_Vulkan, "Vulkan", spv);
    Check(VriGraphicsAPI_OpenGL, "OpenGL", spv);
#if defined(_WIN32)
    const Shaders dxbc {g_triangleDxbcVS, sizeof(g_triangleDxbcVS), g_triangleDxbcPS, sizeof(g_triangleDxbcPS)};
    Check(VriGraphicsAPI_D3D12, "D3D12", dxbc);
#endif
}
