// Multiview (single-pass layered rendering): render one fullscreen triangle to a 2-layer array
// target in a single pass with viewMask = 0b11. The shader colors each view by SV_ViewID, so
// layer 0 comes out red and layer 1 green -- proving the viewMask drove per-layer rendering (the
// basis of single-pass VR stereo). Gated on VriDeviceDesc::hasMultiview: Vulkan + desktop-GL/GLES
// (GL_OVR_multiview2); D3D12 ViewInstancing is a follow-up, so D3D12 self-skips.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include "shaders/tests/multiview_frag_spv.h" // g_multiviewFragSpv
#include "shaders/tests/multiview_vert_spv.h" // g_multiviewVertSpv (split per stage so the fragment
                                              // module carries no ViewIndex - GL OVR_multiview2 rule)

#include <cstdint>

namespace
{
    constexpr uint32_t kW          = 64;
    constexpr uint32_t kH          = 64;
    constexpr uint32_t kViews      = 2;
    constexpr uint64_t kLayerBytes = static_cast<uint64_t>(kW) * kH * 4;

    void RunMultiview(VriGraphicsAPI api, const char* name)
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
        const VriDeviceDesc* dd = c.GetDeviceDesc(dev);
        if (dd->hasMultiview == VRI_FALSE || dd->maxViewCount < kViews)
        {
            MESSAGE("[" << name << "] multiview unsupported - skipped");
            vriDestroyDevice(dev);
            return;
        }
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        // 2-layer color array target + a 2D-array view spanning both layers.
        VriTextureDesc td {};
        td.type           = VriTextureType_2D;
        td.format         = VriFormat_RGBA8_UNORM;
        td.width          = kW;
        td.height         = kH;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = kViews;
        td.sampleNum      = 1;
        td.usage          = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* color = nullptr;
        REQUIRE(c.CreateTexture(dev, &td, &color) == VriResult_Success);

        VriTextureViewDesc vdsc {};
        vdsc.texture        = color;
        vdsc.viewType       = VriTextureViewType_2DArray;
        vdsc.format         = VriFormat_Unknown;
        vdsc.aspect         = VriImageAspect_Color;
        vdsc.layerNum       = kViews;
        VriDescriptor* view = nullptr;
        REQUIRE(c.CreateTextureView(dev, &vdsc, &view) == VriResult_Success);

        VriBufferDesc bd {};
        bd.size             = kLayerBytes * kViews; // both layers, contiguous
        bd.usage            = VriBufferUsage_TransferDst;
        bd.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &readback) == VriResult_Success);

        VriPipelineLayoutDesc ld {};
        VriPipelineLayout*    layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriShaderDesc sh[2] {};
        sh[0].stage          = VriShaderStage_Vertex;
        sh[1].stage          = VriShaderStage_Fragment;
        sh[0].entryPointName = "vertexMain";
        sh[1].entryPointName = "fragmentMain";
        sh[0].bytecode       = g_multiviewVertSpv;
        sh[0].bytecodeSize   = sizeof(g_multiviewVertSpv);
        sh[1].bytecode       = g_multiviewFragSpv;
        sh[1].bytecodeSize   = sizeof(g_multiviewFragSpv);

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
        pd.outputMerger.viewMask   = (1u << kViews) - 1u; // 0b11: views 0 and 1
        VriPipeline* pipeline      = nullptr;
        REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        auto texBarrier = [&](VriAccessFlags        beforeA,
                              VriLayout             beforeL,
                              VriPipelineStageFlags beforeS,
                              VriAccessFlags        afterA,
                              VriLayout             afterL,
                              VriPipelineStageFlags afterS) {
            VriTextureBarrierDesc b {};
            b.texture       = color;
            b.before.access = beforeA;
            b.before.layout = beforeL;
            b.before.stages = beforeS;
            b.after.access  = afterA;
            b.after.layout  = afterL;
            b.after.stages  = afterS;
            b.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g {};
            g.textures   = &b;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        };
        texBarrier(VriAccess_None,
                   VriLayout_Undefined,
                   VriPipelineStage_None,
                   VriAccess_ColorAttachmentWrite,
                   VriLayout_ColorAttachment,
                   VriPipelineStage_ColorAttachmentOutput);

        VriAttachmentDesc rt {};
        rt.view                    = view;
        rt.loadOp                  = VriAttachmentLoadOp_Clear;
        rt.storeOp                 = VriAttachmentStoreOp_Store;
        rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att {};
        att.colors            = &rt;
        att.colorNum          = 1;
        att.renderArea.width  = kW;
        att.renderArea.height = kH;
        att.layerNum          = 1;                   // multiview drives layers via viewMask
        att.viewMask          = (1u << kViews) - 1u; // 0b11
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp {0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1};
        c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc {0, 0, kW, kH};
        c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        VriDrawDesc draw {};
        draw.vertexNum   = 3;
        draw.instanceNum = 1;
        c.CmdDraw(cmd, &draw);
        c.CmdEndRendering(cmd);

        texBarrier(VriAccess_ColorAttachmentWrite,
                   VriLayout_ColorAttachment,
                   VriPipelineStage_ColorAttachmentOutput,
                   VriAccess_CopySourceRead,
                   VriLayout_CopySource,
                   VriPipelineStage_Transfer);
        for (uint32_t layer = 0; layer < kViews; ++layer)
        {
            VriBufferTextureCopyDesc rc {};
            rc.bufferOffset      = layer * kLayerBytes;
            rc.texture.aspect    = VriImageAspect_Color;
            rc.texture.baseLayer = layer;
            rc.texture.layerNum  = 1;
            rc.texture.width     = kW;
            rc.texture.height    = kH;
            c.CmdReadbackTextureToBuffer(cmd, readback, color, &rc);
        }
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

        VriFenceSubmitDesc sig {};
        sig.fence = fence;
        sig.value = 1;
        VriQueueSubmitDesc sub {};
        sub.commandBuffers   = &cmd;
        sub.commandBufferNum = 1;
        sub.signalFences     = &sig;
        sub.signalFenceNum   = 1;
        c.QueueSubmit(queue, &sub);
        c.Wait(fence, 1);

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bd.size));
        // Layer 0 (view 0) -> red; layer 1 (view 1) -> green. Sample each layer's center pixel.
        const uint8_t* v0         = px + 0 * kLayerBytes + (static_cast<size_t>(kH / 2) * kW + kW / 2) * 4;
        const uint8_t* v1         = px + 1 * kLayerBytes + (static_cast<size_t>(kH / 2) * kW + kW / 2) * 4;
        const bool     view0Red   = v0[0] > 240 && v0[1] < 16 && v0[2] < 16;
        const bool     view1Green = v1[0] < 16 && v1[1] > 240 && v1[2] < 16;
        MESSAGE("[" << name << "] view0=(" << int(v0[0]) << "," << int(v0[1]) << "," << int(v0[2]) << ") view1=("
                    << int(v1[0]) << "," << int(v1[1]) << "," << int(v1[2]) << ")");
        c.UnmapBuffer(readback);
        CHECK(view0Red);
        CHECK(view1Green);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(readback);
        c.DestroyDescriptor(view);
        c.DestroyTexture(color);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("Multiview: single-pass stereo colors each view by SV_ViewID")
{
    RunMultiview(VriGraphicsAPI_Vulkan, "Vulkan");
    RunMultiview(VriGraphicsAPI_D3D12, "D3D12");
    RunMultiview(VriGraphicsAPI_OpenGL, "OpenGL");
}
