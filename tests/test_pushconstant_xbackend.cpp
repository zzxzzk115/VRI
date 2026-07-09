// Cross-backend push-constant MATRIX parity. A triangle is transformed by a column-major
// float4x4 (Slang's / glm's default) carried in a PUSH CONSTANT, drawn into a 64x64
// offscreen target and read back. Vulkan / D3D12 / WebGPU upload the push blob as raw
// bytes and get it right; OpenGL has no push constants and emulates them by handing each
// matrix member to glUniformMatrix - which needs the correct transpose flag. Because
// SPIRV-Cross emits the same `v * M` GLSL for a matrix regardless of storage, a column-
// major matrix loaded with GL_FALSE is transposed (its translation column lands in the
// wrong slot and is silently dropped), so the transform is wrong on GL only. This is the
// regression guard for that bug (a plain float4x4 mvp in the model-viewer shader rendered
// nothing sensible on OpenGL until VRI's GL backend transposed column-major matrices).
//
// The identity case alone can't catch it (identity is pass-through), so two cases run:
//   - identity mvp        -> the triangle covers the center pixel (green).
//   - a large +x translation -> the triangle is pushed fully off-screen (center clears).
// A transposed matrix keeps identity looking fine but loses the translation, failing the
// second case on the affected backend only.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstdio>

#include "shaders/tests/triangle_push_spv.h"  // g_trianglePushSpv  (Vulkan + OpenGL via SPIRV-Cross)
#include "shaders/tests/triangle_push_wgsl.h" // g_trianglePushWgsl (WebGPU)
#if defined(_WIN32)
#include "shaders/tests/triangle_push_dxbc.h" // g_trianglePushDxbcVS + g_trianglePushDxbcPS (D3D12)
#endif

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    // Column-major float4x4 (the shader declares `column_major`, matching vshadersystem /
    // the model-viewer). The translation lives in column 3, i.e. flat index 12 (tx) / 13 (ty).
    struct Mat4
    {
        float m[16];
    };
    Mat4 Identity() { return Mat4 {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}}; }
    Mat4 TranslateX(float tx)
    {
        Mat4 r  = Identity();
        r.m[12] = tx; // column 3, row 0
        return r;
    }

    // Draw the SV_VertexID triangle transformed by `mvp` and report whether the center
    // pixel is green. vs/ps may alias the same multi-entry blob (SPIR-V / WGSL) or be
    // separate per-stage blobs (DXBC).
    bool RunPush(VriGraphicsAPI api,
                 const void*    vs,
                 size_t         vsSize,
                 const void*    ps,
                 size_t         psSize,
                 const Mat4&    mvp,
                 bool&          ran)
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
        td.clearValue.color.f32[3] = 1.0f; // match the render-pass clear (opaque black)
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

        // Pipeline layout with a single 64-byte push constant (the float4x4), visible to
        // the vertex stage.
        VriPushConstantDesc pcd {};
        pcd.baseRegister = 0;
        pcd.size         = 64;
        pcd.shaderStages = VriShaderStage_Vertex;
        VriPipelineLayoutDesc ld {};
        ld.pushConstants          = &pcd;
        ld.pushConstantNum        = 1;
        ld.shaderStages           = VriShaderStage_Vertex | VriShaderStage_Fragment;
        VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriShaderDesc sh[2] {};
        sh[0].stage          = VriShaderStage_Vertex;
        sh[0].bytecode       = vs;
        sh[0].bytecodeSize   = vsSize;
        sh[0].entryPointName = "vertexMain";
        sh[1].stage          = VriShaderStage_Fragment;
        sh[1].bytecode       = ps;
        sh[1].bytecodeSize   = psSize;
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
        c.CmdSetPipelineLayout(cmd, layout); // Vulkan/D3D12 push constants target the bound layout
        c.CmdSetConstants(cmd, 0, mvp.m, sizeof(mvp.m));
        VriDrawDesc draw {};
        draw.vertexNum   = 3;
        draw.instanceNum = 1;
        c.CmdDraw(cmd, &draw);
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
        const bool     green = (px[o + 1] == 255 && px[o + 0] < 128 && px[o + 2] < 128);
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

    // Assert the matrix push constant reaches the shader on `api`: identity draws the
    // triangle over the center (green), a large +x translation pushes it off (clear).
    void CheckBackend(const char* name, VriGraphicsAPI api, const void* vs, size_t vsN, const void* ps, size_t psN)
    {
        bool       ran     = false;
        const bool covered = RunPush(api, vs, vsN, ps, psN, Identity(), ran);
        if (!ran)
        {
            std::fprintf(stderr, "[pushtest] %s: unavailable (skipped)\n", name);
            return;
        }
        bool       ran2       = false;
        const bool coveredOff = RunPush(api, vs, vsN, ps, psN, TranslateX(2.0f), ran2);
        std::fprintf(stderr,
                     "[pushtest] %s: identity covered=%d (want 1), translated coveredOff=%d (want 0)\n",
                     name,
                     covered ? 1 : 0,
                     coveredOff ? 1 : 0);
        // identity mvp -> the triangle must cover the center (proves the matrix reached the shader)
        CHECK(covered);
        // +x translation -> the triangle is pushed fully off-screen (proves the value flows through)
        CHECK_FALSE(coveredOff);
    }
} // namespace

TEST_CASE("push-constant matrix parity: a float4x4 transform reaches the shader on each backend")
{
    CheckBackend("Vulkan",
                 VriGraphicsAPI_Vulkan,
                 g_trianglePushSpv,
                 sizeof(g_trianglePushSpv),
                 g_trianglePushSpv,
                 sizeof(g_trianglePushSpv));
    CheckBackend("OpenGL",
                 VriGraphicsAPI_OpenGL,
                 g_trianglePushSpv,
                 sizeof(g_trianglePushSpv),
                 g_trianglePushSpv,
                 sizeof(g_trianglePushSpv));
    CheckBackend("WebGPU",
                 VriGraphicsAPI_WebGPU,
                 g_trianglePushWgsl,
                 sizeof(g_trianglePushWgsl),
                 g_trianglePushWgsl,
                 sizeof(g_trianglePushWgsl));
#if defined(_WIN32)
    CheckBackend("D3D12",
                 VriGraphicsAPI_D3D12,
                 g_trianglePushDxbcVS,
                 sizeof(g_trianglePushDxbcVS),
                 g_trianglePushDxbcPS,
                 sizeof(g_trianglePushDxbcPS));
#endif
}
