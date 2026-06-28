// Occlusion queries (ext/vri_ext_query.h): bracket a triangle draw with CmdBeginQuery/CmdEndQuery
// inside a render pass, resolve the result, and check that samples passed (the triangle covers
// pixels, no depth test -> samplesPassed > 0). Runs on the explicit backends whose occlusion model
// matches VRI's pool-at-query-time API (Vulkan, D3D12, desktop OpenGL); WebGPU's occlusion binds
// the query set at pass-begin, so it reports Unsupported and this self-skips there. Validation-clean.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>

#include "common/shaders/triangle_dxbc.h" // g_triangleDxbcVS + g_triangleDxbcPS (D3D12)
#include "common/shaders/triangle_spv.h"  // g_triangleSpv (Vulkan + OpenGL via SPIRV-Cross)

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    void RunOcclusion(VriGraphicsAPI api, const char* name)
    {
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        VriDevice* dev      = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << name << "] unavailable - skipping occlusion test");
            return;
        }
        struct Guard
        {
            VriDevice* d;
            ~Guard() { vriDestroyDevice(d); }
        } guard {dev};

        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriQueryInterface q {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_QUERY, sizeof(q), &q) == VriResult_Success);

        // Occlusion pool; skip the backend if it does not support occlusion (e.g. WebGPU).
        VriQueryPoolDesc qpd {};
        qpd.type           = VriQueryType_Occlusion;
        qpd.queryCount     = 1;
        VriQueryPool* pool = nullptr;
        if (q.CreateQueryPool(dev, &qpd, &pool) != VriResult_Success)
        {
            MESSAGE("[" << name << "] occlusion queries unsupported - skipping");
            return;
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
        td.usage                   = VriTextureUsage_ColorAttachment;
        td.memoryLocation          = VriMemoryLocation_Device;
        td.clearValue.color.f32[3] = 1.0f;
        VriTexture* color          = nullptr;
        REQUIRE(c.CreateTexture(dev, &td, &color) == VriResult_Success);
        VriTextureViewDesc vdsc {};
        vdsc.texture        = color;
        vdsc.viewType       = VriTextureViewType_2D;
        vdsc.format         = VriFormat_Unknown;
        vdsc.aspect         = VriImageAspect_Color;
        VriDescriptor* view = nullptr;
        REQUIRE(c.CreateTextureView(dev, &vdsc, &view) == VriResult_Success);

        VriBufferDesc bd {};
        bd.size             = sizeof(uint64_t);
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
        if (api == VriGraphicsAPI_D3D12)
        {
            sh[0].bytecode     = g_triangleDxbcVS;
            sh[0].bytecodeSize = sizeof(g_triangleDxbcVS);
            sh[1].bytecode     = g_triangleDxbcPS;
            sh[1].bytecodeSize = sizeof(g_triangleDxbcPS);
        }
        else
        {
            sh[0].bytecode = sh[1].bytecode = g_triangleSpv;
            sh[0].bytecodeSize = sh[1].bytecodeSize = sizeof(g_triangleSpv);
        }
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
            VriTextureBarrierDesc b {};
            b.texture       = color;
            b.before.layout = VriLayout_Undefined;
            b.before.stages = VriPipelineStage_None;
            b.after.access  = VriAccess_ColorAttachmentWrite;
            b.after.layout  = VriLayout_ColorAttachment;
            b.after.stages  = VriPipelineStage_ColorAttachmentOutput;
            b.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g {};
            g.textures   = &b;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        }
        q.CmdResetQueries(cmd, pool, 0, 1); // outside the render pass (Vulkan requirement)

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
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp {0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1};
        c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc {0, 0, kW, kH};
        c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        q.CmdBeginQuery(cmd, pool, 0);
        VriDrawDesc draw {};
        draw.vertexNum   = 3;
        draw.instanceNum = 1;
        c.CmdDraw(cmd, &draw);
        q.CmdEndQuery(cmd, pool, 0);
        c.CmdEndRendering(cmd);
        q.CmdCopyQueries(cmd, pool, 0, 1, readback, 0);
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

        uint64_t        samples = 0;
        const uint64_t* mapped  = static_cast<const uint64_t*>(c.MapBuffer(readback, 0, sizeof(uint64_t)));
        REQUIRE(mapped != nullptr);
        std::memcpy(&samples, mapped, sizeof(samples));
        c.UnmapBuffer(readback);
        CHECK(samples > 0); // the triangle covers pixels and there's no depth test
        MESSAGE("[" << name << "] occlusion samples passed = " << samples);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(readback);
        c.DestroyDescriptor(view);
        c.DestroyTexture(color);
        q.DestroyQueryPool(pool);
    }
} // namespace

TEST_CASE("Occlusion query: Vulkan counts samples passed") { RunOcclusion(VriGraphicsAPI_Vulkan, "Vulkan"); }
TEST_CASE("Occlusion query: D3D12 counts samples passed") { RunOcclusion(VriGraphicsAPI_D3D12, "D3D12"); }
TEST_CASE("Occlusion query: OpenGL counts samples passed") { RunOcclusion(VriGraphicsAPI_OpenGL, "OpenGL"); }
