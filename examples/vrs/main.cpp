// Variable rate shading: a deliberately expensive, high-frequency animated fragment shader
// fills an offscreen target; VriShadingRateInterface::CmdSetShadingRate picks the per-draw
// coarse rate (1x1 .. 4x4) before that draw. The coarse-shading blockiness is directly
// visible on the fine pattern, and a GPU timestamp readout (VRI_INTERFACE_QUERY, the
// example-profiler pattern) shows the pass time drop as the rate coarsens. Requests
// VriFeature_VariableShadingRate at device creation (bestEffort); where
// VriDeviceDesc::hasVariableShadingRate is false (Metal, GL, web) the scene renders at
// full rate and the UI says so. Shared scaffolding: examples/common/example_app.h.
#include "../common/example_app.h"

#include <vri/ext/vri_ext_query.h>
#include <vri/ext/vri_ext_vrs.h>

#include <cmath>

#include "shaders/examples/show_tex_dxbc.h"  // g_showTexDxbcVS / PS (fullscreen composite)
#include "shaders/examples/show_tex_spv.h"   // g_showTexSpv
#include "shaders/examples/show_tex_wgsl.h"  // g_showTexWgsl
#include "shaders/examples/vrs_scene_dxbc.h" // g_vrsSceneDxbcVS / PS
#include "shaders/examples/vrs_scene_spv.h"  // g_vrsSceneSpv
#include "shaders/examples/vrs_scene_wgsl.h" // g_vrsSceneWgsl

namespace
{
    constexpr uint32_t  kWidth = 640, kHeight = 480;
    constexpr VriFormat kColor = VriFormat_RGBA8_UNORM;

    struct Push
    {
        float p[4];
    }; // time, aspect, loop count, unused

    const char* kRateNames[] = {"1x1 (full rate)", "1x2", "2x1", "2x2", "2x4", "4x2", "4x4"};

    bool g_rtInit = false;
} // namespace

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.requestFeatures = VriFeature_VariableShadingRate; // bestEffort: hasVariableShadingRate reports the outcome
    app.Init("vrs", kWidth, kHeight, /*hasDepth*/ false);
    VriCoreInterface& c = app.c;

    static VriShadingRateInterface vrs {};
    const bool                     hasVrs = c.GetDeviceDesc(app.dev)->hasVariableShadingRate == VRI_TRUE &&
                        vriGetInterface(app.dev, VRI_INTERFACE_VRS, sizeof(vrs), &vrs) == VriResult_Success;

    // ---- GPU timing for the scene pass (timestamp queries, where available) ----
    static VriQueryInterface q {};
    static VriQueryPool*     qpool     = nullptr;
    static VriBuffer*        qreadback = nullptr;
    static double            periodNs  = 0.0;
    const bool hasTiming = vriGetInterface(app.dev, VRI_INTERFACE_QUERY, sizeof(q), &q) == VriResult_Success &&
                           c.GetDeviceDesc(app.dev)->hasTimestampQueries == VRI_TRUE;
    if (hasTiming)
    {
        periodNs = c.GetDeviceDesc(app.dev)->timestampPeriodNanoseconds;
        VriQueryPoolDesc qpd {};
        qpd.type       = VriQueryType_Timestamp;
        qpd.queryCount = 2;
        if (q.CreateQueryPool(app.dev, &qpd, &qpool) == VriResult_Success)
        {
            VriBufferDesc bd {};
            bd.size           = 2 * q.GetQuerySize(qpool);
            bd.usage          = VriBufferUsage_TransferDst;
            bd.memoryLocation = VriMemoryLocation_HostReadback;
            c.CreateBuffer(app.dev, &bd, &qreadback);
        }
    }

    // ---- offscreen target the expensive shader fills ----
    VriTextureDesc td {};
    td.type           = VriTextureType_2D;
    td.format         = kColor;
    td.width          = kWidth;
    td.height         = kHeight;
    td.depth          = 1;
    td.mipNum         = 1;
    td.layerNum       = 1;
    td.sampleNum      = 1;
    td.usage          = VriTextureUsage_ColorAttachment | VriTextureUsage_ShaderResource;
    td.memoryLocation = VriMemoryLocation_Device;
    VriTexture* rt    = nullptr;
    if (c.CreateTexture(app.dev, &td, &rt) != VriResult_Success)
        app.Fail("offscreen CreateTexture failed");
    VriTextureViewDesc tvd {};
    tvd.texture           = rt;
    tvd.viewType          = VriTextureViewType_2D;
    tvd.format            = VriFormat_Unknown;
    tvd.aspect            = VriImageAspect_Color;
    VriDescriptor* rtView = nullptr;
    c.CreateTextureView(app.dev, &tvd, &rtView);

    // ---- scene pipeline: push-constant time, expensive FS ----
    VriPushConstantDesc pcd {};
    pcd.baseRegister = 0;
    pcd.size         = sizeof(Push);
    pcd.shaderStages = VriShaderStage_Vertex | VriShaderStage_Fragment;
    VriPipelineLayoutDesc sld {};
    sld.pushConstants              = &pcd;
    sld.pushConstantNum            = 1;
    VriPipelineLayout* sceneLayout = nullptr;
    c.CreatePipelineLayout(app.dev, &sld, &sceneLayout);

    const vriex::ExampleApp::ShaderVariants sceneVS {
        VRI_SHADER_BLOB(g_vrsSceneSpv),
        VRI_SHADER_BLOB(g_vrsSceneWgsl),
        VRI_SHADER_D3D12(g_vrsSceneDxbcVS),
    };
    const vriex::ExampleApp::ShaderVariants scenePS {
        VRI_SHADER_BLOB(g_vrsSceneSpv),
        VRI_SHADER_BLOB(g_vrsSceneWgsl),
        VRI_SHADER_D3D12(g_vrsSceneDxbcPS),
    };
    VriShaderDesc ssh[2] = {
        app.Shader(VriShaderStage_Vertex, "vertexMain", sceneVS),
        app.Shader(VriShaderStage_Fragment, "fragmentMain", scenePS),
    };
    VriColorAttachmentDesc sca {};
    sca.format         = kColor;
    sca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc spd {};
    spd.pipelineLayout          = sceneLayout;
    spd.shaders                 = ssh;
    spd.shaderNum               = 2;
    spd.inputAssembly.topology  = VriPrimitiveTopology_TriangleList;
    spd.rasterization.cullMode  = VriCullMode_None;
    spd.rasterization.lineWidth = 1.0f;
    spd.multisample.sampleNum   = 1;
    spd.outputMerger.colors     = &sca;
    spd.outputMerger.colorNum   = 1;
    VriPipeline* scenePipeline  = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &spd, &scenePipeline) != VriResult_Success)
        app.Fail("scene pipeline failed");

    // ---- composite: sample the offscreen target across the swapchain (show_tex) ----
    VriDescriptorRangeDesc cr[2] {};
    cr[0].baseRegister   = 0;
    cr[0].descriptorNum  = 1;
    cr[0].descriptorType = VriDescriptorType_Texture;
    cr[0].shaderStages   = VriShaderStage_Fragment;
    cr[1].baseRegister   = 1;
    cr[1].descriptorNum  = 1;
    cr[1].descriptorType = VriDescriptorType_Sampler;
    cr[1].shaderStages   = VriShaderStage_Fragment;
    VriDescriptorSetDesc csd {};
    csd.registerSpace = 0;
    csd.ranges        = cr;
    csd.rangeNum      = 2;
    VriPipelineLayoutDesc cld {};
    cld.descriptorSets            = &csd;
    cld.descriptorSetNum          = 1;
    VriPipelineLayout* compLayout = nullptr;
    c.CreatePipelineLayout(app.dev, &cld, &compLayout);

    const vriex::ExampleApp::ShaderVariants compVS {
        VRI_SHADER_BLOB(g_showTexSpv),
        VRI_SHADER_BLOB(g_showTexWgsl),
        VRI_SHADER_D3D12(g_showTexDxbcVS),
    };
    const vriex::ExampleApp::ShaderVariants compPS {
        VRI_SHADER_BLOB(g_showTexSpv),
        VRI_SHADER_BLOB(g_showTexWgsl),
        VRI_SHADER_D3D12(g_showTexDxbcPS),
    };
    VriShaderDesc csh[2] = {
        app.Shader(VriShaderStage_Vertex, "vertexMain", compVS),
        app.Shader(VriShaderStage_Fragment, "fragmentMain", compPS),
    };
    VriColorAttachmentDesc cca {};
    cca.format         = app.swapFormat;
    cca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc cpd {};
    cpd.pipelineLayout          = compLayout;
    cpd.shaders                 = csh;
    cpd.shaderNum               = 2;
    cpd.inputAssembly.topology  = VriPrimitiveTopology_TriangleList;
    cpd.rasterization.cullMode  = VriCullMode_None;
    cpd.rasterization.lineWidth = 1.0f;
    cpd.multisample.sampleNum   = 1;
    cpd.outputMerger.colors     = &cca;
    cpd.outputMerger.colorNum   = 1;
    VriPipeline* compPipeline   = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &cpd, &compPipeline) != VriResult_Success)
        app.Fail("composite pipeline failed");

    VriSamplerDesc smp {};
    smp.magFilter          = VriFilter_Nearest;
    smp.minFilter          = VriFilter_Nearest;
    smp.mipmapMode         = VriMipmapMode_Nearest;
    smp.addressModeU       = VriAddressMode_ClampToEdge;
    smp.addressModeV       = VriAddressMode_ClampToEdge;
    smp.addressModeW       = VriAddressMode_ClampToEdge;
    smp.maxLod             = 1.0f;
    VriDescriptor* sampler = nullptr;
    c.CreateSampler(app.dev, &smp, &sampler);

    VriDescriptorPoolDesc pdsc {};
    pdsc.descriptorSetMaxNum = 1;
    pdsc.textureMaxNum       = 1;
    pdsc.samplerMaxNum       = 1;
    VriDescriptorPool* pool  = nullptr;
    c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* compSet = nullptr;
    c.AllocateDescriptorSets(pool, compLayout, 0, &compSet, 1);
    {
        const VriDescriptor*         t[1] = {rtView};
        const VriDescriptor*         s[1] = {sampler};
        VriDescriptorRangeUpdateDesc u[2] {};
        u[0].descriptors   = t;
        u[0].descriptorNum = 1;
        u[1].descriptors   = s;
        u[1].descriptorNum = 1;
        c.UpdateDescriptorRanges(compSet, 0, 2, u);
    }

    static float time      = 0.0f;
    static int   rate      = 0; // index into VriShadingRate / kRateNames
    static int   loops     = 48;
    static float gpuMs     = 0.0f;
    static bool  timedOnce = false;
    app.onUpdate           = [](uint64_t frame) {
        time += app.dt;
        // The frame loop waits the fence each frame, so last frame's queries are resolved.
        if (qreadback && frame > 1 && timedOnce)
        {
            const uint64_t* ticks = static_cast<const uint64_t*>(app.c.MapBuffer(qreadback, 0, 2 * sizeof(uint64_t)));
            gpuMs                 = float(double(ticks[1] - ticks[0]) * periodNs / 1.0e6);
            app.c.UnmapBuffer(qreadback);
        }
    };
    app.onGui = [=] {
        if (hasVrs)
            ImGui::Combo("shading rate", &rate, kRateNames, 7);
        else
            ImGui::Text("hasVariableShadingRate = false on %s\n(full rate only)", app.apiName);
        ImGui::SliderInt("shader cost (loops)", &loops, 8, 96);
        if (qreadback)
            ImGui::Text("scene pass: %.3f ms", gpuMs);
    };

    app.onPreRender = [=](VriCommandBuffer* cmd) {
        Push pc {};
        pc.p[0] = time;
        pc.p[1] = float(kWidth) / float(kHeight);
        pc.p[2] = float(loops);

        if (qreadback)
            q.CmdResetQueries(cmd, qpool, 0, 2);

        VriTextureBarrierDesc tb {};
        tb.texture       = rt;
        tb.aspect        = VriImageAspect_Color;
        tb.before.layout = g_rtInit ? VriLayout_ShaderResource : VriLayout_Undefined;
        tb.before.stages = g_rtInit ? VriPipelineStage_FragmentShader : VriPipelineStage_None;
        tb.after.access  = VriAccess_ColorAttachmentWrite;
        tb.after.layout  = VriLayout_ColorAttachment;
        tb.after.stages  = VriPipelineStage_ColorAttachmentOutput;
        VriBarrierGroupDesc gw {};
        gw.textures   = &tb;
        gw.textureNum = 1;
        c.CmdBarrier(cmd, &gw);
        g_rtInit = true;

        if (qreadback)
            q.CmdWriteTimestamp(cmd, qpool, 0);

        VriAttachmentDesc col {};
        col.view    = rtView;
        col.loadOp  = VriAttachmentLoadOp_Clear;
        col.storeOp = VriAttachmentStoreOp_Store;
        VriAttachmentsDesc att {};
        att.colors            = &col;
        att.colorNum          = 1;
        att.renderArea.width  = kWidth;
        att.renderArea.height = kHeight;
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp {0, 0, float(kWidth), float(kHeight), 0, 1};
        c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc {0, 0, kWidth, kHeight};
        c.CmdSetScissors(cmd, &sc, 1);
        if (hasVrs)
        {
            VriShadingRateDesc srd {};
            srd.shadingRate        = static_cast<VriShadingRate>(rate);
            srd.primitiveCombiner  = VriShadingRateCombiner_Keep;
            srd.attachmentCombiner = VriShadingRateCombiner_Keep;
            vrs.CmdSetShadingRate(cmd, &srd);
        }
        c.CmdSetPipeline(cmd, scenePipeline);
        c.CmdSetPipelineLayout(cmd, sceneLayout);
        c.CmdSetConstants(cmd, 0, &pc, sizeof(Push));
        VriDrawDesc d {};
        d.vertexNum   = 3;
        d.instanceNum = 1;
        c.CmdDraw(cmd, &d);
        c.CmdEndRendering(cmd);

        if (hasVrs)
        {
            // Back to full rate so the composite + ImGui stay crisp.
            VriShadingRateDesc full {};
            full.shadingRate        = VriShadingRate_1x1;
            full.primitiveCombiner  = VriShadingRateCombiner_Keep;
            full.attachmentCombiner = VriShadingRateCombiner_Keep;
            vrs.CmdSetShadingRate(cmd, &full);
        }
        if (qreadback)
        {
            q.CmdWriteTimestamp(cmd, qpool, 1);
            q.CmdCopyQueries(cmd, qpool, 0, 2, qreadback, 0);
            timedOnce = true;
        }

        VriTextureBarrierDesc ts {};
        ts.texture       = rt;
        ts.aspect        = VriImageAspect_Color;
        ts.before.access = VriAccess_ColorAttachmentWrite;
        ts.before.layout = VriLayout_ColorAttachment;
        ts.before.stages = VriPipelineStage_ColorAttachmentOutput;
        ts.after.access  = VriAccess_ShaderResourceRead;
        ts.after.layout  = VriLayout_ShaderResource;
        ts.after.stages  = VriPipelineStage_FragmentShader;
        VriBarrierGroupDesc gs {};
        gs.textures   = &ts;
        gs.textureNum = 1;
        c.CmdBarrier(cmd, &gs);
    };

    app.onRecord = [=](VriCommandBuffer* cmd) {
        c.CmdSetPipeline(cmd, compPipeline);
        c.CmdSetPipelineLayout(cmd, compLayout);
        c.CmdSetDescriptorSet(cmd, 0, compSet);
        VriDrawDesc d {};
        d.vertexNum   = 3;
        d.instanceNum = 1;
        c.CmdDraw(cmd, &d);
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyDescriptorPool(pool);
    c.DestroyDescriptor(sampler);
    c.DestroyPipeline(compPipeline);
    c.DestroyPipelineLayout(compLayout);
    c.DestroyPipeline(scenePipeline);
    c.DestroyPipelineLayout(sceneLayout);
    c.DestroyDescriptor(rtView);
    c.DestroyTexture(rt);
    if (qreadback)
        c.DestroyBuffer(qreadback);
    if (qpool)
        q.DestroyQueryPool(qpool);
    app.Shutdown();
#endif
    return 0;
}
