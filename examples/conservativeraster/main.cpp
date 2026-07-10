// Conservative rasterization: render a spinning pinwheel of thin sliver triangles twice into a
// tiny (40x30) render target - once with normal rasterization, once with
// VriRasterizationDesc::conservativeRaster - and composite the two low-res results magnified
// side by side (nearest filtering, so every texel is a fat block). Normal rasterization covers a
// texel only when a sliver crosses its center, so the blades break into sparse dashes; the
// conservative half lights every texel the slivers touch, so the blades stay solid. Where
// VriDeviceDesc::hasConservativeRaster is false (Metal, GL, web) both halves render normally and
// the UI says so. ImGui drives the spin. Shared scaffolding: examples/common/example_app.h.
#include "../common/example_app.h"

#include <cmath>

#include "shaders/examples/composite_dxbc.h"      // g_compositeDxbcVS / PS (two textures side by side)
#include "shaders/examples/composite_spv.h"       // g_compositeSpv
#include "shaders/examples/composite_wgsl.h"      // g_compositeWgsl
#include "shaders/examples/consrast_scene_dxbc.h" // g_consrastSceneDxbcVS / PS
#include "shaders/examples/consrast_scene_spv.h"  // g_consrastSceneSpv (Vulkan + OpenGL)
#include "shaders/examples/consrast_scene_wgsl.h" // g_consrastSceneWgsl (WebGPU; desktop-only target, still generated)

namespace
{
    constexpr uint32_t  kWidth = 640, kHeight = 480;
    constexpr uint32_t  kRtWidth = 40, kRtHeight = 30; // tiny on purpose: texels become fat blocks
    constexpr VriFormat kColor = VriFormat_RGBA8_UNORM;

    struct Push
    {
        float rot[4];
    }; // cos, sin, aspect (height/width), unused

    bool g_rtInit[2] = {false, false};
} // namespace

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.Init("conservativeraster", kWidth, kHeight, /*hasDepth*/ false);
    VriCoreInterface& c = app.c;

    const bool hasConsRast = c.GetDeviceDesc(app.dev)->hasConservativeRaster == VRI_TRUE;

    // ---- two tiny offscreen targets: [0] normal raster, [1] conservative raster ----
    VriTexture*    rt[2]     = {};
    VriDescriptor* rtView[2] = {};
    for (int i = 0; i < 2; ++i)
    {
        VriTextureDesc td {};
        td.type                    = VriTextureType_2D;
        td.format                  = kColor;
        td.width                   = kRtWidth;
        td.height                  = kRtHeight;
        td.depth                   = 1;
        td.mipNum                  = 1;
        td.layerNum                = 1;
        td.sampleNum               = 1;
        td.usage                   = VriTextureUsage_ColorAttachment | VriTextureUsage_ShaderResource;
        td.memoryLocation          = VriMemoryLocation_Device;
        td.clearValue.color.f32[0] = 0.06f;
        td.clearValue.color.f32[1] = 0.07f;
        td.clearValue.color.f32[2] = 0.09f;
        td.clearValue.color.f32[3] = 1.0f;
        if (c.CreateTexture(app.dev, &td, &rt[i]) != VriResult_Success)
            app.Fail("offscreen CreateTexture failed");
        VriTextureViewDesc vd {};
        vd.texture  = rt[i];
        vd.viewType = VriTextureViewType_2D;
        vd.format   = VriFormat_Unknown;
        vd.aspect   = VriImageAspect_Color;
        if (c.CreateTextureView(app.dev, &vd, &rtView[i]) != VriResult_Success)
            app.Fail("offscreen view failed");
    }

    // ---- scene: push-constant spin, one pipeline per rasterization mode ----
    VriPushConstantDesc pcd {};
    pcd.baseRegister = 0;
    pcd.size         = sizeof(Push);
    pcd.shaderStages = VriShaderStage_Vertex;
    VriPipelineLayoutDesc sld {};
    sld.pushConstants              = &pcd;
    sld.pushConstantNum            = 1;
    VriPipelineLayout* sceneLayout = nullptr;
    c.CreatePipelineLayout(app.dev, &sld, &sceneLayout);

    const vriex::ExampleApp::ShaderVariants sceneShader {
        VRI_SHADER_BLOB(g_consrastSceneSpv),
        VRI_SHADER_BLOB(g_consrastSceneWgsl),
        VRI_SHADER_D3D12(g_consrastSceneDxbcVS),
    };
    const vriex::ExampleApp::ShaderVariants sceneShaderPS {
        VRI_SHADER_BLOB(g_consrastSceneSpv),
        VRI_SHADER_BLOB(g_consrastSceneWgsl),
        VRI_SHADER_D3D12(g_consrastSceneDxbcPS),
    };

    VriColorAttachmentDesc sca {};
    sca.format             = kColor;
    sca.colorWriteMask     = VriColorWrite_RGBA;
    auto makeScenePipeline = [&](bool conservative) {
        VriShaderDesc sh[2] = {
            app.Shader(VriShaderStage_Vertex, "vertexMain", sceneShader),
            app.Shader(VriShaderStage_Fragment, "fragmentMain", sceneShaderPS),
        };
        VriGraphicsPipelineDesc pd {};
        pd.pipelineLayout                   = sceneLayout;
        pd.shaders                          = sh;
        pd.shaderNum                        = 2;
        pd.inputAssembly.topology           = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode           = VriCullMode_None;
        pd.rasterization.frontFace          = VriFrontFace_CounterClockwise;
        pd.rasterization.lineWidth          = 1.0f;
        pd.rasterization.conservativeRaster = conservative ? VRI_TRUE : VRI_FALSE;
        pd.multisample.sampleNum            = 1;
        pd.outputMerger.colors              = &sca;
        pd.outputMerger.colorNum            = 1;
        VriPipeline* p                      = nullptr;
        if (c.CreateGraphicsPipeline(app.dev, &pd, &p) != VriResult_Success)
            app.Fail("scene pipeline failed");
        return p;
    };
    VriPipeline* sceneNormal = makeScenePipeline(false);
    // Without the capability, the "conservative" half honestly renders normal (and the UI says so)
    // rather than requesting a mode the device would reject.
    VriPipeline* sceneConservative = hasConsRast ? makeScenePipeline(true) : sceneNormal;

    // ---- composite: both tiny targets magnified side by side (nearest = fat texels) ----
    VriDescriptorRangeDesc cr[3] {};
    cr[0].baseRegister   = 0;
    cr[0].descriptorNum  = 1;
    cr[0].descriptorType = VriDescriptorType_Texture;
    cr[0].shaderStages   = VriShaderStage_Fragment;
    cr[1].baseRegister   = 1;
    cr[1].descriptorNum  = 1;
    cr[1].descriptorType = VriDescriptorType_Texture;
    cr[1].shaderStages   = VriShaderStage_Fragment;
    cr[2].baseRegister   = 2;
    cr[2].descriptorNum  = 1;
    cr[2].descriptorType = VriDescriptorType_Sampler;
    cr[2].shaderStages   = VriShaderStage_Fragment;
    VriDescriptorSetDesc csd {};
    csd.registerSpace = 0;
    csd.ranges        = cr;
    csd.rangeNum      = 3;
    VriPipelineLayoutDesc cld {};
    cld.descriptorSets            = &csd;
    cld.descriptorSetNum          = 1;
    VriPipelineLayout* compLayout = nullptr;
    c.CreatePipelineLayout(app.dev, &cld, &compLayout);

    const vriex::ExampleApp::ShaderVariants compVS {
        VRI_SHADER_BLOB(g_compositeSpv),
        VRI_SHADER_BLOB(g_compositeWgsl),
        VRI_SHADER_D3D12(g_compositeDxbcVS),
    };
    const vriex::ExampleApp::ShaderVariants compPS {
        VRI_SHADER_BLOB(g_compositeSpv),
        VRI_SHADER_BLOB(g_compositeWgsl),
        VRI_SHADER_D3D12(g_compositeDxbcPS),
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
    smp.magFilter          = VriFilter_Nearest; // fat blocky texels: the whole point of the demo
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
    pdsc.textureMaxNum       = 2;
    pdsc.samplerMaxNum       = 1;
    VriDescriptorPool* pool  = nullptr;
    c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* compSet = nullptr;
    c.AllocateDescriptorSets(pool, compLayout, 0, &compSet, 1);
    {
        const VriDescriptor*         t0[1] = {rtView[0]};
        const VriDescriptor*         t1[1] = {rtView[1]};
        const VriDescriptor*         s[1]  = {sampler};
        VriDescriptorRangeUpdateDesc u[3] {};
        u[0].descriptors   = t0;
        u[0].descriptorNum = 1;
        u[1].descriptors   = t1;
        u[1].descriptorNum = 1;
        u[2].descriptors   = s;
        u[2].descriptorNum = 1;
        c.UpdateDescriptorRanges(compSet, 0, 3, u);
    }

    static float spin = 1.0f, angle = 0.35f;
    app.onUpdate = [](uint64_t) { angle += 0.25f * spin * app.dt; }; // slow: watch texels pop in/out
    app.onGui    = [=] {
        ImGui::SliderFloat("spin", &spin, 0.0f, 5.0f);
        ImGui::TextUnformatted("left: normal raster | right: conservative");
        if (hasConsRast)
            ImGui::TextUnformatted("conservative raster: native");
        else
            ImGui::Text("hasConservativeRaster = false on %s\n(both halves render normally)", app.apiName);
    };

    app.onPreRender = [=](VriCommandBuffer* cmd) {
        Push pc {};
        pc.rot[0] = std::cos(angle);
        pc.rot[1] = std::sin(angle);
        pc.rot[2] = float(kRtHeight) / float(kRtWidth);

        // Render the same scene into each tiny target with its rasterization mode.
        for (int i = 0; i < 2; ++i)
        {
            VriTextureBarrierDesc tb {};
            tb.texture       = rt[i];
            tb.aspect        = VriImageAspect_Color;
            tb.before.layout = g_rtInit[i] ? VriLayout_ShaderResource : VriLayout_Undefined;
            tb.before.stages = g_rtInit[i] ? VriPipelineStage_FragmentShader : VriPipelineStage_None;
            tb.after.access  = VriAccess_ColorAttachmentWrite;
            tb.after.layout  = VriLayout_ColorAttachment;
            tb.after.stages  = VriPipelineStage_ColorAttachmentOutput;
            VriBarrierGroupDesc gw {};
            gw.textures   = &tb;
            gw.textureNum = 1;
            c.CmdBarrier(cmd, &gw);
            g_rtInit[i] = true;

            VriAttachmentDesc col {};
            col.view                    = rtView[i];
            col.loadOp                  = VriAttachmentLoadOp_Clear;
            col.storeOp                 = VriAttachmentStoreOp_Store;
            col.clearValue.color.f32[0] = 0.06f;
            col.clearValue.color.f32[1] = 0.07f;
            col.clearValue.color.f32[2] = 0.09f;
            col.clearValue.color.f32[3] = 1.0f;
            VriAttachmentsDesc att {};
            att.colors            = &col;
            att.colorNum          = 1;
            att.renderArea.width  = kRtWidth;
            att.renderArea.height = kRtHeight;
            att.layerNum          = 1;
            c.CmdBeginRendering(cmd, &att);
            VriViewport vp {0, 0, float(kRtWidth), float(kRtHeight), 0, 1};
            c.CmdSetViewports(cmd, &vp, 1);
            VriRect sc {0, 0, kRtWidth, kRtHeight};
            c.CmdSetScissors(cmd, &sc, 1);
            c.CmdSetPipeline(cmd, i == 0 ? sceneNormal : sceneConservative);
            c.CmdSetPipelineLayout(cmd, sceneLayout);
            c.CmdSetConstants(cmd, 0, &pc, sizeof(Push));
            VriDrawDesc d {};
            d.vertexNum   = 9; // three sliver blades
            d.instanceNum = 1;
            c.CmdDraw(cmd, &d);
            c.CmdEndRendering(cmd);
        }

        // both tiny targets -> sampleable by the composite pass
        VriTextureBarrierDesc ts[2] {};
        for (int i = 0; i < 2; ++i)
        {
            ts[i].texture       = rt[i];
            ts[i].aspect        = VriImageAspect_Color;
            ts[i].before.access = VriAccess_ColorAttachmentWrite;
            ts[i].before.layout = VriLayout_ColorAttachment;
            ts[i].before.stages = VriPipelineStage_ColorAttachmentOutput;
            ts[i].after.access  = VriAccess_ShaderResourceRead;
            ts[i].after.layout  = VriLayout_ShaderResource;
            ts[i].after.stages  = VriPipelineStage_FragmentShader;
        }
        VriBarrierGroupDesc gs {};
        gs.textures   = ts;
        gs.textureNum = 2;
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
    if (sceneConservative != sceneNormal)
        c.DestroyPipeline(sceneConservative);
    c.DestroyPipeline(sceneNormal);
    c.DestroyPipelineLayout(sceneLayout);
    for (int i = 0; i < 2; ++i)
    {
        c.DestroyDescriptor(rtView[i]);
        c.DestroyTexture(rt[i]);
    }
    app.Shutdown();
#endif
    return 0;
}
