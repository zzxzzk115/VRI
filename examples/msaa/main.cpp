// MSAA: render a spinning pinwheel of thin, gap-separated triangle blades (lots of sharp edges)
// into a 4x multisampled color target, then RESOLVE it to a single-sample texture which the
// swapchain pass samples full-screen. Toggle "4x MSAA" off to render the same scene at 1x into
// the resolve texture directly - the blade edges visibly turn jagged. Exercises multisampled
// attachments + the resolve path (VriAttachmentDesc.resolveView), which run on every backend
// (WebGPU permits sample counts 1 or 4, so 4x is the portable MSAA level). ImGui drives the
// toggle + spin. Shared scaffolding: examples/common/example_app.h (VRI_API / ?backend force one).
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "tests/shaders/msaa_spv.h"        // g_msaaSpv  (Vulkan + OpenGL)
#include "tests/shaders/msaa_wgsl.h"       // g_msaaWgsl (WebGPU)
#include "tests/shaders/msaa_dxbc.h"       // g_msaaDxbcVS / PS (D3D12)
#include "tests/shaders/show_tex_spv.h"    // g_showTexSpv  (fullscreen composite)
#include "tests/shaders/show_tex_wgsl.h"   // g_showTexWgsl
#include "tests/shaders/show_tex_dxbc.h"   // g_showTexDxbcVS / PS

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480; // width*4 is 256-aligned (D3D12 readback pitch)
    constexpr uint32_t kSamples = 4;                // WebGPU allows only 1 or 4; 4 is portable
    constexpr VriFormat kColor = VriFormat_RGBA8_UNORM;

    struct Vertex { float x, y; float r, g, b; };

    // A pinwheel: `blades` thin triangles (each spanning half a slot, so a gap follows), colored
    // around the hue wheel. Centered fan; the rotation/aspect are applied in the shader.
    std::vector<Vertex> BuildPinwheel(int blades)
    {
        std::vector<Vertex> v;
        const float twoPi = 6.2831853f, R = 0.92f, step = twoPi / float(blades);
        for (int i = 0; i < blades; ++i)
        {
            const float h = float(i) * 0.7f;
            const float cr = 0.5f + 0.5f * std::sin(h), cg = 0.5f + 0.5f * std::sin(h + 2.094f), cb = 0.5f + 0.5f * std::sin(h + 4.188f);
            const float a0 = float(i) * step, a1 = a0 + step * 0.5f;
            v.push_back({0.0f, 0.0f, cr, cg, cb});
            v.push_back({R * std::cos(a0), R * std::sin(a0), cr, cg, cb});
            v.push_back({R * std::cos(a1), R * std::sin(a1), cr, cg, cb});
        }
        return v;
    }

    struct Push { float rot[4]; }; // cos, sin, aspect(H/W), unused
    bool g_msaaInit = false, g_resolvedInit = false;
} // namespace

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.Init("msaa", kWidth, kHeight, /*hasDepth*/ false); // swapchain pass is a fullscreen composite
    VriCoreInterface& c = app.c;
    const bool useWgsl = app.useWgsl, useDxbc = app.useDxbc;

    const std::vector<Vertex> verts = BuildPinwheel(24);
    const uint32_t vertexNum = static_cast<uint32_t>(verts.size());
    const uint64_t vbBytes = verts.size() * sizeof(Vertex);
    VriBufferDesc vbd{}; vbd.size = vbBytes; vbd.usage = VriBufferUsage_VertexBuffer | VriBufferUsage_TransferDst; vbd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* vbuf = nullptr; c.CreateBuffer(app.dev, &vbd, &vbuf);

    // a 4x multisampled color target + a single-sample resolve target that the composite samples.
    auto makeTex = [&](uint32_t samples, VriTextureUsageFlags usage, VriDescriptor** outView) {
        VriTextureDesc td{}; td.type = VriTextureType_2D; td.format = kColor; td.width = kWidth; td.height = kHeight; td.depth = 1;
        td.mipNum = 1; td.layerNum = 1; td.sampleNum = samples; td.usage = usage; td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* t = nullptr; if (c.CreateTexture(app.dev, &td, &t) != VriResult_Success) app.Fail("CreateTexture (MSAA) failed");
        VriTextureViewDesc vd{}; vd.texture = t; vd.viewType = VriTextureViewType_2D; vd.format = VriFormat_Unknown; vd.aspect = VriImageAspect_Color;
        if (c.CreateTextureView(app.dev, &vd, outView) != VriResult_Success) app.Fail("view failed");
        return t;
    };
    VriDescriptor* msaaView = nullptr; VriTexture* msaaColor = makeTex(kSamples, VriTextureUsage_ColorAttachment, &msaaView);
    VriDescriptor* resolvedView = nullptr; VriTexture* resolved = makeTex(1, VriTextureUsage_ColorAttachment | VriTextureUsage_ShaderResource, &resolvedView);

    // ---- scene layout/pipelines: push-constant rotation only; 1x + 4x variants ----
    VriPushConstantDesc pcd{}; pcd.baseRegister = 0; pcd.size = sizeof(Push); pcd.shaderStages = VriShaderStage_Vertex;
    VriPipelineLayoutDesc sld{}; sld.pushConstants = &pcd; sld.pushConstantNum = 1;
    VriPipelineLayout* sceneLayout = nullptr; c.CreatePipelineLayout(app.dev, &sld, &sceneLayout);

    VriShaderDesc ssh[2]{};
    ssh[0].stage = VriShaderStage_Vertex;   ssh[0].entryPointName = "vertexMain";
    ssh[1].stage = VriShaderStage_Fragment; ssh[1].entryPointName = "fragmentMain";
    if (useDxbc) { ssh[0].bytecode = g_msaaDxbcVS; ssh[0].bytecodeSize = sizeof(g_msaaDxbcVS); ssh[1].bytecode = g_msaaDxbcPS; ssh[1].bytecodeSize = sizeof(g_msaaDxbcPS); }
    else { ssh[0].bytecode = ssh[1].bytecode = useWgsl ? static_cast<const void*>(g_msaaWgsl) : static_cast<const void*>(g_msaaSpv);
           ssh[0].bytecodeSize = ssh[1].bytecodeSize = useWgsl ? sizeof(g_msaaWgsl) : sizeof(g_msaaSpv); }

    VriVertexAttributeDesc attrs[2]{};
    attrs[0].format = VriFormat_RG32_SFLOAT;  attrs[0].offset = 0; attrs[0].streamIndex = 0; // pos
    attrs[1].format = VriFormat_RGB32_SFLOAT; attrs[1].offset = 8; attrs[1].streamIndex = 0; // color
    VriVertexStreamDesc stream{}; stream.stride = sizeof(Vertex); stream.bindingSlot = 0; stream.stepRate = VriVertexStepRate_PerVertex;

    VriColorAttachmentDesc sca{}; sca.format = kColor; sca.colorWriteMask = VriColorWrite_RGBA;
    auto makeScenePipeline = [&](uint32_t samples) {
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = sceneLayout; pd.shaders = ssh; pd.shaderNum = 2;
        pd.vertexInput.attributes = attrs; pd.vertexInput.attributeNum = 2; pd.vertexInput.streams = &stream; pd.vertexInput.streamNum = 1;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.frontFace = VriFrontFace_CounterClockwise; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = samples;
        pd.outputMerger.colors = &sca; pd.outputMerger.colorNum = 1;
        VriPipeline* p = nullptr; if (c.CreateGraphicsPipeline(app.dev, &pd, &p) != VriResult_Success) app.Fail("scene pipeline failed");
        return p;
    };
    VriPipeline* scene4x = makeScenePipeline(kSamples);
    VriPipeline* scene1x = makeScenePipeline(1);

    // ---- composite: sample the resolve texture across the swapchain (show_tex) ----
    VriDescriptorRangeDesc cr[2]{};
    cr[0].baseRegister = 0; cr[0].descriptorNum = 1; cr[0].descriptorType = VriDescriptorType_Texture; cr[0].shaderStages = VriShaderStage_Fragment;
    cr[1].baseRegister = 1; cr[1].descriptorNum = 1; cr[1].descriptorType = VriDescriptorType_Sampler; cr[1].shaderStages = VriShaderStage_Fragment;
    VriDescriptorSetDesc csd{}; csd.registerSpace = 0; csd.ranges = cr; csd.rangeNum = 2;
    VriPipelineLayoutDesc cld{}; cld.descriptorSets = &csd; cld.descriptorSetNum = 1;
    VriPipelineLayout* compLayout = nullptr; c.CreatePipelineLayout(app.dev, &cld, &compLayout);

    VriShaderDesc csh[2]{};
    csh[0].stage = VriShaderStage_Vertex;   csh[0].entryPointName = "vertexMain";
    csh[1].stage = VriShaderStage_Fragment; csh[1].entryPointName = "fragmentMain";
    if (useDxbc) { csh[0].bytecode = g_showTexDxbcVS; csh[0].bytecodeSize = sizeof(g_showTexDxbcVS); csh[1].bytecode = g_showTexDxbcPS; csh[1].bytecodeSize = sizeof(g_showTexDxbcPS); }
    else { csh[0].bytecode = csh[1].bytecode = useWgsl ? static_cast<const void*>(g_showTexWgsl) : static_cast<const void*>(g_showTexSpv);
           csh[0].bytecodeSize = csh[1].bytecodeSize = useWgsl ? sizeof(g_showTexWgsl) : sizeof(g_showTexSpv); }

    VriColorAttachmentDesc cca{}; cca.format = app.swapFormat; cca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc cpd{};
    cpd.pipelineLayout = compLayout; cpd.shaders = csh; cpd.shaderNum = 2;
    cpd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
    cpd.rasterization.cullMode = VriCullMode_None; cpd.rasterization.lineWidth = 1.0f;
    cpd.multisample.sampleNum = 1; cpd.outputMerger.colors = &cca; cpd.outputMerger.colorNum = 1;
    VriPipeline* compPipeline = nullptr; if (c.CreateGraphicsPipeline(app.dev, &cpd, &compPipeline) != VriResult_Success) app.Fail("composite pipeline failed");

    VriSamplerDesc smp{}; smp.magFilter = VriFilter_Linear; smp.minFilter = VriFilter_Linear; smp.mipmapMode = VriMipmapMode_Nearest;
    smp.addressModeU = VriAddressMode_ClampToEdge; smp.addressModeV = VriAddressMode_ClampToEdge; smp.addressModeW = VriAddressMode_ClampToEdge; smp.maxLod = 1.0f;
    VriDescriptor* sampler = nullptr; c.CreateSampler(app.dev, &smp, &sampler);

    VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 1; pdsc.textureMaxNum = 1; pdsc.samplerMaxNum = 1;
    VriDescriptorPool* pool = nullptr; c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* compSet = nullptr; c.AllocateDescriptorSets(pool, compLayout, 0, &compSet, 1);
    { const VriDescriptor* t[1] = {resolvedView}; const VriDescriptor* s[1] = {sampler};
      VriDescriptorRangeUpdateDesc u[2]{}; u[0].descriptors=t; u[0].descriptorNum=1; u[1].descriptors=s; u[1].descriptorNum=1; c.UpdateDescriptorRanges(compSet, 0, 2, u); }

    // one-time vertex upload
    app.BeginUpload();
    app.UploadBuffer(vbuf, verts.data(), vbBytes, VriAccess_VertexBufferRead, VriPipelineStage_VertexInput);
    app.EndUpload();

    static float spin = 1.0f, angle = 0.0f; static bool msaaOn = true; // ImGui-controlled
    app.onUpdate = [](uint64_t) { angle += 0.01f * spin; };
    app.onGui = [] {
        ImGui::Checkbox("4x MSAA", &msaaOn);
        ImGui::SliderFloat("spin", &spin, 0.0f, 5.0f);
        ImGui::TextUnformatted(msaaOn ? "edges: resolved (smooth)" : "edges: 1x (jagged)");
    };

    app.onPreRender = [=](VriCommandBuffer* cmd) {
        Push pc{}; pc.rot[0] = std::cos(angle); pc.rot[1] = std::sin(angle); pc.rot[2] = float(kHeight) / float(kWidth);

        // The scene target: 4x MSAA color resolving into `resolved`, or `resolved` at 1x directly.
        std::vector<VriTextureBarrierDesc> pre;
        VriTextureBarrierDesc rb{}; rb.texture = resolved; rb.aspect = VriImageAspect_Color;
        rb.before.layout = g_resolvedInit ? VriLayout_ShaderResource : VriLayout_Undefined; rb.before.stages = g_resolvedInit ? VriPipelineStage_FragmentShader : VriPipelineStage_None;
        rb.after.access = VriAccess_ColorAttachmentWrite; rb.after.layout = VriLayout_ColorAttachment; rb.after.stages = VriPipelineStage_ColorAttachmentOutput;
        pre.push_back(rb);
        if (msaaOn)
        {
            VriTextureBarrierDesc mb{}; mb.texture = msaaColor; mb.aspect = VriImageAspect_Color;
            mb.before.layout = g_msaaInit ? VriLayout_ColorAttachment : VriLayout_Undefined; mb.before.stages = g_msaaInit ? VriPipelineStage_ColorAttachmentOutput : VriPipelineStage_None;
            mb.after.access = VriAccess_ColorAttachmentWrite; mb.after.layout = VriLayout_ColorAttachment; mb.after.stages = VriPipelineStage_ColorAttachmentOutput;
            pre.push_back(mb);
        }
        VriBarrierGroupDesc gw{}; gw.textures = pre.data(); gw.textureNum = static_cast<uint32_t>(pre.size()); c.CmdBarrier(cmd, &gw);
        g_resolvedInit = true; g_msaaInit = true;

        VriAttachmentDesc col{};
        col.view = msaaOn ? msaaView : resolvedView;
        col.loadOp = VriAttachmentLoadOp_Clear; col.storeOp = VriAttachmentStoreOp_Store;
        col.clearValue.color.f32[0] = 0.06f; col.clearValue.color.f32[1] = 0.07f; col.clearValue.color.f32[2] = 0.09f; col.clearValue.color.f32[3] = 1.0f;
        if (msaaOn) col.resolveView = resolvedView; // MSAA -> single-sample resolve at EndRendering
        VriAttachmentsDesc att{}; att.colors = &col; att.colorNum = 1; att.renderArea.width = kWidth; att.renderArea.height = kHeight; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, float(kWidth), float(kHeight), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kWidth, kHeight}; c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, msaaOn ? scene4x : scene1x);
        c.CmdSetPipelineLayout(cmd, sceneLayout);
        c.CmdSetConstants(cmd, 0, &pc, sizeof(Push));
        VriVertexBufferBinding vb{}; vb.buffer = vbuf; vb.offset = 0; c.CmdSetVertexBuffers(cmd, 0, &vb, 1);
        VriDrawDesc d{}; d.vertexNum = vertexNum; d.instanceNum = 1; c.CmdDraw(cmd, &d);
        c.CmdEndRendering(cmd);

        // resolve texture -> sampleable by the composite pass
        VriTextureBarrierDesc ts{}; ts.texture = resolved; ts.aspect = VriImageAspect_Color;
        ts.before.access = VriAccess_ColorAttachmentWrite; ts.before.layout = VriLayout_ColorAttachment; ts.before.stages = VriPipelineStage_ColorAttachmentOutput;
        ts.after.access = VriAccess_ShaderResourceRead; ts.after.layout = VriLayout_ShaderResource; ts.after.stages = VriPipelineStage_FragmentShader;
        VriBarrierGroupDesc gs{}; gs.textures = &ts; gs.textureNum = 1; c.CmdBarrier(cmd, &gs);
    };

    app.onRecord = [=](VriCommandBuffer* cmd) {
        c.CmdSetPipeline(cmd, compPipeline);
        c.CmdSetPipelineLayout(cmd, compLayout);
        c.CmdSetDescriptorSet(cmd, 0, compSet);
        VriDrawDesc d{}; d.vertexNum = 3; d.instanceNum = 1; c.CmdDraw(cmd, &d);
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(compPipeline); c.DestroyPipelineLayout(compLayout);
    c.DestroyPipeline(scene1x); c.DestroyPipeline(scene4x); c.DestroyPipelineLayout(sceneLayout);
    c.DestroyDescriptor(sampler); c.DestroyDescriptor(resolvedView); c.DestroyDescriptor(msaaView);
    c.DestroyTexture(resolved); c.DestroyTexture(msaaColor); c.DestroyBuffer(vbuf);
    app.Shutdown();
#endif
    return 0;
}
