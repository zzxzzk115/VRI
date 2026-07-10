// GPU queries: an occluder wall slides between the camera and two spheres. Each sphere's
// draw is bracketed by an OCCLUSION query (CmdBeginQuery/CmdEndQuery); the results resolve
// GPU-side (CmdCopyQueries) and the example reads LAST frame's numbers - no stall - showing
// "N samples visible / OCCLUDED" per sphere and tinting a hidden sphere red. One PIPELINE-
// STATISTICS query wraps the whole scene and an ImGui table lists the per-stage counters
// (VriPipelineStatistics). The scene renders in its own offscreen pass so the queries can
// reset/resolve outside a render pass; the swapchain pass just composites it (show_tex).
// Panels appear only where VRI_INTERFACE_QUERY + the pool type exist (occlusion: Vulkan /
// D3D12 / GL; statistics: needs VriDeviceDesc::hasPipelineStatistics).
// Shared scaffolding: examples/common/example_app.h.
#include "../common/example_app.h"

#include <vri/ext/vri_ext_query.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "../cube/mat4.h"

#include "shaders/examples/occl_scene_dxbc.h" // g_occlSceneDxbcVS / PS (D3D12)
#include "shaders/examples/occl_scene_spv.h"  // g_occlSceneSpv (Vulkan + OpenGL)
#include "shaders/examples/occl_scene_wgsl.h" // g_occlSceneWgsl (WebGPU: scene only, no queries)
#include "shaders/examples/show_tex_dxbc.h"   // g_showTexDxbcVS / PS (fullscreen composite)
#include "shaders/examples/show_tex_spv.h"    // g_showTexSpv
#include "shaders/examples/show_tex_wgsl.h"   // g_showTexWgsl

namespace
{
    constexpr uint32_t  kWidth = 640, kHeight = 480;
    constexpr VriFormat kColor = VriFormat_RGBA8_UNORM;
    constexpr VriFormat kDepth = VriFormat_D32_SFLOAT;

    struct Vertex
    {
        float px, py, pz;
        float nx, ny, nz;
    };
    struct CameraUbo
    {
        Mat4 viewProj;
    };
    struct Push
    {
        Mat4  model;
        float color[4];
    };

    // sphere + occluder quad share one vertex/index buffer; these locate each mesh
    struct MeshRange
    {
        uint32_t baseIndex, indexNum;
        int32_t  vertexOffset;
    };

    void BuildSphere(std::vector<Vertex>& verts, std::vector<uint16_t>& indices)
    {
        constexpr int   kSeg = 24, kRing = 16;
        constexpr float kPi = 3.1415926f, kTwoPi = 6.2831853f;
        for (int r = 0; r <= kRing; ++r)
            for (int s = 0; s <= kSeg; ++s)
            {
                const float phi   = kPi * float(r) / float(kRing) - kPi * 0.5f; // -pi/2..pi/2
                const float theta = kTwoPi * float(s) / float(kSeg);
                Vertex      v {};
                v.nx = std::cos(phi) * std::cos(theta);
                v.ny = std::sin(phi);
                v.nz = std::cos(phi) * std::sin(theta);
                v.px = v.nx;
                v.py = v.ny;
                v.pz = v.nz;
                verts.push_back(v);
            }
        auto at = [](int r, int s) { return uint16_t(r * (kSeg + 1) + s); };
        for (int r = 0; r < kRing; ++r)
            for (int s = 0; s < kSeg; ++s)
            {
                indices.push_back(at(r, s));
                indices.push_back(at(r + 1, s));
                indices.push_back(at(r, s + 1));
                indices.push_back(at(r + 1, s));
                indices.push_back(at(r + 1, s + 1));
                indices.push_back(at(r, s + 1));
            }
    }

    void BuildQuad(std::vector<Vertex>& verts, std::vector<uint16_t>& indices)
    {
        const float n[3]    = {0, 0, 1};
        const float q[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        for (auto& p : q)
            verts.push_back({p[0], p[1], 0.0f, n[0], n[1], n[2]});
        const uint16_t idx[6] = {0, 1, 2, 0, 2, 3};
        indices.insert(indices.end(), idx, idx + 6);
    }

    bool g_rtInit = false;
} // namespace

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.Init("queries", kWidth, kHeight, /*hasDepth*/ false); // the scene pass owns its own depth
    VriCoreInterface& c = app.c;

    // ---- query machinery (each piece degrades independently) ----
    static VriQueryInterface q {};
    const bool hasQueryIface = vriGetInterface(app.dev, VRI_INTERFACE_QUERY, sizeof(q), &q) == VriResult_Success;
    const bool hasStatsCap   = c.GetDeviceDesc(app.dev)->hasPipelineStatistics == VRI_TRUE;

    static VriQueryPool* occlPool     = nullptr;
    static VriBuffer*    occlReadback = nullptr;
    if (hasQueryIface)
    {
        VriQueryPoolDesc qd {};
        qd.type       = VriQueryType_Occlusion;
        qd.queryCount = 2;
        if (q.CreateQueryPool(app.dev, &qd, &occlPool) == VriResult_Success)
        {
            VriBufferDesc bd {};
            bd.size           = 2 * q.GetQuerySize(occlPool);
            bd.usage          = VriBufferUsage_TransferDst;
            bd.memoryLocation = VriMemoryLocation_HostReadback;
            if (c.CreateBuffer(app.dev, &bd, &occlReadback) != VriResult_Success)
                occlReadback = nullptr;
        }
    }
    static VriQueryPool* statsPool     = nullptr;
    static VriBuffer*    statsReadback = nullptr;
    if (hasQueryIface && hasStatsCap)
    {
        VriQueryPoolDesc qd {};
        qd.type       = VriQueryType_PipelineStatistics;
        qd.queryCount = 1;
        if (q.CreateQueryPool(app.dev, &qd, &statsPool) == VriResult_Success)
        {
            VriBufferDesc bd {};
            bd.size           = q.GetQuerySize(statsPool);
            bd.usage          = VriBufferUsage_TransferDst;
            bd.memoryLocation = VriMemoryLocation_HostReadback;
            if (c.CreateBuffer(app.dev, &bd, &statsReadback) != VriResult_Success)
                statsReadback = nullptr;
        }
    }

    // ---- offscreen scene target + depth ----
    auto makeTex = [&](VriFormat fmt, VriTextureUsageFlags usage) {
        VriTextureDesc td {};
        td.type           = VriTextureType_2D;
        td.format         = fmt;
        td.width          = kWidth;
        td.height         = kHeight;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = 1;
        td.sampleNum      = 1;
        td.usage          = usage;
        td.memoryLocation = VriMemoryLocation_Device;
        if (fmt == kDepth)
        {
            td.clearValue.depthStencil.depth = 1.0f;
        }
        else
        {
            // must match the render-pass clear (D3D12 fast-clear, and its debug layer warns per frame)
            td.clearValue.color.f32[0] = 0.07f;
            td.clearValue.color.f32[1] = 0.08f;
            td.clearValue.color.f32[2] = 0.11f;
            td.clearValue.color.f32[3] = 1.0f;
        }
        VriTexture* t = nullptr;
        if (c.CreateTexture(app.dev, &td, &t) != VriResult_Success)
            app.Fail("offscreen CreateTexture failed");
        return t;
    };
    VriTexture* rt      = makeTex(kColor, VriTextureUsage_ColorAttachment | VriTextureUsage_ShaderResource);
    VriTexture* rtDepth = makeTex(kDepth, VriTextureUsage_DepthStencilAttachment);

    auto makeView = [&](VriTexture* t, VriImageAspectFlags aspect) {
        VriTextureViewDesc vd {};
        vd.texture       = t;
        vd.viewType      = VriTextureViewType_2D;
        vd.format        = VriFormat_Unknown;
        vd.aspect        = aspect;
        VriDescriptor* v = nullptr;
        if (c.CreateTextureView(app.dev, &vd, &v) != VriResult_Success)
            app.Fail("view failed");
        return v;
    };
    VriDescriptor* rtView    = makeView(rt, VriImageAspect_Color);
    VriDescriptor* depthView = makeView(rtDepth, VriImageAspect_Depth);

    // ---- geometry: one buffer holds the sphere then the occluder quad ----
    std::vector<Vertex>   verts;
    std::vector<uint16_t> indices;
    BuildSphere(verts, indices);
    const MeshRange sphere {0, uint32_t(indices.size()), 0};
    const int32_t   quadVertexOffset = int32_t(verts.size());
    const uint32_t  quadBaseIndex    = uint32_t(indices.size());
    BuildQuad(verts, indices);
    const MeshRange quad {quadBaseIndex, uint32_t(indices.size()) - quadBaseIndex, quadVertexOffset};

    auto deviceBuf = [&](uint64_t size, VriBufferUsageFlags usage) {
        VriBufferDesc bd {};
        bd.size           = size;
        bd.usage          = usage | VriBufferUsage_TransferDst;
        bd.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* b      = nullptr;
        c.CreateBuffer(app.dev, &bd, &b);
        return b;
    };
    VriBuffer* vbuf = deviceBuf(verts.size() * sizeof(Vertex), VriBufferUsage_VertexBuffer);
    VriBuffer* ibuf = deviceBuf(indices.size() * sizeof(uint16_t), VriBufferUsage_IndexBuffer);
    VriBuffer* ubo  = deviceBuf(sizeof(CameraUbo), VriBufferUsage_ConstantBuffer);

    VriBufferDesc usd {};
    usd.size           = sizeof(CameraUbo);
    usd.usage          = VriBufferUsage_TransferSrc;
    usd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* ustg    = nullptr;
    c.CreateBuffer(app.dev, &usd, &ustg);

    VriBufferViewDesc ubv {};
    ubv.buffer             = ubo;
    ubv.viewType           = VriDescriptorType_ConstantBuffer;
    ubv.offset             = 0;
    ubv.size               = sizeof(CameraUbo);
    VriDescriptor* uboView = nullptr;
    c.CreateBufferView(app.dev, &ubv, &uboView);

    app.BeginUpload();
    app.UploadBuffer(
        vbuf, verts.data(), verts.size() * sizeof(Vertex), VriAccess_VertexBufferRead, VriPipelineStage_VertexInput);
    app.UploadBuffer(ibuf,
                     indices.data(),
                     indices.size() * sizeof(uint16_t),
                     VriAccess_IndexBufferRead,
                     VriPipelineStage_VertexInput);
    app.EndUpload();

    // ---- scene pipeline: CB@0 (viewProj) + push-constant model/color ----
    VriDescriptorRangeDesc range {};
    range.baseRegister   = 0;
    range.descriptorNum  = 1;
    range.descriptorType = VriDescriptorType_ConstantBuffer;
    range.shaderStages   = VriShaderStage_Vertex;
    VriDescriptorSetDesc setDesc {};
    setDesc.registerSpace = 0;
    setDesc.ranges        = &range;
    setDesc.rangeNum      = 1;
    // D3D12: push constants become root constants at a b# register; the UBO already takes b0, so
    // Slang puts the push-constant cbuffer at b1. VK/GL/WebGPU ignore baseRegister for push
    // constants (byte offset / named / reserved group), so b1 is safe everywhere.
    VriPushConstantDesc pcd {};
    pcd.baseRegister = 1;
    pcd.size         = sizeof(Push);
    pcd.shaderStages = VriShaderStage_Vertex | VriShaderStage_Fragment;
    VriPipelineLayoutDesc sld {};
    sld.descriptorSets             = &setDesc;
    sld.descriptorSetNum           = 1;
    sld.pushConstants              = &pcd;
    sld.pushConstantNum            = 1;
    VriPipelineLayout* sceneLayout = nullptr;
    c.CreatePipelineLayout(app.dev, &sld, &sceneLayout);

    const vriex::ExampleApp::ShaderVariants sceneVS {
        VRI_SHADER_BLOB(g_occlSceneSpv),
        VRI_SHADER_BLOB(g_occlSceneWgsl),
        VRI_SHADER_D3D12(g_occlSceneDxbcVS),
    };
    const vriex::ExampleApp::ShaderVariants scenePS {
        VRI_SHADER_BLOB(g_occlSceneSpv),
        VRI_SHADER_BLOB(g_occlSceneWgsl),
        VRI_SHADER_D3D12(g_occlSceneDxbcPS),
    };
    VriShaderDesc ssh[2] = {
        app.Shader(VriShaderStage_Vertex, "vertexMain", sceneVS),
        app.Shader(VriShaderStage_Fragment, "fragmentMain", scenePS),
    };
    VriVertexAttributeDesc attrs[2] {};
    attrs[0].format      = VriFormat_RGB32_SFLOAT;
    attrs[0].offset      = 0;
    attrs[0].streamIndex = 0; // position
    attrs[1].format      = VriFormat_RGB32_SFLOAT;
    attrs[1].offset      = 12;
    attrs[1].streamIndex = 0; // normal
    VriVertexStreamDesc stream {};
    stream.stride      = sizeof(Vertex);
    stream.bindingSlot = 0;
    stream.stepRate    = VriVertexStepRate_PerVertex;
    VriColorAttachmentDesc sca {};
    sca.format         = kColor;
    sca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc spd {};
    spd.pipelineLayout                  = sceneLayout;
    spd.shaders                         = ssh;
    spd.shaderNum                       = 2;
    spd.vertexInput.attributes          = attrs;
    spd.vertexInput.attributeNum        = 2;
    spd.vertexInput.streams             = &stream;
    spd.vertexInput.streamNum           = 1;
    spd.inputAssembly.topology          = VriPrimitiveTopology_TriangleList;
    spd.rasterization.cullMode          = VriCullMode_None; // the occluder quad must block from the front
    spd.rasterization.frontFace         = VriFrontFace_CounterClockwise;
    spd.rasterization.lineWidth         = 1.0f;
    spd.depthStencil.depthTest          = VRI_TRUE;
    spd.depthStencil.depthWrite         = VRI_TRUE;
    spd.depthStencil.depthCompareOp     = VriCompareOp_Less;
    spd.multisample.sampleNum           = 1;
    spd.outputMerger.colors             = &sca;
    spd.outputMerger.colorNum           = 1;
    spd.outputMerger.depthStencilFormat = kDepth;
    VriPipeline* scenePipeline          = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &spd, &scenePipeline) != VriResult_Success)
        app.Fail("scene pipeline failed");

    // ---- composite (show_tex) ----
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
    smp.magFilter          = VriFilter_Linear;
    smp.minFilter          = VriFilter_Linear;
    smp.mipmapMode         = VriMipmapMode_Nearest;
    smp.addressModeU       = VriAddressMode_ClampToEdge;
    smp.addressModeV       = VriAddressMode_ClampToEdge;
    smp.addressModeW       = VriAddressMode_ClampToEdge;
    smp.maxLod             = 1.0f;
    VriDescriptor* sampler = nullptr;
    c.CreateSampler(app.dev, &smp, &sampler);

    VriDescriptorPoolDesc pdsc {};
    pdsc.descriptorSetMaxNum  = 2;
    pdsc.textureMaxNum        = 1;
    pdsc.samplerMaxNum        = 1;
    pdsc.constantBufferMaxNum = 1;
    VriDescriptorPool* pool   = nullptr;
    c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* sceneSet = nullptr;
    c.AllocateDescriptorSets(pool, sceneLayout, 0, &sceneSet, 1);
    VriDescriptorSet* compSet = nullptr;
    c.AllocateDescriptorSets(pool, compLayout, 0, &compSet, 1);
    {
        const VriDescriptor*         u[1] = {uboView};
        VriDescriptorRangeUpdateDesc up {};
        up.descriptors   = u;
        up.descriptorNum = 1;
        c.UpdateDescriptorRanges(sceneSet, 0, 1, &up);
        const VriDescriptor*         t[1] = {rtView};
        const VriDescriptor*         s[1] = {sampler};
        VriDescriptorRangeUpdateDesc uc[2] {};
        uc[0].descriptors   = t;
        uc[0].descriptorNum = 1;
        uc[1].descriptors   = s;
        uc[1].descriptorNum = 1;
        c.UpdateDescriptorRanges(compSet, 0, 2, uc);
    }

    // ---- per-frame state ----
    static float                 occluderX = 0.0f, autoT = 0.0f;
    static bool                  autoMove          = true;
    static uint64_t              visibleSamples[2] = {~0ull, ~0ull}; // last frame's occlusion results
    static VriPipelineStatistics stats {};
    static bool                  resultsValid = false, queriesRecorded = false;

    app.onUpdate = [ustg](uint64_t frame) {
        if (autoMove)
        {
            autoT += 0.5f * app.dt;
            occluderX = std::sin(autoT) * 1.6f;
        }
        const float eye[3] = {0, 0.4f, 4.2f}, ctr[3] = {0, 0, 0}, up[3] = {0, 1, 0};
        CameraUbo   u {};
        u.viewProj =
            Transpose(Mul(Perspective(0.85f, float(kWidth) / float(kHeight), 0.1f, 100.0f), LookAt(eye, ctr, up)));
        std::memcpy(app.c.MapBuffer(ustg, 0, sizeof(CameraUbo)), &u, sizeof(CameraUbo));
        app.c.UnmapBuffer(ustg);

        // The frame loop waits the fence each frame, so last frame's resolves are ready.
        if (frame > 1 && queriesRecorded)
        {
            if (occlReadback)
            {
                const uint64_t* r =
                    static_cast<const uint64_t*>(app.c.MapBuffer(occlReadback, 0, 2 * sizeof(uint64_t)));
                visibleSamples[0] = r[0];
                visibleSamples[1] = r[1];
                app.c.UnmapBuffer(occlReadback);
            }
            if (statsReadback)
            {
                std::memcpy(&stats,
                            app.c.MapBuffer(statsReadback, 0, sizeof(VriPipelineStatistics)),
                            sizeof(VriPipelineStatistics));
                app.c.UnmapBuffer(statsReadback);
            }
            resultsValid = true;
        }
    };

    app.onGui = [=] {
        ImGui::Checkbox("slide occluder", &autoMove);
        ImGui::SliderFloat("occluder x", &occluderX, -2.0f, 2.0f);
        if (occlReadback)
        {
            ImGui::SeparatorText("occlusion (last frame)");
            const char* names[2] = {"green sphere", "blue sphere"};
            for (int i = 0; i < 2; ++i)
            {
                if (!resultsValid)
                    ImGui::Text("%s: ...", names[i]);
                else if (visibleSamples[i] == 0)
                    ImGui::Text("%s: OCCLUDED", names[i]);
                else
                    ImGui::Text(
                        "%s: %llu samples visible", names[i], static_cast<unsigned long long>(visibleSamples[i]));
            }
        }
        else
        {
            ImGui::Text("occlusion queries unavailable on %s", app.apiName);
        }
        if (statsReadback)
        {
            ImGui::SeparatorText("pipeline statistics (last frame)");
            auto row = [](const char* n, uint64_t v) {
                ImGui::Text("%-26s %llu", n, static_cast<unsigned long long>(v));
            };
            row("input vertices", stats.inputVertices);
            row("input primitives", stats.inputPrimitives);
            row("VS invocations", stats.vertexShaderInvocations);
            row("clipping invocations", stats.clippingInvocations);
            row("clipping primitives", stats.clippingPrimitives);
            row("FS invocations", stats.fragmentShaderInvocations);
        }
        else if (!hasStatsCap)
        {
            ImGui::Text("hasPipelineStatistics = false on %s", app.apiName);
        }
    };

    app.onPreRender = [=](VriCommandBuffer* cmd) {
        // camera CB upload
        VriBufferCopyDesc ucp {};
        ucp.size = sizeof(CameraUbo);
        c.CmdCopyBuffer(cmd, ubo, ustg, &ucp);
        VriBufferBarrierDesc ub {};
        ub.buffer        = ubo;
        ub.before.access = VriAccess_CopyDestinationWrite;
        ub.before.stages = VriPipelineStage_Transfer;
        ub.after.access  = VriAccess_ConstantBufferRead;
        ub.after.stages  = VriPipelineStage_VertexShader;
        VriBarrierGroupDesc ubg {};
        ubg.buffers   = &ub;
        ubg.bufferNum = 1;
        c.CmdBarrier(cmd, &ubg);

        if (occlPool)
            q.CmdResetQueries(cmd, occlPool, 0, 2);
        if (statsPool)
            q.CmdResetQueries(cmd, statsPool, 0, 1);

        // offscreen target + its depth -> attachments
        VriTextureBarrierDesc tb[2] {};
        tb[0].texture       = rt;
        tb[0].aspect        = VriImageAspect_Color;
        tb[0].before.layout = g_rtInit ? VriLayout_ShaderResource : VriLayout_Undefined;
        tb[0].before.stages = g_rtInit ? VriPipelineStage_FragmentShader : VriPipelineStage_None;
        tb[0].after.access  = VriAccess_ColorAttachmentWrite;
        tb[0].after.layout  = VriLayout_ColorAttachment;
        tb[0].after.stages  = VriPipelineStage_ColorAttachmentOutput;
        tb[1].texture       = rtDepth;
        tb[1].aspect        = VriImageAspect_Depth;
        tb[1].before.layout = g_rtInit ? VriLayout_DepthStencilAttachment : VriLayout_Undefined;
        tb[1].before.stages = g_rtInit ? VriPipelineStage_LateFragmentTests : VriPipelineStage_None;
        tb[1].after.access  = VriAccess_DepthStencilAttachmentWrite;
        tb[1].after.layout  = VriLayout_DepthStencilAttachment;
        tb[1].after.stages  = VriPipelineStage_EarlyFragmentTests;
        VriBarrierGroupDesc gw {};
        gw.textures   = tb;
        gw.textureNum = 2;
        c.CmdBarrier(cmd, &gw);
        g_rtInit = true;

        VriAttachmentDesc col {};
        col.view                    = rtView;
        col.loadOp                  = VriAttachmentLoadOp_Clear;
        col.storeOp                 = VriAttachmentStoreOp_Store;
        col.clearValue.color.f32[0] = 0.07f;
        col.clearValue.color.f32[1] = 0.08f;
        col.clearValue.color.f32[2] = 0.11f;
        col.clearValue.color.f32[3] = 1.0f;
        VriAttachmentDesc dep {};
        dep.view                          = depthView;
        dep.loadOp                        = VriAttachmentLoadOp_Clear;
        dep.storeOp                       = VriAttachmentStoreOp_DontCare;
        dep.clearValue.depthStencil.depth = 1.0f;
        VriAttachmentsDesc att {};
        att.colors            = &col;
        att.colorNum          = 1;
        att.depth             = &dep;
        att.renderArea.width  = kWidth;
        att.renderArea.height = kHeight;
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp {0, 0, float(kWidth), float(kHeight), 0, 1};
        c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc {0, 0, kWidth, kHeight};
        c.CmdSetScissors(cmd, &sc, 1);

        c.CmdSetPipeline(cmd, scenePipeline);
        c.CmdSetPipelineLayout(cmd, sceneLayout);
        c.CmdSetDescriptorSet(cmd, 0, sceneSet);
        VriVertexBufferBinding vb {};
        vb.buffer = vbuf;
        vb.offset = 0;
        c.CmdSetVertexBuffers(cmd, 0, &vb, 1);
        c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);

        auto drawMesh = [&](const MeshRange& m, const Mat4& model, float r, float g, float b) {
            Push pc {};
            pc.model    = Transpose(model);
            pc.color[0] = r;
            pc.color[1] = g;
            pc.color[2] = b;
            pc.color[3] = 1.0f;
            c.CmdSetConstants(cmd, 0, &pc, sizeof(Push));
            VriDrawIndexedDesc di {};
            di.indexNum     = m.indexNum;
            di.instanceNum  = 1;
            di.baseIndex    = m.baseIndex;
            di.vertexOffset = m.vertexOffset;
            c.CmdDrawIndexed(cmd, &di);
        };

        if (statsPool)
            q.CmdBeginQuery(cmd, statsPool, 0);

        // occluder wall (front, depth-writes first)
        drawMesh(quad, Mul(Translate(occluderX, 0.0f, 1.4f), Scale(0.9f)), 0.45f, 0.42f, 0.4f);

        // the two spheres behind it, each bracketed by an occlusion query;
        // a sphere that was hidden last frame renders red this frame
        const float sx[2]           = {-0.9f, 0.9f};
        const float baseColor[2][3] = {{0.3f, 0.85f, 0.45f}, {0.35f, 0.55f, 0.95f}};
        for (int i = 0; i < 2; ++i)
        {
            const bool hidden = resultsValid && visibleSamples[i] == 0;
            if (occlPool)
                q.CmdBeginQuery(cmd, occlPool, uint32_t(i));
            drawMesh(sphere,
                     Mul(Translate(sx[i], 0.0f, -0.6f), Scale(0.55f)),
                     hidden ? 0.9f : baseColor[i][0],
                     hidden ? 0.25f : baseColor[i][1],
                     hidden ? 0.25f : baseColor[i][2]);
            if (occlPool)
                q.CmdEndQuery(cmd, occlPool, uint32_t(i));
        }

        if (statsPool)
            q.CmdEndQuery(cmd, statsPool, 0);

        c.CmdEndRendering(cmd);

        if (occlPool && occlReadback)
            q.CmdCopyQueries(cmd, occlPool, 0, 2, occlReadback, 0);
        if (statsPool && statsReadback)
            q.CmdCopyQueries(cmd, statsPool, 0, 1, statsReadback, 0);
        queriesRecorded = occlReadback || statsReadback;

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
    c.DestroyDescriptor(uboView);
    c.DestroyBuffer(ustg);
    c.DestroyBuffer(ubo);
    c.DestroyBuffer(ibuf);
    c.DestroyBuffer(vbuf);
    c.DestroyDescriptor(depthView);
    c.DestroyDescriptor(rtView);
    c.DestroyTexture(rtDepth);
    c.DestroyTexture(rt);
    if (statsReadback)
        c.DestroyBuffer(statsReadback);
    if (statsPool)
        q.DestroyQueryPool(statsPool);
    if (occlReadback)
        c.DestroyBuffer(occlReadback);
    if (occlPool)
        q.DestroyQueryPool(occlPool);
    app.Shutdown();
#endif
    return 0;
}
