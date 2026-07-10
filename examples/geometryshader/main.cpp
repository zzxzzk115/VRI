// Geometry shaders: a spinning Blinn-Phong torus, plus a second pass whose GEOMETRY stage
// extrudes every vertex normal into a line segment (yellow base -> red tip) - the classic
// normal-debugging use of the stage (Sascha Willems' geometryshader). The normals pipeline
// exists only where VriDeviceDesc::hasGeometryShader (Vulkan / D3D12 / desktop GL); elsewhere
// (Metal, WebGPU) the lit torus still renders and the UI says the stage is unavailable.
// ImGui drives spin, the toggle, and the normal length (a constant buffer the GS reads).
// Shared scaffolding: examples/common/example_app.h.
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "../cube/mat4.h"

#include "shaders/examples/geom_normals_dxbc.h" // g_geomNormalsDxbcVS / GS / PS (D3D12)
#include "shaders/examples/geom_normals_spv.h"  // g_geomNormalsSpv (Vulkan + OpenGL; no WGSL: geometry stage)
#include "shaders/examples/geom_scene_dxbc.h"   // g_geomSceneDxbcVS / PS
#include "shaders/examples/geom_scene_spv.h"    // g_geomSceneSpv
#include "shaders/examples/geom_scene_wgsl.h"   // g_geomSceneWgsl (WebGPU renders the lit torus only)

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480;
    constexpr int      kSegU = 48, kSegV = 24; // torus tessellation
    constexpr float    kRingR = 0.65f, kTubeR = 0.28f;

    struct Vertex
    {
        float px, py, pz;
        float nx, ny, nz;
    };

    struct Ubo
    {
        Mat4  mvp;
        Mat4  model;
        float misc[4]; // x = normal length
    };

    void BuildTorus(std::vector<Vertex>& verts, std::vector<uint16_t>& indices)
    {
        const float twoPi = 6.2831853f;
        for (int u = 0; u < kSegU; ++u)
            for (int v = 0; v < kSegV; ++v)
            {
                const float theta = twoPi * float(u) / float(kSegU);
                const float phi   = twoPi * float(v) / float(kSegV);
                const float nx = std::cos(theta) * std::cos(phi), ny = std::sin(phi),
                            nz = std::sin(theta) * std::cos(phi);
                Vertex vert {};
                vert.px = std::cos(theta) * kRingR + nx * kTubeR;
                vert.py = ny * kTubeR;
                vert.pz = std::sin(theta) * kRingR + nz * kTubeR;
                vert.nx = nx;
                vert.ny = ny;
                vert.nz = nz;
                verts.push_back(vert);
            }
        auto at = [](int u, int v) { return uint16_t(((u % kSegU) * kSegV) + (v % kSegV)); };
        for (int u = 0; u < kSegU; ++u)
            for (int v = 0; v < kSegV; ++v)
            {
                indices.push_back(at(u, v));
                indices.push_back(at(u + 1, v));
                indices.push_back(at(u, v + 1));
                indices.push_back(at(u + 1, v));
                indices.push_back(at(u + 1, v + 1));
                indices.push_back(at(u, v + 1));
            }
    }
} // namespace

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.Init("geometryshader", kWidth, kHeight, /*hasDepth*/ true);
    VriCoreInterface& c = app.c;

    const bool hasGS = c.GetDeviceDesc(app.dev)->hasGeometryShader == VRI_TRUE;

    // ---- torus geometry ----
    std::vector<Vertex>   verts;
    std::vector<uint16_t> indices;
    BuildTorus(verts, indices);
    const uint32_t indexNum = uint32_t(indices.size());

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
    VriBuffer* ubo  = deviceBuf(sizeof(Ubo), VriBufferUsage_ConstantBuffer);

    VriBufferDesc usd {};
    usd.size           = sizeof(Ubo);
    usd.usage          = VriBufferUsage_TransferSrc;
    usd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* ustg    = nullptr;
    c.CreateBuffer(app.dev, &usd, &ustg);

    VriBufferViewDesc ubv {};
    ubv.buffer             = ubo;
    ubv.viewType           = VriDescriptorType_ConstantBuffer;
    ubv.offset             = 0;
    ubv.size               = sizeof(Ubo);
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

    // ---- layout: one CB, read by the vertex stage (+ geometry where the stage exists) ----
    VriDescriptorRangeDesc range {};
    range.baseRegister   = 0;
    range.descriptorNum  = 1;
    range.descriptorType = VriDescriptorType_ConstantBuffer;
    range.shaderStages   = VriShaderStage_Vertex | (hasGS ? VriShaderStage_Geometry : 0u);
    VriDescriptorSetDesc setDesc {};
    setDesc.registerSpace = 0;
    setDesc.ranges        = &range;
    setDesc.rangeNum      = 1;
    VriPipelineLayoutDesc ld {};
    ld.descriptorSets         = &setDesc;
    ld.descriptorSetNum       = 1;
    VriPipelineLayout* layout = nullptr;
    c.CreatePipelineLayout(app.dev, &ld, &layout);

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

    VriColorAttachmentDesc ca {};
    ca.format         = app.swapFormat;
    ca.colorWriteMask = VriColorWrite_RGBA;

    // ---- lit torus pipeline (all backends) ----
    const vriex::ExampleApp::ShaderVariants sceneVS {
        VRI_SHADER_BLOB(g_geomSceneSpv),
        VRI_SHADER_BLOB(g_geomSceneWgsl),
        VRI_SHADER_D3D12(g_geomSceneDxbcVS),
    };
    const vriex::ExampleApp::ShaderVariants scenePS {
        VRI_SHADER_BLOB(g_geomSceneSpv),
        VRI_SHADER_BLOB(g_geomSceneWgsl),
        VRI_SHADER_D3D12(g_geomSceneDxbcPS),
    };
    VriShaderDesc ssh[2] = {
        app.Shader(VriShaderStage_Vertex, "vertexMain", sceneVS),
        app.Shader(VriShaderStage_Fragment, "fragmentMain", scenePS),
    };
    VriGraphicsPipelineDesc pd {};
    pd.pipelineLayout                  = layout;
    pd.shaders                         = ssh;
    pd.shaderNum                       = 2;
    pd.vertexInput.attributes          = attrs;
    pd.vertexInput.attributeNum        = 2;
    pd.vertexInput.streams             = &stream;
    pd.vertexInput.streamNum           = 1;
    pd.inputAssembly.topology          = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode          = VriCullMode_Back;
    pd.rasterization.frontFace         = VriFrontFace_CounterClockwise;
    pd.rasterization.lineWidth         = 1.0f;
    pd.depthStencil.depthTest          = VRI_TRUE;
    pd.depthStencil.depthWrite         = VRI_TRUE;
    pd.depthStencil.depthCompareOp     = VriCompareOp_Less;
    pd.multisample.sampleNum           = 1;
    pd.outputMerger.colors             = &ca;
    pd.outputMerger.colorNum           = 1;
    pd.outputMerger.depthStencilFormat = app.depthFormat;
    VriPipeline* scenePipeline         = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &pd, &scenePipeline) != VriResult_Success)
        app.Fail("scene pipeline failed");

    // ---- normal-visualization pipeline (VS + GS + FS) where the stage exists ----
    VriPipeline* normalsPipeline = nullptr;
    if (hasGS)
    {
        const vriex::ExampleApp::ShaderVariants nVS {
            VRI_SHADER_BLOB(g_geomNormalsSpv), nullptr, 0, VRI_SHADER_D3D12(g_geomNormalsDxbcVS)};
        const vriex::ExampleApp::ShaderVariants nGS {
            VRI_SHADER_BLOB(g_geomNormalsSpv), nullptr, 0, VRI_SHADER_D3D12(g_geomNormalsDxbcGS)};
        const vriex::ExampleApp::ShaderVariants nPS {
            VRI_SHADER_BLOB(g_geomNormalsSpv), nullptr, 0, VRI_SHADER_D3D12(g_geomNormalsDxbcPS)};
        VriShaderDesc nsh[3] = {
            app.Shader(VriShaderStage_Vertex, "vertexMain", nVS),
            app.Shader(VriShaderStage_Geometry, "geometryMain", nGS),
            app.Shader(VriShaderStage_Fragment, "fragmentMain", nPS),
        };
        VriGraphicsPipelineDesc nd = pd;
        nd.shaders                 = nsh;
        nd.shaderNum               = 3;
        nd.rasterization.cullMode  = VriCullMode_None; // lines
        nd.depthStencil.depthWrite = VRI_FALSE;        // test against the torus, don't occlude it
        if (c.CreateGraphicsPipeline(app.dev, &nd, &normalsPipeline) != VriResult_Success)
            app.Fail("normals pipeline failed");
    }

    VriDescriptorPoolDesc pdsc {};
    pdsc.descriptorSetMaxNum  = 1;
    pdsc.constantBufferMaxNum = 1;
    VriDescriptorPool* pool   = nullptr;
    c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* set = nullptr;
    c.AllocateDescriptorSets(pool, layout, 0, &set, 1);
    {
        const VriDescriptor*         u[1] = {uboView};
        VriDescriptorRangeUpdateDesc up {};
        up.descriptors   = u;
        up.descriptorNum = 1;
        c.UpdateDescriptorRanges(set, 0, 1, &up);
    }

    static float spin = 1.0f, angle = 0.6f, normalLen = 0.12f;
    static bool  showNormals = true;
    app.onUpdate             = [ustg](uint64_t) {
        angle += 0.5f * spin * app.dt;
        const float eye[3] = {0, 0.6f, 2.6f}, ctr[3] = {0, 0, 0}, up[3] = {0, 1, 0};
        const Mat4  model = Mul(RotateY(angle), RotateX(0.45f));
        const Mat4  vp    = Mul(Perspective(0.9f, float(kWidth) / float(kHeight), 0.1f, 100.0f), LookAt(eye, ctr, up));
        Ubo         u {};
        u.mvp     = Transpose(Mul(vp, model));
        u.model   = Transpose(model);
        u.misc[0] = normalLen;
        std::memcpy(app.c.MapBuffer(ustg, 0, sizeof(Ubo)), &u, sizeof(Ubo));
        app.c.UnmapBuffer(ustg);
    };
    app.onGui = [=] {
        ImGui::SliderFloat("spin", &spin, 0.0f, 5.0f);
        if (hasGS)
        {
            ImGui::Checkbox("show normals (GS)", &showNormals);
            ImGui::SliderFloat("normal length", &normalLen, 0.02f, 0.4f);
        }
        else
        {
            ImGui::Text("hasGeometryShader = false on %s\n(lit torus only)", app.apiName);
        }
    };
    app.onPreRender = [ubo, ustg](VriCommandBuffer* cmd) {
        VriBufferCopyDesc ucp {};
        ucp.size = sizeof(Ubo);
        app.c.CmdCopyBuffer(cmd, ubo, ustg, &ucp);
        VriBufferBarrierDesc ub {};
        ub.buffer        = ubo;
        ub.before.access = VriAccess_CopyDestinationWrite;
        ub.before.stages = VriPipelineStage_Transfer;
        ub.after.access  = VriAccess_ConstantBufferRead;
        ub.after.stages  = VriPipelineStage_VertexShader;
        VriBarrierGroupDesc ubg {};
        ubg.buffers   = &ub;
        ubg.bufferNum = 1;
        app.c.CmdBarrier(cmd, &ubg);
    };
    app.onRecord = [=](VriCommandBuffer* cmd) {
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, set);
        VriVertexBufferBinding vb {};
        vb.buffer = vbuf;
        vb.offset = 0;
        c.CmdSetVertexBuffers(cmd, 0, &vb, 1);
        c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);
        VriDrawIndexedDesc di {};
        di.indexNum    = indexNum;
        di.instanceNum = 1;
        c.CmdSetPipeline(cmd, scenePipeline);
        c.CmdDrawIndexed(cmd, &di);
        if (normalsPipeline && showNormals)
        {
            c.CmdSetPipeline(cmd, normalsPipeline);
            c.CmdDrawIndexed(cmd, &di); // same mesh; the GS emits the lines
        }
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyDescriptorPool(pool);
    if (normalsPipeline)
        c.DestroyPipeline(normalsPipeline);
    c.DestroyPipeline(scenePipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyDescriptor(uboView);
    c.DestroyBuffer(ustg);
    c.DestroyBuffer(ubo);
    c.DestroyBuffer(ibuf);
    c.DestroyBuffer(vbuf);
    app.Shutdown();
#endif
    return 0;
}
