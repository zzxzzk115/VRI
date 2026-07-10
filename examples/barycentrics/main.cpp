// Fragment-shader barycentrics: a spinning torus whose fragment stage reads the
// rasterizer-generated SV_Barycentrics. Three ImGui modes: the raw barycentric weights as
// RGB, a solid antialiased wireframe computed from the distance to the nearest edge (the
// classic use - no line primitives, no geometry pass), and shading with the wireframe
// overlaid. Vulkan maps to VK_KHR_fragment_shader_barycentric, D3D12 to SM6.1 SV_Barycentrics
// (DXIL). Where VriDeviceDesc::hasFragmentShaderBarycentric is false (Metal, GL, WebGPU) the
// torus renders plain-shaded and the UI says so.
// Shared scaffolding: examples/common/example_app.h.
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "../cube/mat4.h"

#include "shaders/examples/barywire_basic_dxbc.h" // g_barywireBasicDxbcVS / PS (D3D12 fallback, unused when DXIL runs)
#include "shaders/examples/barywire_basic_spv.h"  // g_barywireBasicSpv (Vulkan + OpenGL fallback)
#include "shaders/examples/barywire_basic_wgsl.h" // g_barywireBasicWgsl (WebGPU fallback)
#include "shaders/examples/barywire_dxil.h"       // g_barywireDxilVS / PS (D3D12, SM6.1)
#include "shaders/examples/barywire_spv.h"        // g_barywireSpv (Vulkan; no WGSL/DXBC: barycentrics)

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480;
    constexpr int      kSegU = 32, kSegV = 16; // chunky triangles: the wireframe should read clearly
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
        float misc[4]; // x = mode
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

    const char* kModeNames[] = {"raw barycentrics", "solid wireframe", "shaded + wireframe"};
} // namespace

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.Init("barycentrics", kWidth, kHeight, /*hasDepth*/ true);
    VriCoreInterface& c = app.c;

    const bool hasBary = c.GetDeviceDesc(app.dev)->hasFragmentShaderBarycentric == VRI_TRUE;

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

    VriDescriptorRangeDesc range {};
    range.baseRegister   = 0;
    range.descriptorNum  = 1;
    range.descriptorType = VriDescriptorType_ConstantBuffer;
    range.shaderStages   = VriShaderStage_Vertex | VriShaderStage_Fragment; // FS reads the mode
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

    const vriex::ExampleApp::ShaderVariants vsVar =
        hasBary ?
            vriex::ExampleApp::ShaderVariants {
                VRI_SHADER_BLOB(g_barywireSpv), nullptr, 0, VRI_SHADER_D3D12(g_barywireDxilVS)} :
            vriex::ExampleApp::ShaderVariants {VRI_SHADER_BLOB(g_barywireBasicSpv),
                                               VRI_SHADER_BLOB(g_barywireBasicWgsl),
                                               VRI_SHADER_D3D12(g_barywireBasicDxbcVS)};
    const vriex::ExampleApp::ShaderVariants psVar =
        hasBary ?
            vriex::ExampleApp::ShaderVariants {
                VRI_SHADER_BLOB(g_barywireSpv), nullptr, 0, VRI_SHADER_D3D12(g_barywireDxilPS)} :
            vriex::ExampleApp::ShaderVariants {VRI_SHADER_BLOB(g_barywireBasicSpv),
                                               VRI_SHADER_BLOB(g_barywireBasicWgsl),
                                               VRI_SHADER_D3D12(g_barywireBasicDxbcPS)};
    VriShaderDesc sh[2] = {
        app.Shader(VriShaderStage_Vertex, "vertexMain", vsVar),
        app.Shader(VriShaderStage_Fragment, "fragmentMain", psVar),
    };

    VriColorAttachmentDesc ca {};
    ca.format         = app.swapFormat;
    ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd {};
    pd.pipelineLayout                  = layout;
    pd.shaders                         = sh;
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
    VriPipeline* pipeline              = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &pd, &pipeline) != VriResult_Success)
        app.Fail("pipeline failed");

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

    static float spin = 1.0f, angle = 0.6f;
    static int   mode = 1; // solid wireframe: the flagship use
    app.onUpdate      = [ustg](uint64_t) {
        angle += 0.5f * spin * app.dt;
        const float eye[3] = {0, 0.6f, 2.6f}, ctr[3] = {0, 0, 0}, up[3] = {0, 1, 0};
        const Mat4  model = Mul(RotateY(angle), RotateX(0.45f));
        Ubo         u {};
        u.mvp = Transpose(
            Mul(Mul(Perspective(0.9f, float(kWidth) / float(kHeight), 0.1f, 100.0f), LookAt(eye, ctr, up)), model));
        u.model   = Transpose(model);
        u.misc[0] = float(mode);
        std::memcpy(app.c.MapBuffer(ustg, 0, sizeof(Ubo)), &u, sizeof(Ubo));
        app.c.UnmapBuffer(ustg);
    };
    app.onGui = [=] {
        if (hasBary)
            ImGui::Combo("mode", &mode, kModeNames, 3);
        else
            ImGui::Text("hasFragmentShaderBarycentric = false on %s\n(plain shading)", app.apiName);
        ImGui::SliderFloat("spin", &spin, 0.0f, 5.0f);
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
        ub.after.stages  = VriPipelineStage_VertexShader | VriPipelineStage_FragmentShader;
        VriBarrierGroupDesc ubg {};
        ubg.buffers   = &ub;
        ubg.bufferNum = 1;
        app.c.CmdBarrier(cmd, &ubg);
    };
    app.onRecord = [=](VriCommandBuffer* cmd) {
        c.CmdSetPipeline(cmd, pipeline);
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
        c.CmdDrawIndexed(cmd, &di);
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(pipeline);
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
