// Tessellation: an 8x8 grid of 4-control-point QUAD patches displaced into rolling terrain
// by the domain shader (hull sets the ImGui-driven factors, VriInputAssemblyDesc::
// patchControlPoints = 4). Slide the factor from 1 to 32 to watch the faceted patch grid
// turn smooth - flip on wireframe to count the triangles. Where VriDeviceDesc::
// hasTessellation is false (Metal, WebGPU) a pre-triangulated fixed grid with vertex-stage
// displacement renders the same terrain and the UI says so. The height field lives in
// shaders/common/terrain_height.slangh, shared by both paths.
// Shared scaffolding: examples/common/example_app.h.
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "../cube/mat4.h"

#include "shaders/examples/terrain_basic_dxbc.h" // g_terrainBasicDxbcVS / PS (D3D12)
#include "shaders/examples/terrain_basic_spv.h"  // g_terrainBasicSpv (Vulkan + OpenGL)
#include "shaders/examples/terrain_basic_wgsl.h" // g_terrainBasicWgsl (WebGPU fallback path)
#include "shaders/examples/terrain_tess_dxbc.h"  // g_terrainTessDxbcVS / HS / DS / PS (D3D12)
#include "shaders/examples/terrain_tess_spv.h"   // g_terrainTessSpv (no WGSL: hull/domain stages)

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480;
    constexpr int      kPatches = 8;  // patch grid (9x9 control points)
    constexpr int      kDense   = 64; // fallback grid density
    constexpr float    kExtent  = 1.4f;

    struct Ubo
    {
        Mat4  mvp;
        float params[4]; // x = tess factor, y = height scale
    };

    float GridCoord(int i, int n) { return (float(i) / float(n) * 2.0f - 1.0f) * kExtent; }
} // namespace

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.Init("tessellation", kWidth, kHeight, /*hasDepth*/ true);
    VriCoreInterface& c = app.c;

    const bool hasTess = c.GetDeviceDesc(app.dev)->hasTessellation == VRI_TRUE;

    // ---- geometry: control-point patches (tess path) or a dense triangulated grid (fallback) ----
    std::vector<float>    verts; // float3 per vertex, y = 0 (displaced in the shader)
    std::vector<uint16_t> indices;
    if (hasTess)
    {
        for (int z = 0; z <= kPatches; ++z)
            for (int x = 0; x <= kPatches; ++x)
            {
                verts.push_back(GridCoord(x, kPatches));
                verts.push_back(0.0f);
                verts.push_back(GridCoord(z, kPatches));
            }
        auto at = [](int x, int z) { return uint16_t(z * (kPatches + 1) + x); };
        for (int z = 0; z < kPatches; ++z)
            for (int x = 0; x < kPatches; ++x)
            {
                // counter-clockwise quad, matching the domain shader's bilinear corner order
                indices.push_back(at(x, z));
                indices.push_back(at(x + 1, z));
                indices.push_back(at(x + 1, z + 1));
                indices.push_back(at(x, z + 1));
            }
    }
    else
    {
        for (int z = 0; z <= kDense; ++z)
            for (int x = 0; x <= kDense; ++x)
            {
                verts.push_back(GridCoord(x, kDense));
                verts.push_back(0.0f);
                verts.push_back(GridCoord(z, kDense));
            }
        auto at = [](int x, int z) { return uint16_t(z * (kDense + 1) + x); };
        for (int z = 0; z < kDense; ++z)
            for (int x = 0; x < kDense; ++x)
            {
                indices.push_back(at(x, z));
                indices.push_back(at(x + 1, z));
                indices.push_back(at(x, z + 1));
                indices.push_back(at(x + 1, z));
                indices.push_back(at(x + 1, z + 1));
                indices.push_back(at(x, z + 1));
            }
    }
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
    VriBuffer* vbuf = deviceBuf(verts.size() * sizeof(float), VriBufferUsage_VertexBuffer);
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
        vbuf, verts.data(), verts.size() * sizeof(float), VriAccess_VertexBufferRead, VriPipelineStage_VertexInput);
    app.UploadBuffer(ibuf,
                     indices.data(),
                     indices.size() * sizeof(uint16_t),
                     VriAccess_IndexBufferRead,
                     VriPipelineStage_VertexInput);
    app.EndUpload();

    // ---- layout: one CB, read by VS (fallback) or hull+domain (tess path) ----
    VriDescriptorRangeDesc range {};
    range.baseRegister   = 0;
    range.descriptorNum  = 1;
    range.descriptorType = VriDescriptorType_ConstantBuffer;
    range.shaderStages =
        VriShaderStage_Vertex | (hasTess ? (VriShaderStage_TessControl | VriShaderStage_TessEval) : 0u);
    VriDescriptorSetDesc setDesc {};
    setDesc.registerSpace = 0;
    setDesc.ranges        = &range;
    setDesc.rangeNum      = 1;
    VriPipelineLayoutDesc ld {};
    ld.descriptorSets         = &setDesc;
    ld.descriptorSetNum       = 1;
    VriPipelineLayout* layout = nullptr;
    c.CreatePipelineLayout(app.dev, &ld, &layout);

    VriVertexAttributeDesc attr {};
    attr.format      = VriFormat_RGB32_SFLOAT;
    attr.offset      = 0;
    attr.streamIndex = 0;
    VriVertexStreamDesc stream {};
    stream.stride      = sizeof(float) * 3;
    stream.bindingSlot = 0;
    stream.stepRate    = VriVertexStepRate_PerVertex;

    VriColorAttachmentDesc ca {};
    ca.format         = app.swapFormat;
    ca.colorWriteMask = VriColorWrite_RGBA;

    auto makePipeline = [&](bool wireframe) {
        VriShaderDesc sh[4] {};
        uint32_t      shaderNum = 0;
        if (hasTess)
        {
            const vriex::ExampleApp::ShaderVariants tVS {
                VRI_SHADER_BLOB(g_terrainTessSpv), nullptr, 0, VRI_SHADER_D3D12(g_terrainTessDxbcVS)};
            const vriex::ExampleApp::ShaderVariants tHS {
                VRI_SHADER_BLOB(g_terrainTessSpv), nullptr, 0, VRI_SHADER_D3D12(g_terrainTessDxbcHS)};
            const vriex::ExampleApp::ShaderVariants tDS {
                VRI_SHADER_BLOB(g_terrainTessSpv), nullptr, 0, VRI_SHADER_D3D12(g_terrainTessDxbcDS)};
            const vriex::ExampleApp::ShaderVariants tPS {
                VRI_SHADER_BLOB(g_terrainTessSpv), nullptr, 0, VRI_SHADER_D3D12(g_terrainTessDxbcPS)};
            sh[shaderNum++] = app.Shader(VriShaderStage_Vertex, "vertexMain", tVS);
            sh[shaderNum++] = app.Shader(VriShaderStage_TessControl, "hullMain", tHS);
            sh[shaderNum++] = app.Shader(VriShaderStage_TessEval, "domainMain", tDS);
            sh[shaderNum++] = app.Shader(VriShaderStage_Fragment, "fragmentMain", tPS);
        }
        else
        {
            const vriex::ExampleApp::ShaderVariants bVS {
                VRI_SHADER_BLOB(g_terrainBasicSpv),
                VRI_SHADER_BLOB(g_terrainBasicWgsl),
                VRI_SHADER_D3D12(g_terrainBasicDxbcVS),
            };
            const vriex::ExampleApp::ShaderVariants bPS {
                VRI_SHADER_BLOB(g_terrainBasicSpv),
                VRI_SHADER_BLOB(g_terrainBasicWgsl),
                VRI_SHADER_D3D12(g_terrainBasicDxbcPS),
            };
            sh[shaderNum++] = app.Shader(VriShaderStage_Vertex, "vertexMain", bVS);
            sh[shaderNum++] = app.Shader(VriShaderStage_Fragment, "fragmentMain", bPS);
        }
        VriGraphicsPipelineDesc pd {};
        pd.pipelineLayout           = layout;
        pd.shaders                  = sh;
        pd.shaderNum                = shaderNum;
        pd.vertexInput.attributes   = &attr;
        pd.vertexInput.attributeNum = 1;
        pd.vertexInput.streams      = &stream;
        pd.vertexInput.streamNum    = 1;
        pd.inputAssembly.topology   = hasTess ? VriPrimitiveTopology_PatchList : VriPrimitiveTopology_TriangleList;
        pd.tessellation.patchControlPoints = hasTess ? 4 : 0;
        pd.rasterization.polygonMode       = wireframe ? VriPolygonMode_Line : VriPolygonMode_Fill;
        pd.rasterization.cullMode          = VriCullMode_None; // terrain is viewed from above; keep it simple
        pd.rasterization.frontFace         = VriFrontFace_CounterClockwise;
        pd.rasterization.lineWidth         = 1.0f;
        pd.depthStencil.depthTest          = VRI_TRUE;
        pd.depthStencil.depthWrite         = VRI_TRUE;
        pd.depthStencil.depthCompareOp     = VriCompareOp_Less;
        pd.multisample.sampleNum           = 1;
        pd.outputMerger.colors             = &ca;
        pd.outputMerger.colorNum           = 1;
        pd.outputMerger.depthStencilFormat = app.depthFormat;
        VriPipeline* p                     = nullptr;
        if (c.CreateGraphicsPipeline(app.dev, &pd, &p) != VriResult_Success)
            app.Fail("terrain pipeline failed");
        return p;
    };
    VriPipeline* fillPipeline = makePipeline(false);
    VriPipeline* wirePipeline = makePipeline(true);

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

    static float spin = 0.6f, angle = 0.5f, heightScale = 0.55f;
    static int   tessFactor = 12;
    static bool  wireframe  = false;
    app.onUpdate            = [ustg](uint64_t) {
        angle += 0.3f * spin * app.dt;
        const float eye[3] = {std::cos(angle) * 2.6f, 1.7f, std::sin(angle) * 2.6f};
        const float ctr[3] = {0, 0, 0}, up[3] = {0, 1, 0};
        Ubo         u {};
        u.mvp = Transpose(Mul(Perspective(0.9f, float(kWidth) / float(kHeight), 0.1f, 100.0f), LookAt(eye, ctr, up)));
        u.params[0] = float(tessFactor);
        u.params[1] = heightScale;
        std::memcpy(app.c.MapBuffer(ustg, 0, sizeof(Ubo)), &u, sizeof(Ubo));
        app.c.UnmapBuffer(ustg);
    };
    app.onGui = [=] {
        if (hasTess)
            ImGui::SliderInt("tess factor", &tessFactor, 1, 32);
        else
            ImGui::Text("hasTessellation = false on %s\n(fixed-density vertex displacement)", app.apiName);
        ImGui::SliderFloat("height", &heightScale, 0.0f, 1.2f);
        ImGui::Checkbox("wireframe", &wireframe);
        ImGui::SliderFloat("orbit", &spin, 0.0f, 5.0f);
    };
    app.onPreRender = [ubo, ustg, hasTess](VriCommandBuffer* cmd) {
        VriBufferCopyDesc ucp {};
        ucp.size = sizeof(Ubo);
        app.c.CmdCopyBuffer(cmd, ubo, ustg, &ucp);
        VriBufferBarrierDesc ub {};
        ub.buffer        = ubo;
        ub.before.access = VriAccess_CopyDestinationWrite;
        ub.before.stages = VriPipelineStage_Transfer;
        ub.after.access  = VriAccess_ConstantBufferRead;
        ub.after.stages  = hasTess ? (VriPipelineStage_TessControlShader | VriPipelineStage_TessEvalShader) :
                                     VriPipelineStage_VertexShader;
        VriBarrierGroupDesc ubg {};
        ubg.buffers   = &ub;
        ubg.bufferNum = 1;
        app.c.CmdBarrier(cmd, &ubg);
    };
    app.onRecord = [=](VriCommandBuffer* cmd) {
        c.CmdSetPipeline(cmd, wireframe ? wirePipeline : fillPipeline);
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
    c.DestroyPipeline(wirePipeline);
    c.DestroyPipeline(fillPipeline);
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
