// Stencil outlines: the classic two-pass silhouette technique, exercising the STENCIL buffer
// across all backends. Pass 1 (FILL) draws each lit cube and, via the pipeline's stencil state
// (compareOp Always, passOp Replace, ref 1), stamps stencil = 1 wherever a cube covers. Pass 2
// (OUTLINE) redraws each cube slightly enlarged (the scale baked into the per-object push-constant
// model) in a flat colour with depth test off and stencil compareOp NotEqual (ref 1, writeMask 0),
// so only the rim outside the already-stamped body survives - a clean border around each cube.
// Needs a depth+STENCIL target: the example asks for it just by setting a D24S8 depthFormat before
// Init (example_app then makes the depth view + barriers Depth|Stencil). The cubes are kept apart so
// no enlarged outline reaches into a neighbour's stencil-stamped body (which would clip it).
// All through the shared scaffolding in examples/common/example_app.h (VRI_API / ?backend force one).
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "../cube/mat4.h"

#include "examples/shaders/stencil_fill_dxbc.h"    // g_stencilFillDxbcVS / PS
#include "examples/shaders/stencil_fill_spv.h"     // g_stencilFillSpv
#include "examples/shaders/stencil_fill_wgsl.h"    // g_stencilFillWgsl
#include "examples/shaders/stencil_outline_dxbc.h" // g_stencilOutlineDxbcVS / PS
#include "examples/shaders/stencil_outline_spv.h"  // g_stencilOutlineSpv
#include "examples/shaders/stencil_outline_wgsl.h" // g_stencilOutlineWgsl

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480; // width*4 is 256-aligned (D3D12 readback pitch)

    struct Vertex
    {
        float px, py, pz;
        float nx, ny, nz;
    };

    // A unit cube, 24 verts (per-face normals) + 36 indices. Standalone (no base-vertex needed).
    const Vertex kVerts[] = {
        {-0.5f, -0.5f, 0.5f, 0, 0, 1},   {0.5f, -0.5f, 0.5f, 0, 0, 1},
        {0.5f, 0.5f, 0.5f, 0, 0, 1},     {-0.5f, 0.5f, 0.5f, 0, 0, 1}, // +Z
        {0.5f, -0.5f, -0.5f, 0, 0, -1},  {-0.5f, -0.5f, -0.5f, 0, 0, -1},
        {-0.5f, 0.5f, -0.5f, 0, 0, -1},  {0.5f, 0.5f, -0.5f, 0, 0, -1}, // -Z
        {0.5f, -0.5f, 0.5f, 1, 0, 0},    {0.5f, -0.5f, -0.5f, 1, 0, 0},
        {0.5f, 0.5f, -0.5f, 1, 0, 0},    {0.5f, 0.5f, 0.5f, 1, 0, 0}, // +X
        {-0.5f, -0.5f, -0.5f, -1, 0, 0}, {-0.5f, -0.5f, 0.5f, -1, 0, 0},
        {-0.5f, 0.5f, 0.5f, -1, 0, 0},   {-0.5f, 0.5f, -0.5f, -1, 0, 0}, // -X
        {-0.5f, 0.5f, 0.5f, 0, 1, 0},    {0.5f, 0.5f, 0.5f, 0, 1, 0},
        {0.5f, 0.5f, -0.5f, 0, 1, 0},    {-0.5f, 0.5f, -0.5f, 0, 1, 0}, // +Y
        {-0.5f, -0.5f, -0.5f, 0, -1, 0}, {0.5f, -0.5f, -0.5f, 0, -1, 0},
        {0.5f, -0.5f, 0.5f, 0, -1, 0},   {-0.5f, -0.5f, 0.5f, 0, -1, 0}, // -Y
    };
    uint16_t kIndices[36];
    void     BuildIndices()
    {
        for (uint16_t f = 0; f < 6; ++f)
        {
            const uint16_t b = uint16_t(f * 4);
            uint16_t*      o = kIndices + f * 6;
            o[0]             = b;
            o[1]             = uint16_t(b + 1);
            o[2]             = uint16_t(b + 2);
            o[3]             = b;
            o[4]             = uint16_t(b + 2);
            o[5]             = uint16_t(b + 3);
        }
    }

    // Three cubes in a row, each at its own x with a per-cube spin phase. Kept >2 units apart so
    // enlarged outlines never reach a neighbour's body.
    constexpr int kCubeNum             = 3;
    const float   kCubeX[kCubeNum]     = {-2.4f, 0.0f, 2.4f};
    const float   kSpinPhase[kCubeNum] = {0.0f, 1.1f, 2.2f};

    struct SceneUbo
    {
        Mat4  viewProj;
        float lightDir[4];
        float outlineColor[4];
    };
} // namespace

int main(int, char**)
{
    BuildIndices();

    static vriex::ExampleApp app;
    app.depthFormat = VriFormat_D24_UNORM_S8_UINT; // ask for a depth+stencil target (set before Init)
    app.Init("stenciloutline", kWidth, kHeight, /*hasDepth*/ true);
    VriCoreInterface& c       = app.c;
    const bool        useWgsl = app.useWgsl, useDxbc = app.useDxbc;

    // geometry (device-local via the upload helper)
    VriBufferDesc vbd {};
    vbd.size           = sizeof(kVerts);
    vbd.usage          = VriBufferUsage_VertexBuffer | VriBufferUsage_TransferDst;
    vbd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* vbuf    = nullptr;
    c.CreateBuffer(app.dev, &vbd, &vbuf);
    VriBufferDesc ibd {};
    ibd.size           = sizeof(kIndices);
    ibd.usage          = VriBufferUsage_IndexBuffer | VriBufferUsage_TransferDst;
    ibd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* ibuf    = nullptr;
    c.CreateBuffer(app.dev, &ibd, &ibuf);

    // scene constant buffer (refreshed each frame from host staging)
    VriBufferDesc ubd {};
    ubd.size           = sizeof(SceneUbo);
    ubd.usage          = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst;
    ubd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* ubo     = nullptr;
    c.CreateBuffer(app.dev, &ubd, &ubo);
    VriBufferDesc usd {};
    usd.size           = sizeof(SceneUbo);
    usd.usage          = VriBufferUsage_TransferSrc;
    usd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* ustg    = nullptr;
    c.CreateBuffer(app.dev, &usd, &ustg);
    VriBufferViewDesc ubv {};
    ubv.buffer             = ubo;
    ubv.viewType           = VriDescriptorType_ConstantBuffer;
    ubv.offset             = 0;
    ubv.size               = sizeof(SceneUbo);
    VriDescriptor* uboView = nullptr;
    c.CreateBufferView(app.dev, &ubv, &uboView);

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

    // CB@0 (both stages) + per-object model push constant. b1: the UBO takes b0, so Slang puts the
    // push-constant cbuffer at b1 on D3D12; VK/GL/WebGPU ignore baseRegister for push constants.
    VriPushConstantDesc pc {};
    pc.baseRegister = 1;
    pc.size         = sizeof(Mat4);
    pc.shaderStages = VriShaderStage_Vertex;
    VriDescriptorRangeDesc dr {};
    dr.baseRegister   = 0;
    dr.descriptorNum  = 1;
    dr.descriptorType = VriDescriptorType_ConstantBuffer;
    dr.shaderStages   = VriShaderStage_Vertex | VriShaderStage_Fragment;
    VriDescriptorSetDesc dss {};
    dss.registerSpace = 0;
    dss.ranges        = &dr;
    dss.rangeNum      = 1;
    VriPipelineLayoutDesc pld {};
    pld.descriptorSets        = &dss;
    pld.descriptorSetNum      = 1;
    pld.pushConstants         = &pc;
    pld.pushConstantNum       = 1;
    VriPipelineLayout* layout = nullptr;
    c.CreatePipelineLayout(app.dev, &pld, &layout); // both pipelines share it

    // front==back stencil face state (cullMode None means both faces are rasterized).
    auto stencilFace = [](VriCompareOp op, VriStencilOp passOp, uint32_t writeMask) {
        VriStencilOpDesc f {};
        f.compareOp   = op;
        f.passOp      = passOp;
        f.failOp      = VriStencilOp_Keep;
        f.depthFailOp = VriStencilOp_Keep;
        f.compareMask = 0xFFu;
        f.writeMask   = writeMask;
        f.reference   = 1u;
        return f;
    };

    auto makePipeline = [&](const void*      spv,
                            size_t           spvLen,
                            const void*      wgsl,
                            size_t           wgslLen,
                            const void*      vs,
                            size_t           vsLen,
                            const void*      ps,
                            size_t           psLen,
                            bool             depthTest,
                            bool             depthWrite,
                            VriCompareOp     depthCmp,
                            VriStencilOpDesc face,
                            uint32_t         attributeNum) {
        VriShaderDesc sh[2] {};
        sh[0].stage          = VriShaderStage_Vertex;
        sh[0].entryPointName = "vertexMain";
        sh[1].stage          = VriShaderStage_Fragment;
        sh[1].entryPointName = "fragmentMain";
        if (useDxbc)
        {
            sh[0].bytecode     = vs;
            sh[0].bytecodeSize = vsLen;
            sh[1].bytecode     = ps;
            sh[1].bytecodeSize = psLen;
        }
        else
        {
            sh[0].bytecode = sh[1].bytecode = useWgsl ? wgsl : spv;
            sh[0].bytecodeSize = sh[1].bytecodeSize = useWgsl ? wgslLen : spvLen;
        }

        VriColorAttachmentDesc ca {};
        ca.format         = app.swapFormat;
        ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd {};
        pd.pipelineLayout                  = layout;
        pd.shaders                         = sh;
        pd.shaderNum                       = 2;
        pd.vertexInput.attributes          = attrs;
        pd.vertexInput.attributeNum        = attributeNum;
        pd.vertexInput.streams             = &stream;
        pd.vertexInput.streamNum           = 1;
        pd.inputAssembly.topology          = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode          = VriCullMode_None;
        pd.rasterization.frontFace         = VriFrontFace_CounterClockwise;
        pd.rasterization.lineWidth         = 1.0f;
        pd.multisample.sampleNum           = 1;
        pd.depthStencil.depthTest          = depthTest ? VRI_TRUE : VRI_FALSE;
        pd.depthStencil.depthWrite         = depthWrite ? VRI_TRUE : VRI_FALSE;
        pd.depthStencil.depthCompareOp     = depthCmp;
        pd.depthStencil.stencilTest        = VRI_TRUE;
        pd.depthStencil.front              = face;
        pd.depthStencil.back               = face;
        pd.outputMerger.colors             = &ca;
        pd.outputMerger.colorNum           = 1;
        pd.outputMerger.depthStencilFormat = app.depthFormat;
        VriPipeline* p                     = nullptr;
        if (c.CreateGraphicsPipeline(app.dev, &pd, &p) != VriResult_Success)
            app.Fail("CreateGraphicsPipeline failed");
        return p;
    };
    // FILL: depth on, stencil Always->Replace (stamp 1). OUTLINE: depth off, stencil NotEqual 1 (keep the rim, don't
    // write).
    VriPipeline* fillPipe    = makePipeline(g_stencilFillSpv,
                                         sizeof(g_stencilFillSpv),
                                         g_stencilFillWgsl,
                                         sizeof(g_stencilFillWgsl),
                                         g_stencilFillDxbcVS,
                                         sizeof(g_stencilFillDxbcVS),
                                         g_stencilFillDxbcPS,
                                         sizeof(g_stencilFillDxbcPS),
                                         /*depthTest*/ true,
                                         /*depthWrite*/ true,
                                         VriCompareOp_Less,
                                         stencilFace(VriCompareOp_Always, VriStencilOp_Replace, 0xFFu),
                                         2);
    VriPipeline* outlinePipe = makePipeline(g_stencilOutlineSpv,
                                            sizeof(g_stencilOutlineSpv),
                                            g_stencilOutlineWgsl,
                                            sizeof(g_stencilOutlineWgsl),
                                            g_stencilOutlineDxbcVS,
                                            sizeof(g_stencilOutlineDxbcVS),
                                            g_stencilOutlineDxbcPS,
                                            sizeof(g_stencilOutlineDxbcPS),
                                            /*depthTest*/ false,
                                            /*depthWrite*/ false,
                                            VriCompareOp_Less,
                                            stencilFace(VriCompareOp_NotEqual, VriStencilOp_Keep, 0x00u),
                                            useDxbc ? 2u : 1u);

    VriDescriptorPoolDesc pdsc {};
    pdsc.descriptorSetMaxNum  = 1;
    pdsc.constantBufferMaxNum = 1;
    VriDescriptorPool* pool   = nullptr;
    c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* set = nullptr;
    c.AllocateDescriptorSets(pool, layout, 0, &set, 1);
    {
        const VriDescriptor*         a[1] = {uboView};
        VriDescriptorRangeUpdateDesc u[1] {};
        u[0].descriptors   = a;
        u[0].descriptorNum = 1;
        c.UpdateDescriptorRanges(set, 0, 1, u);
    }

    // one-time geometry upload
    app.BeginUpload();
    app.UploadBuffer(vbuf, kVerts, sizeof(kVerts), VriAccess_VertexBufferRead, VriPipelineStage_VertexInput);
    app.UploadBuffer(ibuf, kIndices, sizeof(kIndices), VriAccess_IndexBufferRead, VriPipelineStage_VertexInput);
    app.EndUpload();

    static float spinSpeed = 0.6f, outlineScale = 1.08f;
    static float outlineCol[3] = {1.0f, 0.85f, 0.10f};
    static float t             = 0.0f;

    // per-cube fill model (translate * spin) and the enlarged outline model (fill * uniform scale).
    auto fillModel = [](int i, float time) {
        return Mul(Translate(kCubeX[i], 0.0f, 0.0f), Mul(RotateY(time * spinSpeed + kSpinPhase[i]), RotateX(0.5f)));
    };

    app.onUpdate = [=](uint64_t) {
        t += app.dt;
        SceneUbo    s {};
        const float eye[3] = {0.0f, 2.2f, 7.5f}, ctr[3] = {0, 0, 0}, up[3] = {0, 1, 0};
        Mat4        view  = LookAt(eye, ctr, up);
        Mat4        proj  = Perspective(0.7f, float(kWidth) / float(kHeight), 0.1f, 100.0f);
        s.viewProj        = Transpose(Mul(proj, view));
        const float ld[3] = {0.4f, 0.8f, 0.45f};
        const float len   = std::sqrt(ld[0] * ld[0] + ld[1] * ld[1] + ld[2] * ld[2]);
        s.lightDir[0]     = ld[0] / len;
        s.lightDir[1]     = ld[1] / len;
        s.lightDir[2]     = ld[2] / len;
        s.lightDir[3]     = 0;
        s.outlineColor[0] = outlineCol[0];
        s.outlineColor[1] = outlineCol[1];
        s.outlineColor[2] = outlineCol[2];
        s.outlineColor[3] = 1.0f;
        std::memcpy(c.MapBuffer(ustg, 0, sizeof(SceneUbo)), &s, sizeof(SceneUbo));
        c.UnmapBuffer(ustg);
    };

    app.onGui = [] {
        ImGui::SliderFloat("spin speed", &spinSpeed, 0.0f, 2.0f);
        ImGui::SliderFloat("outline thickness", &outlineScale, 1.0f, 1.25f);
        ImGui::ColorEdit3("outline color", outlineCol);
    };

    app.onPreRender = [=](VriCommandBuffer* cmd) {
        VriBufferCopyDesc ucp {};
        ucp.size = sizeof(SceneUbo);
        c.CmdCopyBuffer(cmd, ubo, ustg, &ucp);
        VriBufferBarrierDesc ub {};
        ub.buffer        = ubo;
        ub.before.access = VriAccess_CopyDestinationWrite;
        ub.before.stages = VriPipelineStage_Transfer;
        ub.after.access  = VriAccess_ConstantBufferRead;
        ub.after.stages  = VriPipelineStage_VertexShader | VriPipelineStage_FragmentShader;
        VriBarrierGroupDesc ubg {};
        ubg.buffers   = &ub;
        ubg.bufferNum = 1;
        c.CmdBarrier(cmd, &ubg);
    };

    app.onRecord = [=](VriCommandBuffer* cmd) {
        VriVertexBufferBinding vb {};
        vb.buffer = vbuf;
        vb.offset = 0;

        // Pass 1: fill (lit, stamps stencil = 1)
        c.CmdSetPipeline(cmd, fillPipe);
        c.CmdSetVertexBuffers(cmd, 0, &vb, 1);
        c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, set);
        for (int i = 0; i < kCubeNum; ++i)
        {
            Mat4 m = Transpose(fillModel(i, t));
            c.CmdSetConstants(cmd, 0, &m, sizeof(Mat4));
            VriDrawIndexedDesc di {};
            di.indexNum     = 36;
            di.instanceNum  = 1;
            di.baseIndex    = 0;
            di.vertexOffset = 0;
            c.CmdDrawIndexed(cmd, &di);
        }

        // Pass 2: outline (enlarged, flat color, only where stencil != 1)
        c.CmdSetPipeline(cmd, outlinePipe);
        c.CmdSetVertexBuffers(cmd, 0, &vb, 1);
        c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, set);
        for (int i = 0; i < kCubeNum; ++i)
        {
            Mat4 m = Transpose(Mul(fillModel(i, t), Scale(outlineScale)));
            c.CmdSetConstants(cmd, 0, &m, sizeof(Mat4));
            VriDrawIndexedDesc di {};
            di.indexNum     = 36;
            di.instanceNum  = 1;
            di.baseIndex    = 0;
            di.vertexOffset = 0;
            c.CmdDrawIndexed(cmd, &di);
        }
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(fillPipe);
    c.DestroyPipeline(outlinePipe);
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
