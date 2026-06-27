// Cubemap skybox + a reflective chrome sphere. A six-face CUBEMAP (VriTextureType_Cube,
// layerNum 6, sampled through a VriTextureViewType_Cube view) is generated procedurally on the
// CPU - one distinct base colour per face plus a grid + radial gradient so the environment has
// legible structure. Two pipelines share that one cubemap: the skybox draws a box wrapped around
// the eye (the view matrix's translation is stripped on the CPU) and samples by the interpolated
// look direction with depth test/write OFF (it is the background); the sphere draws on top and
// samples by reflect(eye->surface, normal), so the surroundings appear mirrored on it. The cube
// face convention (layer order +X,-X,+Y,-Y,+Z,-Z) and the direction-based sampling are identical
// on every backend - this is the first example to exercise cube textures across all five.
// All through the shared scaffolding in examples/common/example_app.h (VRI_API / ?backend force one).
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "../cube/mat4.h"

#include "tests/shaders/skybox_dxbc.h"         // g_skyboxDxbcVS / PS
#include "tests/shaders/skybox_reflect_dxbc.h" // g_skyboxReflectDxbcVS / PS
#include "tests/shaders/skybox_reflect_spv.h"  // g_skyboxReflectSpv
#include "tests/shaders/skybox_reflect_wgsl.h" // g_skyboxReflectWgsl
#include "tests/shaders/skybox_spv.h"          // g_skyboxSpv
#include "tests/shaders/skybox_wgsl.h"         // g_skyboxWgsl

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480; // width*4 is 256-aligned (D3D12 readback pitch)
    constexpr uint32_t kFace   = 256;               // cubemap face edge (square)
    constexpr int      kStacks = 24, kSlices = 48;  // reflective sphere tessellation

    struct Vertex
    {
        float px, py, pz;
        float nx, ny, nz;
    };

    // ---- geometry: a unit box (skybox) then a unit sphere (reflective object), one buffer ----
    // 8 box corners; normals unused by the skybox (it samples by position = look direction).
    const Vertex kBox[8] = {
        {-1, -1, -1, -1, -1, -1},
        {1, -1, -1, 1, -1, -1},
        {1, 1, -1, 1, 1, -1},
        {-1, 1, -1, -1, 1, -1},
        {-1, -1, 1, -1, -1, 1},
        {1, -1, 1, 1, -1, 1},
        {1, 1, 1, 1, 1, 1},
        {-1, 1, 1, -1, 1, 1},
    };
    // 36 box indices (12 triangles). Winding is irrelevant: both pipelines cull nothing.
    const uint16_t kBoxIdx[36] = {
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2, 3, 2, 6, 3, 6, 7, 0, 4, 5, 0, 5, 1,
    };

    std::vector<Vertex>   g_verts;   // box (8) + sphere
    std::vector<uint16_t> g_indices; // box (36) + sphere (base vertex baked in: WebGL2 has no baseVertex)
    uint32_t              g_sphereBaseIndex = 0, g_sphereIndexNum = 0;

    void BuildGeometry()
    {
        g_verts.assign(kBox, kBox + 8);
        g_indices.assign(kBoxIdx, kBoxIdx + 36);

        const uint16_t base = 8; // sphere verts follow the 8 box corners
        for (int i = 0; i <= kStacks; ++i)
        {
            const float theta = 3.14159265f * float(i) / float(kStacks);
            const float st = std::sin(theta), ct = std::cos(theta);
            for (int j = 0; j <= kSlices; ++j)
            {
                const float phi = 6.28318531f * float(j) / float(kSlices);
                const float x = st * std::cos(phi), y = ct, z = st * std::sin(phi);
                g_verts.push_back({x, y, z, x, y, z}); // unit sphere: position == normal
            }
        }
        g_sphereBaseIndex = 36;
        const int ring    = kSlices + 1;
        for (int i = 0; i < kStacks; ++i)
            for (int j = 0; j < kSlices; ++j)
            {
                const uint16_t a = uint16_t(base + i * ring + j), b = uint16_t(a + ring);
                g_indices.push_back(a);
                g_indices.push_back(b);
                g_indices.push_back(uint16_t(a + 1));
                g_indices.push_back(uint16_t(a + 1));
                g_indices.push_back(b);
                g_indices.push_back(uint16_t(b + 1));
            }
        g_sphereIndexNum = uint32_t(g_indices.size()) - g_sphereBaseIndex;
    }

    // Generate the six RGBA8 faces (layer order +X,-X,+Y,-Y,+Z,-Z). Each: a face base colour
    // modulated by a centre-bright radial gradient, with a bright grid so reflections read clearly.
    void BuildCubemap(std::vector<uint8_t>& out)
    {
        const float baseCol[6][3] = {
            {0.90f, 0.28f, 0.22f}, // +X warm red
            {0.20f, 0.70f, 0.90f}, // -X cyan
            {0.55f, 0.72f, 1.00f}, // +Y sky blue (up)
            {0.42f, 0.32f, 0.22f}, // -Y ground brown (down)
            {0.32f, 0.80f, 0.38f}, // +Z green
            {0.62f, 0.36f, 0.86f}, // -Z purple
        };
        out.resize(size_t(kFace) * kFace * 4 * 6);
        for (int f = 0; f < 6; ++f)
        {
            uint8_t* dst = out.data() + size_t(f) * kFace * kFace * 4;
            for (uint32_t y = 0; y < kFace; ++y)
                for (uint32_t x = 0; x < kFace; ++x)
                {
                    const float cx = (x + 0.5f) / kFace - 0.5f, cy = (y + 0.5f) / kFace - 0.5f;
                    const float r = std::sqrt(cx * cx + cy * cy);
                    float       g = 0.55f + 0.6f * (1.0f - r * 1.5f); // brighter toward the centre
                    if (g < 0.25f)
                        g = 0.25f;
                    if (g > 1.0f)
                        g = 1.0f;
                    const bool grid = (x % 32 < 2) || (y % 32 < 2);
                    uint8_t*   p    = dst + (size_t(y) * kFace + x) * 4;
                    for (int ch = 0; ch < 3; ++ch)
                    {
                        float v = baseCol[f][ch] * g;
                        if (grid)
                            v = v * 0.4f + 0.6f; // whiten grid lines
                        if (v > 1.0f)
                            v = 1.0f;
                        p[ch] = uint8_t(v * 255.0f + 0.5f);
                    }
                    p[3] = 255;
                }
        }
    }

    struct SkyUbo
    {
        Mat4 viewProj;
    };
    struct ReflectUbo
    {
        Mat4  viewProj;
        Mat4  model;
        float cameraPos[4];
    };
} // namespace

int main(int, char**)
{
    BuildGeometry();
    std::vector<uint8_t> faces;
    BuildCubemap(faces);

    static vriex::ExampleApp app;
    app.Init("cubemap", kWidth, kHeight, /*hasDepth*/ true);
    VriCoreInterface& c       = app.c;
    const bool        useWgsl = app.useWgsl, useDxbc = app.useDxbc;

    // ---- geometry (device-local via the upload helper) ----
    VriBufferDesc vbd {};
    vbd.size           = g_verts.size() * sizeof(Vertex);
    vbd.usage          = VriBufferUsage_VertexBuffer | VriBufferUsage_TransferDst;
    vbd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* vbuf    = nullptr;
    c.CreateBuffer(app.dev, &vbd, &vbuf);
    VriBufferDesc ibd {};
    ibd.size           = g_indices.size() * sizeof(uint16_t);
    ibd.usage          = VriBufferUsage_IndexBuffer | VriBufferUsage_TransferDst;
    ibd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* ibuf    = nullptr;
    c.CreateBuffer(app.dev, &ibd, &ibuf);

    // ---- the cubemap: 6 square faces, sampled through a Cube view ----
    VriTextureDesc td {};
    td.type             = VriTextureType_Cube;
    td.format           = VriFormat_RGBA8_UNORM;
    td.width            = kFace;
    td.height           = kFace;
    td.depth            = 1;
    td.mipNum           = 1;
    td.layerNum         = 6;
    td.sampleNum        = 1;
    td.usage            = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst;
    td.memoryLocation   = VriMemoryLocation_Device;
    VriTexture* cubeTex = nullptr;
    if (c.CreateTexture(app.dev, &td, &cubeTex) != VriResult_Success)
        app.Fail("CreateTexture (cubemap) failed");
    VriTextureViewDesc cvd {};
    cvd.texture             = cubeTex;
    cvd.viewType            = VriTextureViewType_Cube;
    cvd.format              = VriFormat_Unknown;
    cvd.aspect              = VriImageAspect_Color;
    cvd.layerNum            = 6;
    VriDescriptor* cubeView = nullptr;
    if (c.CreateTextureView(app.dev, &cvd, &cubeView) != VriResult_Success)
        app.Fail("CreateTextureView (cube) failed");

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

    // ---- two constant buffers (skybox: viewProj; reflect: viewProj + model + cameraPos) ----
    auto makeUbo = [&](uint64_t size, VriBuffer*& ubo, VriBuffer*& stg, VriDescriptor*& view) {
        VriBufferDesc ud {};
        ud.size           = size;
        ud.usage          = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst;
        ud.memoryLocation = VriMemoryLocation_Device;
        c.CreateBuffer(app.dev, &ud, &ubo);
        VriBufferDesc sd {};
        sd.size           = size;
        sd.usage          = VriBufferUsage_TransferSrc;
        sd.memoryLocation = VriMemoryLocation_HostUpload;
        c.CreateBuffer(app.dev, &sd, &stg);
        VriBufferViewDesc bv {};
        bv.buffer   = ubo;
        bv.viewType = VriDescriptorType_ConstantBuffer;
        bv.offset   = 0;
        bv.size     = size;
        c.CreateBufferView(app.dev, &bv, &view);
    };
    VriBuffer *    skyUbo = nullptr, *skyStg = nullptr, *refUbo = nullptr, *refStg = nullptr;
    VriDescriptor *skyUboView = nullptr, *refUboView = nullptr;
    makeUbo(sizeof(SkyUbo), skyUbo, skyStg, skyUboView);
    makeUbo(sizeof(ReflectUbo), refUbo, refStg, refUboView);

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

    // Shared layout shape: CB@0 + cubemap Texture@1 (Cube view) + Sampler@2. WebGPU bakes the
    // Cube view dimension into the bind-group layout, so the texture range must declare it.
    VriDescriptorRangeDesc r[3] {};
    r[0].baseRegister   = 0;
    r[0].descriptorNum  = 1;
    r[0].descriptorType = VriDescriptorType_ConstantBuffer;
    r[0].shaderStages   = VriShaderStage_Vertex | VriShaderStage_Fragment;
    r[1].baseRegister   = 1;
    r[1].descriptorNum  = 1;
    r[1].descriptorType = VriDescriptorType_Texture;
    r[1].shaderStages   = VriShaderStage_Fragment;
    r[1].viewType       = VriTextureViewType_Cube;
    r[2].baseRegister   = 2;
    r[2].descriptorNum  = 1;
    r[2].descriptorType = VriDescriptorType_Sampler;
    r[2].shaderStages   = VriShaderStage_Fragment;
    VriDescriptorSetDesc setDesc {};
    setDesc.registerSpace = 0;
    setDesc.ranges        = r;
    setDesc.rangeNum      = 3;
    VriPipelineLayoutDesc pld {};
    pld.descriptorSets        = &setDesc;
    pld.descriptorSetNum      = 1;
    VriPipelineLayout* layout = nullptr;
    c.CreatePipelineLayout(app.dev, &pld, &layout); // both pipelines share it

    auto makePipeline = [&](const void* spv,
                            size_t      spvLen,
                            const void* wgsl,
                            size_t      wgslLen,
                            const void* vs,
                            size_t      vsLen,
                            const void* ps,
                            size_t      psLen,
                            bool        depthOn,
                            VriCullMode cull,
                            uint32_t    attributeNum) {
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
        pd.rasterization.cullMode          = cull;
        pd.rasterization.frontFace         = VriFrontFace_CounterClockwise;
        pd.rasterization.lineWidth         = 1.0f;
        pd.multisample.sampleNum           = 1;
        pd.depthStencil.depthTest          = depthOn ? VRI_TRUE : VRI_FALSE;
        pd.depthStencil.depthWrite         = depthOn ? VRI_TRUE : VRI_FALSE;
        pd.depthStencil.depthCompareOp     = VriCompareOp_Less;
        ca.format                          = app.swapFormat;
        pd.outputMerger.colors             = &ca;
        pd.outputMerger.colorNum           = 1;
        pd.outputMerger.depthStencilFormat = app.depthFormat; // pass has a depth attachment
        VriPipeline* p                     = nullptr;
        if (c.CreateGraphicsPipeline(app.dev, &pd, &p) != VriResult_Success)
            app.Fail("CreateGraphicsPipeline failed");
        return p;
    };
    // skybox: viewed from inside the box, so cull FRONT faces (keep the inward-facing back). sphere:
    // solid, so cull BACK faces - drawing both faces (None) leaves the inward-normal back faces to
    // z-fight the front on D3D12, speckling the reflection.
    VriPipeline* skyPipe = makePipeline(g_skyboxSpv,
                                        sizeof(g_skyboxSpv),
                                        g_skyboxWgsl,
                                        sizeof(g_skyboxWgsl),
                                        g_skyboxDxbcVS,
                                        sizeof(g_skyboxDxbcVS),
                                        g_skyboxDxbcPS,
                                        sizeof(g_skyboxDxbcPS),
                                        /*depthOn*/ false,
                                        VriCullMode_None,
                                        useDxbc ? 2u : 1u);
    VriPipeline* refPipe = makePipeline(g_skyboxReflectSpv,
                                        sizeof(g_skyboxReflectSpv),
                                        g_skyboxReflectWgsl,
                                        sizeof(g_skyboxReflectWgsl),
                                        g_skyboxReflectDxbcVS,
                                        sizeof(g_skyboxReflectDxbcVS),
                                        g_skyboxReflectDxbcPS,
                                        sizeof(g_skyboxReflectDxbcPS),
                                        /*depthOn*/ true,
                                        VriCullMode_Back,
                                        useDxbc ? 2u : 1u);

    // ---- descriptor sets (one per pipeline; both bind the same cubemap + sampler) ----
    VriDescriptorPoolDesc pdsc {};
    pdsc.descriptorSetMaxNum  = 2;
    pdsc.constantBufferMaxNum = 2;
    pdsc.textureMaxNum        = 2;
    pdsc.samplerMaxNum        = 2;
    VriDescriptorPool* pool   = nullptr;
    c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* skySet = nullptr;
    c.AllocateDescriptorSets(pool, layout, 0, &skySet, 1);
    VriDescriptorSet* refSet = nullptr;
    c.AllocateDescriptorSets(pool, layout, 0, &refSet, 1);
    auto writeSet = [&](VriDescriptorSet* set, VriDescriptor* uboView) {
        const VriDescriptor*         a[1] = {uboView};
        const VriDescriptor*         b[1] = {cubeView};
        const VriDescriptor*         s[1] = {sampler};
        VriDescriptorRangeUpdateDesc u[3] {};
        u[0].descriptors   = a;
        u[0].descriptorNum = 1;
        u[1].descriptors   = b;
        u[1].descriptorNum = 1;
        u[2].descriptors   = s;
        u[2].descriptorNum = 1;
        c.UpdateDescriptorRanges(set, 0, 3, u);
    };
    writeSet(skySet, skyUboView);
    writeSet(refSet, refUboView);

    // ---- one-time uploads: geometry + the six cube faces ----
    app.BeginUpload();
    app.UploadBuffer(vbuf,
                     g_verts.data(),
                     g_verts.size() * sizeof(Vertex),
                     VriAccess_VertexBufferRead,
                     VriPipelineStage_VertexInput);
    app.UploadBuffer(ibuf,
                     g_indices.data(),
                     g_indices.size() * sizeof(uint16_t),
                     VriAccess_IndexBufferRead,
                     VriPipelineStage_VertexInput);
    app.UploadTexture(cubeTex, faces.data(), kFace, kFace, 6, kFace * kFace * 4);
    app.EndUpload();

    // ---- per-frame camera (orbits) + reflective sphere (spins) ----
    static float orbitSpeed = 0.4f, spinSpeed = 0.25f, sphereRadius = 1.3f;
    static bool  showSphere = true;
    static float t          = 0.0f;

    app.onUpdate = [=](uint64_t) {
        t += app.dt;
        const float up[3] = {0, 1, 0}, ctr[3] = {0, 0, 0};
        const float ang    = t * orbitSpeed;
        const float eye[3] = {std::cos(ang) * 4.0f, 1.6f, std::sin(ang) * 4.0f};
        Mat4        view   = LookAt(eye, ctr, up);
        Mat4        proj   = Perspective(0.8f, float(kWidth) / float(kHeight), 0.1f, 100.0f);

        // skybox: same view with translation removed so the box stays centred on the eye
        Mat4 viewNoT  = view;
        viewNoT.m[12] = viewNoT.m[13] = viewNoT.m[14] = 0.0f;
        SkyUbo sky {};
        sky.viewProj = Transpose(Mul(proj, viewNoT));
        std::memcpy(c.MapBuffer(skyStg, 0, sizeof(SkyUbo)), &sky, sizeof(SkyUbo));
        c.UnmapBuffer(skyStg);

        ReflectUbo ref {};
        ref.viewProj     = Transpose(Mul(proj, view));
        ref.model        = Transpose(Mul(RotateY(t * spinSpeed), Scale(sphereRadius)));
        ref.cameraPos[0] = eye[0];
        ref.cameraPos[1] = eye[1];
        ref.cameraPos[2] = eye[2];
        ref.cameraPos[3] = 1.0f;
        std::memcpy(c.MapBuffer(refStg, 0, sizeof(ReflectUbo)), &ref, sizeof(ReflectUbo));
        c.UnmapBuffer(refStg);
    };

    app.onGui = [] {
        ImGui::SliderFloat("orbit speed", &orbitSpeed, 0.0f, 1.5f);
        ImGui::SliderFloat("sphere spin", &spinSpeed, 0.0f, 1.5f);
        ImGui::SliderFloat("sphere size", &sphereRadius, 0.5f, 2.0f);
        ImGui::Checkbox("show sphere", &showSphere);
    };

    // refresh both constant buffers (staging -> device) before the render pass
    app.onPreRender = [=](VriCommandBuffer* cmd) {
        VriBuffer* ubos[2]  = {skyUbo, refUbo};
        VriBuffer* stgs[2]  = {skyStg, refStg};
        uint64_t   sizes[2] = {sizeof(SkyUbo), sizeof(ReflectUbo)};
        for (int i = 0; i < 2; ++i)
        {
            VriBufferCopyDesc cp {};
            cp.size = sizes[i];
            c.CmdCopyBuffer(cmd, ubos[i], stgs[i], &cp);
            VriBufferBarrierDesc bb {};
            bb.buffer        = ubos[i];
            bb.before.access = VriAccess_CopyDestinationWrite;
            bb.before.stages = VriPipelineStage_Transfer;
            bb.after.access  = VriAccess_ConstantBufferRead;
            bb.after.stages  = VriPipelineStage_VertexShader | VriPipelineStage_FragmentShader;
            VriBarrierGroupDesc g {};
            g.buffers   = &bb;
            g.bufferNum = 1;
            c.CmdBarrier(cmd, &g);
        }
    };

    app.onRecord = [=](VriCommandBuffer* cmd) {
        VriVertexBufferBinding vb {};
        vb.buffer = vbuf;
        vb.offset = 0;

        // skybox first (depth off -> fills the background), reflective sphere on top
        c.CmdSetPipeline(cmd, skyPipe);
        c.CmdSetVertexBuffers(cmd, 0, &vb, 1);
        c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, skySet);
        VriDrawIndexedDesc sky {};
        sky.indexNum     = 36;
        sky.instanceNum  = 1;
        sky.baseIndex    = 0;
        sky.vertexOffset = 0;
        c.CmdDrawIndexed(cmd, &sky);

        if (showSphere)
        {
            c.CmdSetPipeline(cmd, refPipe);
            c.CmdSetVertexBuffers(cmd, 0, &vb, 1);
            c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);
            c.CmdSetPipelineLayout(cmd, layout);
            c.CmdSetDescriptorSet(cmd, 0, refSet);
            VriDrawIndexedDesc s {};
            s.indexNum     = g_sphereIndexNum;
            s.instanceNum  = 1;
            s.baseIndex    = g_sphereBaseIndex;
            s.vertexOffset = 0;
            c.CmdDrawIndexed(cmd, &s);
        }
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(skyPipe);
    c.DestroyPipeline(refPipe);
    c.DestroyPipelineLayout(layout);
    c.DestroyDescriptor(sampler);
    c.DestroyDescriptor(cubeView);
    c.DestroyTexture(cubeTex);
    c.DestroyDescriptor(skyUboView);
    c.DestroyDescriptor(refUboView);
    c.DestroyBuffer(skyStg);
    c.DestroyBuffer(skyUbo);
    c.DestroyBuffer(refStg);
    c.DestroyBuffer(refUbo);
    c.DestroyBuffer(ibuf);
    c.DestroyBuffer(vbuf);
    app.Shutdown();
#endif
    return 0;
}
