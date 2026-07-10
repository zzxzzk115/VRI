// GPU-driven rendering: 4096 cubes are frustum-culled ON THE GPU each frame. A compute pass
// (shaders/examples/cull.slang) sphere-tests every instance, compacts the survivors into a
// per-instance vertex stream, appends one indirect draw record each (atomic counter), and
// CmdDrawIndexedIndirectCount consumes records + counter - the CPU never knows how many draws
// happen. CmdClearStorageBuffer zeroes the counter each frame (buffer copy fallback where
// hasClearStorageBuffer is false), and CmdClearStorageTexture clears a top-down minimap the
// culling pass plots into (green = drawn, grey = culled; shown via ImGui::Image). Where
// hasDrawIndirectCount is false the CS instead keeps one record slot per instance, zeroing
// indexNum on the culled ones, and a plain CmdDrawIndexedIndirect(drawNum = total) is issued.
// The classic proof: freeze culling, keep orbiting - a cube-shaped hole hangs in the view.
// Shared scaffolding: examples/common/example_app.h.
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "../cube/mat4.h"

#include "shaders/examples/cull_dxbc.h"            // g_cullDxbcCS (D3D12)
#include "shaders/examples/cull_spv.h"             // g_cullSpv (Vulkan + OpenGL; no WGSL variant -
                                                   // WebGPU desktop draws everything, uncullled)
#include "shaders/examples/gpudriven_scene_dxbc.h" // g_gpudrivenSceneDxbcVS / PS
#include "shaders/examples/gpudriven_scene_spv.h"  // g_gpudrivenSceneSpv
#include "shaders/examples/gpudriven_scene_wgsl.h" // g_gpudrivenSceneWgsl

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480;
    constexpr int      kGrid         = 64; // kGrid^2 = 4096 instances
    constexpr uint32_t kInstances    = kGrid * kGrid;
    constexpr float    kSpacing      = 3.0f;
    constexpr float    kExtent       = kGrid * kSpacing * 0.5f;
    constexpr uint32_t kMapSize      = 64; // top-down minimap
    constexpr uint32_t kRecordU32    = 5;  // one VriDrawIndexedDesc record = 5 uint32
    constexpr uint32_t kCubeIndexNum = 36;

    struct Instance
    {
        float posScale[4]; // xyz + uniform scale
        float color[4];
    };
    struct CameraUbo
    {
        Mat4 viewProj;
    };
    struct CullUbo
    {
        Mat4  cullViewProj;
        float misc0[4]; // total, compact mode, minimap size, field extent
        float misc1[4]; // bounding radius at scale 1
    };
    struct Vertex
    {
        float px, py, pz;
        float nx, ny, nz;
    };

    void BuildCube(std::vector<Vertex>& verts, std::vector<uint16_t>& indices)
    {
        const float n[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        for (int f = 0; f < 6; ++f)
        {
            const float* fn   = n[f];
            float        u[3] = {fn[1], fn[2], fn[0]}; // any tangent
            float        v[3] = {fn[2] * u[1] - fn[1] * u[2], fn[0] * u[2] - fn[2] * u[0], fn[1] * u[0] - fn[0] * u[1]};
            const float  corner[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
            const auto   base         = uint16_t(verts.size());
            for (auto& cxy : corner)
            {
                Vertex vert {};
                vert.px = fn[0] + u[0] * cxy[0] + v[0] * cxy[1];
                vert.py = fn[1] + u[1] * cxy[0] + v[1] * cxy[1];
                vert.pz = fn[2] + u[2] * cxy[0] + v[2] * cxy[1];
                vert.nx = fn[0];
                vert.ny = fn[1];
                vert.nz = fn[2];
                verts.push_back(vert);
            }
            const uint16_t quad[6] = {0, 1, 2, 0, 2, 3};
            for (uint16_t q : quad)
                indices.push_back(uint16_t(base + q));
        }
    }

    std::vector<Instance> BuildInstances()
    {
        std::vector<Instance> out;
        out.reserve(kInstances);
        for (int z = 0; z < kGrid; ++z)
            for (int x = 0; x < kGrid; ++x)
            {
                const uint32_t h = uint32_t(x * 374761393 + z * 668265263) ^ 0x5bf03635u;
                Instance       i {};
                i.posScale[0]   = (float(x) + 0.5f - kGrid * 0.5f) * kSpacing;
                i.posScale[1]   = 0.0f;
                i.posScale[2]   = (float(z) + 0.5f - kGrid * 0.5f) * kSpacing;
                i.posScale[3]   = 0.35f + float(h % 400) * 0.001f; // 0.35 .. 0.75
                const float hue = float((h >> 8) % 360) / 360.0f;
                i.color[0]      = 0.5f + 0.5f * std::sin(hue * 6.2832f);
                i.color[1]      = 0.5f + 0.5f * std::sin(hue * 6.2832f + 2.094f);
                i.color[2]      = 0.5f + 0.5f * std::sin(hue * 6.2832f + 4.189f);
                i.color[3]      = 1.0f;
                out.push_back(i);
            }
        return out;
    }

    bool g_mapInit = false, g_bufInit = false;
} // namespace

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.Init("gpudriven", kWidth, kHeight, /*hasDepth*/ true);
    VriCoreInterface&    c  = app.c;
    const VriDeviceDesc* dd = c.GetDeviceDesc(app.dev);

    // GPU culling needs compute AND a cull-shader variant for this backend: SPIR-V (Vulkan/GL)
    // or DXBC (D3D12). There is no WGSL variant (Slang can't emit this CS for WGSL), so on the
    // desktop WebGPU backend - which also lacks hasDrawIndirectCount - everything draws unculled.
    const bool canCull     = dd->hasComputeShader == VRI_TRUE && !app.useWgsl;
    const bool hasCount    = canCull && dd->hasDrawIndirectCount == VRI_TRUE;
    const bool hasClearBuf = dd->hasClearStorageBuffer == VRI_TRUE;
    const bool hasClearTex = canCull && dd->hasClearStorageTexture == VRI_TRUE;

    // ---- geometry + instances ----
    std::vector<Vertex>   verts;
    std::vector<uint16_t> indices;
    BuildCube(verts, indices);
    const std::vector<Instance> instances = BuildInstances();

    auto makeBuf = [&](uint64_t size, VriBufferUsageFlags usage, VriMemoryLocation loc) {
        VriBufferDesc bd {};
        bd.size           = size;
        bd.usage          = usage;
        bd.memoryLocation = loc;
        VriBuffer* b      = nullptr;
        if (c.CreateBuffer(app.dev, &bd, &b) != VriResult_Success)
            app.Fail("CreateBuffer failed");
        return b;
    };
    VriBuffer* vbuf    = makeBuf(verts.size() * sizeof(Vertex),
                              VriBufferUsage_VertexBuffer | VriBufferUsage_TransferDst,
                              VriMemoryLocation_Device);
    VriBuffer* ibuf    = makeBuf(indices.size() * sizeof(uint16_t),
                              VriBufferUsage_IndexBuffer | VriBufferUsage_TransferDst,
                              VriMemoryLocation_Device);
    VriBuffer* instBuf = makeBuf(kInstances * sizeof(Instance),
                                 VriBufferUsage_StorageBuffer | VriBufferUsage_TransferDst,
                                 VriMemoryLocation_Device);
    // survivors: written as a storage buffer by the CS, read as the per-instance vertex stream
    VriBuffer* visBuf        = makeBuf(kInstances * sizeof(Instance),
                                VriBufferUsage_StorageBuffer | VriBufferUsage_VertexBuffer,
                                VriMemoryLocation_Device);
    VriBuffer* recBuf        = makeBuf(uint64_t(kInstances) * kRecordU32 * sizeof(uint32_t),
                                VriBufferUsage_StorageBuffer | VriBufferUsage_IndirectBuffer,
                                VriMemoryLocation_Device);
    VriBuffer* countBuf      = makeBuf(sizeof(uint32_t),
                                  VriBufferUsage_StorageBuffer | VriBufferUsage_IndirectBuffer |
                                      VriBufferUsage_TransferDst | VriBufferUsage_TransferSrc,
                                  VriMemoryLocation_Device);
    VriBuffer* countReadback = makeBuf(sizeof(uint32_t), VriBufferUsage_TransferDst, VriMemoryLocation_HostReadback);
    VriBuffer* zeroBuf       = nullptr; // counter-clear fallback where CmdClearStorageBuffer is unavailable
    if (!hasClearBuf)
    {
        zeroBuf = makeBuf(sizeof(uint32_t), VriBufferUsage_TransferSrc, VriMemoryLocation_HostUpload);
        *static_cast<uint32_t*>(c.MapBuffer(zeroBuf, 0, sizeof(uint32_t))) = 0;
        c.UnmapBuffer(zeroBuf);
    }

    VriBuffer* camUbo = makeBuf(
        sizeof(CameraUbo), VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst, VriMemoryLocation_Device);
    VriBuffer* cullUbo =
        makeBuf(sizeof(CullUbo), VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst, VriMemoryLocation_Device);
    VriBuffer* camStg  = makeBuf(sizeof(CameraUbo), VriBufferUsage_TransferSrc, VriMemoryLocation_HostUpload);
    VriBuffer* cullStg = makeBuf(sizeof(CullUbo), VriBufferUsage_TransferSrc, VriMemoryLocation_HostUpload);

    auto makeCbView = [&](VriBuffer* b, uint64_t size) {
        VriBufferViewDesc vd {};
        vd.buffer        = b;
        vd.viewType      = VriDescriptorType_ConstantBuffer;
        vd.offset        = 0;
        vd.size          = size;
        VriDescriptor* v = nullptr;
        c.CreateBufferView(app.dev, &vd, &v);
        return v;
    };
    auto makeStorageView = [&](VriBuffer* b, VriDescriptorType type, uint64_t size) {
        VriBufferViewDesc vd {};
        vd.buffer        = b;
        vd.viewType      = type;
        vd.offset        = 0;
        vd.size          = size;
        VriDescriptor* v = nullptr;
        c.CreateBufferView(app.dev, &vd, &v);
        return v;
    };
    VriDescriptor* camView  = makeCbView(camUbo, sizeof(CameraUbo));
    VriDescriptor* cullView = makeCbView(cullUbo, sizeof(CullUbo));
    VriDescriptor* instView =
        makeStorageView(instBuf, VriDescriptorType_StructuredBuffer, kInstances * sizeof(Instance));
    VriDescriptor* visView = makeStorageView(visBuf, VriDescriptorType_StorageBuffer, kInstances * sizeof(Instance));
    VriDescriptor* recView =
        makeStorageView(recBuf, VriDescriptorType_StorageBuffer, uint64_t(kInstances) * kRecordU32 * sizeof(uint32_t));
    VriDescriptor* countView = makeStorageView(countBuf, VriDescriptorType_StorageBuffer, sizeof(uint32_t));

    // ---- top-down minimap (storage texture the CS plots into) ----
    VriTexture*    mapTex  = nullptr;
    VriDescriptor* mapView = nullptr;
    if (hasClearTex)
    {
        VriTextureDesc td {};
        td.type      = VriTextureType_2D;
        td.format    = VriFormat_RGBA8_UNORM;
        td.width     = kMapSize;
        td.height    = kMapSize;
        td.depth     = 1;
        td.mipNum    = 1;
        td.layerNum  = 1;
        td.sampleNum = 1;
        td.usage = VriTextureUsage_ShaderResourceStorage | VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst;
        td.memoryLocation = VriMemoryLocation_Device;
        if (c.CreateTexture(app.dev, &td, &mapTex) != VriResult_Success)
            app.Fail("minimap CreateTexture failed");
        VriTextureViewDesc vd {};
        vd.texture  = mapTex;
        vd.viewType = VriTextureViewType_2D;
        vd.format   = VriFormat_Unknown;
        vd.aspect   = VriImageAspect_Color;
        c.CreateTextureView(app.dev, &vd, &mapView);
    }

    // ---- uploads ----
    app.BeginUpload();
    app.UploadBuffer(
        vbuf, verts.data(), verts.size() * sizeof(Vertex), VriAccess_VertexBufferRead, VriPipelineStage_VertexInput);
    app.UploadBuffer(ibuf,
                     indices.data(),
                     indices.size() * sizeof(uint16_t),
                     VriAccess_IndexBufferRead,
                     VriPipelineStage_VertexInput);
    app.UploadBuffer(instBuf,
                     instances.data(),
                     kInstances * sizeof(Instance),
                     VriAccess_ShaderResourceRead,
                     VriPipelineStage_ComputeShader);
    if (!canCull) // no culling pass: the per-instance stream is just all instances, uploaded once
        app.UploadBuffer(visBuf,
                         instances.data(),
                         kInstances * sizeof(Instance),
                         VriAccess_VertexBufferRead,
                         VriPipelineStage_VertexInput);
    app.EndUpload();

    // ---- culling compute pipeline (only where a cull-shader variant exists) ----
    VriPipelineLayout* cullLayout   = nullptr;
    VriPipeline*       cullPipeline = nullptr;
    if (canCull)
    {
        VriDescriptorRangeDesc cr[6] {};
        for (uint32_t i = 0; i < 6; ++i)
        {
            cr[i].baseRegister  = i;
            cr[i].descriptorNum = 1;
            cr[i].shaderStages  = VriShaderStage_Compute;
        }
        cr[0].descriptorType = VriDescriptorType_ConstantBuffer;
        cr[1].descriptorType = VriDescriptorType_StructuredBuffer;
        cr[2].descriptorType = VriDescriptorType_StorageBuffer;
        cr[3].descriptorType = VriDescriptorType_StorageBuffer;
        cr[4].descriptorType = VriDescriptorType_StorageBuffer;
        cr[5].descriptorType = VriDescriptorType_StorageTexture;
        VriDescriptorSetDesc csd {};
        csd.registerSpace = 0;
        csd.ranges        = cr;
        csd.rangeNum      = hasClearTex ? 6u : 5u; // no minimap -> no storage-texture binding
        VriPipelineLayoutDesc cld {};
        cld.descriptorSets   = &csd;
        cld.descriptorSetNum = 1;
        c.CreatePipelineLayout(app.dev, &cld, &cullLayout);

        const vriex::ExampleApp::ShaderVariants cullCS {
            VRI_SHADER_BLOB(g_cullSpv), nullptr, 0, VRI_SHADER_D3D12(g_cullDxbcCS)};
        VriComputePipelineDesc cpd {};
        cpd.pipelineLayout = cullLayout;
        cpd.shader         = app.Shader(VriShaderStage_Compute, "computeMain", cullCS);
        if (c.CreateComputePipeline(app.dev, &cpd, &cullPipeline) != VriResult_Success)
            app.Fail("cull pipeline failed");
    }

    // ---- scene pipeline: per-vertex stream 0 + per-instance stream 1 (the compacted buffer) ----
    VriDescriptorRangeDesc sr {};
    sr.baseRegister   = 0;
    sr.descriptorNum  = 1;
    sr.descriptorType = VriDescriptorType_ConstantBuffer;
    sr.shaderStages   = VriShaderStage_Vertex;
    VriDescriptorSetDesc ssd {};
    ssd.registerSpace = 0;
    ssd.ranges        = &sr;
    ssd.rangeNum      = 1;
    VriPipelineLayoutDesc sld {};
    sld.descriptorSets             = &ssd;
    sld.descriptorSetNum           = 1;
    VriPipelineLayout* sceneLayout = nullptr;
    c.CreatePipelineLayout(app.dev, &sld, &sceneLayout);

    const vriex::ExampleApp::ShaderVariants sceneVS {VRI_SHADER_BLOB(g_gpudrivenSceneSpv),
                                                     VRI_SHADER_BLOB(g_gpudrivenSceneWgsl),
                                                     VRI_SHADER_D3D12(g_gpudrivenSceneDxbcVS)};
    const vriex::ExampleApp::ShaderVariants scenePS {VRI_SHADER_BLOB(g_gpudrivenSceneSpv),
                                                     VRI_SHADER_BLOB(g_gpudrivenSceneWgsl),
                                                     VRI_SHADER_D3D12(g_gpudrivenSceneDxbcPS)};
    VriShaderDesc                           sh[2] = {
        app.Shader(VriShaderStage_Vertex, "vertexMain", sceneVS),
        app.Shader(VriShaderStage_Fragment, "fragmentMain", scenePS),
    };

    VriVertexAttributeDesc attrs[4] {};
    attrs[0].format      = VriFormat_RGB32_SFLOAT;
    attrs[0].offset      = 0;
    attrs[0].streamIndex = 0; // pos
    attrs[1].format      = VriFormat_RGB32_SFLOAT;
    attrs[1].offset      = 12;
    attrs[1].streamIndex = 0; // normal
    attrs[2].format      = VriFormat_RGBA32_SFLOAT;
    attrs[2].offset      = 0;
    attrs[2].streamIndex = 1; // posScale (per instance)
    attrs[3].format      = VriFormat_RGBA32_SFLOAT;
    attrs[3].offset      = 16;
    attrs[3].streamIndex = 1; // color (per instance)
    VriVertexStreamDesc streams[2] {};
    streams[0].stride      = sizeof(Vertex);
    streams[0].bindingSlot = 0;
    streams[0].stepRate    = VriVertexStepRate_PerVertex;
    streams[1].stride      = sizeof(Instance);
    streams[1].bindingSlot = 1;
    streams[1].stepRate    = VriVertexStepRate_PerInstance;

    VriColorAttachmentDesc ca {};
    ca.format         = app.swapFormat;
    ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd {};
    pd.pipelineLayout                  = sceneLayout;
    pd.shaders                         = sh;
    pd.shaderNum                       = 2;
    pd.vertexInput.attributes          = attrs;
    pd.vertexInput.attributeNum        = 4;
    pd.vertexInput.streams             = streams;
    pd.vertexInput.streamNum           = 2;
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

    // ---- descriptor sets ----
    VriDescriptorPoolDesc pdsc {};
    pdsc.descriptorSetMaxNum    = 2;
    pdsc.constantBufferMaxNum   = 2;
    pdsc.structuredBufferMaxNum = 1;
    pdsc.storageBufferMaxNum    = 3;
    pdsc.storageTextureMaxNum   = 1;
    VriDescriptorPool* pool     = nullptr;
    c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* cullSet  = nullptr;
    VriDescriptorSet* sceneSet = nullptr;
    c.AllocateDescriptorSets(pool, sceneLayout, 0, &sceneSet, 1);
    {
        const VriDescriptor*         s0[1] = {camView};
        VriDescriptorRangeUpdateDesc su {};
        su.descriptors   = s0;
        su.descriptorNum = 1;
        c.UpdateDescriptorRanges(sceneSet, 0, 1, &su);
    }
    if (canCull)
    {
        c.AllocateDescriptorSets(pool, cullLayout, 0, &cullSet, 1);
        const VriDescriptor*         d0[1] = {cullView};
        const VriDescriptor*         d1[1] = {instView};
        const VriDescriptor*         d2[1] = {recView};
        const VriDescriptor*         d3[1] = {countView};
        const VriDescriptor*         d4[1] = {visView};
        const VriDescriptor*         d5[1] = {mapView};
        VriDescriptorRangeUpdateDesc u[6] {};
        u[0].descriptors   = d0;
        u[0].descriptorNum = 1;
        u[1].descriptors   = d1;
        u[1].descriptorNum = 1;
        u[2].descriptors   = d2;
        u[2].descriptorNum = 1;
        u[3].descriptors   = d3;
        u[3].descriptorNum = 1;
        u[4].descriptors   = d4;
        u[4].descriptorNum = 1;
        u[5].descriptors   = d5;
        u[5].descriptorNum = 1;
        c.UpdateDescriptorRanges(cullSet, 0, hasClearTex ? 6u : 5u, u);
    }

    // ---- per-frame state ----
    static float    orbit = 0.6f, angle = 0.4f;
    static bool     freezeCull = false;
    static Mat4     frozenCullVP {};
    static bool     haveFrozen     = false;
    static uint32_t visibleCount   = 0;
    static bool     countAvailable = false;

    app.onUpdate = [camStg, cullStg, countReadback](uint64_t frame) {
        angle += 0.25f * orbit * app.dt;
        const float eye[3] = {std::cos(angle) * 55.0f, 26.0f, std::sin(angle) * 55.0f};
        const float ctr[3] = {0, 0, 0}, up[3] = {0, 1, 0};
        const Mat4  vp = Mul(Perspective(0.9f, float(kWidth) / float(kHeight), 0.5f, 400.0f), LookAt(eye, ctr, up));

        CameraUbo cu {};
        cu.viewProj = Transpose(vp);
        std::memcpy(app.c.MapBuffer(camStg, 0, sizeof(CameraUbo)), &cu, sizeof(CameraUbo));
        app.c.UnmapBuffer(camStg);

        if (!freezeCull || !haveFrozen)
        {
            frozenCullVP = vp;
            haveFrozen   = true;
        }
        CullUbo ku {};
        ku.cullViewProj = Transpose(frozenCullVP);
        ku.misc0[0]     = float(kInstances);
        ku.misc0[1]     = 0.0f; // set per capability below (constant per run)
        ku.misc0[2]     = 0.0f;
        ku.misc0[3]     = kExtent;
        ku.misc1[0]     = 1.7321f; // unit-cube half diagonal: bounding-sphere radius at scale 1
        std::memcpy(app.c.MapBuffer(cullStg, 0, sizeof(CullUbo)), &ku, sizeof(CullUbo));
        app.c.UnmapBuffer(cullStg);

        // last frame's resolved counter (the frame loop waits the fence, so it's ready)
        if (frame > 1 && countAvailable)
        {
            visibleCount = *static_cast<const uint32_t*>(app.c.MapBuffer(countReadback, 0, sizeof(uint32_t)));
            app.c.UnmapBuffer(countReadback);
        }
    };

    // misc0.y/z are capability-constant; patch them into the staged data via a tiny lambda hook:
    // simpler to fold into onUpdate - rewrite with capture (done here once for clarity)
    const float compactMode = hasCount ? 1.0f : 0.0f;
    const float mapSizeF    = hasClearTex ? float(kMapSize) : 0.0f;

    app.onGui = [=] {
        if (!canCull)
        {
            ImGui::Text(
                "GPU culling unavailable on %s\n(all %u cubes drawn with one instanced draw)", app.apiName, kInstances);
            ImGui::SliderFloat("orbit", &orbit, 0.0f, 4.0f);
            return;
        }
        ImGui::Text("visible: %u / %u cubes", visibleCount, kInstances);
        ImGui::Checkbox("freeze culling (then orbit!)", &freezeCull);
        ImGui::SliderFloat("orbit", &orbit, 0.0f, 4.0f);
        ImGui::Text("draw path: %s",
                    hasCount ? "CmdDrawIndexedIndirectCount (GPU count)" :
                               "CmdDrawIndexedIndirect (empty records; no hasDrawIndirectCount)");
        ImGui::Text("counter clear: %s",
                    hasClearBuf ? "CmdClearStorageBuffer" : "buffer-copy fallback (no hasClearStorageBuffer)");
        if (mapView)
        {
            ImGui::TextUnformatted("minimap (CmdClearStorageTexture each frame):");
            ImGui::Image(reinterpret_cast<ImTextureID>(mapView), ImVec2(128, 128));
        }
        else
        {
            ImGui::Text("hasClearStorageTexture = false on %s (no minimap)", app.apiName);
        }
    };

    app.onPreRender = [=](VriCommandBuffer* cmd) {
        // --- refresh the two constant buffers, patching the capability-constant fields ---
        // (misc0.y/z live at fixed offsets right after the matrix)
        {
            VriBufferCopyDesc cp {};
            cp.size = sizeof(CameraUbo);
            c.CmdCopyBuffer(cmd, camUbo, camStg, &cp);
        }
        if (!canCull) // no culling pass: just make the camera UBO visible to the vertex shader
        {
            VriBufferBarrierDesc ub {};
            ub.buffer        = camUbo;
            ub.before.access = VriAccess_CopyDestinationWrite;
            ub.before.stages = VriPipelineStage_Transfer;
            ub.after.access  = VriAccess_ConstantBufferRead;
            ub.after.stages  = VriPipelineStage_VertexShader;
            VriBarrierGroupDesc g {};
            g.buffers   = &ub;
            g.bufferNum = 1;
            c.CmdBarrier(cmd, &g);
            return;
        }
        {
            // patch the staged CullUbo before copying (map is cheap on HostUpload)
            auto* ku     = static_cast<CullUbo*>(c.MapBuffer(cullStg, 0, sizeof(CullUbo)));
            ku->misc0[1] = compactMode;
            ku->misc0[2] = mapSizeF;
            c.UnmapBuffer(cullStg);
            VriBufferCopyDesc cp {};
            cp.size = sizeof(CullUbo);
            c.CmdCopyBuffer(cmd, cullUbo, cullStg, &cp);
        }
        {
            VriBufferBarrierDesc ub[2] {};
            ub[0].buffer        = camUbo;
            ub[0].before.access = VriAccess_CopyDestinationWrite;
            ub[0].before.stages = VriPipelineStage_Transfer;
            ub[0].after.access  = VriAccess_ConstantBufferRead;
            ub[0].after.stages  = VriPipelineStage_VertexShader;
            ub[1].buffer        = cullUbo;
            ub[1].before.access = VriAccess_CopyDestinationWrite;
            ub[1].before.stages = VriPipelineStage_Transfer;
            ub[1].after.access  = VriAccess_ConstantBufferRead;
            ub[1].after.stages  = VriPipelineStage_ComputeShader;
            VriBarrierGroupDesc g {};
            g.buffers   = ub;
            g.bufferNum = 2;
            c.CmdBarrier(cmd, &g);
        }

        // --- zero the draw counter (the feature, or the copy fallback) ---
        {
            VriBufferBarrierDesc bb {};
            bb.buffer        = countBuf;
            bb.before.access = g_bufInit ? VriAccess_IndirectBufferRead : VriAccess_None;
            bb.before.stages = g_bufInit ? VriPipelineStage_DrawIndirect : VriPipelineStage_None;
            bb.after.access  = VriAccess_CopyDestinationWrite;
            bb.after.stages  = VriPipelineStage_Transfer;
            VriBarrierGroupDesc g {};
            g.buffers   = &bb;
            g.bufferNum = 1;
            c.CmdBarrier(cmd, &g);
            if (hasClearBuf)
            {
                c.CmdClearStorageBuffer(cmd, countBuf, 0, sizeof(uint32_t), 0);
            }
            else
            {
                VriBufferCopyDesc cp {};
                cp.size = sizeof(uint32_t);
                c.CmdCopyBuffer(cmd, countBuf, zeroBuf, &cp);
            }
        }

        // --- minimap -> cleared, then writable by the CS ---
        if (mapTex)
        {
            VriTextureBarrierDesc tb {};
            tb.texture       = mapTex;
            tb.aspect        = VriImageAspect_Color;
            tb.before.layout = g_mapInit ? VriLayout_ShaderResource : VriLayout_Undefined;
            tb.before.stages = g_mapInit ? VriPipelineStage_FragmentShader : VriPipelineStage_None;
            tb.after.access  = VriAccess_CopyDestinationWrite;
            tb.after.layout  = VriLayout_CopyDestination;
            tb.after.stages  = VriPipelineStage_Transfer;
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
            VriClearColor cc {};
            cc.f32[0] = 0.08f;
            cc.f32[1] = 0.09f;
            cc.f32[2] = 0.12f;
            cc.f32[3] = 1.0f;
            c.CmdClearStorageTexture(cmd, mapTex, &cc);
            VriTextureBarrierDesc tw {};
            tw.texture       = mapTex;
            tw.aspect        = VriImageAspect_Color;
            tw.before.access = VriAccess_CopyDestinationWrite;
            tw.before.layout = VriLayout_CopyDestination;
            tw.before.stages = VriPipelineStage_Transfer;
            tw.after.access  = VriAccess_ShaderResourceStorageWrite;
            tw.after.layout  = VriLayout_ShaderResourceStorage;
            tw.after.stages  = VriPipelineStage_ComputeShader;
            VriBarrierGroupDesc gw {};
            gw.textures   = &tw;
            gw.textureNum = 1;
            c.CmdBarrier(cmd, &gw);
            g_mapInit = true;
        }

        // --- records/survivors/counter -> writable by the CS ---
        {
            VriBufferBarrierDesc bb[3] {};
            bb[0].buffer        = recBuf;
            bb[0].before.access = g_bufInit ? VriAccess_IndirectBufferRead : VriAccess_None;
            bb[0].before.stages = g_bufInit ? VriPipelineStage_DrawIndirect : VriPipelineStage_None;
            bb[0].after.access  = VriAccess_ShaderResourceStorageWrite;
            bb[0].after.stages  = VriPipelineStage_ComputeShader;
            bb[1].buffer        = visBuf;
            bb[1].before.access = g_bufInit ? VriAccess_VertexBufferRead : VriAccess_None;
            bb[1].before.stages = g_bufInit ? VriPipelineStage_VertexInput : VriPipelineStage_None;
            bb[1].after.access  = VriAccess_ShaderResourceStorageWrite;
            bb[1].after.stages  = VriPipelineStage_ComputeShader;
            bb[2].buffer        = countBuf;
            bb[2].before.access = VriAccess_CopyDestinationWrite;
            bb[2].before.stages = VriPipelineStage_Transfer;
            bb[2].after.access  = VriAccess_ShaderResourceStorageWrite;
            bb[2].after.stages  = VriPipelineStage_ComputeShader;
            VriBarrierGroupDesc g {};
            g.buffers   = bb;
            g.bufferNum = 3;
            c.CmdBarrier(cmd, &g);
        }
        g_bufInit = true;

        // --- GPU culling ---
        c.CmdSetPipeline(cmd, cullPipeline);
        c.CmdSetPipelineLayout(cmd, cullLayout);
        c.CmdSetDescriptorSet(cmd, 0, cullSet);
        VriDispatchDesc disp {};
        disp.x = (kInstances + 63) / 64;
        disp.y = 1;
        disp.z = 1;
        c.CmdDispatch(cmd, &disp);

        // --- results -> indirect args + instance stream; counter also snapshots to the UI ---
        {
            VriBufferBarrierDesc bb[3] {};
            bb[0].buffer        = recBuf;
            bb[0].before.access = VriAccess_ShaderResourceStorageWrite;
            bb[0].before.stages = VriPipelineStage_ComputeShader;
            bb[0].after.access  = VriAccess_IndirectBufferRead;
            bb[0].after.stages  = VriPipelineStage_DrawIndirect;
            bb[1].buffer        = visBuf;
            bb[1].before.access = VriAccess_ShaderResourceStorageWrite;
            bb[1].before.stages = VriPipelineStage_ComputeShader;
            bb[1].after.access  = VriAccess_VertexBufferRead;
            bb[1].after.stages  = VriPipelineStage_VertexInput;
            bb[2].buffer        = countBuf;
            bb[2].before.access = VriAccess_ShaderResourceStorageWrite;
            bb[2].before.stages = VriPipelineStage_ComputeShader;
            bb[2].after.access  = VriAccess_CopySourceRead;
            bb[2].after.stages  = VriPipelineStage_Transfer;
            VriBarrierGroupDesc g {};
            g.buffers   = bb;
            g.bufferNum = 3;
            c.CmdBarrier(cmd, &g);
        }
        {
            VriBufferCopyDesc cp {};
            cp.size = sizeof(uint32_t);
            c.CmdCopyBuffer(cmd, countReadback, countBuf, &cp);
            countAvailable = true;
            VriBufferBarrierDesc bb {};
            bb.buffer        = countBuf;
            bb.before.access = VriAccess_CopySourceRead;
            bb.before.stages = VriPipelineStage_Transfer;
            bb.after.access  = VriAccess_IndirectBufferRead;
            bb.after.stages  = VriPipelineStage_DrawIndirect;
            VriBarrierGroupDesc g {};
            g.buffers   = &bb;
            g.bufferNum = 1;
            c.CmdBarrier(cmd, &g);
        }

        // --- minimap -> sampled by ImGui in the swapchain pass ---
        if (mapTex)
        {
            VriTextureBarrierDesc tr {};
            tr.texture       = mapTex;
            tr.aspect        = VriImageAspect_Color;
            tr.before.access = VriAccess_ShaderResourceStorageWrite;
            tr.before.layout = VriLayout_ShaderResourceStorage;
            tr.before.stages = VriPipelineStage_ComputeShader;
            tr.after.access  = VriAccess_ShaderResourceRead;
            tr.after.layout  = VriLayout_ShaderResource;
            tr.after.stages  = VriPipelineStage_FragmentShader;
            VriBarrierGroupDesc g {};
            g.textures   = &tr;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        }
    };

    app.onRecord = [=](VriCommandBuffer* cmd) {
        c.CmdSetPipeline(cmd, scenePipeline);
        c.CmdSetPipelineLayout(cmd, sceneLayout);
        c.CmdSetDescriptorSet(cmd, 0, sceneSet);
        VriVertexBufferBinding vb[2] {};
        vb[0].buffer = vbuf;
        vb[0].offset = 0;
        vb[1].buffer = visBuf;
        vb[1].offset = 0;
        c.CmdSetVertexBuffers(cmd, 0, vb, 2);
        c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);
        const uint32_t stride = kRecordU32 * sizeof(uint32_t);
        if (hasCount)
            c.CmdDrawIndexedIndirectCount(cmd, recBuf, 0, countBuf, 0, kInstances, stride);
        else
            c.CmdDrawIndexedIndirect(cmd, recBuf, 0, kInstances, stride); // culled records draw 0 indices
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(scenePipeline);
    c.DestroyPipelineLayout(sceneLayout);
    c.DestroyPipeline(cullPipeline);
    c.DestroyPipelineLayout(cullLayout);
    if (mapView)
        c.DestroyDescriptor(mapView);
    if (mapTex)
        c.DestroyTexture(mapTex);
    for (VriDescriptor* v : {camView, cullView, instView, visView, recView, countView})
        c.DestroyDescriptor(v);
    if (zeroBuf)
        c.DestroyBuffer(zeroBuf);
    for (VriBuffer* b :
         {vbuf, ibuf, instBuf, visBuf, recBuf, countBuf, countReadback, camUbo, cullUbo, camStg, cullStg})
        c.DestroyBuffer(b);
    app.Shutdown();
#endif
    return 0;
}
