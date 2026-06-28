// PBR (metallic-roughness) basic: a grid of procedurally generated UV spheres, metallic
// varying along X and roughness along Y, lit by 4 point lights with a Cook-Torrance BRDF.
// Ported from Sascha Willems' pbrbasic; the per-object material that the original passes via
// push constants is delivered here through the (already cross-backend-proven) per-instance
// vertex stream, so this runs on every backend incl. the web. No assets - the sphere mesh and
// the instance grid are generated at startup. Shared host scaffolding in examples/common.
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include "../cube/mat4.h"

#include "examples/shaders/pbr_dxbc.h" // g_pbrDxbcVS / PS (D3D12)
#include "examples/shaders/pbr_spv.h"  // g_pbrSpv  (Vulkan + OpenGL)
#include "examples/shaders/pbr_wgsl.h" // g_pbrWgsl (WebGPU)

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480; // width*4 is 256-aligned (readback pitch)
    constexpr int      kGrid    = 7;                // kGrid x kGrid spheres
    constexpr float    kSpacing = 2.5f;
    constexpr float    kRadius  = 1.0f;
    constexpr float    kPI      = 3.14159265358979323846f;

    struct Vertex
    {
        float px, py, pz;
        float nx, ny, nz;
    };
    struct Instance
    {
        float a[4];
        float b[4];
    }; // a: pos.xyz + roughness, b: color.rgb + metallic

    // UV sphere (rings x sectors) centered at origin; normal == normalized position.
    void BuildSphere(std::vector<Vertex>& verts, std::vector<uint16_t>& indices, int rings = 32, int sectors = 32)
    {
        for (int r = 0; r <= rings; ++r)
        {
            const float theta = kPI * float(r) / float(rings);
            const float st = std::sin(theta), ct = std::cos(theta);
            for (int s = 0; s <= sectors; ++s)
            {
                const float phi = 2.0f * kPI * float(s) / float(sectors);
                const float nx = st * std::cos(phi), ny = ct, nz = st * std::sin(phi);
                verts.push_back({nx * kRadius, ny * kRadius, nz * kRadius, nx, ny, nz});
            }
        }
        const int stride = sectors + 1;
        for (int r = 0; r < rings; ++r)
            for (int s = 0; s < sectors; ++s)
            {
                const uint16_t a = uint16_t(r * stride + s);
                const uint16_t b = uint16_t((r + 1) * stride + s);
                indices.push_back(a);
                indices.push_back(b);
                indices.push_back(uint16_t(a + 1));
                indices.push_back(uint16_t(a + 1));
                indices.push_back(b);
                indices.push_back(uint16_t(b + 1));
            }
    }

    // Matches the shader Ubo: transposed view-projection + camera + 4 light positions.
    struct Ubo
    {
        Mat4  viewProj;
        float camPos[4];
        float lights[4][4];
    };
} // namespace

int main(int, char**)
{
    std::vector<Vertex>   sphereVerts;
    std::vector<uint16_t> sphereIdx;
    BuildSphere(sphereVerts, sphereIdx);
    const uint64_t vbBytes  = sphereVerts.size() * sizeof(Vertex);
    const uint64_t ibBytes  = sphereIdx.size() * sizeof(uint16_t);
    const uint32_t indexNum = static_cast<uint32_t>(sphereIdx.size());

    // Instance grid: metallic across X, roughness down Y, a fixed gold-ish base color.
    std::vector<Instance> instances;
    const float           half = (kGrid - 1) * kSpacing * 0.5f;
    for (int y = 0; y < kGrid; ++y)
        for (int x = 0; x < kGrid; ++x)
        {
            Instance in {};
            in.a[0] = x * kSpacing - half;
            in.a[1] = y * kSpacing - half;
            in.a[2] = 0.0f;
            in.a[3] = std::fmax(0.05f, float(y) / float(kGrid - 1)); // roughness
            in.b[0] = 1.0f;
            in.b[1] = 0.86f;
            in.b[2] = 0.57f;                       // base color (gold)
            in.b[3] = float(x) / float(kGrid - 1); // metallic
            instances.push_back(in);
        }
    const uint64_t instBytes   = instances.size() * sizeof(Instance);
    const uint32_t instanceNum = static_cast<uint32_t>(instances.size());

    static vriex::ExampleApp app;
    app.Init("pbrbasic", kWidth, kHeight, /*hasDepth*/ true);
    app.SetClearColor(0.02f, 0.02f, 0.03f);
    VriCoreInterface& c = app.c;

    auto makeDeviceBuf = [&](uint64_t size, VriBufferUsageFlags usage, const void* src) {
        VriBufferDesc bd {};
        bd.size           = size;
        bd.usage          = usage | VriBufferUsage_TransferDst;
        bd.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* b      = nullptr;
        c.CreateBuffer(app.dev, &bd, &b);
        VriBufferDesc sd {};
        sd.size           = size;
        sd.usage          = VriBufferUsage_TransferSrc;
        sd.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* s      = nullptr;
        c.CreateBuffer(app.dev, &sd, &s);
        std::memcpy(c.MapBuffer(s, 0, size), src, size);
        c.UnmapBuffer(s);
        return std::pair<VriBuffer*, VriBuffer*>(b, s);
    };
    auto [vbuf, vstg]    = makeDeviceBuf(vbBytes, VriBufferUsage_VertexBuffer, sphereVerts.data());
    auto [ibuf, istg]    = makeDeviceBuf(ibBytes, VriBufferUsage_IndexBuffer, sphereIdx.data());
    auto [inst, inststg] = makeDeviceBuf(instBytes, VriBufferUsage_VertexBuffer, instances.data());

    // camera + lights constant buffer (device-local, refreshed each frame from staging)
    VriBufferDesc ubd {};
    ubd.size           = sizeof(Ubo);
    ubd.usage          = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst;
    ubd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* ubo     = nullptr;
    c.CreateBuffer(app.dev, &ubd, &ubo);
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

    // pipeline layout: one constant buffer @0 visible to vertex + fragment
    VriDescriptorRangeDesc range {};
    range.baseRegister   = 0;
    range.descriptorNum  = 1;
    range.descriptorType = VriDescriptorType_ConstantBuffer;
    range.shaderStages   = VriShaderStage_Vertex | VriShaderStage_Fragment;
    VriDescriptorSetDesc setDesc {};
    setDesc.registerSpace = 0;
    setDesc.ranges        = &range;
    setDesc.rangeNum      = 1;
    VriPipelineLayoutDesc ld {};
    ld.descriptorSets         = &setDesc;
    ld.descriptorSetNum       = 1;
    VriPipelineLayout* layout = nullptr;
    c.CreatePipelineLayout(app.dev, &ld, &layout);

    VriShaderDesc sh[2] {};
    sh[0].stage          = VriShaderStage_Vertex;
    sh[0].entryPointName = "vertexMain";
    sh[1].stage          = VriShaderStage_Fragment;
    sh[1].entryPointName = "fragmentMain";
    if (app.useDxbc)
    {
        sh[0].bytecode     = g_pbrDxbcVS;
        sh[0].bytecodeSize = sizeof(g_pbrDxbcVS);
        sh[1].bytecode     = g_pbrDxbcPS;
        sh[1].bytecodeSize = sizeof(g_pbrDxbcPS);
    }
    else
    {
        sh[0].bytecode = sh[1].bytecode =
            app.useWgsl ? static_cast<const void*>(g_pbrWgsl) : static_cast<const void*>(g_pbrSpv);
        sh[0].bytecodeSize = sh[1].bytecodeSize = app.useWgsl ? sizeof(g_pbrWgsl) : sizeof(g_pbrSpv);
    }

    VriVertexAttributeDesc attrs[4] {};
    attrs[0].format      = VriFormat_RGB32_SFLOAT;
    attrs[0].offset      = 0;
    attrs[0].streamIndex = 0; // position
    attrs[1].format      = VriFormat_RGB32_SFLOAT;
    attrs[1].offset      = 12;
    attrs[1].streamIndex = 0; // normal
    attrs[2].format      = VriFormat_RGBA32_SFLOAT;
    attrs[2].offset      = 0;
    attrs[2].streamIndex = 1; // instA
    attrs[3].format      = VriFormat_RGBA32_SFLOAT;
    attrs[3].offset      = 16;
    attrs[3].streamIndex = 1; // instB
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
    pd.pipelineLayout                  = layout;
    pd.shaders                         = sh;
    pd.shaderNum                       = 2;
    pd.vertexInput.attributes          = attrs;
    pd.vertexInput.attributeNum        = 4;
    pd.vertexInput.streams             = streams;
    pd.vertexInput.streamNum           = 2;
    pd.inputAssembly.topology          = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode          = VriCullMode_None;
    pd.rasterization.frontFace         = VriFrontFace_CounterClockwise;
    pd.rasterization.lineWidth         = 1.0f;
    pd.multisample.sampleNum           = 1;
    pd.depthStencil.depthTest          = VRI_TRUE;
    pd.depthStencil.depthWrite         = VRI_TRUE;
    pd.depthStencil.depthCompareOp     = VriCompareOp_Less;
    pd.outputMerger.colors             = &ca;
    pd.outputMerger.colorNum           = 1;
    pd.outputMerger.depthStencilFormat = app.depthFormat;
    VriPipeline* pipeline              = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &pd, &pipeline) != VriResult_Success)
        app.Fail("CreateGraphicsPipeline failed");

    VriDescriptorPoolDesc pdsc {};
    pdsc.descriptorSetMaxNum  = 1;
    pdsc.constantBufferMaxNum = 1;
    VriDescriptorPool* pool   = nullptr;
    c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* set = nullptr;
    c.AllocateDescriptorSets(pool, layout, 0, &set, 1);
    const VriDescriptor*         d0[1] = {uboView};
    VriDescriptorRangeUpdateDesc upd {};
    upd.descriptors   = d0;
    upd.descriptorNum = 1;
    c.UpdateDescriptorRanges(set, 0, 1, &upd);

    // one-time upload: sphere vertices / indices / instances -> read state (fence value 1)
    {
        VriCommandBuffer* cmd = app.cmd;
        c.BeginCommandBuffer(cmd);
        VriBufferCopyDesc cp {};
        cp.size = vbBytes;
        c.CmdCopyBuffer(cmd, vbuf, vstg, &cp);
        cp.size = ibBytes;
        c.CmdCopyBuffer(cmd, ibuf, istg, &cp);
        cp.size = instBytes;
        c.CmdCopyBuffer(cmd, inst, inststg, &cp);
        VriBufferBarrierDesc gb[3] {};
        gb[0].buffer       = vbuf;
        gb[0].after.access = VriAccess_VertexBufferRead;
        gb[1].buffer       = ibuf;
        gb[1].after.access = VriAccess_IndexBufferRead;
        gb[2].buffer       = inst;
        gb[2].after.access = VriAccess_VertexBufferRead;
        for (auto& b : gb)
        {
            b.before.access = VriAccess_CopyDestinationWrite;
            b.before.stages = VriPipelineStage_Transfer;
            b.after.stages  = VriPipelineStage_VertexInput;
        }
        VriBarrierGroupDesc gbg {};
        gbg.buffers   = gb;
        gbg.bufferNum = 3;
        c.CmdBarrier(cmd, &gbg);
        c.EndCommandBuffer(cmd);
        VriFenceSubmitDesc sig {};
        sig.fence = app.fence;
        sig.value = 1;
        VriQueueSubmitDesc sub {};
        sub.commandBuffers   = &cmd;
        sub.commandBufferNum = 1;
        sub.signalFences     = &sig;
        sub.signalFenceNum   = 1;
        c.QueueSubmit(app.queue, &sub);
        c.Wait(app.fence, 1);
    }

    static float spin = 1.0f, t = 0.0f; // ImGui-controlled camera orbit (metallic/roughness vary per sphere)
    app.onUpdate = [ustg](uint64_t) {
        t += 0.36f * spin * app.dt; // 0.36/s (= 0.006/frame at 60fps)
        const float R      = kGrid * kSpacing;
        const float eye[3] = {std::sin(t) * R, R * 0.25f, std::cos(t) * R}, ctr[3] = {0, 0, 0}, up[3] = {0, 1, 0};
        Ubo         u {};
        u.viewProj =
            Transpose(Mul(Perspective(0.9f, float(kWidth) / float(kHeight), 0.1f, 200.0f), LookAt(eye, ctr, up)));
        u.camPos[0]          = eye[0];
        u.camPos[1]          = eye[1];
        u.camPos[2]          = eye[2];
        const float L        = kGrid * kSpacing;
        const float lp[4][3] = {{-L, L, L}, {L, L, L}, {-L, -L, L}, {L, -L, -L}};
        for (int i = 0; i < 4; ++i)
        {
            u.lights[i][0] = lp[i][0];
            u.lights[i][1] = lp[i][1];
            u.lights[i][2] = lp[i][2];
        }
        std::memcpy(app.c.MapBuffer(ustg, 0, sizeof(Ubo)), &u, sizeof(Ubo));
        app.c.UnmapBuffer(ustg);
    };
    app.onGui = [] {
        ImGui::SliderFloat("orbit", &spin, 0.0f, 5.0f);
        ImGui::Text("X: metallic   Y: roughness");
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
    app.onRecord = [pipeline, layout, set, vbuf, inst, ibuf, indexNum, instanceNum](VriCommandBuffer* cmd) {
        app.c.CmdSetPipeline(cmd, pipeline);
        app.c.CmdSetPipelineLayout(cmd, layout);
        app.c.CmdSetDescriptorSet(cmd, 0, set);
        VriVertexBufferBinding vbs[2] {};
        vbs[0].buffer = vbuf;
        vbs[0].offset = 0;
        vbs[1].buffer = inst;
        vbs[1].offset = 0;
        app.c.CmdSetVertexBuffers(cmd, 0, vbs, 2);
        app.c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);
        VriDrawIndexedDesc di {};
        di.indexNum    = indexNum;
        di.instanceNum = instanceNum;
        app.c.CmdDrawIndexed(cmd, &di);
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
    c.DestroyBuffer(inststg);
    c.DestroyBuffer(inst);
    c.DestroyBuffer(istg);
    c.DestroyBuffer(ibuf);
    c.DestroyBuffer(vstg);
    c.DestroyBuffer(vbuf);
    app.Shutdown();
#endif
    return 0;
}
