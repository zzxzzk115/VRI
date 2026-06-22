// Instanced textured cubes: a field of cubes drawn in one indexed instanced draw, each
// placed by a per-instance model matrix (vertex stream 1), with a shared view-projection
// (constant buffer) that orbits the field. Builds on examples/cube; the shared host
// scaffolding lives in examples/common/example_app.h. VRI_API / ?backend force a backend;
// VRI_MAX_FRAMES / ?frames=N auto-exit.
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "../cube/mat4.h"
#include "../cube/ktx.h"

#include "tests/shaders/cube_inst_spv.h"  // g_cubeInstSpv
#include "tests/shaders/cube_inst_wgsl.h" // g_cubeInstWgsl
#include "tests/shaders/cube_inst_dxbc.h" // g_cubeInstDxbcVS / PS

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480; // width*4 is 256-aligned (readback pitch)
    constexpr int kGrid = 4;            // kGrid^3 cubes
    constexpr float kSpacing = 2.2f;

    struct Vertex { float px, py, pz; float u, v; };

    const Vertex kCube[24] = {
        {-0.5f,-0.5f, 0.5f, 0,1}, { 0.5f,-0.5f, 0.5f, 1,1}, { 0.5f, 0.5f, 0.5f, 1,0}, {-0.5f, 0.5f, 0.5f, 0,0},
        { 0.5f,-0.5f,-0.5f, 0,1}, {-0.5f,-0.5f,-0.5f, 1,1}, {-0.5f, 0.5f,-0.5f, 1,0}, { 0.5f, 0.5f,-0.5f, 0,0},
        { 0.5f,-0.5f, 0.5f, 0,1}, { 0.5f,-0.5f,-0.5f, 1,1}, { 0.5f, 0.5f,-0.5f, 1,0}, { 0.5f, 0.5f, 0.5f, 0,0},
        {-0.5f,-0.5f,-0.5f, 0,1}, {-0.5f,-0.5f, 0.5f, 1,1}, {-0.5f, 0.5f, 0.5f, 1,0}, {-0.5f, 0.5f,-0.5f, 0,0},
        {-0.5f, 0.5f, 0.5f, 0,1}, { 0.5f, 0.5f, 0.5f, 1,1}, { 0.5f, 0.5f,-0.5f, 1,0}, {-0.5f, 0.5f,-0.5f, 0,0},
        {-0.5f,-0.5f,-0.5f, 0,1}, { 0.5f,-0.5f,-0.5f, 1,1}, { 0.5f,-0.5f, 0.5f, 1,0}, {-0.5f,-0.5f, 0.5f, 0,0},
    };
    uint16_t kIndices[36];
    void BuildIndices()
    {
        for (uint32_t f = 0; f < 6; ++f)
        {
            const uint16_t b = static_cast<uint16_t>(f * 4);
            uint16_t* o = kIndices + f * 6;
            o[0]=b; o[1]=uint16_t(b+1); o[2]=uint16_t(b+2); o[3]=b; o[4]=uint16_t(b+2); o[5]=uint16_t(b+3);
        }
    }
} // namespace

int main(int, char**)
{
    BuildIndices();

    // per-instance model matrices (static grid; each cube fixed-rotated and scaled)
    std::vector<Mat4> instances;
    instances.reserve(kGrid * kGrid * kGrid);
    const float half = (kGrid - 1) * kSpacing * 0.5f;
    for (int z = 0; z < kGrid; ++z)
        for (int y = 0; y < kGrid; ++y)
            for (int x = 0; x < kGrid; ++x)
            {
                const float phase = float(x + y * 3 + z * 7) * 0.5f;
                instances.push_back(Mul(Translate(x * kSpacing - half, y * kSpacing - half, z * kSpacing - half), Mul(RotateY(phase), Scale(0.7f))));
            }
    const uint64_t instBytes = instances.size() * sizeof(Mat4);
    const uint32_t instanceNum = static_cast<uint32_t>(instances.size());

    static vriex::ExampleApp app;
    app.Init("instancing", kWidth, kHeight, /*hasDepth*/ true);
    VriCoreInterface& c = app.c;

    // load the texture (desktop: next to the exe; web: preloaded into MEMFS at root)
    KtxImage tex;
#if defined(__EMSCRIPTEN__)
    if (!LoadKtxRgba("/metalplate01_rgba.ktx", tex) && !LoadKtxRgba("metalplate01_rgba.ktx", tex))
        app.Fail("failed to load metalplate01_rgba.ktx");
#else
    {
        const char* base = SDL_GetBasePath();
        std::string path = (base ? std::string(base) : std::string()) + "metalplate01_rgba.ktx";
        if (!LoadKtxRgba(path.c_str(), tex) && !LoadKtxRgba("metalplate01_rgba.ktx", tex))
            app.Fail("failed to load metalplate01_rgba.ktx");
    }
#endif

    auto makeDeviceBuf = [&](uint64_t size, VriBufferUsageFlags usage, const void* src) {
        VriBufferDesc bd{}; bd.size = size; bd.usage = usage | VriBufferUsage_TransferDst; bd.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* b = nullptr; c.CreateBuffer(app.dev, &bd, &b);
        VriBufferDesc sd{}; sd.size = size; sd.usage = VriBufferUsage_TransferSrc; sd.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* s = nullptr; c.CreateBuffer(app.dev, &sd, &s);
        std::memcpy(c.MapBuffer(s, 0, size), src, size); c.UnmapBuffer(s);
        return std::pair<VriBuffer*, VriBuffer*>(b, s);
    };
    auto [vbuf, vstg] = makeDeviceBuf(sizeof(kCube), VriBufferUsage_VertexBuffer, kCube);
    auto [ibuf, istg] = makeDeviceBuf(sizeof(kIndices), VriBufferUsage_IndexBuffer, kIndices);
    auto [inst, inststg] = makeDeviceBuf(instBytes, VriBufferUsage_VertexBuffer, instances.data());

    // shared view-projection constant buffer (orbits the field), refreshed each frame
    VriBufferDesc ubd{}; ubd.size = sizeof(Mat4); ubd.usage = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst; ubd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* ubo = nullptr; c.CreateBuffer(app.dev, &ubd, &ubo);
    VriBufferDesc usd{}; usd.size = sizeof(Mat4); usd.usage = VriBufferUsage_TransferSrc; usd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* ustg = nullptr; c.CreateBuffer(app.dev, &usd, &ustg);
    VriBufferViewDesc ubv{}; ubv.buffer = ubo; ubv.viewType = VriDescriptorType_ConstantBuffer; ubv.offset = 0; ubv.size = sizeof(Mat4);
    VriDescriptor* uboView = nullptr; c.CreateBufferView(app.dev, &ubv, &uboView);

    // texture (device-local) + staging upload + view + sampler
    VriTextureDesc ttd{};
    ttd.type = VriTextureType_2D; ttd.format = VriFormat_RGBA8_UNORM; ttd.width = tex.width; ttd.height = tex.height; ttd.depth = 1;
    ttd.mipNum = 1; ttd.layerNum = 1; ttd.sampleNum = 1; ttd.usage = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst; ttd.memoryLocation = VriMemoryLocation_Device;
    VriTexture* texture = nullptr; c.CreateTexture(app.dev, &ttd, &texture);
    VriTextureViewDesc tvd{}; tvd.texture = texture; tvd.viewType = VriTextureViewType_2D; tvd.format = VriFormat_Unknown; tvd.aspect = VriImageAspect_Color;
    VriDescriptor* texView = nullptr; c.CreateTextureView(app.dev, &tvd, &texView);
    VriBufferDesc stg{}; stg.size = tex.rgba.size(); stg.usage = VriBufferUsage_TransferSrc; stg.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* staging = nullptr; c.CreateBuffer(app.dev, &stg, &staging);
    std::memcpy(c.MapBuffer(staging, 0, tex.rgba.size()), tex.rgba.data(), tex.rgba.size()); c.UnmapBuffer(staging);
    VriSamplerDesc smp{}; smp.magFilter = VriFilter_Linear; smp.minFilter = VriFilter_Linear; smp.mipmapMode = VriMipmapMode_Linear;
    smp.addressModeU = VriAddressMode_Repeat; smp.addressModeV = VriAddressMode_Repeat; smp.addressModeW = VriAddressMode_Repeat; smp.maxLod = 1.0f;
    VriDescriptor* sampler = nullptr; c.CreateSampler(app.dev, &smp, &sampler);

    // pipeline layout: CB@0 (vertex), Texture@1 + Sampler@2 (fragment)
    VriDescriptorRangeDesc ranges[3]{};
    ranges[0].baseRegister = 0; ranges[0].descriptorNum = 1; ranges[0].descriptorType = VriDescriptorType_ConstantBuffer; ranges[0].shaderStages = VriShaderStage_Vertex;
    ranges[1].baseRegister = 1; ranges[1].descriptorNum = 1; ranges[1].descriptorType = VriDescriptorType_Texture;        ranges[1].shaderStages = VriShaderStage_Fragment;
    ranges[2].baseRegister = 2; ranges[2].descriptorNum = 1; ranges[2].descriptorType = VriDescriptorType_Sampler;        ranges[2].shaderStages = VriShaderStage_Fragment;
    VriDescriptorSetDesc setDesc{}; setDesc.registerSpace = 0; setDesc.ranges = ranges; setDesc.rangeNum = 3;
    VriPipelineLayoutDesc ld{}; ld.descriptorSets = &setDesc; ld.descriptorSetNum = 1;
    VriPipelineLayout* layout = nullptr; c.CreatePipelineLayout(app.dev, &ld, &layout);

    VriShaderDesc sh[2]{};
    sh[0].stage = VriShaderStage_Vertex;   sh[0].entryPointName = "vertexMain";
    sh[1].stage = VriShaderStage_Fragment; sh[1].entryPointName = "fragmentMain";
    if (app.useDxbc) { sh[0].bytecode = g_cubeInstDxbcVS; sh[0].bytecodeSize = sizeof(g_cubeInstDxbcVS); sh[1].bytecode = g_cubeInstDxbcPS; sh[1].bytecodeSize = sizeof(g_cubeInstDxbcPS); }
    else { sh[0].bytecode = sh[1].bytecode = app.useWgsl ? static_cast<const void*>(g_cubeInstWgsl) : static_cast<const void*>(g_cubeInstSpv);
           sh[0].bytecodeSize = sh[1].bytecodeSize = app.useWgsl ? sizeof(g_cubeInstWgsl) : sizeof(g_cubeInstSpv); }

    VriVertexAttributeDesc attrs[6]{};
    attrs[0].format = VriFormat_RGB32_SFLOAT;  attrs[0].offset = 0;  attrs[0].streamIndex = 0;
    attrs[1].format = VriFormat_RG32_SFLOAT;   attrs[1].offset = 12; attrs[1].streamIndex = 0;
    for (int i = 0; i < 4; ++i) { attrs[2 + i].format = VriFormat_RGBA32_SFLOAT; attrs[2 + i].offset = uint32_t(i * 16); attrs[2 + i].streamIndex = 1; }
    VriVertexStreamDesc streams[2]{};
    streams[0].stride = sizeof(Vertex); streams[0].bindingSlot = 0; streams[0].stepRate = VriVertexStepRate_PerVertex;
    streams[1].stride = sizeof(Mat4);   streams[1].bindingSlot = 1; streams[1].stepRate = VriVertexStepRate_PerInstance;

    VriColorAttachmentDesc ca{}; ca.format = app.swapFormat; ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd{};
    pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
    pd.vertexInput.attributes = attrs; pd.vertexInput.attributeNum = 6; pd.vertexInput.streams = streams; pd.vertexInput.streamNum = 2;
    pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode = VriCullMode_Back; pd.rasterization.frontFace = VriFrontFace_CounterClockwise; pd.rasterization.lineWidth = 1.0f;
    pd.multisample.sampleNum = 1;
    pd.depthStencil.depthTest = VRI_TRUE; pd.depthStencil.depthWrite = VRI_TRUE; pd.depthStencil.depthCompareOp = VriCompareOp_Less;
    pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1; pd.outputMerger.depthStencilFormat = app.depthFormat;
    VriPipeline* pipeline = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &pd, &pipeline) != VriResult_Success) app.Fail("CreateGraphicsPipeline failed");

    VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 1; pdsc.constantBufferMaxNum = 1; pdsc.textureMaxNum = 1; pdsc.samplerMaxNum = 1;
    VriDescriptorPool* pool = nullptr; c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* set = nullptr; c.AllocateDescriptorSets(pool, layout, 0, &set, 1);
    const VriDescriptor* d0[1] = {uboView}; const VriDescriptor* d1[1] = {texView}; const VriDescriptor* d2[1] = {sampler};
    VriDescriptorRangeUpdateDesc upd[3]{};
    upd[0].descriptors = d0; upd[0].descriptorNum = 1; upd[1].descriptors = d1; upd[1].descriptorNum = 1; upd[2].descriptors = d2; upd[2].descriptorNum = 1;
    c.UpdateDescriptorRanges(set, 0, 3, upd);

    // one-time upload: vertex / index / instance / texture -> read state (fence value 1)
    {
        VriCommandBuffer* cmd = app.cmd;
        c.BeginCommandBuffer(cmd);
        VriBufferCopyDesc cp{}; cp.size = sizeof(kCube); c.CmdCopyBuffer(cmd, vbuf, vstg, &cp);
        cp.size = sizeof(kIndices); c.CmdCopyBuffer(cmd, ibuf, istg, &cp);
        cp.size = instBytes; c.CmdCopyBuffer(cmd, inst, inststg, &cp);
        VriBufferBarrierDesc gb[3]{};
        gb[0].buffer = vbuf; gb[0].after.access = VriAccess_VertexBufferRead;
        gb[1].buffer = ibuf; gb[1].after.access = VriAccess_IndexBufferRead;
        gb[2].buffer = inst; gb[2].after.access = VriAccess_VertexBufferRead;
        for (auto& b : gb) { b.before.access = VriAccess_CopyDestinationWrite; b.before.stages = VriPipelineStage_Transfer; b.after.stages = VriPipelineStage_VertexInput; }
        VriBarrierGroupDesc gbg{}; gbg.buffers = gb; gbg.bufferNum = 3; c.CmdBarrier(cmd, &gbg);

        VriTextureBarrierDesc tb{}; tb.texture = texture; tb.before.layout = VriLayout_Undefined; tb.before.stages = VriPipelineStage_None;
        tb.after.access = VriAccess_CopyDestinationWrite; tb.after.layout = VriLayout_CopyDestination; tb.after.stages = VriPipelineStage_Transfer; tb.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc g0{}; g0.textures = &tb; g0.textureNum = 1; c.CmdBarrier(cmd, &g0);
        VriBufferTextureCopyDesc up{}; up.texture.aspect = VriImageAspect_Color; up.texture.layerNum = 1; up.texture.width = tex.width; up.texture.height = tex.height;
        c.CmdUploadBufferToTexture(cmd, texture, staging, &up);
        VriTextureBarrierDesc tb2{}; tb2.texture = texture; tb2.before.access = VriAccess_CopyDestinationWrite; tb2.before.layout = VriLayout_CopyDestination; tb2.before.stages = VriPipelineStage_Transfer;
        tb2.after.access = VriAccess_ShaderResourceRead; tb2.after.layout = VriLayout_ShaderResource; tb2.after.stages = VriPipelineStage_FragmentShader; tb2.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc g1{}; g1.textures = &tb2; g1.textureNum = 1; c.CmdBarrier(cmd, &g1);
        c.EndCommandBuffer(cmd);
        VriFenceSubmitDesc sig{}; sig.fence = app.fence; sig.value = 1;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(app.queue, &sub); c.Wait(app.fence, 1);
    }

    // per-frame: orbit the camera, refresh the shared view-projection, draw all instances
    app.onUpdate = [ustg](uint64_t frame) {
        const float t = static_cast<float>(frame) * 0.01f;
        const float r = kGrid * kSpacing + 6.0f;
        const float eye[3] = {std::cos(t) * r, 4.0f, std::sin(t) * r}, ctr[3] = {0, 0, 0}, up[3] = {0, 1, 0};
        // transpose for Slang's mul(viewProj, world); the per-instance columns stay column-major
        Mat4 viewProj = Transpose(Mul(Perspective(0.9f, float(kWidth) / float(kHeight), 0.1f, 200.0f), LookAt(eye, ctr, up)));
        std::memcpy(app.c.MapBuffer(ustg, 0, sizeof(Mat4)), &viewProj, sizeof(Mat4)); app.c.UnmapBuffer(ustg);
    };
    app.onPreRender = [ubo, ustg](VriCommandBuffer* cmd) {
        VriBufferCopyDesc ucp{}; ucp.size = sizeof(Mat4); app.c.CmdCopyBuffer(cmd, ubo, ustg, &ucp);
        VriBufferBarrierDesc ub{}; ub.buffer = ubo; ub.before.access = VriAccess_CopyDestinationWrite; ub.before.stages = VriPipelineStage_Transfer;
        ub.after.access = VriAccess_ConstantBufferRead; ub.after.stages = VriPipelineStage_VertexShader;
        VriBarrierGroupDesc ubg{}; ubg.buffers = &ub; ubg.bufferNum = 1; app.c.CmdBarrier(cmd, &ubg);
    };
    app.onRecord = [pipeline, layout, set, vbuf, ibuf, inst, instanceNum](VriCommandBuffer* cmd) {
        app.c.CmdSetPipeline(cmd, pipeline);
        app.c.CmdSetPipelineLayout(cmd, layout);
        app.c.CmdSetDescriptorSet(cmd, 0, set);
        VriVertexBufferBinding vbs[2]{}; vbs[0].buffer = vbuf; vbs[0].offset = 0; vbs[1].buffer = inst; vbs[1].offset = 0;
        app.c.CmdSetVertexBuffers(cmd, 0, vbs, 2);
        app.c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);
        VriDrawIndexedDesc di{}; di.indexNum = 36; di.instanceNum = instanceNum; app.c.CmdDrawIndexed(cmd, &di);
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
    c.DestroyDescriptor(sampler); c.DestroyDescriptor(texView); c.DestroyTexture(texture); c.DestroyBuffer(staging);
    c.DestroyDescriptor(uboView); c.DestroyBuffer(ustg); c.DestroyBuffer(ubo);
    c.DestroyBuffer(inststg); c.DestroyBuffer(inst); c.DestroyBuffer(istg); c.DestroyBuffer(vstg); c.DestroyBuffer(ibuf); c.DestroyBuffer(vbuf);
    app.Shutdown();
#endif
    return 0;
}
