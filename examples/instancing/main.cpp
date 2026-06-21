// Instanced textured cubes: a field of cubes drawn in one indexed instanced draw, each
// placed by a per-instance model matrix (vertex stream 1), with a shared view-projection
// (constant buffer) that orbits the field. Builds on examples/cube - same texture and
// helpers - and adds instancing. Set VRI_API=vulkan|webgpu|opengl|d3d12 (default vulkan)
// and VRI_MAX_FRAMES=N to auto-exit.
#include <SDL3/SDL.h>

#include <vri/vri.h>
#include <vri/integration/vri_sdl3.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../cube/mat4.h"
#include "../cube/ktx.h"

#include "tests/shaders/cube_inst_spv.h"  // g_cube_instSpv
#include "tests/shaders/cube_inst_wgsl.h" // g_cube_instWgsl
#include "tests/shaders/cube_inst_dxbc.h" // g_cube_instDxbcVS / PS

namespace
{
    constexpr uint32_t kWidth = 800, kHeight = 600;
    constexpr VriFormat kSwapFormat = VriFormat_BGRA8_UNORM;
    constexpr VriFormat kDepthFormat = VriFormat_D32_SFLOAT;
    constexpr int kGrid = 4; // kGrid^3 cubes

    void Fail(const char* msg) { std::fprintf(stderr, "[instancing] %s\n", msg); std::exit(1); }

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
    const float spacing = 2.2f, half = (kGrid - 1) * spacing * 0.5f;
    for (int z = 0; z < kGrid; ++z)
        for (int y = 0; y < kGrid; ++y)
            for (int x = 0; x < kGrid; ++x)
            {
                const float phase = float(x + y * 3 + z * 7) * 0.5f;
                Mat4 m = Mul(Translate(x * spacing - half, y * spacing - half, z * spacing - half), Mul(RotateY(phase), Scale(0.7f)));
                instances.push_back(m);
            }
    const uint32_t instanceNum = static_cast<uint32_t>(instances.size());
    const uint64_t instBytes = instanceNum * sizeof(Mat4);

    const char* apiEnv = std::getenv("VRI_API");
    VriGraphicsAPI api = VriGraphicsAPI_Vulkan;
    if (apiEnv && std::strcmp(apiEnv, "webgpu") == 0) api = VriGraphicsAPI_WebGPU;
    else if (apiEnv && (std::strcmp(apiEnv, "opengl") == 0 || std::strcmp(apiEnv, "gl") == 0)) api = VriGraphicsAPI_OpenGL;
    else if (apiEnv && (std::strcmp(apiEnv, "d3d12") == 0 || std::strcmp(apiEnv, "dx12") == 0)) api = VriGraphicsAPI_D3D12;
    const bool useWgsl = api == VriGraphicsAPI_WebGPU;
    const bool useDxbc = api == VriGraphicsAPI_D3D12;
    const char* apiName = api == VriGraphicsAPI_WebGPU ? "WebGPU" : (api == VriGraphicsAPI_OpenGL ? "OpenGL" : (api == VriGraphicsAPI_D3D12 ? "D3D12" : "Vulkan"));

    if (!SDL_Init(SDL_INIT_VIDEO)) Fail("SDL_Init failed");
    char title[64]; std::snprintf(title, sizeof(title), "VRI Instancing (%s)", apiName);
    SDL_Window* window = SDL_CreateWindow(title, kWidth, kHeight, 0);
    if (!window) Fail("SDL_CreateWindow failed");

    KtxImage tex;
    {
        const char* base = SDL_GetBasePath();
        std::string path = (base ? std::string(base) : std::string()) + "metalplate01_rgba.ktx";
        if (!LoadKtxRgba(path.c_str(), tex) && !LoadKtxRgba("metalplate01_rgba.ktx", tex))
            Fail("failed to load metalplate01_rgba.ktx");
    }

    VriDeviceCreationDesc dd{};
    dd.graphicsAPI = api; dd.enableValidation = VRI_TRUE; dd.bestEffort = VRI_TRUE;
    VriDevice* dev = nullptr;
    if (vriCreateDevice(&dd, &dev) != VriResult_Success) Fail("vriCreateDevice failed");

    VriCoreInterface c{}; VriSwapChainInterface swap{};
    if (vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) != VriResult_Success ||
        vriGetInterface(dev, VRI_INTERFACE_SWAPCHAIN, sizeof(swap), &swap) != VriResult_Success)
        Fail("vriGetInterface failed");

    VriQueue* queue = nullptr; c.GetQueue(dev, VriQueueType_Graphics, 0, &queue);

    VriSwapChainDesc scd{};
    scd.window = vriWindowHandleFromSDL3(window); scd.queue = queue; scd.format = kSwapFormat;
    scd.width = kWidth; scd.height = kHeight; scd.textureNum = 3; scd.vsync = VRI_TRUE;
    VriSwapChain* swapchain = nullptr;
    if (swap.CreateSwapChain(dev, &scd, &swapchain) != VriResult_Success) Fail("CreateSwapChain failed");

    VriTextureDesc dtd{};
    dtd.type = VriTextureType_2D; dtd.format = kDepthFormat; dtd.width = kWidth; dtd.height = kHeight; dtd.depth = 1;
    dtd.mipNum = 1; dtd.layerNum = 1; dtd.sampleNum = 1; dtd.usage = VriTextureUsage_DepthStencilAttachment; dtd.memoryLocation = VriMemoryLocation_Device;
    VriTexture* depth = nullptr;
    if (c.CreateTexture(dev, &dtd, &depth) != VriResult_Success) Fail("depth CreateTexture failed");
    VriTextureViewDesc dvd{}; dvd.texture = depth; dvd.viewType = VriTextureViewType_2D; dvd.format = VriFormat_Unknown; dvd.aspect = VriImageAspect_Depth;
    VriDescriptor* depthView = nullptr;
    if (c.CreateTextureView(dev, &dvd, &depthView) != VriResult_Success) Fail("depth view failed");

    // device-local vertex / index / instance buffers + their staging
    auto makeDeviceBuf = [&](uint64_t size, VriBufferUsageFlags usage, const void* src) {
        VriBufferDesc bd{}; bd.size = size; bd.usage = usage | VriBufferUsage_TransferDst; bd.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* b = nullptr; c.CreateBuffer(dev, &bd, &b);
        VriBufferDesc sd{}; sd.size = size; sd.usage = VriBufferUsage_TransferSrc; sd.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* s = nullptr; c.CreateBuffer(dev, &sd, &s);
        std::memcpy(c.MapBuffer(s, 0, size), src, size); c.UnmapBuffer(s);
        return std::pair<VriBuffer*, VriBuffer*>(b, s);
    };
    auto [vbuf, vstg] = makeDeviceBuf(sizeof(kCube), VriBufferUsage_VertexBuffer, kCube);
    auto [ibuf, istg] = makeDeviceBuf(sizeof(kIndices), VriBufferUsage_IndexBuffer, kIndices);
    auto [inst, inststg] = makeDeviceBuf(instBytes, VriBufferUsage_VertexBuffer, instances.data());

    // view-projection constant buffer (refreshed each frame via staging)
    VriBufferDesc ubd{}; ubd.size = sizeof(Mat4); ubd.usage = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst; ubd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* ubo = nullptr; c.CreateBuffer(dev, &ubd, &ubo);
    VriBufferDesc usd{}; usd.size = sizeof(Mat4); usd.usage = VriBufferUsage_TransferSrc; usd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* ustg = nullptr; c.CreateBuffer(dev, &usd, &ustg);
    VriBufferViewDesc ubv{}; ubv.buffer = ubo; ubv.viewType = VriDescriptorType_ConstantBuffer; ubv.offset = 0; ubv.size = sizeof(Mat4);
    VriDescriptor* uboView = nullptr; c.CreateBufferView(dev, &ubv, &uboView);

    // texture + staging + view + sampler
    VriTextureDesc ttd{};
    ttd.type = VriTextureType_2D; ttd.format = VriFormat_RGBA8_UNORM; ttd.width = tex.width; ttd.height = tex.height; ttd.depth = 1;
    ttd.mipNum = 1; ttd.layerNum = 1; ttd.sampleNum = 1; ttd.usage = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst; ttd.memoryLocation = VriMemoryLocation_Device;
    VriTexture* texture = nullptr; c.CreateTexture(dev, &ttd, &texture);
    VriTextureViewDesc tvd{}; tvd.texture = texture; tvd.viewType = VriTextureViewType_2D; tvd.format = VriFormat_Unknown; tvd.aspect = VriImageAspect_Color;
    VriDescriptor* texView = nullptr; c.CreateTextureView(dev, &tvd, &texView);
    VriBufferDesc stg{}; stg.size = tex.rgba.size(); stg.usage = VriBufferUsage_TransferSrc; stg.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* staging = nullptr; c.CreateBuffer(dev, &stg, &staging);
    std::memcpy(c.MapBuffer(staging, 0, tex.rgba.size()), tex.rgba.data(), tex.rgba.size()); c.UnmapBuffer(staging);
    VriSamplerDesc smp{}; smp.magFilter = VriFilter_Linear; smp.minFilter = VriFilter_Linear; smp.mipmapMode = VriMipmapMode_Linear;
    smp.addressModeU = VriAddressMode_Repeat; smp.addressModeV = VriAddressMode_Repeat; smp.addressModeW = VriAddressMode_Repeat; smp.maxLod = 1.0f;
    VriDescriptor* sampler = nullptr; c.CreateSampler(dev, &smp, &sampler);

    VriDescriptorRangeDesc ranges[3]{};
    ranges[0].baseRegister = 0; ranges[0].descriptorNum = 1; ranges[0].descriptorType = VriDescriptorType_ConstantBuffer; ranges[0].shaderStages = VriShaderStage_Vertex;
    ranges[1].baseRegister = 1; ranges[1].descriptorNum = 1; ranges[1].descriptorType = VriDescriptorType_Texture;        ranges[1].shaderStages = VriShaderStage_Fragment;
    ranges[2].baseRegister = 2; ranges[2].descriptorNum = 1; ranges[2].descriptorType = VriDescriptorType_Sampler;        ranges[2].shaderStages = VriShaderStage_Fragment;
    VriDescriptorSetDesc setDesc{}; setDesc.registerSpace = 0; setDesc.ranges = ranges; setDesc.rangeNum = 3;
    VriPipelineLayoutDesc ld{}; ld.descriptorSets = &setDesc; ld.descriptorSetNum = 1;
    VriPipelineLayout* layout = nullptr; c.CreatePipelineLayout(dev, &ld, &layout);

    VriShaderDesc sh[2]{};
    sh[0].stage = VriShaderStage_Vertex;   sh[0].entryPointName = "vertexMain";
    sh[1].stage = VriShaderStage_Fragment; sh[1].entryPointName = "fragmentMain";
    if (useDxbc) { sh[0].bytecode = g_cubeInstDxbcVS; sh[0].bytecodeSize = sizeof(g_cubeInstDxbcVS); sh[1].bytecode = g_cubeInstDxbcPS; sh[1].bytecodeSize = sizeof(g_cubeInstDxbcPS); }
    else { sh[0].bytecode = sh[1].bytecode = useWgsl ? static_cast<const void*>(g_cubeInstWgsl) : static_cast<const void*>(g_cubeInstSpv);
           sh[0].bytecodeSize = sh[1].bytecodeSize = useWgsl ? sizeof(g_cubeInstWgsl) : sizeof(g_cubeInstSpv); }

    // stream 0: per-vertex {pos, uv}; stream 1: per-instance model matrix (4 columns)
    VriVertexAttributeDesc attrs[6]{};
    attrs[0].format = VriFormat_RGB32_SFLOAT;  attrs[0].offset = 0;  attrs[0].streamIndex = 0;
    attrs[1].format = VriFormat_RG32_SFLOAT;   attrs[1].offset = 12; attrs[1].streamIndex = 0;
    for (int i = 0; i < 4; ++i) { attrs[2 + i].format = VriFormat_RGBA32_SFLOAT; attrs[2 + i].offset = uint32_t(i * 16); attrs[2 + i].streamIndex = 1; }
    VriVertexStreamDesc streams[2]{};
    streams[0].stride = sizeof(Vertex); streams[0].bindingSlot = 0; streams[0].stepRate = VriVertexStepRate_PerVertex;
    streams[1].stride = sizeof(Mat4);   streams[1].bindingSlot = 1; streams[1].stepRate = VriVertexStepRate_PerInstance;

    VriColorAttachmentDesc ca{}; ca.format = kSwapFormat; ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd{};
    pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
    pd.vertexInput.attributes = attrs; pd.vertexInput.attributeNum = 6; pd.vertexInput.streams = streams; pd.vertexInput.streamNum = 2;
    pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode = VriCullMode_Back; pd.rasterization.frontFace = VriFrontFace_CounterClockwise; pd.rasterization.lineWidth = 1.0f;
    pd.multisample.sampleNum = 1;
    pd.depthStencil.depthTest = VRI_TRUE; pd.depthStencil.depthWrite = VRI_TRUE; pd.depthStencil.depthCompareOp = VriCompareOp_Less;
    pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1; pd.outputMerger.depthStencilFormat = kDepthFormat;
    VriPipeline* pipeline = nullptr;
    if (c.CreateGraphicsPipeline(dev, &pd, &pipeline) != VriResult_Success) Fail("CreateGraphicsPipeline failed");

    VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 1; pdsc.constantBufferMaxNum = 1; pdsc.textureMaxNum = 1; pdsc.samplerMaxNum = 1;
    VriDescriptorPool* pool = nullptr; c.CreateDescriptorPool(dev, &pdsc, &pool);
    VriDescriptorSet* set = nullptr; c.AllocateDescriptorSets(pool, layout, 0, &set, 1);
    const VriDescriptor* d0[1] = {uboView}; const VriDescriptor* d1[1] = {texView}; const VriDescriptor* d2[1] = {sampler};
    VriDescriptorRangeUpdateDesc upd[3]{};
    upd[0].descriptors = d0; upd[0].descriptorNum = 1; upd[1].descriptors = d1; upd[1].descriptorNum = 1; upd[2].descriptors = d2; upd[2].descriptorNum = 1;
    c.UpdateDescriptorRanges(set, 0, 3, upd);

    VriCommandAllocator* alloc = nullptr; c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
    VriCommandBuffer* cmd = nullptr; c.CreateCommandBuffer(alloc, &cmd);
    VriFence* fence = nullptr; c.CreateFence(dev, 0, &fence);

    // one-time upload: vertex / index / instance / texture -> read state
    {
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
        VriFenceSubmitDesc sig{}; sig.fence = fence; sig.value = 1;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(queue, &sub); c.Wait(fence, 1);
    }

    const char* maxFramesEnv = std::getenv("VRI_MAX_FRAMES");
    const uint64_t maxFrames = maxFramesEnv ? std::strtoull(maxFramesEnv, nullptr, 10) : 0;
    uint64_t frameValue = 1;
    bool depthInit = false, running = true;
    float t = 0.0f;
    while (running)
    {
        SDL_Event e;
        while (SDL_PollEvent(&e)) if (e.type == SDL_EVENT_QUIT || e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) running = false;
        if (!running) break;

        // orbit the camera around the field
        t += 0.01f;
        const float r = kGrid * spacing + 6.0f;
        const float eye[3] = {std::cos(t) * r, 4.0f, std::sin(t) * r}, ctr[3] = {0, 0, 0}, up[3] = {0, 1, 0};
        Mat4 viewProj = Mul(Perspective(0.9f, float(kWidth) / float(kHeight), 0.1f, 200.0f), LookAt(eye, ctr, up));
        std::memcpy(c.MapBuffer(ustg, 0, sizeof(Mat4)), &viewProj, sizeof(Mat4)); c.UnmapBuffer(ustg);

        uint32_t index = 0;
        if (swap.AcquireNextTexture(swapchain, nullptr, 0, &index) == VriResult_OutOfDate) { swap.Resize(swapchain, kWidth, kHeight); continue; }
        VriTexture* backbuffers[8] = {}; uint32_t count = 8; swap.GetSwapChainTextures(swapchain, backbuffers, &count);
        VriTexture* backbuffer = backbuffers[index];
        VriTextureViewDesc bvd{}; bvd.texture = backbuffer; bvd.viewType = VriTextureViewType_2D; bvd.format = VriFormat_Unknown; bvd.aspect = VriImageAspect_Color;
        VriDescriptor* bbView = nullptr; c.CreateTextureView(dev, &bvd, &bbView);

        c.BeginCommandBuffer(cmd);
        VriBufferCopyDesc ucp{}; ucp.size = sizeof(Mat4); c.CmdCopyBuffer(cmd, ubo, ustg, &ucp);
        VriBufferBarrierDesc ub{}; ub.buffer = ubo; ub.before.access = VriAccess_CopyDestinationWrite; ub.before.stages = VriPipelineStage_Transfer;
        ub.after.access = VriAccess_ConstantBufferRead; ub.after.stages = VriPipelineStage_VertexShader;
        VriBarrierGroupDesc ubg{}; ubg.buffers = &ub; ubg.bufferNum = 1; c.CmdBarrier(cmd, &ubg);

        VriTextureBarrierDesc toColor{}; toColor.texture = backbuffer; toColor.before.layout = VriLayout_Undefined; toColor.before.stages = VriPipelineStage_None;
        toColor.after.access = VriAccess_ColorAttachmentWrite; toColor.after.layout = VriLayout_ColorAttachment; toColor.after.stages = VriPipelineStage_ColorAttachmentOutput; toColor.aspect = VriImageAspect_Color;
        VriTextureBarrierDesc toDepth{}; toDepth.texture = depth; toDepth.before.layout = depthInit ? VriLayout_DepthStencilAttachment : VriLayout_Undefined; toDepth.before.stages = VriPipelineStage_None;
        toDepth.after.access = VriAccess_DepthStencilAttachmentWrite; toDepth.after.layout = VriLayout_DepthStencilAttachment; toDepth.after.stages = VriPipelineStage_EarlyFragmentTests; toDepth.aspect = VriImageAspect_Depth;
        VriTextureBarrierDesc bgr[2] = {toColor, toDepth};
        VriBarrierGroupDesc g{}; g.textures = bgr; g.textureNum = 2; c.CmdBarrier(cmd, &g);
        depthInit = true;

        VriAttachmentDesc colorRT{}; colorRT.view = bbView; colorRT.loadOp = VriAttachmentLoadOp_Clear; colorRT.storeOp = VriAttachmentStoreOp_Store;
        colorRT.clearValue.color.f32[0] = 0.08f; colorRT.clearValue.color.f32[1] = 0.10f; colorRT.clearValue.color.f32[2] = 0.14f; colorRT.clearValue.color.f32[3] = 1.0f;
        VriAttachmentDesc depthRT{}; depthRT.view = depthView; depthRT.loadOp = VriAttachmentLoadOp_Clear; depthRT.storeOp = VriAttachmentStoreOp_DontCare; depthRT.clearValue.depthStencil.depth = 1.0f;
        VriAttachmentsDesc att{}; att.colors = &colorRT; att.colorNum = 1; att.depth = &depthRT; att.renderArea.width = kWidth; att.renderArea.height = kHeight; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, float(kWidth), float(kHeight), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect scis{0, 0, kWidth, kHeight}; c.CmdSetScissors(cmd, &scis, 1);
        c.CmdSetPipeline(cmd, pipeline);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, set);
        VriVertexBufferBinding vbs[2]{}; vbs[0].buffer = vbuf; vbs[0].offset = 0; vbs[1].buffer = inst; vbs[1].offset = 0;
        c.CmdSetVertexBuffers(cmd, 0, vbs, 2);
        c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);
        VriDrawIndexedDesc di{}; di.indexNum = 36; di.instanceNum = instanceNum; c.CmdDrawIndexed(cmd, &di);
        c.CmdEndRendering(cmd);

        VriTextureBarrierDesc toPresent{}; toPresent.texture = backbuffer; toPresent.before.access = VriAccess_ColorAttachmentWrite; toPresent.before.layout = VriLayout_ColorAttachment; toPresent.before.stages = VriPipelineStage_ColorAttachmentOutput;
        toPresent.after.layout = VriLayout_Present; toPresent.after.stages = VriPipelineStage_AllCommands; toPresent.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc gp{}; gp.textures = &toPresent; gp.textureNum = 1; c.CmdBarrier(cmd, &gp);
        c.EndCommandBuffer(cmd);

        VriFenceSubmitDesc sig{}; sig.fence = fence; sig.value = ++frameValue; sig.stages = VriPipelineStage_AllCommands;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(queue, &sub);
        c.Wait(fence, frameValue);
        swap.Present(swapchain, nullptr, 0);
        c.DestroyDescriptor(bbView);

        if (maxFrames != 0 && frameValue - 1 >= maxFrames) running = false;
    }

    c.DeviceWaitIdle(dev);
    c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
    c.DestroyDescriptor(sampler); c.DestroyDescriptor(texView); c.DestroyTexture(texture); c.DestroyBuffer(staging);
    c.DestroyDescriptor(uboView); c.DestroyBuffer(ustg); c.DestroyBuffer(ubo);
    c.DestroyBuffer(inststg); c.DestroyBuffer(inst); c.DestroyBuffer(istg); c.DestroyBuffer(vstg); c.DestroyBuffer(ibuf); c.DestroyBuffer(vbuf);
    c.DestroyDescriptor(depthView); c.DestroyTexture(depth);
    swap.DestroySwapChain(swapchain);
    vriDestroyDevice(dev);
    SDL_DestroyWindow(window); SDL_Quit();
    std::printf("[instancing] %s: %u cubes, %llu frames presented\n", apiName, instanceNum, static_cast<unsigned long long>(frameValue - 1));
    return 0;
}
