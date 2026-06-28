// Windowed textured cube: vertex+index buffers, an animated MVP constant buffer, a sampled
// texture (Sascha Willems .ktx), a depth buffer, and a 3-descriptor set - all through the
// same VRI calls on every backend. The host scaffolding (backend select, windowing, the
// present loop, headless capture) lives in examples/common/example_app.h; this file only
// builds the cube's resources + records its draw. VRI_API / ?backend force a backend.
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <string>

#include "ktx.h"
#include "mat4.h"

#include "common/shaders/cube_dxbc.h" // g_cubeDxbcVS / PS (D3D12)
#include "common/shaders/cube_spv.h"  // g_cubeSpv  (Vulkan + OpenGL)
#include "common/shaders/cube_wgsl.h" // g_cubeWgsl (WebGPU)

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480; // width*4 is 256-aligned (D3D12 readback pitch)

    struct Vertex
    {
        float px, py, pz;
        float u, v;
    };

    // 24 vertices (4 per face, per-face UVs), 36 indices. Unit cube centered at origin.
    const Vertex kCube[24] = {
        {-0.5f, -0.5f, 0.5f, 0, 1},  {0.5f, -0.5f, 0.5f, 1, 1},
        {0.5f, 0.5f, 0.5f, 1, 0},    {-0.5f, 0.5f, 0.5f, 0, 0}, // +Z
        {0.5f, -0.5f, -0.5f, 0, 1},  {-0.5f, -0.5f, -0.5f, 1, 1},
        {-0.5f, 0.5f, -0.5f, 1, 0},  {0.5f, 0.5f, -0.5f, 0, 0}, // -Z
        {0.5f, -0.5f, 0.5f, 0, 1},   {0.5f, -0.5f, -0.5f, 1, 1},
        {0.5f, 0.5f, -0.5f, 1, 0},   {0.5f, 0.5f, 0.5f, 0, 0}, // +X
        {-0.5f, -0.5f, -0.5f, 0, 1}, {-0.5f, -0.5f, 0.5f, 1, 1},
        {-0.5f, 0.5f, 0.5f, 1, 0},   {-0.5f, 0.5f, -0.5f, 0, 0}, // -X
        {-0.5f, 0.5f, 0.5f, 0, 1},   {0.5f, 0.5f, 0.5f, 1, 1},
        {0.5f, 0.5f, -0.5f, 1, 0},   {-0.5f, 0.5f, -0.5f, 0, 0}, // +Y
        {-0.5f, -0.5f, -0.5f, 0, 1}, {0.5f, -0.5f, -0.5f, 1, 1},
        {0.5f, -0.5f, 0.5f, 1, 0},   {-0.5f, -0.5f, 0.5f, 0, 0}, // -Y
    };
    uint16_t kIndices[36];
    void     BuildIndices()
    {
        for (uint32_t f = 0; f < 6; ++f)
        {
            const uint16_t b = static_cast<uint16_t>(f * 4);
            uint16_t*      o = kIndices + f * 6;
            o[0]             = b;
            o[1]             = uint16_t(b + 1);
            o[2]             = uint16_t(b + 2);
            o[3]             = b;
            o[4]             = uint16_t(b + 2);
            o[5]             = uint16_t(b + 3);
        }
    }
} // namespace

int main(int, char**)
{
    BuildIndices();

    static vriex::ExampleApp app;
    app.Init("cube", kWidth, kHeight, /*hasDepth*/ true);
    VriCoreInterface& c = app.c;

    const bool useWgsl = app.useWgsl, useDxbc = app.useDxbc;

    // load the .ktx texture (desktop: next to the exe; web: preloaded into MEMFS at root)
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

    // device-local vertex/index buffers uploaded via staging (WebGPU forbids MAP_WRITE on usage buffers)
    VriBufferDesc vbd {};
    vbd.size           = sizeof(kCube);
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
    VriBufferDesc vsd {};
    vsd.size           = sizeof(kCube);
    vsd.usage          = VriBufferUsage_TransferSrc;
    vsd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* vstg    = nullptr;
    c.CreateBuffer(app.dev, &vsd, &vstg);
    std::memcpy(c.MapBuffer(vstg, 0, sizeof(kCube)), kCube, sizeof(kCube));
    c.UnmapBuffer(vstg);
    VriBufferDesc isd {};
    isd.size           = sizeof(kIndices);
    isd.usage          = VriBufferUsage_TransferSrc;
    isd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* istg    = nullptr;
    c.CreateBuffer(app.dev, &isd, &istg);
    std::memcpy(c.MapBuffer(istg, 0, sizeof(kIndices)), kIndices, sizeof(kIndices));
    c.UnmapBuffer(istg);

    // MVP constant buffer: device-local, refreshed each frame from a host-mapped staging buffer
    VriBufferDesc ubd {};
    ubd.size           = sizeof(Mat4);
    ubd.usage          = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst;
    ubd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* ubo     = nullptr;
    c.CreateBuffer(app.dev, &ubd, &ubo);
    VriBufferDesc usd {};
    usd.size           = sizeof(Mat4);
    usd.usage          = VriBufferUsage_TransferSrc;
    usd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* ustg    = nullptr;
    c.CreateBuffer(app.dev, &usd, &ustg);
    VriBufferViewDesc ubv {};
    ubv.buffer             = ubo;
    ubv.viewType           = VriDescriptorType_ConstantBuffer;
    ubv.offset             = 0;
    ubv.size               = sizeof(Mat4);
    VriDescriptor* uboView = nullptr;
    c.CreateBufferView(app.dev, &ubv, &uboView);

    // texture (device-local) + staging upload + view + sampler
    VriTextureDesc ttd {};
    ttd.type            = VriTextureType_2D;
    ttd.format          = VriFormat_RGBA8_UNORM;
    ttd.width           = tex.width;
    ttd.height          = tex.height;
    ttd.depth           = 1;
    ttd.mipNum          = 1;
    ttd.layerNum        = 1;
    ttd.sampleNum       = 1;
    ttd.usage           = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst;
    ttd.memoryLocation  = VriMemoryLocation_Device;
    VriTexture* texture = nullptr;
    c.CreateTexture(app.dev, &ttd, &texture);
    VriTextureViewDesc tvd {};
    tvd.texture            = texture;
    tvd.viewType           = VriTextureViewType_2D;
    tvd.format             = VriFormat_Unknown;
    tvd.aspect             = VriImageAspect_Color;
    VriDescriptor* texView = nullptr;
    c.CreateTextureView(app.dev, &tvd, &texView);
    VriBufferDesc stg {};
    stg.size           = tex.rgba.size();
    stg.usage          = VriBufferUsage_TransferSrc;
    stg.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* staging = nullptr;
    c.CreateBuffer(app.dev, &stg, &staging);
    std::memcpy(c.MapBuffer(staging, 0, tex.rgba.size()), tex.rgba.data(), tex.rgba.size());
    c.UnmapBuffer(staging);
    VriSamplerDesc smp {};
    smp.magFilter          = VriFilter_Linear;
    smp.minFilter          = VriFilter_Linear;
    smp.mipmapMode         = VriMipmapMode_Linear;
    smp.addressModeU       = VriAddressMode_Repeat;
    smp.addressModeV       = VriAddressMode_Repeat;
    smp.addressModeW       = VriAddressMode_Repeat;
    smp.maxLod             = 1.0f;
    VriDescriptor* sampler = nullptr;
    c.CreateSampler(app.dev, &smp, &sampler);

    // pipeline layout: CB@0 (vertex), Texture@1 + Sampler@2 (fragment)
    VriDescriptorRangeDesc ranges[3] {};
    ranges[0].baseRegister   = 0;
    ranges[0].descriptorNum  = 1;
    ranges[0].descriptorType = VriDescriptorType_ConstantBuffer;
    ranges[0].shaderStages   = VriShaderStage_Vertex;
    ranges[1].baseRegister   = 1;
    ranges[1].descriptorNum  = 1;
    ranges[1].descriptorType = VriDescriptorType_Texture;
    ranges[1].shaderStages   = VriShaderStage_Fragment;
    ranges[2].baseRegister   = 2;
    ranges[2].descriptorNum  = 1;
    ranges[2].descriptorType = VriDescriptorType_Sampler;
    ranges[2].shaderStages   = VriShaderStage_Fragment;
    VriDescriptorSetDesc setDesc {};
    setDesc.registerSpace = 0;
    setDesc.ranges        = ranges;
    setDesc.rangeNum      = 3;
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
    if (useDxbc)
    {
        sh[0].bytecode     = g_cubeDxbcVS;
        sh[0].bytecodeSize = sizeof(g_cubeDxbcVS);
        sh[1].bytecode     = g_cubeDxbcPS;
        sh[1].bytecodeSize = sizeof(g_cubeDxbcPS);
    }
    else
    {
        sh[0].bytecode = sh[1].bytecode =
            useWgsl ? static_cast<const void*>(g_cubeWgsl) : static_cast<const void*>(g_cubeSpv);
        sh[0].bytecodeSize = sh[1].bytecodeSize = useWgsl ? sizeof(g_cubeWgsl) : sizeof(g_cubeSpv);
    }

    VriVertexAttributeDesc attrs[2] {};
    attrs[0].format      = VriFormat_RGB32_SFLOAT;
    attrs[0].offset      = 0;
    attrs[0].streamIndex = 0;
    attrs[1].format      = VriFormat_RG32_SFLOAT;
    attrs[1].offset      = 12;
    attrs[1].streamIndex = 0;
    VriVertexStreamDesc stream {};
    stream.stride      = sizeof(Vertex);
    stream.bindingSlot = 0;
    stream.stepRate    = VriVertexStepRate_PerVertex;

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
    pdsc.textureMaxNum        = 1;
    pdsc.samplerMaxNum        = 1;
    VriDescriptorPool* pool   = nullptr;
    c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* set = nullptr;
    c.AllocateDescriptorSets(pool, layout, 0, &set, 1);
    const VriDescriptor*         d0[1] = {uboView};
    const VriDescriptor*         d1[1] = {texView};
    const VriDescriptor*         d2[1] = {sampler};
    VriDescriptorRangeUpdateDesc upd[3] {};
    upd[0].descriptors   = d0;
    upd[0].descriptorNum = 1;
    upd[1].descriptors   = d1;
    upd[1].descriptorNum = 1;
    upd[2].descriptors   = d2;
    upd[2].descriptorNum = 1;
    c.UpdateDescriptorRanges(set, 0, 3, upd);

    // one-time upload: staging -> vertex/index/texture, transition to read state (fence value 1)
    {
        VriCommandBuffer* cmd = app.cmd;
        c.BeginCommandBuffer(cmd);
        VriBufferCopyDesc vcp {};
        vcp.size = sizeof(kCube);
        c.CmdCopyBuffer(cmd, vbuf, vstg, &vcp);
        VriBufferCopyDesc icp {};
        icp.size = sizeof(kIndices);
        c.CmdCopyBuffer(cmd, ibuf, istg, &icp);
        VriBufferBarrierDesc gb[2] {};
        gb[0].buffer        = vbuf;
        gb[0].before.access = VriAccess_CopyDestinationWrite;
        gb[0].before.stages = VriPipelineStage_Transfer;
        gb[0].after.access  = VriAccess_VertexBufferRead;
        gb[0].after.stages  = VriPipelineStage_VertexInput;
        gb[1].buffer        = ibuf;
        gb[1].before.access = VriAccess_CopyDestinationWrite;
        gb[1].before.stages = VriPipelineStage_Transfer;
        gb[1].after.access  = VriAccess_IndexBufferRead;
        gb[1].after.stages  = VriPipelineStage_VertexInput;
        VriBarrierGroupDesc gbg {};
        gbg.buffers   = gb;
        gbg.bufferNum = 2;
        c.CmdBarrier(cmd, &gbg);

        VriTextureBarrierDesc tb {};
        tb.texture       = texture;
        tb.before.layout = VriLayout_Undefined;
        tb.before.stages = VriPipelineStage_None;
        tb.after.access  = VriAccess_CopyDestinationWrite;
        tb.after.layout  = VriLayout_CopyDestination;
        tb.after.stages  = VriPipelineStage_Transfer;
        tb.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc g0 {};
        g0.textures   = &tb;
        g0.textureNum = 1;
        c.CmdBarrier(cmd, &g0);
        VriBufferTextureCopyDesc up {};
        up.texture.aspect   = VriImageAspect_Color;
        up.texture.layerNum = 1;
        up.texture.width    = tex.width;
        up.texture.height   = tex.height;
        c.CmdUploadBufferToTexture(cmd, texture, staging, &up);
        VriTextureBarrierDesc tb2 {};
        tb2.texture       = texture;
        tb2.before.access = VriAccess_CopyDestinationWrite;
        tb2.before.layout = VriLayout_CopyDestination;
        tb2.before.stages = VriPipelineStage_Transfer;
        tb2.after.access  = VriAccess_ShaderResourceRead;
        tb2.after.layout  = VriLayout_ShaderResource;
        tb2.after.stages  = VriPipelineStage_FragmentShader;
        tb2.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc g1 {};
        g1.textures   = &tb2;
        g1.textureNum = 1;
        c.CmdBarrier(cmd, &g1);
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

    // per-frame: refresh the MVP (Transpose for Slang's mul(mvp,pos)) then draw the cube.
    // The CPU staging write (onUpdate) runs before acquire; the copy into the device-local
    // constant buffer (onPreRender) is recorded before the render pass.
    static float spin = 1.0f, angle = 0.0f;
    static bool  paused = false; // ImGui-controlled
    app.onUpdate        = [ustg](uint64_t) {
        if (!paused)
            angle += 1.2f * spin * app.dt; // 1.2 rad/s (= 0.02/frame at 60fps)
        const float eye[3] = {0, 0, 3.0f}, ctr[3] = {0, 0, 0}, up[3] = {0, 1, 0};
        Mat4        model = Mul(RotateY(angle), RotateX(angle * 0.5f));
        Mat4        view  = LookAt(eye, ctr, up);
        Mat4        proj  = Perspective(0.9f, float(kWidth) / float(kHeight), 0.1f, 100.0f);
        Mat4        mvp   = Transpose(Mul(proj, Mul(view, model)));
        std::memcpy(app.c.MapBuffer(ustg, 0, sizeof(Mat4)), &mvp, sizeof(Mat4));
        app.c.UnmapBuffer(ustg);
    };
    app.onGui = [] {
        ImGui::SliderFloat("spin", &spin, 0.0f, 5.0f);
        ImGui::Checkbox("paused", &paused);
    };
    app.onPreRender = [ubo, ustg](VriCommandBuffer* cmd) {
        VriBufferCopyDesc ucp {};
        ucp.size = sizeof(Mat4);
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
    app.onRecord = [pipeline, layout, set, vbuf, ibuf](VriCommandBuffer* cmd) {
        app.c.CmdSetPipeline(cmd, pipeline);
        app.c.CmdSetPipelineLayout(cmd, layout);
        app.c.CmdSetDescriptorSet(cmd, 0, set);
        VriVertexBufferBinding vb {};
        vb.buffer = vbuf;
        vb.offset = 0;
        app.c.CmdSetVertexBuffers(cmd, 0, &vb, 1);
        app.c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);
        VriDrawIndexedDesc di {};
        di.indexNum    = 36;
        di.instanceNum = 1;
        app.c.CmdDrawIndexed(cmd, &di);
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyDescriptor(sampler);
    c.DestroyDescriptor(texView);
    c.DestroyTexture(texture);
    c.DestroyBuffer(staging);
    c.DestroyDescriptor(uboView);
    c.DestroyBuffer(ustg);
    c.DestroyBuffer(ubo);
    c.DestroyBuffer(istg);
    c.DestroyBuffer(vstg);
    c.DestroyBuffer(ibuf);
    c.DestroyBuffer(vbuf);
    app.Shutdown();
#endif
    return 0;
}
