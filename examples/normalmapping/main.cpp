// Tangent-space normal mapping: a textured quad lit by an orbiting point light, with the
// per-pixel normal taken from a normal map (so the flat quad shows surface relief). Two
// sampled textures (color + normal) + a constant buffer, ported in spirit from Sascha
// Willems' normalmapping. Assets: stonefloor01 color/normal .ktx. Shared host scaffolding in
// examples/common; runs on every backend.
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <string>
#include <utility>

#include "../cube/ktx.h"
#include "../cube/mat4.h"

#include "shaders/examples/normalmap_dxbc.h" // g_normalmapDxbcVS / PS (D3D12)
#include "shaders/examples/normalmap_spv.h"  // g_normalmapSpv  (Vulkan + OpenGL)
#include "shaders/examples/normalmap_wgsl.h" // g_normalmapWgsl (WebGPU)

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480;
    constexpr float    kPI = 3.14159265358979323846f;

    struct Vertex
    {
        float px, py, pz;
        float nx, ny, nz;
        float tx, ty, tz;
        float u, v;
    };

    // A quad in the XY plane facing +Z; normal +Z, tangent +X.
    const Vertex kQuad[4] = {
        {-1, -1, 0, 0, 0, 1, 1, 0, 0, 0, 1},
        {1, -1, 0, 0, 0, 1, 1, 0, 0, 1, 1},
        {1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0},
        {-1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0},
    };
    const uint16_t kIndices[6] = {0, 1, 2, 0, 2, 3};

    struct Ubo
    {
        Mat4  mvp;
        float lightPos[4];
        float camPos[4];
    };

    // device-local texture from a loaded RGBA8 image (staging upload + view).
    struct TexUpload
    {
        VriTexture*    tex;
        VriDescriptor* view;
        VriBuffer*     staging;
        uint32_t       w, h;
    };
    TexUpload MakeTexture(vriex::ExampleApp& app, const KtxImage& img)
    {
        VriCoreInterface& c = app.c;
        VriTextureDesc    td {};
        td.type           = VriTextureType_2D;
        td.format         = VriFormat_RGBA8_UNORM;
        td.width          = img.width;
        td.height         = img.height;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = 1;
        td.sampleNum      = 1;
        td.usage          = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* tex   = nullptr;
        c.CreateTexture(app.dev, &td, &tex);
        VriTextureViewDesc vd {};
        vd.texture          = tex;
        vd.viewType         = VriTextureViewType_2D;
        vd.format           = VriFormat_Unknown;
        vd.aspect           = VriImageAspect_Color;
        VriDescriptor* view = nullptr;
        c.CreateTextureView(app.dev, &vd, &view);
        VriBufferDesc sd {};
        sd.size            = img.rgba.size();
        sd.usage           = VriBufferUsage_TransferSrc;
        sd.memoryLocation  = VriMemoryLocation_HostUpload;
        VriBuffer* staging = nullptr;
        c.CreateBuffer(app.dev, &sd, &staging);
        std::memcpy(c.MapBuffer(staging, 0, img.rgba.size()), img.rgba.data(), img.rgba.size());
        c.UnmapBuffer(staging);
        return {tex, view, staging, img.width, img.height};
    }
} // namespace

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.Init("normalmapping", kWidth, kHeight, /*hasDepth*/ false);
    app.SetClearColor(0.02f, 0.02f, 0.03f);
    VriCoreInterface& c = app.c;

    auto loadTex = [&](const char* file) {
        KtxImage img;
#if defined(__EMSCRIPTEN__)
        std::string p = std::string("/") + file;
        if (!LoadKtxRgba(p.c_str(), img) && !LoadKtxRgba(file, img))
            app.Fail("failed to load texture");
#else
        const char* base = SDL_GetBasePath();
        std::string p    = (base ? std::string(base) : std::string()) + file;
        if (!LoadKtxRgba(p.c_str(), img) && !LoadKtxRgba(file, img))
            app.Fail("failed to load texture");
#endif
        return img;
    };
    const KtxImage colorImg  = loadTex("stonefloor01_color_rgba.ktx");
    const KtxImage normalImg = loadTex("stonefloor01_normal_rgba.ktx");

    // quad vertex/index buffers (device-local, via staging)
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
    auto [vbuf, vstg] = makeDeviceBuf(sizeof(kQuad), VriBufferUsage_VertexBuffer, kQuad);
    auto [ibuf, istg] = makeDeviceBuf(sizeof(kIndices), VriBufferUsage_IndexBuffer, kIndices);

    // constant buffer (mvp + light + camera), refreshed each frame
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

    TexUpload      color  = MakeTexture(app, colorImg);
    TexUpload      normal = MakeTexture(app, normalImg);
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

    // layout: CB@0 (vertex+fragment), colorTex@1, normalTex@2, sampler@3 (fragment)
    VriDescriptorRangeDesc ranges[4] {};
    ranges[0].baseRegister   = 0;
    ranges[0].descriptorNum  = 1;
    ranges[0].descriptorType = VriDescriptorType_ConstantBuffer;
    ranges[0].shaderStages   = VriShaderStage_Vertex | VriShaderStage_Fragment;
    ranges[1].baseRegister   = 1;
    ranges[1].descriptorNum  = 1;
    ranges[1].descriptorType = VriDescriptorType_Texture;
    ranges[1].shaderStages   = VriShaderStage_Fragment;
    ranges[2].baseRegister   = 2;
    ranges[2].descriptorNum  = 1;
    ranges[2].descriptorType = VriDescriptorType_Texture;
    ranges[2].shaderStages   = VriShaderStage_Fragment;
    ranges[3].baseRegister   = 3;
    ranges[3].descriptorNum  = 1;
    ranges[3].descriptorType = VriDescriptorType_Sampler;
    ranges[3].shaderStages   = VriShaderStage_Fragment;
    VriDescriptorSetDesc setDesc {};
    setDesc.registerSpace = 0;
    setDesc.ranges        = ranges;
    setDesc.rangeNum      = 4;
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
        sh[0].bytecode     = g_normalmapDxbcVS;
        sh[0].bytecodeSize = sizeof(g_normalmapDxbcVS);
        sh[1].bytecode     = g_normalmapDxbcPS;
        sh[1].bytecodeSize = sizeof(g_normalmapDxbcPS);
    }
    else
    {
        sh[0].bytecode = sh[1].bytecode =
            app.useWgsl ? static_cast<const void*>(g_normalmapWgsl) : static_cast<const void*>(g_normalmapSpv);
        sh[0].bytecodeSize = sh[1].bytecodeSize = app.useWgsl ? sizeof(g_normalmapWgsl) : sizeof(g_normalmapSpv);
    }

    VriVertexAttributeDesc attrs[4] {};
    attrs[0].format      = VriFormat_RGB32_SFLOAT;
    attrs[0].offset      = 0;
    attrs[0].streamIndex = 0; // position
    attrs[1].format      = VriFormat_RGB32_SFLOAT;
    attrs[1].offset      = 12;
    attrs[1].streamIndex = 0; // normal
    attrs[2].format      = VriFormat_RGB32_SFLOAT;
    attrs[2].offset      = 24;
    attrs[2].streamIndex = 0; // tangent
    attrs[3].format      = VriFormat_RG32_SFLOAT;
    attrs[3].offset      = 36;
    attrs[3].streamIndex = 0; // uv
    VriVertexStreamDesc stream {};
    stream.stride      = sizeof(Vertex);
    stream.bindingSlot = 0;
    stream.stepRate    = VriVertexStepRate_PerVertex;

    VriColorAttachmentDesc ca {};
    ca.format         = app.swapFormat;
    ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd {};
    pd.pipelineLayout           = layout;
    pd.shaders                  = sh;
    pd.shaderNum                = 2;
    pd.vertexInput.attributes   = attrs;
    pd.vertexInput.attributeNum = 4;
    pd.vertexInput.streams      = &stream;
    pd.vertexInput.streamNum    = 1;
    pd.inputAssembly.topology   = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode   = VriCullMode_None;
    pd.rasterization.frontFace  = VriFrontFace_CounterClockwise;
    pd.rasterization.lineWidth  = 1.0f;
    pd.multisample.sampleNum    = 1;
    pd.outputMerger.colors      = &ca;
    pd.outputMerger.colorNum    = 1;
    VriPipeline* pipeline       = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &pd, &pipeline) != VriResult_Success)
        app.Fail("CreateGraphicsPipeline failed");

    VriDescriptorPoolDesc pdsc {};
    pdsc.descriptorSetMaxNum  = 1;
    pdsc.constantBufferMaxNum = 1;
    pdsc.textureMaxNum        = 2;
    pdsc.samplerMaxNum        = 1;
    VriDescriptorPool* pool   = nullptr;
    c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* set = nullptr;
    c.AllocateDescriptorSets(pool, layout, 0, &set, 1);
    const VriDescriptor*         d0[1] = {uboView};
    const VriDescriptor*         d1[1] = {color.view};
    const VriDescriptor*         d2[1] = {normal.view};
    const VriDescriptor*         d3[1] = {sampler};
    VriDescriptorRangeUpdateDesc upd[4] {};
    upd[0].descriptors   = d0;
    upd[0].descriptorNum = 1;
    upd[1].descriptors   = d1;
    upd[1].descriptorNum = 1;
    upd[2].descriptors   = d2;
    upd[2].descriptorNum = 1;
    upd[3].descriptors   = d3;
    upd[3].descriptorNum = 1;
    c.UpdateDescriptorRanges(set, 0, 4, upd);

    // one-time upload: quad buffers + both textures -> read state (fence value 1)
    {
        VriCommandBuffer* cmd = app.cmd;
        c.BeginCommandBuffer(cmd);
        VriBufferCopyDesc cp {};
        cp.size = sizeof(kQuad);
        c.CmdCopyBuffer(cmd, vbuf, vstg, &cp);
        cp.size = sizeof(kIndices);
        c.CmdCopyBuffer(cmd, ibuf, istg, &cp);
        VriBufferBarrierDesc gb[2] {};
        gb[0].buffer       = vbuf;
        gb[0].after.access = VriAccess_VertexBufferRead;
        gb[1].buffer       = ibuf;
        gb[1].after.access = VriAccess_IndexBufferRead;
        for (auto& b : gb)
        {
            b.before.access = VriAccess_CopyDestinationWrite;
            b.before.stages = VriPipelineStage_Transfer;
            b.after.stages  = VriPipelineStage_VertexInput;
        }
        VriBarrierGroupDesc gbg {};
        gbg.buffers   = gb;
        gbg.bufferNum = 2;
        c.CmdBarrier(cmd, &gbg);

        TexUpload* texes[2] = {&color, &normal};
        for (TexUpload* t : texes)
        {
            VriTextureBarrierDesc tb {};
            tb.texture       = t->tex;
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
            up.texture.width    = t->w;
            up.texture.height   = t->h;
            c.CmdUploadBufferToTexture(cmd, t->tex, t->staging, &up);
            VriTextureBarrierDesc tb2 {};
            tb2.texture       = t->tex;
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
        }
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

    static float lightSpeed = 1.0f, t = 0.0f;
    static float lightZ = 1.3f; // ImGui-controlled light orbit
    app.onUpdate        = [ustg](uint64_t) {
        t += 1.2f * lightSpeed * app.dt; // 1.2/s (= 0.02/frame at 60fps)
        const float eye[3] = {0, 0, 2.8f}, ctr[3] = {0, 0, 0}, up[3] = {0, 1, 0};
        Ubo         u {};
        u.mvp = Transpose(Mul(Perspective(0.9f, float(kWidth) / float(kHeight), 0.1f, 100.0f), LookAt(eye, ctr, up)));
        u.lightPos[0] = std::cos(t) * 1.6f;
        u.lightPos[1] = std::sin(t) * 1.6f;
        u.lightPos[2] = lightZ;
        u.camPos[0]   = eye[0];
        u.camPos[1]   = eye[1];
        u.camPos[2]   = eye[2];
        std::memcpy(app.c.MapBuffer(ustg, 0, sizeof(Ubo)), &u, sizeof(Ubo));
        app.c.UnmapBuffer(ustg);
    };
    app.onGui = [] {
        ImGui::SliderFloat("light speed", &lightSpeed, 0.0f, 5.0f);
        ImGui::SliderFloat("light Z", &lightZ, 0.2f, 3.0f);
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
        di.indexNum    = 6;
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
    c.DestroyDescriptor(color.view);
    c.DestroyTexture(color.tex);
    c.DestroyBuffer(color.staging);
    c.DestroyDescriptor(normal.view);
    c.DestroyTexture(normal.tex);
    c.DestroyBuffer(normal.staging);
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
