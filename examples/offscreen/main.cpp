// Offscreen render-to-texture + MRT: pass 1 draws a textured cube into TWO offscreen color
// targets at once (albedo + a screen-derived face normal) plus an offscreen depth buffer;
// pass 2 (the swapchain pass) samples both offscreen targets and shows them side by side
// (albedo left, normals right). Exercises non-swapchain color attachments, multiple render
// targets in one pass, offscreen depth, and the ColorAttachment->ShaderResource barrier - all
// through the shared scaffolding in examples/common/example_app.h. VRI_API / ?backend force a
// backend. Geometry + the .ktx texture are reused from the cube example (no asset dup).
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <string>

#include "../cube/ktx.h"
#include "../cube/mat4.h"

#include "shaders/examples/composite_dxbc.h" // g_compositeDxbcVS / PS
#include "shaders/examples/composite_spv.h"  // g_compositeSpv
#include "shaders/examples/composite_wgsl.h" // g_compositeWgsl
#include "shaders/examples/offscreen_dxbc.h" // g_offscreenDxbcVS / PS (D3D12)
#include "shaders/examples/offscreen_spv.h"  // g_offscreenSpv  (Vulkan + OpenGL)
#include "shaders/examples/offscreen_wgsl.h" // g_offscreenWgsl (WebGPU)

namespace
{
    constexpr uint32_t  kWidth = 640, kHeight = 480;
    constexpr VriFormat kOffFormat = VriFormat_RGBA8_UNORM;

    struct Vertex
    {
        float px, py, pz;
        float u, v;
    };
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

    bool g_offInit      = false; // offscreen color targets written at least once (layout tracking)
    bool g_offDepthInit = false;

    // Create a device-local 2D color/depth attachment + its view in one shot.
    VriTexture* MakeAttachment(vriex::ExampleApp&   app,
                               VriFormat            fmt,
                               VriTextureUsageFlags usage,
                               VriImageAspectFlags  aspect,
                               VriDescriptor**      outView,
                               VriClearValue        clear = {})
    {
        VriTextureDesc td {};
        td.type           = VriTextureType_2D;
        td.format         = fmt;
        td.width          = kWidth;
        td.height         = kHeight;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = 1;
        td.sampleNum      = 1;
        td.usage          = usage;
        td.memoryLocation = VriMemoryLocation_Device;
        td.clearValue     = clear; // match the render-pass clear (D3D12 fast-clear)
        VriTexture* t     = nullptr;
        if (app.c.CreateTexture(app.dev, &td, &t) != VriResult_Success)
            app.Fail("offscreen CreateTexture failed");
        VriTextureViewDesc vd {};
        vd.texture  = t;
        vd.viewType = VriTextureViewType_2D;
        vd.format   = VriFormat_Unknown;
        vd.aspect   = aspect;
        if (app.c.CreateTextureView(app.dev, &vd, outView) != VriResult_Success)
            app.Fail("offscreen view failed");
        return t;
    }
} // namespace

int main(int, char**)
{
    BuildIndices();

    static vriex::ExampleApp app;
    app.Init("offscreen", kWidth, kHeight, /*hasDepth*/ false); // swapchain pass is a fullscreen composite, no depth
    VriCoreInterface& c       = app.c;
    const bool        useWgsl = app.useWgsl, useDxbc = app.useDxbc;

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

    // cube geometry (device-local via staging)
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

    // MVP constant buffer
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

    // albedo texture (device-local) + staging
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

    VriSamplerDesc rsmp {};
    rsmp.magFilter               = VriFilter_Linear;
    rsmp.minFilter               = VriFilter_Linear;
    rsmp.mipmapMode              = VriMipmapMode_Linear;
    rsmp.addressModeU            = VriAddressMode_Repeat;
    rsmp.addressModeV            = VriAddressMode_Repeat;
    rsmp.addressModeW            = VriAddressMode_Repeat;
    rsmp.maxLod                  = 1.0f;
    VriDescriptor* repeatSampler = nullptr;
    c.CreateSampler(app.dev, &rsmp, &repeatSampler);
    VriSamplerDesc csmp {};
    csmp.magFilter              = VriFilter_Linear;
    csmp.minFilter              = VriFilter_Linear;
    csmp.mipmapMode             = VriMipmapMode_Nearest;
    csmp.addressModeU           = VriAddressMode_ClampToEdge;
    csmp.addressModeV           = VriAddressMode_ClampToEdge;
    csmp.addressModeW           = VriAddressMode_ClampToEdge;
    csmp.maxLod                 = 1.0f;
    VriDescriptor* clampSampler = nullptr;
    c.CreateSampler(app.dev, &csmp, &clampSampler);

    // offscreen targets: 2 color (sampled later) + depth
    VriDescriptor* off0View     = nullptr;
    VriDescriptor* off1View     = nullptr;
    VriDescriptor* offDepthView = nullptr;
    VriClearValue  off0Clear {};
    off0Clear.color.f32[0] = 0.08f;
    off0Clear.color.f32[1] = 0.10f;
    off0Clear.color.f32[2] = 0.14f;
    off0Clear.color.f32[3] = 1.0f;
    VriClearValue offDepthClear {};
    offDepthClear.depthStencil.depth = 1.0f;
    VriTexture* off0                 = MakeAttachment(app,
                                      kOffFormat,
                                      VriTextureUsage_ColorAttachment | VriTextureUsage_ShaderResource,
                                      VriImageAspect_Color,
                                      &off0View,
                                      off0Clear);
    VriTexture* off1                 = MakeAttachment(app,
                                      kOffFormat,
                                      VriTextureUsage_ColorAttachment | VriTextureUsage_ShaderResource,
                                      VriImageAspect_Color,
                                      &off1View); // clears to black (default)
    VriTexture* offDepth             = MakeAttachment(app,
                                          app.depthFormat,
                                          VriTextureUsage_DepthStencilAttachment,
                                          VriImageAspect_Depth,
                                          &offDepthView,
                                          offDepthClear);

    // ---- offscreen pipeline: CB@0 (vertex), Texture@1 + Sampler@2 (fragment); 2 color targets + depth
    VriDescriptorRangeDesc or_[3] {};
    or_[0].baseRegister   = 0;
    or_[0].descriptorNum  = 1;
    or_[0].descriptorType = VriDescriptorType_ConstantBuffer;
    or_[0].shaderStages   = VriShaderStage_Vertex;
    or_[1].baseRegister   = 1;
    or_[1].descriptorNum  = 1;
    or_[1].descriptorType = VriDescriptorType_Texture;
    or_[1].shaderStages   = VriShaderStage_Fragment;
    or_[2].baseRegister   = 2;
    or_[2].descriptorNum  = 1;
    or_[2].descriptorType = VriDescriptorType_Sampler;
    or_[2].shaderStages   = VriShaderStage_Fragment;
    VriDescriptorSetDesc osd {};
    osd.registerSpace = 0;
    osd.ranges        = or_;
    osd.rangeNum      = 3;
    VriPipelineLayoutDesc old_ {};
    old_.descriptorSets          = &osd;
    old_.descriptorSetNum        = 1;
    VriPipelineLayout* offLayout = nullptr;
    c.CreatePipelineLayout(app.dev, &old_, &offLayout);

    VriShaderDesc osh[2] {};
    osh[0].stage          = VriShaderStage_Vertex;
    osh[0].entryPointName = "vertexMain";
    osh[1].stage          = VriShaderStage_Fragment;
    osh[1].entryPointName = "fragmentMain";
    if (useDxbc)
    {
        osh[0].bytecode     = g_offscreenDxbcVS;
        osh[0].bytecodeSize = sizeof(g_offscreenDxbcVS);
        osh[1].bytecode     = g_offscreenDxbcPS;
        osh[1].bytecodeSize = sizeof(g_offscreenDxbcPS);
    }
    else
    {
        osh[0].bytecode = osh[1].bytecode =
            useWgsl ? static_cast<const void*>(g_offscreenWgsl) : static_cast<const void*>(g_offscreenSpv);
        osh[0].bytecodeSize = osh[1].bytecodeSize = useWgsl ? sizeof(g_offscreenWgsl) : sizeof(g_offscreenSpv);
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

    VriColorAttachmentDesc oca[2] {};
    oca[0].format         = kOffFormat;
    oca[0].colorWriteMask = VriColorWrite_RGBA;
    oca[1].format         = kOffFormat;
    oca[1].colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc opd {};
    opd.pipelineLayout                  = offLayout;
    opd.shaders                         = osh;
    opd.shaderNum                       = 2;
    opd.vertexInput.attributes          = attrs;
    opd.vertexInput.attributeNum        = 2;
    opd.vertexInput.streams             = &stream;
    opd.vertexInput.streamNum           = 1;
    opd.inputAssembly.topology          = VriPrimitiveTopology_TriangleList;
    opd.rasterization.cullMode          = VriCullMode_None;
    opd.rasterization.frontFace         = VriFrontFace_CounterClockwise;
    opd.rasterization.lineWidth         = 1.0f;
    opd.multisample.sampleNum           = 1;
    opd.depthStencil.depthTest          = VRI_TRUE;
    opd.depthStencil.depthWrite         = VRI_TRUE;
    opd.depthStencil.depthCompareOp     = VriCompareOp_Less;
    opd.outputMerger.colors             = oca;
    opd.outputMerger.colorNum           = 2;
    opd.outputMerger.depthStencilFormat = app.depthFormat;
    VriPipeline* offPipeline            = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &opd, &offPipeline) != VriResult_Success)
        app.Fail("offscreen CreateGraphicsPipeline failed");

    // ---- composite pipeline: Texture@0 + Texture@1 + Sampler@2 (fragment); 1 swapchain color
    VriDescriptorRangeDesc cr[3] {};
    cr[0].baseRegister   = 0;
    cr[0].descriptorNum  = 1;
    cr[0].descriptorType = VriDescriptorType_Texture;
    cr[0].shaderStages   = VriShaderStage_Fragment;
    cr[1].baseRegister   = 1;
    cr[1].descriptorNum  = 1;
    cr[1].descriptorType = VriDescriptorType_Texture;
    cr[1].shaderStages   = VriShaderStage_Fragment;
    cr[2].baseRegister   = 2;
    cr[2].descriptorNum  = 1;
    cr[2].descriptorType = VriDescriptorType_Sampler;
    cr[2].shaderStages   = VriShaderStage_Fragment;
    VriDescriptorSetDesc csd {};
    csd.registerSpace = 0;
    csd.ranges        = cr;
    csd.rangeNum      = 3;
    VriPipelineLayoutDesc cld {};
    cld.descriptorSets            = &csd;
    cld.descriptorSetNum          = 1;
    VriPipelineLayout* compLayout = nullptr;
    c.CreatePipelineLayout(app.dev, &cld, &compLayout);

    VriShaderDesc csh[2] {};
    csh[0].stage          = VriShaderStage_Vertex;
    csh[0].entryPointName = "vertexMain";
    csh[1].stage          = VriShaderStage_Fragment;
    csh[1].entryPointName = "fragmentMain";
    if (useDxbc)
    {
        csh[0].bytecode     = g_compositeDxbcVS;
        csh[0].bytecodeSize = sizeof(g_compositeDxbcVS);
        csh[1].bytecode     = g_compositeDxbcPS;
        csh[1].bytecodeSize = sizeof(g_compositeDxbcPS);
    }
    else
    {
        csh[0].bytecode = csh[1].bytecode =
            useWgsl ? static_cast<const void*>(g_compositeWgsl) : static_cast<const void*>(g_compositeSpv);
        csh[0].bytecodeSize = csh[1].bytecodeSize = useWgsl ? sizeof(g_compositeWgsl) : sizeof(g_compositeSpv);
    }

    VriColorAttachmentDesc cca {};
    cca.format         = app.swapFormat;
    cca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc cpd {};
    cpd.pipelineLayout          = compLayout;
    cpd.shaders                 = csh;
    cpd.shaderNum               = 2;
    cpd.inputAssembly.topology  = VriPrimitiveTopology_TriangleList;
    cpd.rasterization.cullMode  = VriCullMode_None;
    cpd.rasterization.lineWidth = 1.0f;
    cpd.multisample.sampleNum   = 1;
    cpd.outputMerger.colors     = &cca;
    cpd.outputMerger.colorNum   = 1;
    VriPipeline* compPipeline   = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &cpd, &compPipeline) != VriResult_Success)
        app.Fail("composite CreateGraphicsPipeline failed");

    // descriptor sets
    VriDescriptorPoolDesc pdsc {};
    pdsc.descriptorSetMaxNum  = 2;
    pdsc.constantBufferMaxNum = 1;
    pdsc.textureMaxNum        = 3;
    pdsc.samplerMaxNum        = 2;
    VriDescriptorPool* pool   = nullptr;
    c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* offSet = nullptr;
    c.AllocateDescriptorSets(pool, offLayout, 0, &offSet, 1);
    VriDescriptorSet* compSet = nullptr;
    c.AllocateDescriptorSets(pool, compLayout, 0, &compSet, 1);
    {
        const VriDescriptor*         a[1] = {uboView};
        const VriDescriptor*         b[1] = {texView};
        const VriDescriptor*         s[1] = {repeatSampler};
        VriDescriptorRangeUpdateDesc u[3] {};
        u[0].descriptors   = a;
        u[0].descriptorNum = 1;
        u[1].descriptors   = b;
        u[1].descriptorNum = 1;
        u[2].descriptors   = s;
        u[2].descriptorNum = 1;
        c.UpdateDescriptorRanges(offSet, 0, 3, u);
    }
    {
        const VriDescriptor*         a[1] = {off0View};
        const VriDescriptor*         b[1] = {off1View};
        const VriDescriptor*         s[1] = {clampSampler};
        VriDescriptorRangeUpdateDesc u[3] {};
        u[0].descriptors   = a;
        u[0].descriptorNum = 1;
        u[1].descriptors   = b;
        u[1].descriptorNum = 1;
        u[2].descriptors   = s;
        u[2].descriptorNum = 1;
        c.UpdateDescriptorRanges(compSet, 0, 3, u);
    }

    // one-time upload (fence value 1)
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

    // onPreRender records the ENTIRE offscreen MRT pass (it runs after BeginCommandBuffer but
    // before the swapchain render pass opens) and leaves both color targets sampleable.
    app.onPreRender = [=](VriCommandBuffer* cmd) {
        // refresh MVP
        VriBufferCopyDesc ucp {};
        ucp.size = sizeof(Mat4);
        c.CmdCopyBuffer(cmd, ubo, ustg, &ucp);
        VriBufferBarrierDesc ub {};
        ub.buffer        = ubo;
        ub.before.access = VriAccess_CopyDestinationWrite;
        ub.before.stages = VriPipelineStage_Transfer;
        ub.after.access  = VriAccess_ConstantBufferRead;
        ub.after.stages  = VriPipelineStage_VertexShader;
        VriBarrierGroupDesc ubg {};
        ubg.buffers   = &ub;
        ubg.bufferNum = 1;
        c.CmdBarrier(cmd, &ubg);

        // offscreen targets -> writable (colors + depth)
        VriTextureBarrierDesc tb[3] {};
        tb[0].texture       = off0;
        tb[0].before.layout = g_offInit ? VriLayout_ShaderResource : VriLayout_Undefined;
        tb[0].before.stages = g_offInit ? VriPipelineStage_FragmentShader : VriPipelineStage_None;
        tb[0].after.access  = VriAccess_ColorAttachmentWrite;
        tb[0].after.layout  = VriLayout_ColorAttachment;
        tb[0].after.stages  = VriPipelineStage_ColorAttachmentOutput;
        tb[0].aspect        = VriImageAspect_Color;
        tb[1]               = tb[0];
        tb[1].texture       = off1;
        tb[2].texture       = offDepth;
        tb[2].before.layout = g_offDepthInit ? VriLayout_DepthStencilAttachment : VriLayout_Undefined;
        tb[2].before.stages = VriPipelineStage_None;
        tb[2].after.access  = VriAccess_DepthStencilAttachmentWrite;
        tb[2].after.layout  = VriLayout_DepthStencilAttachment;
        tb[2].after.stages  = VriPipelineStage_EarlyFragmentTests;
        tb[2].aspect        = VriImageAspect_Depth;
        VriBarrierGroupDesc gw {};
        gw.textures   = tb;
        gw.textureNum = 3;
        c.CmdBarrier(cmd, &gw);
        g_offInit      = true;
        g_offDepthInit = true;

        VriAttachmentDesc colors[2] {};
        colors[0].view                    = off0View;
        colors[0].loadOp                  = VriAttachmentLoadOp_Clear;
        colors[0].storeOp                 = VriAttachmentStoreOp_Store;
        colors[0].clearValue.color.f32[0] = 0.08f;
        colors[0].clearValue.color.f32[1] = 0.10f;
        colors[0].clearValue.color.f32[2] = 0.14f;
        colors[0].clearValue.color.f32[3] = 1.0f;
        colors[1].view                    = off1View;
        colors[1].loadOp                  = VriAttachmentLoadOp_Clear;
        colors[1].storeOp                 = VriAttachmentStoreOp_Store; // normal target clears to black
        VriAttachmentDesc depthRT {};
        depthRT.view                          = offDepthView;
        depthRT.loadOp                        = VriAttachmentLoadOp_Clear;
        depthRT.storeOp                       = VriAttachmentStoreOp_DontCare;
        depthRT.clearValue.depthStencil.depth = 1.0f;
        VriAttachmentsDesc att {};
        att.colors            = colors;
        att.colorNum          = 2;
        att.depth             = &depthRT;
        att.renderArea.width  = kWidth;
        att.renderArea.height = kHeight;
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp {0, 0, float(kWidth), float(kHeight), 0, 1};
        c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc {0, 0, kWidth, kHeight};
        c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, offPipeline);
        c.CmdSetPipelineLayout(cmd, offLayout);
        c.CmdSetDescriptorSet(cmd, 0, offSet);
        VriVertexBufferBinding vb {};
        vb.buffer = vbuf;
        vb.offset = 0;
        c.CmdSetVertexBuffers(cmd, 0, &vb, 1);
        c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);
        VriDrawIndexedDesc di {};
        di.indexNum    = 36;
        di.instanceNum = 1;
        c.CmdDrawIndexed(cmd, &di);
        c.CmdEndRendering(cmd);

        // offscreen colors -> sampleable by the composite pass
        VriTextureBarrierDesc ts[2] {};
        ts[0].texture       = off0;
        ts[0].before.access = VriAccess_ColorAttachmentWrite;
        ts[0].before.layout = VriLayout_ColorAttachment;
        ts[0].before.stages = VriPipelineStage_ColorAttachmentOutput;
        ts[0].after.access  = VriAccess_ShaderResourceRead;
        ts[0].after.layout  = VriLayout_ShaderResource;
        ts[0].after.stages  = VriPipelineStage_FragmentShader;
        ts[0].aspect        = VriImageAspect_Color;
        ts[1]               = ts[0];
        ts[1].texture       = off1;
        VriBarrierGroupDesc gs {};
        gs.textures   = ts;
        gs.textureNum = 2;
        c.CmdBarrier(cmd, &gs);
    };

    app.onRecord = [=](VriCommandBuffer* cmd) {
        c.CmdSetPipeline(cmd, compPipeline);
        c.CmdSetPipelineLayout(cmd, compLayout);
        c.CmdSetDescriptorSet(cmd, 0, compSet);
        VriDrawDesc d {};
        d.vertexNum   = 3;
        d.instanceNum = 1;
        c.CmdDraw(cmd, &d);
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(compPipeline);
    c.DestroyPipelineLayout(compLayout);
    c.DestroyPipeline(offPipeline);
    c.DestroyPipelineLayout(offLayout);
    c.DestroyDescriptor(off0View);
    c.DestroyDescriptor(off1View);
    c.DestroyDescriptor(offDepthView);
    c.DestroyTexture(off0);
    c.DestroyTexture(off1);
    c.DestroyTexture(offDepth);
    c.DestroyDescriptor(clampSampler);
    c.DestroyDescriptor(repeatSampler);
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
