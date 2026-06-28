// Shadow mapping: a two-pass example. Pass 1 renders the scene's depth from a directional
// light's point of view into an offscreen DEPTH texture (depth-only render pass, no color).
// Pass 2 (the swapchain pass) renders the scene from the camera and, per pixel, projects into
// the light's clip space and reads the shadow map through a COMPARISON sampler (hardware PCF,
// SampleCmpLevelZero) to decide lit vs. shadowed. This exercises three RHI features no other
// example uses: comparison samplers, sampling a depth texture as a shader resource, and a
// depth-only pass. Each object's model is a per-draw PUSH CONSTANT (the same call works on every
// backend, including WebGPU, where it is emulated as a ring of dynamic-offset uniform slots).
// All through the shared scaffolding in examples/common/example_app.h (VRI_API / ?backend force one).
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "../cube/mat4.h"

#include "examples/shaders/shadow_depth_dxbc.h" // g_shadowDepthDxbcVS / PS (D3D12)
#include "examples/shaders/shadow_depth_spv.h"  // g_shadowDepthSpv  (Vulkan + OpenGL)
#include "examples/shaders/shadow_depth_wgsl.h" // g_shadowDepthWgsl (WebGPU)
#include "examples/shaders/shadow_scene_dxbc.h" // g_shadowSceneDxbcVS / PS
#include "examples/shaders/shadow_scene_spv.h"  // g_shadowSceneSpv
#include "examples/shaders/shadow_scene_wgsl.h" // g_shadowSceneWgsl

namespace
{
    constexpr uint32_t  kWidth = 640, kHeight = 480; // width*4 is 256-aligned (D3D12 readback pitch)
    constexpr uint32_t  kShadow       = 1024;        // shadow map resolution (square)
    constexpr VriFormat kShadowFormat = VriFormat_D32_SFLOAT;

    struct Vertex
    {
        float px, py, pz;
        float nx, ny, nz;
    };

    // A ground plane (4 verts) followed by a unit cube (24 verts, per-face normals), local space.
    const Vertex kVerts[] = {
        {-4, 0, -4, 0, 1, 0},
        {4, 0, -4, 0, 1, 0},
        {4, 0, 4, 0, 1, 0},
        {-4, 0, 4, 0, 1, 0}, // plane (+Y)
        {-0.5f, -0.5f, 0.5f, 0, 0, 1},
        {0.5f, -0.5f, 0.5f, 0, 0, 1},
        {0.5f, 0.5f, 0.5f, 0, 0, 1},
        {-0.5f, 0.5f, 0.5f, 0, 0, 1}, // +Z
        {0.5f, -0.5f, -0.5f, 0, 0, -1},
        {-0.5f, -0.5f, -0.5f, 0, 0, -1},
        {-0.5f, 0.5f, -0.5f, 0, 0, -1},
        {0.5f, 0.5f, -0.5f, 0, 0, -1}, // -Z
        {0.5f, -0.5f, 0.5f, 1, 0, 0},
        {0.5f, -0.5f, -0.5f, 1, 0, 0},
        {0.5f, 0.5f, -0.5f, 1, 0, 0},
        {0.5f, 0.5f, 0.5f, 1, 0, 0}, // +X
        {-0.5f, -0.5f, -0.5f, -1, 0, 0},
        {-0.5f, -0.5f, 0.5f, -1, 0, 0},
        {-0.5f, 0.5f, 0.5f, -1, 0, 0},
        {-0.5f, 0.5f, -0.5f, -1, 0, 0}, // -X
        {-0.5f, 0.5f, 0.5f, 0, 1, 0},
        {0.5f, 0.5f, 0.5f, 0, 1, 0},
        {0.5f, 0.5f, -0.5f, 0, 1, 0},
        {-0.5f, 0.5f, -0.5f, 0, 1, 0}, // +Y
        {-0.5f, -0.5f, -0.5f, 0, -1, 0},
        {0.5f, -0.5f, -0.5f, 0, -1, 0},
        {0.5f, -0.5f, 0.5f, 0, -1, 0},
        {-0.5f, -0.5f, 0.5f, 0, -1, 0}, // -Y
    };
    // 6 plane indices (verts 0..3) then 36 cube indices. The cube's base vertex (4, after the
    // plane's 4 verts) is baked into the index VALUES rather than passed as CmdDrawIndexed's
    // vertexOffset, because WebGL2's glDrawElements has no baseVertex.
    uint16_t kIndices[6 + 36];
    void     BuildIndices()
    {
        kIndices[0] = 0;
        kIndices[1] = 1;
        kIndices[2] = 2;
        kIndices[3] = 0;
        kIndices[4] = 2;
        kIndices[5] = 3;
        for (uint32_t f = 0; f < 6; ++f)
        {
            const uint16_t b = static_cast<uint16_t>(4 + f * 4); // +4: cube verts follow the plane's 4
            uint16_t*      o = kIndices + 6 + f * 6;
            o[0]             = b;
            o[1]             = uint16_t(b + 1);
            o[2]             = uint16_t(b + 2);
            o[3]             = b;
            o[4]             = uint16_t(b + 2);
            o[5]             = uint16_t(b + 3);
        }
    }

    struct Obj
    {
        Mat4     model;
        uint32_t indexNum, baseIndex;
        int32_t  vertexOffset;
    };
    std::vector<Obj> g_objs;

    struct SceneUbo
    {
        Mat4  camViewProj;
        Mat4  lightViewProj;
        float lightDir[4];
        float params[4];
    };
    bool g_shadowInit = false; // shadow map written at least once (layout tracking)
} // namespace

int main(int, char**)
{
    BuildIndices();

    static vriex::ExampleApp app;
    app.Init("shadowmapping", kWidth, kHeight, /*hasDepth*/ true);
    VriCoreInterface& c       = app.c;
    const bool        useWgsl = app.useWgsl, useDxbc = app.useDxbc;

    // The scene: one ground plane + three cubes resting/floating above it (static models).
    g_objs.push_back({Mat4 {}, 6, 0, 0}); // plane (identity)
    g_objs.push_back({Mul(Translate(-1.3f, 0.5f, -0.6f), RotateY(0.5f)), 36, 6, 0});
    g_objs.push_back({Translate(1.4f, 0.5f, 0.7f), 36, 6, 0});
    g_objs.push_back({Mul(Translate(0.1f, 1.1f, 1.2f), RotateY(0.9f)), 36, 6, 0});

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

    // the shadow map: a depth texture used BOTH as a depth attachment (pass 1) and a sampled
    // resource (pass 2). One view serves both (DSV on attach, SRV on sample).
    VriTextureDesc sd {};
    sd.type                          = VriTextureType_2D;
    sd.format                        = kShadowFormat;
    sd.width                         = kShadow;
    sd.height                        = kShadow;
    sd.depth                         = 1;
    sd.mipNum                        = 1;
    sd.layerNum                      = 1;
    sd.sampleNum                     = 1;
    sd.usage                         = VriTextureUsage_DepthStencilAttachment | VriTextureUsage_ShaderResource;
    sd.memoryLocation                = VriMemoryLocation_Device;
    sd.clearValue.depthStencil.depth = 1.0f; // matches the shadow-pass depth clear (D3D12 fast-clear)
    VriTexture* shadowTex            = nullptr;
    if (c.CreateTexture(app.dev, &sd, &shadowTex) != VriResult_Success)
        app.Fail("CreateTexture (shadow map) failed");
    VriTextureViewDesc svd {};
    svd.texture               = shadowTex;
    svd.viewType              = VriTextureViewType_2D;
    svd.format                = VriFormat_Unknown;
    svd.aspect                = VriImageAspect_Depth;
    VriDescriptor* shadowView = nullptr;
    c.CreateTextureView(app.dev, &svd, &shadowView);

    // a comparison sampler (PCF): compareEnable + LessOrEqual; clamp so off-map lookups stay lit.
    VriSamplerDesc cmp {};
    cmp.magFilter                = VriFilter_Linear;
    cmp.minFilter                = VriFilter_Linear;
    cmp.mipmapMode               = VriMipmapMode_Nearest;
    cmp.addressModeU             = VriAddressMode_ClampToEdge;
    cmp.addressModeV             = VriAddressMode_ClampToEdge;
    cmp.addressModeW             = VriAddressMode_ClampToEdge;
    cmp.compareEnable            = VRI_TRUE;
    cmp.compareOp                = VriCompareOp_LessOrEqual;
    cmp.maxLod                   = 1.0f;
    VriDescriptor* shadowSampler = nullptr;
    c.CreateSampler(app.dev, &cmp, &shadowSampler);

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

    // D3D12: push constants become root constants at a b# register; the UBO already takes b0, so
    // Slang puts the push-constant cbuffer at b1. VK/GL/WebGPU ignore baseRegister for push
    // constants (byte offset / named / reserved group), so b1 is safe everywhere.
    VriPushConstantDesc pc {};
    pc.baseRegister = 1;
    pc.size         = sizeof(Mat4);
    pc.shaderStages = VriShaderStage_Vertex;

    // ---- shadow pass layout/pipeline: CB@0 (vertex) + push-constant model; depth-only ----
    VriDescriptorRangeDesc dr {};
    dr.baseRegister   = 0;
    dr.descriptorNum  = 1;
    dr.descriptorType = VriDescriptorType_ConstantBuffer;
    dr.shaderStages   = VriShaderStage_Vertex;
    VriDescriptorSetDesc dss {};
    dss.registerSpace = 0;
    dss.ranges        = &dr;
    dss.rangeNum      = 1;
    VriPipelineLayoutDesc dld {};
    dld.descriptorSets              = &dss;
    dld.descriptorSetNum            = 1;
    dld.pushConstants               = &pc;
    dld.pushConstantNum             = 1;
    VriPipelineLayout* shadowLayout = nullptr;
    c.CreatePipelineLayout(app.dev, &dld, &shadowLayout);

    VriShaderDesc dsh[2] {};
    dsh[0].stage          = VriShaderStage_Vertex;
    dsh[0].entryPointName = "vertexMain";
    dsh[1].stage          = VriShaderStage_Fragment;
    dsh[1].entryPointName = "fragmentMain";
    if (useDxbc)
    {
        dsh[0].bytecode     = g_shadowDepthDxbcVS;
        dsh[0].bytecodeSize = sizeof(g_shadowDepthDxbcVS);
        dsh[1].bytecode     = g_shadowDepthDxbcPS;
        dsh[1].bytecodeSize = sizeof(g_shadowDepthDxbcPS);
    }
    else
    {
        dsh[0].bytecode = dsh[1].bytecode =
            useWgsl ? static_cast<const void*>(g_shadowDepthWgsl) : static_cast<const void*>(g_shadowDepthSpv);
        dsh[0].bytecodeSize = dsh[1].bytecodeSize = useWgsl ? sizeof(g_shadowDepthWgsl) : sizeof(g_shadowDepthSpv);
    }

    VriGraphicsPipelineDesc dpd {};
    dpd.pipelineLayout                  = shadowLayout;
    dpd.shaders                         = dsh;
    dpd.shaderNum                       = 2;
    dpd.vertexInput.attributes          = attrs;
    dpd.vertexInput.attributeNum        = useDxbc ? 2u : 1u;
    dpd.vertexInput.streams             = &stream;
    dpd.vertexInput.streamNum           = 1;
    dpd.inputAssembly.topology          = VriPrimitiveTopology_TriangleList;
    dpd.rasterization.cullMode          = VriCullMode_None;
    dpd.rasterization.frontFace         = VriFrontFace_CounterClockwise;
    dpd.rasterization.lineWidth         = 1.0f;
    dpd.multisample.sampleNum           = 1;
    dpd.depthStencil.depthTest          = VRI_TRUE;
    dpd.depthStencil.depthWrite         = VRI_TRUE;
    dpd.depthStencil.depthCompareOp     = VriCompareOp_Less;
    dpd.outputMerger.colorNum           = 0;
    dpd.outputMerger.depthStencilFormat = kShadowFormat; // depth-only
    VriPipeline* shadowPipeline         = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &dpd, &shadowPipeline) != VriResult_Success)
        app.Fail("shadow CreateGraphicsPipeline failed");

    // ---- main pass layout/pipeline: CB@0 + shadow Texture@1 + comparison Sampler@2 + push model ----
    VriDescriptorRangeDesc mr[3] {};
    mr[0].baseRegister   = 0;
    mr[0].descriptorNum  = 1;
    mr[0].descriptorType = VriDescriptorType_ConstantBuffer;
    mr[0].shaderStages   = VriShaderStage_Vertex | VriShaderStage_Fragment;
    mr[1].baseRegister   = 1;
    mr[1].descriptorNum  = 1;
    mr[1].descriptorType = VriDescriptorType_Texture;
    mr[1].shaderStages   = VriShaderStage_Fragment;
    mr[1].flags          = VriDescriptorRange_Comparison; // depth sampleType
    mr[2].baseRegister   = 2;
    mr[2].descriptorNum  = 1;
    mr[2].descriptorType = VriDescriptorType_Sampler;
    mr[2].shaderStages   = VriShaderStage_Fragment;
    mr[2].flags          = VriDescriptorRange_Comparison; // comparison sampler
    VriDescriptorSetDesc msd {};
    msd.registerSpace = 0;
    msd.ranges        = mr;
    msd.rangeNum      = 3;
    VriPipelineLayoutDesc mld {};
    mld.descriptorSets            = &msd;
    mld.descriptorSetNum          = 1;
    mld.pushConstants             = &pc;
    mld.pushConstantNum           = 1;
    VriPipelineLayout* mainLayout = nullptr;
    c.CreatePipelineLayout(app.dev, &mld, &mainLayout);

    VriShaderDesc msh[2] {};
    msh[0].stage          = VriShaderStage_Vertex;
    msh[0].entryPointName = "vertexMain";
    msh[1].stage          = VriShaderStage_Fragment;
    msh[1].entryPointName = "fragmentMain";
    if (useDxbc)
    {
        msh[0].bytecode     = g_shadowSceneDxbcVS;
        msh[0].bytecodeSize = sizeof(g_shadowSceneDxbcVS);
        msh[1].bytecode     = g_shadowSceneDxbcPS;
        msh[1].bytecodeSize = sizeof(g_shadowSceneDxbcPS);
    }
    else
    {
        msh[0].bytecode = msh[1].bytecode =
            useWgsl ? static_cast<const void*>(g_shadowSceneWgsl) : static_cast<const void*>(g_shadowSceneSpv);
        msh[0].bytecodeSize = msh[1].bytecodeSize = useWgsl ? sizeof(g_shadowSceneWgsl) : sizeof(g_shadowSceneSpv);
    }

    VriColorAttachmentDesc ca {};
    ca.format         = app.swapFormat;
    ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc mpd {};
    mpd.pipelineLayout                  = mainLayout;
    mpd.shaders                         = msh;
    mpd.shaderNum                       = 2;
    mpd.vertexInput.attributes          = attrs;
    mpd.vertexInput.attributeNum        = 2;
    mpd.vertexInput.streams             = &stream;
    mpd.vertexInput.streamNum           = 1;
    mpd.inputAssembly.topology          = VriPrimitiveTopology_TriangleList;
    mpd.rasterization.cullMode          = VriCullMode_None;
    mpd.rasterization.frontFace         = VriFrontFace_CounterClockwise;
    mpd.rasterization.lineWidth         = 1.0f;
    mpd.multisample.sampleNum           = 1;
    mpd.depthStencil.depthTest          = VRI_TRUE;
    mpd.depthStencil.depthWrite         = VRI_TRUE;
    mpd.depthStencil.depthCompareOp     = VriCompareOp_Less;
    mpd.outputMerger.colors             = &ca;
    mpd.outputMerger.colorNum           = 1;
    mpd.outputMerger.depthStencilFormat = app.depthFormat;
    VriPipeline* mainPipeline           = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &mpd, &mainPipeline) != VriResult_Success)
        app.Fail("main CreateGraphicsPipeline failed");

    // descriptor sets (shadow set: UBO only; main set: UBO + shadow map + comparison sampler)
    VriDescriptorPoolDesc pdsc {};
    pdsc.descriptorSetMaxNum  = 2;
    pdsc.constantBufferMaxNum = 2;
    pdsc.textureMaxNum        = 1;
    pdsc.samplerMaxNum        = 1;
    VriDescriptorPool* pool   = nullptr;
    c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* shadowSet = nullptr;
    c.AllocateDescriptorSets(pool, shadowLayout, 0, &shadowSet, 1);
    VriDescriptorSet* mainSet = nullptr;
    c.AllocateDescriptorSets(pool, mainLayout, 0, &mainSet, 1);
    {
        const VriDescriptor*         a[1] = {uboView};
        VriDescriptorRangeUpdateDesc u[1] {};
        u[0].descriptors   = a;
        u[0].descriptorNum = 1;
        c.UpdateDescriptorRanges(shadowSet, 0, 1, u);
    }
    {
        const VriDescriptor*         a[1] = {uboView};
        const VriDescriptor*         b[1] = {shadowView};
        const VriDescriptor*         s[1] = {shadowSampler};
        VriDescriptorRangeUpdateDesc u[3] {};
        u[0].descriptors   = a;
        u[0].descriptorNum = 1;
        u[1].descriptors   = b;
        u[1].descriptorNum = 1;
        u[2].descriptors   = s;
        u[2].descriptorNum = 1;
        c.UpdateDescriptorRanges(mainSet, 0, 3, u);
    }

    // one-time geometry upload
    app.BeginUpload();
    app.UploadBuffer(vbuf, kVerts, sizeof(kVerts), VriAccess_VertexBufferRead, VriPipelineStage_VertexInput);
    app.UploadBuffer(ibuf, kIndices, sizeof(kIndices), VriAccess_IndexBufferRead, VriPipelineStage_VertexInput);
    app.EndUpload();

    // light + camera. The light orbits (ImGui); the camera is fixed. Every backend now has [0,1]
    // window depth: the GL backend applies SPIRV-Cross fixup_clipspace on the no-clipControl path
    // (ES/WebGL2/pre-4.5 desktop), so WebGL2 is uncompressed [0,1] like all the rest - no remap.
    static float lightAzimuth = 0.9f, lightElev = 0.85f, depthBias = 0.0020f, ambient = 0.30f;
    float        zeroToOne = 1.0f;

    app.onUpdate = [ustg, zeroToOne](uint64_t) {
        SceneUbo    s {};
        const float ce = std::cos(lightElev), se = std::sin(lightElev);
        const float dist    = 9.0f;
        const float lpos[3] = {std::cos(lightAzimuth) * ce * dist, se * dist, std::sin(lightAzimuth) * ce * dist};
        const float ctr[3] = {0, 0, 0}, up[3] = {0, 1, 0};
        Mat4        lightView = LookAt(lpos, ctr, up);
        Mat4        lightProj = Ortho(-5.5f, 5.5f, -5.5f, 5.5f, 0.1f, 22.0f);
        s.lightViewProj       = Transpose(Mul(lightProj, lightView));

        const float eye[3] = {5.0f, 5.0f, 7.0f}, look[3] = {0, 0.4f, 0};
        Mat4        view = LookAt(eye, look, up);
        Mat4        proj = Perspective(0.7f, float(kWidth) / float(kHeight), 0.1f, 100.0f);
        s.camViewProj    = Transpose(Mul(proj, view));

        const float len = std::sqrt(lpos[0] * lpos[0] + lpos[1] * lpos[1] + lpos[2] * lpos[2]);
        s.lightDir[0]   = lpos[0] / len;
        s.lightDir[1]   = lpos[1] / len;
        s.lightDir[2]   = lpos[2] / len;
        s.lightDir[3]   = 0;
        s.params[0]     = depthBias;
        s.params[1]     = 1.0f / float(kShadow);
        s.params[2]     = zeroToOne;
        s.params[3]     = ambient;
        std::memcpy(app.c.MapBuffer(ustg, 0, sizeof(SceneUbo)), &s, sizeof(SceneUbo));
        app.c.UnmapBuffer(ustg);
    };
    app.onGui = [] {
        ImGui::SliderFloat("light azimuth", &lightAzimuth, 0.0f, 6.28f);
        ImGui::SliderFloat("light elevation", &lightElev, 0.25f, 1.45f);
        ImGui::SliderFloat("depth bias", &depthBias, 0.0f, 0.01f, "%.4f");
        ImGui::SliderFloat("ambient", &ambient, 0.0f, 0.6f);
    };

    // draw every object with its model push constant (the layout must already be bound)
    auto drawScene = [vbuf, ibuf](VriCommandBuffer* cmd, VriPipelineLayout* layout, VriDescriptorSet* set) {
        app.c.CmdSetPipelineLayout(cmd, layout);
        app.c.CmdSetDescriptorSet(cmd, 0, set);
        VriVertexBufferBinding vb {};
        vb.buffer = vbuf;
        vb.offset = 0;
        app.c.CmdSetVertexBuffers(cmd, 0, &vb, 1);
        app.c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);
        for (const Obj& o : g_objs)
        {
            Mat4 m = Transpose(o.model);
            app.c.CmdSetConstants(cmd, 0, &m, sizeof(Mat4));
            VriDrawIndexedDesc di {};
            di.indexNum     = o.indexNum;
            di.instanceNum  = 1;
            di.baseIndex    = o.baseIndex;
            di.vertexOffset = o.vertexOffset;
            app.c.CmdDrawIndexed(cmd, &di);
        }
    };

    // onPreRender refreshes the UBO then records the WHOLE shadow (depth-only) pass, leaving the
    // shadow map sampleable for the main pass.
    app.onPreRender = [=](VriCommandBuffer* cmd) {
        VriBufferCopyDesc ucp {};
        ucp.size = sizeof(SceneUbo);
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

        // shadow map -> depth-writable
        VriTextureBarrierDesc tw {};
        tw.texture       = shadowTex;
        tw.aspect        = VriImageAspect_Depth;
        tw.before.layout = g_shadowInit ? VriLayout_ShaderResource : VriLayout_Undefined;
        tw.before.stages = g_shadowInit ? VriPipelineStage_FragmentShader : VriPipelineStage_None;
        tw.after.access  = VriAccess_DepthStencilAttachmentWrite;
        tw.after.layout  = VriLayout_DepthStencilAttachment;
        tw.after.stages  = VriPipelineStage_EarlyFragmentTests;
        VriBarrierGroupDesc gw {};
        gw.textures   = &tw;
        gw.textureNum = 1;
        c.CmdBarrier(cmd, &gw);
        g_shadowInit = true;

        VriAttachmentDesc depthRT {};
        depthRT.view                          = shadowView;
        depthRT.loadOp                        = VriAttachmentLoadOp_Clear;
        depthRT.storeOp                       = VriAttachmentStoreOp_Store;
        depthRT.clearValue.depthStencil.depth = 1.0f;
        VriAttachmentsDesc att {};
        att.colorNum          = 0;
        att.depth             = &depthRT;
        att.renderArea.width  = kShadow;
        att.renderArea.height = kShadow;
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp {0, 0, float(kShadow), float(kShadow), 0, 1};
        c.CmdSetViewports(cmd, &vp, 1);
        VriRect scr {0, 0, kShadow, kShadow};
        c.CmdSetScissors(cmd, &scr, 1);
        c.CmdSetPipeline(cmd, shadowPipeline);
        drawScene(cmd, shadowLayout, shadowSet);
        c.CmdEndRendering(cmd);

        // shadow map -> sampleable by the main pass
        VriTextureBarrierDesc ts {};
        ts.texture       = shadowTex;
        ts.aspect        = VriImageAspect_Depth;
        ts.before.access = VriAccess_DepthStencilAttachmentWrite;
        ts.before.layout = VriLayout_DepthStencilAttachment;
        ts.before.stages = VriPipelineStage_LateFragmentTests;
        ts.after.access  = VriAccess_ShaderResourceRead;
        ts.after.layout  = VriLayout_ShaderResource;
        ts.after.stages  = VriPipelineStage_FragmentShader;
        VriBarrierGroupDesc gs {};
        gs.textures   = &ts;
        gs.textureNum = 1;
        c.CmdBarrier(cmd, &gs);
    };

    app.onRecord = [=](VriCommandBuffer* cmd) {
        c.CmdSetPipeline(cmd, mainPipeline);
        drawScene(cmd, mainLayout, mainSet);
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(mainPipeline);
    c.DestroyPipelineLayout(mainLayout);
    c.DestroyPipeline(shadowPipeline);
    c.DestroyPipelineLayout(shadowLayout);
    c.DestroyDescriptor(shadowSampler);
    c.DestroyDescriptor(shadowView);
    c.DestroyTexture(shadowTex);
    c.DestroyDescriptor(uboView);
    c.DestroyBuffer(ustg);
    c.DestroyBuffer(ubo);
    c.DestroyBuffer(ibuf);
    c.DestroyBuffer(vbuf);
    app.Shutdown();
#endif
    return 0;
}
