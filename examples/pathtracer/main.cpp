// Progressive path tracer with diffuse global illumination over the FlightHelmet glTF model. A single
// compute megakernel (rt_pathtrace.slang) does ray generation + inline ray-query traversal + the
// multi-bounce shading loop - Metal's native ray-tracing model, and the natural shape for a path
// tracer. The sky gradient is the only light, so indirect bounces give real GI (color bleeding,
// contact shadows, sky occlusion). Each frame adds one jittered sample per pixel into a persistent
// accumulation buffer; the image converges while the camera is still and RESETS when it/lighting moves.
//
// Needs ray query (Metal, Vulkan, D3D12 inline). OpenGL/WebGPU/WebGL2 (no ray query) degrade explicitly.
#include "../common/example_app.h"

#if !defined(__EMSCRIPTEN__)
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "../common/gltf_model.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tests/shaders/show_tex_spv.h"
#include "tests/shaders/show_tex_dxbc.h"
#include "tests/shaders/rt_pathtrace_spv.h"      // g_rtPathtraceSpv (Metal/VK inline ray-query megakernel)
#if defined(_WIN32)
#    include "tests/shaders/rt_pathtrace_dxil.h" // g_rtPathtraceDxil (D3D12 inline)
#endif

#if !defined(VRI_GLTF_MODEL_PATH)
#    define VRI_GLTF_MODEL_PATH "FlightHelmet/glTF/FlightHelmet.gltf"
#endif

namespace
{
    constexpr uint32_t kWidth = 960, kHeight = 720;
    constexpr uint32_t kMaxTextures = 64; // must match rt_pathtrace.slang

    struct PtCamera { float eye[4]; float camRight[4]; float camUp[4]; float camFwd[4]; float sky[4]; float control[4]; };
    struct GeometryNode { uint32_t firstIndex; int32_t baseColorTexture; int32_t occlusionTexture; uint32_t pad; };
    struct AsInstance { float transform[12]; uint32_t instanceIdAndMask; uint32_t sbtOffsetAndFlags; uint64_t blasReference; };

    struct V3 { float x, y, z; };
    V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    V3 cross3(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    V3 norm3(V3 a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? V3{a.x / l, a.y / l, a.z / l} : a; }
    void set4(float* d, V3 v, float w = 0.0f) { d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = w; }
} // namespace
#endif

int main(int argc, char** argv)
{
#if defined(__EMSCRIPTEN__)
    (void)argc; (void)argv;
    static vriex::ExampleApp app;
    app.Init("pathtracer", 960, 720, /*hasDepth*/ false);
    app.SetClearColor(0.0f, 0.0f, 0.0f);
    std::printf("[pathtracer] ray query is unavailable on the web - path tracing unsupported\n");
    app.SetupCapture(); app.Run(); return 0;
#else
    static vriex::ExampleApp app;
    app.requestFeatures = VriFeature_RayQuery | VriFeature_RayTracing; // ray query is what we use
    app.Init("pathtracer", kWidth, kHeight, /*hasDepth*/ false);
    app.SetClearColor(0.0f, 0.0f, 0.0f);
    VriCoreInterface& c = app.c;
    const VriDeviceDesc* dd = c.GetDeviceDesc(app.dev);

    VriRayTracingInterface rt{};
    const bool hasRayQuery = dd->hasRayQuery != VRI_FALSE &&
        vriGetInterface(app.dev, VRI_INTERFACE_RAYTRACING, sizeof(rt), &rt) == VriResult_Success;

    static float orbit = 1.4f, lightAzimuth = 0.7f, skyIntensity = 1.2f;
    static int maxBounces = 4;
    static int sampleCount = 0;   // accumulated samples (drives convergence + the accumulate/reset logic)
    static bool dirty = true;     // camera/lighting changed -> restart accumulation

    app.onGui = [hasRayQuery] {
        if (!hasRayQuery) { ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "path tracing unsupported"); ImGui::Text("needs ray query (Metal/Vulkan/D3D12)"); return; }
        ImGui::Text("megakernel path tracer (inline ray query)");
        ImGui::Text("samples: %d", sampleCount);
        if (ImGui::SliderFloat("camera orbit", &orbit, 0.0f, 6.28f)) dirty = true;
        if (ImGui::SliderInt("max bounces", &maxBounces, 1, 8)) dirty = true;
        if (ImGui::SliderFloat("sky intensity", &skyIntensity, 0.0f, 3.0f)) dirty = true;
        if (ImGui::SliderFloat("light azimuth", &lightAzimuth, 0.0f, 6.28f)) dirty = true;
    };

    if (!hasRayQuery)
    {
        std::printf("[pathtracer] no ray query on this backend - degrading\n"); std::fflush(stdout);
        app.SetupCapture(); app.Run(); return 0;
    }

    // ---- model + BLAS/TLAS (identical to examples/raytracing's shared setup) ----
    const char* modelPath = argc > 1 ? argv[1] : VRI_GLTF_MODEL_PATH;
    vriex::GltfModel model;
    if (!model.Load(app, modelPath)) app.Fail("failed to load glTF model");

    std::vector<VriAsGeometryDesc> geoms(model.primitives.size());
    std::vector<GeometryNode> nodes(model.primitives.size());
    for (size_t i = 0; i < model.primitives.size(); ++i)
    {
        const vriex::GltfPrimitive& p = model.primitives[i];
        VriAsGeometryDesc& g = geoms[i];
        g.type = VriAsGeometryType_Triangles; g.flags = VriAsGeometry_Opaque;
        g.triangles.vertexBuffer = model.vertexBuffer; g.triangles.vertexCount = model.vertexCount;
        g.triangles.vertexStride = sizeof(vriex::GltfVertex); g.triangles.vertexFormat = VriFormat_RGB32_SFLOAT;
        g.triangles.indexBuffer = model.indexBuffer; g.triangles.indexOffset = uint64_t(p.firstIndex) * sizeof(uint32_t);
        g.triangles.indexCount = p.indexCount; g.triangles.indexType = VriIndexType_UInt32;
        nodes[i] = {p.firstIndex, p.baseColorTexture, p.occlusionTexture, 0};
    }
    VriAccelerationStructureDesc blasDesc{}; blasDesc.type = VriAccelerationStructureType_BottomLevel; blasDesc.flags = VriAccelerationStructureBuild_PreferFastTrace;
    blasDesc.geometryCount = uint32_t(geoms.size()); blasDesc.geometries = geoms.data();
    VriAccelerationStructure* blas = nullptr;
    if (rt.CreateAccelerationStructure(app.dev, &blasDesc, &blas) != VriResult_Success) app.Fail("CreateAccelerationStructure (BLAS) failed");

    AsInstance inst{}; inst.transform[0] = 1.0f; inst.transform[5] = 1.0f; inst.transform[10] = 1.0f;
    inst.instanceIdAndMask = 0xFFu << 24; inst.sbtOffsetAndFlags = 0x1u << 24;
    inst.blasReference = rt.GetAccelerationStructureDeviceAddress(blas);
    VriBufferDesc ibd{}; ibd.size = sizeof(AsInstance); ibd.usage = VriBufferUsage_AccelerationBuildInput; ibd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* instBuf = nullptr; c.CreateBuffer(app.dev, &ibd, &instBuf);
    std::memcpy(c.MapBuffer(instBuf, 0, sizeof(inst)), &inst, sizeof(inst)); c.UnmapBuffer(instBuf);

    VriAsGeometryDesc tlasGeom{}; tlasGeom.type = VriAsGeometryType_Instances; tlasGeom.instances.instanceBuffer = instBuf; tlasGeom.instances.instanceCount = 1;
    VriAccelerationStructureDesc tlasDesc{}; tlasDesc.type = VriAccelerationStructureType_TopLevel; tlasDesc.flags = VriAccelerationStructureBuild_PreferFastTrace;
    tlasDesc.geometryCount = 1; tlasDesc.geometries = &tlasGeom;
    VriAccelerationStructure* tlas = nullptr;
    if (rt.CreateAccelerationStructure(app.dev, &tlasDesc, &tlas) != VriResult_Success) app.Fail("CreateAccelerationStructure (TLAS) failed");

    VriBufferDesc gnd{}; gnd.size = nodes.size() * sizeof(GeometryNode); gnd.usage = VriBufferUsage_StorageBuffer; gnd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* geomNodeBuf = nullptr; c.CreateBuffer(app.dev, &gnd, &geomNodeBuf);
    std::memcpy(c.MapBuffer(geomNodeBuf, 0, gnd.size), nodes.data(), gnd.size); c.UnmapBuffer(geomNodeBuf);

    c.BeginCommandBuffer(app.cmd);
    VriBuildAccelerationStructureDesc bb{}; bb.dst = blas; bb.geometry = &blasDesc; rt.CmdBuildAccelerationStructure(app.cmd, &bb);
    VriBuildAccelerationStructureDesc tb{}; tb.dst = tlas; tb.geometry = &tlasDesc; rt.CmdBuildAccelerationStructure(app.cmd, &tb);
    c.EndCommandBuffer(app.cmd);
    { VriFenceSubmitDesc sig{}; sig.fence = app.fence; sig.value = 1; VriQueueSubmitDesc sub{}; sub.commandBuffers = &app.cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1; c.QueueSubmit(app.queue, &sub); c.Wait(app.fence, 1); }

    VriDescriptor* tlasDescriptor = nullptr;
    if (rt.CreateAccelerationStructureDescriptor(app.dev, tlas, &tlasDescriptor) != VriResult_Success) app.Fail("CreateAccelerationStructureDescriptor failed");

    auto storageView = [&](VriBuffer* buf, uint64_t size) { VriBufferViewDesc v{}; v.buffer = buf; v.viewType = VriDescriptorType_StorageBuffer; v.offset = 0; v.size = size; VriDescriptor* d = nullptr; c.CreateBufferView(app.dev, &v, &d); return d; };
    VriDescriptor* vtxView = storageView(model.vertexBuffer, uint64_t(model.vertexCount) * sizeof(vriex::GltfVertex));
    VriDescriptor* idxView = storageView(model.indexBuffer, uint64_t(model.indexCount) * sizeof(uint32_t));
    VriDescriptor* gnView = storageView(geomNodeBuf, gnd.size);

    // persistent radiance accumulator (one float4 per pixel; never re-initialised on the host - the
    // shader resets it when control.z says so). Device-local storage buffer.
    VriBufferDesc abd{}; abd.size = uint64_t(kWidth) * kHeight * sizeof(float) * 4; abd.usage = VriBufferUsage_StorageBuffer; abd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* accumBuf = nullptr; c.CreateBuffer(app.dev, &abd, &accumBuf);
    VriDescriptor* accumView = storageView(accumBuf, abd.size);

    VriBufferDesc cbd{}; cbd.size = sizeof(PtCamera); cbd.usage = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst; cbd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* camBuf = nullptr; c.CreateBuffer(app.dev, &cbd, &camBuf);
    VriBufferDesc csd{}; csd.size = sizeof(PtCamera); csd.usage = VriBufferUsage_TransferSrc; csd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* camStg = nullptr; c.CreateBuffer(app.dev, &csd, &camStg);
    VriBufferViewDesc cbv{}; cbv.buffer = camBuf; cbv.viewType = VriDescriptorType_ConstantBuffer; cbv.offset = 0; cbv.size = sizeof(PtCamera);
    VriDescriptor* camView = nullptr; c.CreateBufferView(app.dev, &cbv, &camView);

    VriTextureDesc otd{}; otd.type = VriTextureType_2D; otd.format = VriFormat_RGBA8_UNORM; otd.width = kWidth; otd.height = kHeight; otd.depth = 1;
    otd.mipNum = 1; otd.layerNum = 1; otd.sampleNum = 1; otd.usage = VriTextureUsage_ShaderResourceStorage | VriTextureUsage_ShaderResource; otd.memoryLocation = VriMemoryLocation_Device;
    VriTexture* outImg = nullptr; if (c.CreateTexture(app.dev, &otd, &outImg) != VriResult_Success) app.Fail("CreateTexture (output) failed");
    VriTextureViewDesc ovd{}; ovd.texture = outImg; ovd.viewType = VriTextureViewType_2D; ovd.format = VriFormat_Unknown; ovd.aspect = VriImageAspect_Color;
    VriDescriptor* outView = nullptr; c.CreateTextureView(app.dev, &ovd, &outView);
    VriSamplerDesc smp{}; smp.magFilter = VriFilter_Linear; smp.minFilter = VriFilter_Linear; smp.mipmapMode = VriMipmapMode_Nearest;
    smp.addressModeU = VriAddressMode_ClampToEdge; smp.addressModeV = VriAddressMode_ClampToEdge; smp.addressModeW = VriAddressMode_ClampToEdge; smp.maxLod = 1.0f;
    VriDescriptor* dispSampler = nullptr; c.CreateSampler(app.dev, &smp, &dispSampler);
    VriSamplerDesc msmp{}; msmp.magFilter = VriFilter_Linear; msmp.minFilter = VriFilter_Linear; msmp.mipmapMode = VriMipmapMode_Linear;
    msmp.addressModeU = VriAddressMode_Repeat; msmp.addressModeV = VriAddressMode_Repeat; msmp.addressModeW = VriAddressMode_Repeat; msmp.maxLod = 1.0f;
    VriDescriptor* modelSampler = nullptr; c.CreateSampler(app.dev, &msmp, &modelSampler);

    std::vector<const VriDescriptor*> texArr(kMaxTextures, nullptr);
    for (uint32_t i = 0; i < kMaxTextures; ++i)
        texArr[i] = model.textureViews.empty() ? nullptr : model.textureViews[i < model.textureViews.size() ? i : 0];

    // ---- compute layout: 9 bindings (the 8 trace bindings + the accumulation buffer) ----
    VriDescriptorRangeDesc rr[9]{};
    rr[0].baseRegister = 0; rr[0].descriptorNum = 1;            rr[0].descriptorType = VriDescriptorType_AccelerationStructure;
    rr[1].baseRegister = 1; rr[1].descriptorNum = 1;            rr[1].descriptorType = VriDescriptorType_StorageTexture;
    rr[2].baseRegister = 2; rr[2].descriptorNum = 1;            rr[2].descriptorType = VriDescriptorType_ConstantBuffer;
    rr[3].baseRegister = 3; rr[3].descriptorNum = 1;            rr[3].descriptorType = VriDescriptorType_StorageBuffer;
    rr[4].baseRegister = 4; rr[4].descriptorNum = 1;            rr[4].descriptorType = VriDescriptorType_StorageBuffer;
    rr[5].baseRegister = 5; rr[5].descriptorNum = 1;            rr[5].descriptorType = VriDescriptorType_StorageBuffer;
    rr[6].baseRegister = 6; rr[6].descriptorNum = 1;            rr[6].descriptorType = VriDescriptorType_Sampler;
    rr[7].baseRegister = 7; rr[7].descriptorNum = kMaxTextures; rr[7].descriptorType = VriDescriptorType_Texture;
    rr[8].baseRegister = 8; rr[8].descriptorNum = 1;            rr[8].descriptorType = VriDescriptorType_StorageBuffer;
    for (auto& r : rr) r.shaderStages = VriShaderStage_Compute;
    VriDescriptorSetDesc tsd{}; tsd.registerSpace = 0; tsd.ranges = rr; tsd.rangeNum = 9;
    VriPipelineLayoutDesc tld{}; tld.descriptorSets = &tsd; tld.descriptorSetNum = 1;
    VriPipelineLayout* ptLayout = nullptr; if (c.CreatePipelineLayout(app.dev, &tld, &ptLayout) != VriResult_Success) app.Fail("CreatePipelineLayout failed");

    VriComputePipelineDesc cpd{}; cpd.pipelineLayout = ptLayout; cpd.shader.stage = VriShaderStage_Compute; cpd.shader.entryPointName = "computeMain";
#if defined(_WIN32)
    if (app.useDxbc) { cpd.shader.bytecode = g_rtPathtraceDxil; cpd.shader.bytecodeSize = sizeof(g_rtPathtraceDxil); }
    else
#endif
    { cpd.shader.bytecode = g_rtPathtraceSpv; cpd.shader.bytecodeSize = sizeof(g_rtPathtraceSpv); }
    VriPipeline* ptPipeline = nullptr; if (c.CreateComputePipeline(app.dev, &cpd, &ptPipeline) != VriResult_Success) app.Fail("CreateComputePipeline failed");

    VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 2; pdsc.accelerationStructureMaxNum = 1; pdsc.storageTextureMaxNum = 1;
    pdsc.constantBufferMaxNum = 1; pdsc.storageBufferMaxNum = 4; pdsc.samplerMaxNum = 2; pdsc.textureMaxNum = kMaxTextures + 1;
    VriDescriptorPool* pool = nullptr; c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* ptSet = nullptr; c.AllocateDescriptorSets(pool, ptLayout, 0, &ptSet, 1);
    {
        const VriDescriptor* as[1] = {tlasDescriptor}; const VriDescriptor* oi[1] = {outView}; const VriDescriptor* cb[1] = {camView};
        const VriDescriptor* vb[1] = {vtxView}; const VriDescriptor* ib[1] = {idxView}; const VriDescriptor* gb[1] = {gnView}; const VriDescriptor* sm[1] = {modelSampler}; const VriDescriptor* ab[1] = {accumView};
        VriDescriptorRangeUpdateDesc u[9]{};
        u[0].descriptors = as; u[0].descriptorNum = 1; u[1].descriptors = oi; u[1].descriptorNum = 1; u[2].descriptors = cb; u[2].descriptorNum = 1;
        u[3].descriptors = vb; u[3].descriptorNum = 1; u[4].descriptors = ib; u[4].descriptorNum = 1; u[5].descriptors = gb; u[5].descriptorNum = 1;
        u[6].descriptors = sm; u[6].descriptorNum = 1; u[7].descriptors = texArr.data(); u[7].descriptorNum = kMaxTextures; u[8].descriptors = ab; u[8].descriptorNum = 1;
        c.UpdateDescriptorRanges(ptSet, 0, 9, u);
    }

    // ---- display pipeline ----
    VriDescriptorRangeDesc dr[2]{};
    dr[0].baseRegister = 0; dr[0].descriptorNum = 1; dr[0].descriptorType = VriDescriptorType_Texture; dr[0].shaderStages = VriShaderStage_Fragment;
    dr[1].baseRegister = 1; dr[1].descriptorNum = 1; dr[1].descriptorType = VriDescriptorType_Sampler; dr[1].shaderStages = VriShaderStage_Fragment;
    VriDescriptorSetDesc dsd{}; dsd.registerSpace = 0; dsd.ranges = dr; dsd.rangeNum = 2;
    VriPipelineLayoutDesc dld{}; dld.descriptorSets = &dsd; dld.descriptorSetNum = 1;
    VriPipelineLayout* displayLayout = nullptr; c.CreatePipelineLayout(app.dev, &dld, &displayLayout);
    VriShaderDesc dsh[2]{};
    dsh[0].stage = VriShaderStage_Vertex;   dsh[0].entryPointName = "vertexMain";
    dsh[1].stage = VriShaderStage_Fragment; dsh[1].entryPointName = "fragmentMain";
    if (app.useDxbc) { dsh[0].bytecode = g_showTexDxbcVS; dsh[0].bytecodeSize = sizeof(g_showTexDxbcVS); dsh[1].bytecode = g_showTexDxbcPS; dsh[1].bytecodeSize = sizeof(g_showTexDxbcPS); }
    else { dsh[0].bytecode = dsh[1].bytecode = g_showTexSpv; dsh[0].bytecodeSize = dsh[1].bytecodeSize = sizeof(g_showTexSpv); }
    VriColorAttachmentDesc ca{}; ca.format = app.swapFormat; ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd{}; pd.pipelineLayout = displayLayout; pd.shaders = dsh; pd.shaderNum = 2;
    pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList; pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
    pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
    VriPipeline* displayPipeline = nullptr; if (c.CreateGraphicsPipeline(app.dev, &pd, &displayPipeline) != VriResult_Success) app.Fail("display CreateGraphicsPipeline failed");
    VriDescriptorSet* displaySet = nullptr; c.AllocateDescriptorSets(pool, displayLayout, 0, &displaySet, 1);
    { const VriDescriptor* a[1] = {outView}; const VriDescriptor* b[1] = {dispSampler}; VriDescriptorRangeUpdateDesc u[2]{}; u[0].descriptors = a; u[0].descriptorNum = 1; u[1].descriptors = b; u[1].descriptorNum = 1; c.UpdateDescriptorRanges(displaySet, 0, 2, u); }

    const V3 center = {(model.boundsMin[0] + model.boundsMax[0]) * 0.5f, (model.boundsMin[1] + model.boundsMax[1]) * 0.5f, (model.boundsMin[2] + model.boundsMax[2]) * 0.5f};
    const V3 ext = {model.boundsMax[0] - model.boundsMin[0], model.boundsMax[1] - model.boundsMin[1], model.boundsMax[2] - model.boundsMin[2]};
    const float radius = 0.5f * std::sqrt(ext.x * ext.x + ext.y * ext.y + ext.z * ext.z);

    app.onUpdate = [=](uint64_t) {
        if (dirty) { sampleCount = 0; dirty = false; }      // restart accumulation on any change
        PtCamera cam{};
        const float dist = radius * 3.6f; // far enough that the whole model fits the frame
        const V3 eye = {center.x + std::cos(orbit) * dist, center.y + radius * 0.5f, center.z + std::sin(orbit) * dist};
        const V3 fwd = norm3(sub(center, eye));
        const V3 wup = {0, 1, 0};
        const V3 right = norm3(cross3(fwd, wup));
        const V3 up = cross3(right, fwd);
        const float tanHalf = std::tan(0.6f * 0.5f), aspect = float(kWidth) / float(kHeight);
        set4(cam.eye, eye);
        set4(cam.camRight, {right.x * tanHalf * aspect, right.y * tanHalf * aspect, right.z * tanHalf * aspect});
        set4(cam.camUp, {up.x * tanHalf, up.y * tanHalf, up.z * tanHalf});
        set4(cam.camFwd, fwd);
        // warm sky tint that shifts with the "light azimuth" so the GI direction is visible
        const float a = lightAzimuth;
        set4(cam.sky, {0.7f + 0.3f * std::cos(a), 0.75f, 0.85f + 0.15f * std::sin(a)}, skyIntensity);
        cam.control[0] = float(sampleCount);
        cam.control[1] = float(maxBounces);
        cam.control[2] = (sampleCount == 0) ? 1.0f : 0.0f; // reset the accumulator on the first sample
        std::memcpy(c.MapBuffer(camStg, 0, sizeof(PtCamera)), &cam, sizeof(PtCamera)); c.UnmapBuffer(camStg);
        ++sampleCount;
    };

    static bool g_imgInit = false;
    app.onPreRender = [=](VriCommandBuffer* cmd) {
        VriBufferCopyDesc ucp{}; ucp.size = sizeof(PtCamera); c.CmdCopyBuffer(cmd, camBuf, camStg, &ucp);
        VriBufferBarrierDesc ub{}; ub.buffer = camBuf; ub.before.access = VriAccess_CopyDestinationWrite; ub.before.stages = VriPipelineStage_Transfer;
        ub.after.access = VriAccess_ConstantBufferRead; ub.after.stages = VriPipelineStage_ComputeShader;
        VriBarrierGroupDesc ubg{}; ubg.buffers = &ub; ubg.bufferNum = 1; c.CmdBarrier(cmd, &ubg);

        VriTextureBarrierDesc tw{}; tw.texture = outImg; tw.before.layout = g_imgInit ? VriLayout_ShaderResource : VriLayout_Undefined; tw.before.stages = g_imgInit ? VriPipelineStage_FragmentShader : VriPipelineStage_None;
        tw.after.access = VriAccess_ShaderResourceStorageWrite; tw.after.layout = VriLayout_ShaderResourceStorage; tw.after.stages = VriPipelineStage_ComputeShader; tw.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc gw{}; gw.textures = &tw; gw.textureNum = 1; c.CmdBarrier(cmd, &gw);
        g_imgInit = true;

        c.CmdSetPipeline(cmd, ptPipeline);
        c.CmdSetPipelineLayout(cmd, ptLayout);
        c.CmdSetDescriptorSet(cmd, 0, ptSet);
        VriDispatchDesc disp{}; disp.x = (kWidth + 7) / 8; disp.y = (kHeight + 7) / 8; disp.z = 1; c.CmdDispatch(cmd, &disp);

        VriTextureBarrierDesc tr{}; tr.texture = outImg; tr.before.access = VriAccess_ShaderResourceStorageWrite; tr.before.layout = VriLayout_ShaderResourceStorage; tr.before.stages = VriPipelineStage_ComputeShader;
        tr.after.access = VriAccess_ShaderResourceRead; tr.after.layout = VriLayout_ShaderResource; tr.after.stages = VriPipelineStage_FragmentShader; tr.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc gr{}; gr.textures = &tr; gr.textureNum = 1; c.CmdBarrier(cmd, &gr);
    };
    app.onRecord = [=](VriCommandBuffer* cmd) {
        c.CmdSetPipeline(cmd, displayPipeline);
        c.CmdSetPipelineLayout(cmd, displayLayout);
        c.CmdSetDescriptorSet(cmd, 0, displaySet);
        VriDrawDesc d{}; d.vertexNum = 3; d.instanceNum = 1; c.CmdDraw(cmd, &d);
    };

    std::printf("[pathtracer] megakernel path tracer (inline ray query) over '%s' (%zu primitives), %ux%u\n", modelPath, model.primitives.size(), kWidth, kHeight); std::fflush(stdout);
    app.SetupCapture();
    app.Run();

    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(displayPipeline); c.DestroyPipelineLayout(displayLayout);
    c.DestroyPipeline(ptPipeline); c.DestroyPipelineLayout(ptLayout);
    c.DestroyDescriptor(dispSampler); c.DestroyDescriptor(modelSampler);
    c.DestroyDescriptor(outView); c.DestroyTexture(outImg);
    c.DestroyDescriptor(accumView); c.DestroyBuffer(accumBuf);
    c.DestroyDescriptor(camView); c.DestroyBuffer(camStg); c.DestroyBuffer(camBuf);
    c.DestroyDescriptor(vtxView); c.DestroyDescriptor(idxView); c.DestroyDescriptor(gnView);
    c.DestroyBuffer(geomNodeBuf);
    c.DestroyDescriptor(tlasDescriptor);
    rt.DestroyAccelerationStructure(tlas); rt.DestroyAccelerationStructure(blas);
    c.DestroyBuffer(instBuf);
    model.Destroy(app);
    app.Shutdown();
    return 0;
#endif
}
