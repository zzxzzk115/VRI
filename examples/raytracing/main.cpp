// Ray tracing the FlightHelmet glTF model. Two paths render the SAME image from the SAME loaded
// model + BLAS/TLAS + material textures, chosen by what the backend supports:
//   - RT PIPELINE (Vulkan, D3D12): a real raygen/miss/closesthit pipeline + shader binding table;
//     CmdTraceRays writes a lit, textured image (rt_gltf.slang). This is the DXR / VK_KHR_ray_tracing
//     _pipeline path - the thing examples/rayquery's inline ray query is NOT.
//   - INLINE RAY QUERY in a compute shader (Metal, or any backend with ray query but no RT pipeline):
//     ONE compute kernel does ray generation + traversal + shading (rt_gltf_rayquery.slang). This is
//     Metal's native ray-tracing model (Apple WWDC 2020 "Discover ray tracing with Metal"): Metal has
//     no raygen/SBT/TraceRays, so the pipeline collapses into a single intersector kernel.
// The closesthit/kernel fetches the hit triangle from the model's global vertex/index SSBOs and samples
// the per-primitive material textures by index, so both paths are pixel-identical.
//
// Backends with neither RT pipeline nor ray query (OpenGL / WebGPU / WebGL2) degrade EXPLICITLY.
#include "../common/example_app.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "../common/gltf_model.h"
#include "../common/bvh.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tests/shaders/show_tex_spv.h"           // g_showTexSpv         (display: sample the result)
#include "tests/shaders/show_tex_wgsl.h"          // g_showTexWgsl        (WebGPU display)
#include "tests/shaders/show_tex_dxbc.h"          // g_showTexDxbcVS / PS
#include "tests/shaders/rt_gltf_spv.h"            // g_rtGltfSpv          (Vulkan: raygen/miss/closesthit)
#include "tests/shaders/rt_gltf_rayquery_spv.h"   // g_rtGltfRayquerySpv  (Metal/VK: inline ray query CS)
#include "tests/shaders/rt_gltf_software_spv.h"   // g_rtGltfSoftwareSpv  (compute BVH fallback, GL/VK)
#include "tests/shaders/rt_gltf_software_wgsl.h"  // g_rtGltfSoftwareWgsl (compute BVH fallback, WebGPU)
#if defined(_WIN32)
#    include "tests/shaders/rt_gltf_dxil.h"          // g_rtGltfDxilRGEN/MISS/CHIT (D3D12 DXR library, per stage)
#    include "tests/shaders/rt_gltf_rayquery_dxil.h" // g_rtGltfRayqueryDxilCS     (D3D12 inline ray query CS)
#    include "tests/shaders/rt_gltf_software_dxil.h" // g_rtGltfSoftwareDxilCS      (D3D12 compute BVH fallback)
#endif

#if !defined(VRI_GLTF_MODEL_PATH)
#    define VRI_GLTF_MODEL_PATH "FlightHelmet/glTF/FlightHelmet.gltf" // overridable via xmake / argv[1]
#endif

namespace
{
    constexpr uint32_t kWidth = 960, kHeight = 720;
    constexpr uint32_t kMaxTextures = 64; // must match rt_gltf.slang / rt_gltf_rayquery.slang

    struct Camera { float eye[4]; float camRight[4]; float camUp[4]; float camFwd[4]; float lightDir[4]; float params[4]; };
    struct GeometryNode { uint32_t firstIndex; int32_t baseColorTexture; int32_t occlusionTexture; uint32_t pad; };
    struct AsInstance { float transform[12]; uint32_t instanceIdAndMask; uint32_t sbtOffsetAndFlags; uint64_t blasReference; };

    struct V3 { float x, y, z; };
    V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    V3 cross3(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    V3 norm3(V3 a) { float l = std::sqrt(dot3(a, a)); return l > 0 ? V3{a.x / l, a.y / l, a.z / l} : a; }
    void set4(float* d, V3 v, float w = 0.0f) { d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = w; }
} // namespace

int main(int argc, char** argv)
{
    static vriex::ExampleApp app;
    app.requestFeatures = VriFeature_RayTracing | VriFeature_RayQuery; // grant whichever the adapter has
    app.Init("raytracing", kWidth, kHeight, /*hasDepth*/ false);
    app.SetClearColor(0.13f, 0.16f, 0.22f); // shows through only on the unsupported path
    VriCoreInterface& c = app.c;
    const VriDeviceDesc* dd = c.GetDeviceDesc(app.dev);

    // The ray-tracing interface (acceleration structures + RT pipeline) is registered when EITHER
    // ray tracing or ray query was granted. hasRtPipeline gates the DXR/VK pipeline path; otherwise
    // ray query drives the compute path. Both build/use BLAS/TLAS through the same interface.
    VriRayTracingInterface rt{};
    const bool rtIface = (dd->hasRayTracing != VRI_FALSE || dd->hasRayQuery != VRI_FALSE) &&
        vriGetInterface(app.dev, VRI_INTERFACE_RAYTRACING, sizeof(rt), &rt) == VriResult_Success;
    const bool hasRtPipeline = rtIface && dd->hasRayTracing != VRI_FALSE;
    const bool hasRayQuery = rtIface && dd->hasRayQuery != VRI_FALSE;
    const bool hasCompute = dd->hasComputeShader != VRI_FALSE;
    const bool useSoftware = !hasRtPipeline && !hasRayQuery && hasCompute;

    static float orbit = 0.0f, orbitSpeed = 0.2f, lightAzimuth = 0.7f, lightElev = 0.9f, ambient = 0.22f;
    app.onGui = [hasRtPipeline, hasRayQuery, useSoftware] {
        if (!hasRtPipeline && !hasRayQuery && !useSoftware) { ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "ray tracing unsupported"); ImGui::Text("needs RT/ray query or compute"); return; }
        ImGui::Text("%s", hasRtPipeline ? "RT pipeline (raygen/miss/hit + SBT)" : (hasRayQuery ? "inline ray query (compute, Metal-style)" : "software BVH (compute)"));
        ImGui::SliderFloat("orbit speed", &orbitSpeed, 0.0f, 1.0f);
        ImGui::SliderFloat("light azimuth", &lightAzimuth, 0.0f, 6.28f);
        ImGui::SliderFloat("light elevation", &lightElev, 0.2f, 1.5f);
        ImGui::SliderFloat("ambient", &ambient, 0.0f, 0.6f);
    };

    if (!hasRtPipeline && !hasRayQuery && !useSoftware)
    {
        std::printf("[raytracing] no ray tracing or compute on this backend - degrading\n"); std::fflush(stdout);
        app.SetupCapture();
        app.Run();
        return 0;
    }

    // ================= shared setup (works on every RT/ray-query backend, Metal included) =================
#if defined(__EMSCRIPTEN__)
    (void)argc; (void)argv; // web: argv carries the URL query string (?backend=…), not a CLI path; the model is preloaded at a fixed MEMFS path
    const char* modelPath = VRI_GLTF_MODEL_PATH;
#else
    const char* modelPath = argc > 1 ? argv[1] : VRI_GLTF_MODEL_PATH;
#endif
    vriex::GltfModel model;
    if (!model.Load(app, modelPath, useSoftware, useSoftware)) app.Fail("failed to load glTF model");

    std::vector<GeometryNode> nodes(model.primitives.size());
    for (size_t i = 0; i < model.primitives.size(); ++i)
    {
        const vriex::GltfPrimitive& p = model.primitives[i];
        nodes[i] = {p.firstIndex, p.baseColorTexture, p.occlusionTexture, 0};
    }

    VriAccelerationStructure* blas = nullptr;
    VriAccelerationStructure* tlas = nullptr;
    VriBuffer* instBuf = nullptr;
    VriDescriptor* tlasDescriptor = nullptr;
    if (!useSoftware)
    {
        std::vector<VriAsGeometryDesc> geoms(model.primitives.size());
        for (size_t i = 0; i < model.primitives.size(); ++i)
        {
            const vriex::GltfPrimitive& p = model.primitives[i];
            VriAsGeometryDesc& g = geoms[i];
            g.type = VriAsGeometryType_Triangles; g.flags = VriAsGeometry_Opaque;
            g.triangles.vertexBuffer = model.vertexBuffer; g.triangles.vertexCount = model.vertexCount;
            g.triangles.vertexStride = sizeof(vriex::GltfVertex); g.triangles.vertexFormat = VriFormat_RGB32_SFLOAT;
            g.triangles.indexBuffer = model.indexBuffer; g.triangles.indexOffset = uint64_t(p.firstIndex) * sizeof(uint32_t);
            g.triangles.indexCount = p.indexCount; g.triangles.indexType = VriIndexType_UInt32;
        }
        VriAccelerationStructureDesc blasDesc{}; blasDesc.type = VriAccelerationStructureType_BottomLevel; blasDesc.flags = VriAccelerationStructureBuild_PreferFastTrace;
        blasDesc.geometryCount = uint32_t(geoms.size()); blasDesc.geometries = geoms.data();
        if (rt.CreateAccelerationStructure(app.dev, &blasDesc, &blas) != VriResult_Success) app.Fail("CreateAccelerationStructure (BLAS) failed");

        AsInstance inst{}; inst.transform[0] = 1.0f; inst.transform[5] = 1.0f; inst.transform[10] = 1.0f;
        inst.instanceIdAndMask = 0xFFu << 24; inst.sbtOffsetAndFlags = 0x1u << 24; // cull-disable
        inst.blasReference = rt.GetAccelerationStructureDeviceAddress(blas);
        VriBufferDesc ibd{}; ibd.size = sizeof(AsInstance); ibd.usage = VriBufferUsage_AccelerationBuildInput; ibd.memoryLocation = VriMemoryLocation_HostUpload;
        c.CreateBuffer(app.dev, &ibd, &instBuf);
        std::memcpy(c.MapBuffer(instBuf, 0, sizeof(inst)), &inst, sizeof(inst)); c.UnmapBuffer(instBuf);

        VriAsGeometryDesc tlasGeom{}; tlasGeom.type = VriAsGeometryType_Instances; tlasGeom.instances.instanceBuffer = instBuf; tlasGeom.instances.instanceCount = 1;
        VriAccelerationStructureDesc tlasDesc{}; tlasDesc.type = VriAccelerationStructureType_TopLevel; tlasDesc.flags = VriAccelerationStructureBuild_PreferFastTrace;
        tlasDesc.geometryCount = 1; tlasDesc.geometries = &tlasGeom;
        if (rt.CreateAccelerationStructure(app.dev, &tlasDesc, &tlas) != VriResult_Success) app.Fail("CreateAccelerationStructure (TLAS) failed");

        c.ResetCommandAllocator(app.alloc);
        c.BeginCommandBuffer(app.cmd);
        VriBuildAccelerationStructureDesc bb{}; bb.dst = blas; bb.geometry = &blasDesc; rt.CmdBuildAccelerationStructure(app.cmd, &bb);
        VriBuildAccelerationStructureDesc tb{}; tb.dst = tlas; tb.geometry = &tlasDesc; rt.CmdBuildAccelerationStructure(app.cmd, &tb);
        c.EndCommandBuffer(app.cmd);
        { VriFence* buildFence = nullptr; c.CreateFence(app.dev, 0, &buildFence); VriFenceSubmitDesc sig{}; sig.fence = buildFence; sig.value = 1; VriQueueSubmitDesc sub{}; sub.commandBuffers = &app.cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1; c.QueueSubmit(app.queue, &sub); c.Wait(buildFence, 1); c.DestroyFence(buildFence); }

        if (rt.CreateAccelerationStructureDescriptor(app.dev, tlas, &tlasDescriptor) != VriResult_Success) app.Fail("CreateAccelerationStructureDescriptor failed");
    }

    VriBufferDesc gnd{}; gnd.size = nodes.size() * sizeof(GeometryNode); gnd.usage = VriBufferUsage_StorageBuffer; gnd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* geomNodeBuf = nullptr; c.CreateBuffer(app.dev, &gnd, &geomNodeBuf);
    std::memcpy(c.MapBuffer(geomNodeBuf, 0, gnd.size), nodes.data(), gnd.size); c.UnmapBuffer(geomNodeBuf);

    auto storageView = [&](VriBuffer* buf, uint64_t size) { VriBufferViewDesc v{}; v.buffer = buf; v.viewType = VriDescriptorType_StorageBuffer; v.offset = 0; v.size = size; VriDescriptor* d = nullptr; c.CreateBufferView(app.dev, &v, &d); return d; };
    VriDescriptor* vtxView = storageView(model.vertexBuffer, uint64_t(model.vertexCount) * sizeof(vriex::GltfVertex));
    VriDescriptor* idxView = storageView(model.indexBuffer, uint64_t(model.indexCount) * sizeof(uint32_t));
    VriDescriptor* gnView = storageView(geomNodeBuf, gnd.size);
    VriBuffer* bvhNodeBuf = nullptr; VriBuffer* triRefBuf = nullptr;
    VriDescriptor* bvhNodeView = nullptr; VriDescriptor* triRefView = nullptr;
    if (useSoftware)
    {
        vriex::Bvh bvh = vriex::BuildBvh(model);
        VriBufferDesc bd{}; bd.size = bvh.nodes.size() * sizeof(vriex::BvhNode); bd.usage = VriBufferUsage_StorageBuffer; bd.memoryLocation = VriMemoryLocation_HostUpload;
        c.CreateBuffer(app.dev, &bd, &bvhNodeBuf);
        std::memcpy(c.MapBuffer(bvhNodeBuf, 0, bd.size), bvh.nodes.data(), bd.size); c.UnmapBuffer(bvhNodeBuf);
        VriBufferDesc td{}; td.size = bvh.triRefs.size() * sizeof(vriex::BvhTriRef); td.usage = VriBufferUsage_StorageBuffer; td.memoryLocation = VriMemoryLocation_HostUpload;
        c.CreateBuffer(app.dev, &td, &triRefBuf);
        std::memcpy(c.MapBuffer(triRefBuf, 0, td.size), bvh.triRefs.data(), td.size); c.UnmapBuffer(triRefBuf);
        bvhNodeView = storageView(bvhNodeBuf, bd.size);
        triRefView = storageView(triRefBuf, td.size);
        std::printf("[raytracing] software BVH: %zu nodes, %zu triangles\n", bvh.nodes.size(), bvh.triRefs.size()); std::fflush(stdout);
    }
    VriBufferDesc cbd{}; cbd.size = sizeof(Camera); cbd.usage = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst; cbd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* camBuf = nullptr; c.CreateBuffer(app.dev, &cbd, &camBuf);
    VriBufferDesc csd{}; csd.size = sizeof(Camera); csd.usage = VriBufferUsage_TransferSrc; csd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* camStg = nullptr; c.CreateBuffer(app.dev, &csd, &camStg);
    VriBufferViewDesc cbv{}; cbv.buffer = camBuf; cbv.viewType = VriDescriptorType_ConstantBuffer; cbv.offset = 0; cbv.size = sizeof(Camera);
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

    // Hardware (bindless) paths bind a kMaxTextures-wide descriptor array (slots fully populated:
    // real textures, padded by repeating slot 0). The software (GL/WebGPU) path instead samples the
    // model's 2D-array texture (one layer per glTF texture) as a single descriptor, since those
    // backends have no descriptor-array support.
    const uint32_t texCount = useSoftware ? 1u : kMaxTextures;
    std::vector<const VriDescriptor*> texArr(kMaxTextures, nullptr);
    if (!useSoftware)
        for (uint32_t i = 0; i < kMaxTextures; ++i)
            texArr[i] = model.textureViews.empty() ? nullptr : model.textureViews[i < model.textureViews.size() ? i : 0];
    const VriDescriptor* texArrayView[1] = {model.textureArrayView};

    const VriShaderStageFlags rgStage = hasRtPipeline ? VriShaderStage_RayGen : VriShaderStage_Compute;
    const VriShaderStageFlags chStage = hasRtPipeline ? VriShaderStage_ClosestHit : VriShaderStage_Compute;
    const VriShaderStageFlags rgChStage = rgStage | chStage;
    VriDescriptorRangeDesc rr[10]{};
    uint32_t rn = 0;
    if (!useSoftware) { rr[rn].baseRegister = 0; rr[rn].descriptorNum = 1; rr[rn].descriptorType = VriDescriptorType_AccelerationStructure; rr[rn].shaderStages = rgChStage; ++rn; }
    rr[rn].baseRegister = 1; rr[rn].descriptorNum = 1;            rr[rn].descriptorType = VriDescriptorType_StorageTexture; rr[rn].shaderStages = rgStage; ++rn;
    rr[rn].baseRegister = 2; rr[rn].descriptorNum = 1;            rr[rn].descriptorType = VriDescriptorType_ConstantBuffer; rr[rn].shaderStages = rgChStage; ++rn;
    rr[rn].baseRegister = 3; rr[rn].descriptorNum = 1;            rr[rn].descriptorType = VriDescriptorType_StorageBuffer; rr[rn].shaderStages = chStage; ++rn;
    rr[rn].baseRegister = 4; rr[rn].descriptorNum = 1;            rr[rn].descriptorType = VriDescriptorType_StorageBuffer; rr[rn].shaderStages = chStage; ++rn;
    rr[rn].baseRegister = 5; rr[rn].descriptorNum = 1;            rr[rn].descriptorType = VriDescriptorType_StorageBuffer; rr[rn].shaderStages = chStage; ++rn;
    rr[rn].baseRegister = 6; rr[rn].descriptorNum = 1;            rr[rn].descriptorType = VriDescriptorType_Sampler;       rr[rn].shaderStages = chStage; ++rn;
    rr[rn].baseRegister = 7; rr[rn].descriptorNum = texCount;     rr[rn].descriptorType = VriDescriptorType_Texture;       rr[rn].shaderStages = chStage; if (useSoftware) rr[rn].viewType = VriTextureViewType_2DArray; ++rn;
    if (useSoftware)
    {
        rr[rn].baseRegister = 8; rr[rn].descriptorNum = 1; rr[rn].descriptorType = VriDescriptorType_StorageBuffer; rr[rn].shaderStages = VriShaderStage_Compute; ++rn;
        rr[rn].baseRegister = 9; rr[rn].descriptorNum = 1; rr[rn].descriptorType = VriDescriptorType_StorageBuffer; rr[rn].shaderStages = VriShaderStage_Compute; ++rn;
    }
    VriDescriptorSetDesc tsd{}; tsd.registerSpace = 0; tsd.ranges = rr; tsd.rangeNum = rn;
    VriPipelineLayoutDesc tld{}; tld.descriptorSets = &tsd; tld.descriptorSetNum = 1;
    VriPipelineLayout* traceLayout = nullptr; if (c.CreatePipelineLayout(app.dev, &tld, &traceLayout) != VriResult_Success) app.Fail("CreatePipelineLayout (trace) failed");

    VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 2; pdsc.accelerationStructureMaxNum = useSoftware ? 0 : 1; pdsc.storageTextureMaxNum = 1;
    pdsc.constantBufferMaxNum = 1; pdsc.storageBufferMaxNum = useSoftware ? 5 : 3; pdsc.samplerMaxNum = 2; pdsc.textureMaxNum = kMaxTextures + 1;
    VriDescriptorPool* pool = nullptr; c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* traceSet = nullptr; c.AllocateDescriptorSets(pool, traceLayout, 0, &traceSet, 1);
    {
        const VriDescriptor* as[1] = {tlasDescriptor}; const VriDescriptor* oi[1] = {outView}; const VriDescriptor* cb[1] = {camView};
        const VriDescriptor* vb[1] = {vtxView}; const VriDescriptor* ib[1] = {idxView}; const VriDescriptor* gb[1] = {gnView}; const VriDescriptor* sm[1] = {modelSampler};
        const VriDescriptor* bn[1] = {bvhNodeView}; const VriDescriptor* tr[1] = {triRefView};
        VriDescriptorRangeUpdateDesc u[10]{}; uint32_t un = 0;
        if (!useSoftware) { u[un].descriptors = as; u[un].descriptorNum = 1; ++un; }
        u[un].descriptors = oi; u[un].descriptorNum = 1; ++un; u[un].descriptors = cb; u[un].descriptorNum = 1; ++un;
        u[un].descriptors = vb; u[un].descriptorNum = 1; ++un; u[un].descriptors = ib; u[un].descriptorNum = 1; ++un; u[un].descriptors = gb; u[un].descriptorNum = 1; ++un;
        u[un].descriptors = sm; u[un].descriptorNum = 1; ++un; u[un].descriptors = useSoftware ? texArrayView : texArr.data(); u[un].descriptorNum = texCount; ++un;
        if (useSoftware) { u[un].descriptors = bn; u[un].descriptorNum = 1; ++un; u[un].descriptors = tr; u[un].descriptorNum = 1; ++un; }
        c.UpdateDescriptorRanges(traceSet, 0, un, u);
    }
    // ================= path-specific: RT pipeline (VK/D3D12) or compute ray query (Metal) =================
    VriPipeline* rtPipeline = nullptr; VriBuffer* sbt = nullptr; uint32_t baseAlign = 0;
    VriPipeline* computePipeline = nullptr;
    if (hasRtPipeline)
    {
        VriShaderDesc sh[3]{};
        sh[0].stage = VriShaderStage_RayGen;     sh[0].entryPointName = "rayGenMain";
        sh[1].stage = VriShaderStage_Miss;       sh[1].entryPointName = "missMain";
        sh[2].stage = VriShaderStage_ClosestHit; sh[2].entryPointName = "closestHitMain";
#if defined(_WIN32)
        if (app.useDxbc) {
            sh[0].bytecode = g_rtGltfDxilRGEN; sh[0].bytecodeSize = sizeof(g_rtGltfDxilRGEN);
            sh[1].bytecode = g_rtGltfDxilMISS; sh[1].bytecodeSize = sizeof(g_rtGltfDxilMISS);
            sh[2].bytecode = g_rtGltfDxilCHIT; sh[2].bytecodeSize = sizeof(g_rtGltfDxilCHIT);
        }
        else
#endif
        { for (int i = 0; i < 3; ++i) { sh[i].bytecode = g_rtGltfSpv; sh[i].bytecodeSize = sizeof(g_rtGltfSpv); } }
        VriShaderGroupDesc groups[3]{};
        groups[0].type = VriShaderGroupType_General;           groups[0].generalShader = 0; groups[0].closestHitShader = VRI_SHADER_UNUSED; groups[0].anyHitShader = VRI_SHADER_UNUSED; groups[0].intersectionShader = VRI_SHADER_UNUSED;
        groups[1].type = VriShaderGroupType_General;           groups[1].generalShader = 1; groups[1].closestHitShader = VRI_SHADER_UNUSED; groups[1].anyHitShader = VRI_SHADER_UNUSED; groups[1].intersectionShader = VRI_SHADER_UNUSED;
        groups[2].type = VriShaderGroupType_TrianglesHitGroup; groups[2].generalShader = VRI_SHADER_UNUSED; groups[2].closestHitShader = 2; groups[2].anyHitShader = VRI_SHADER_UNUSED; groups[2].intersectionShader = VRI_SHADER_UNUSED;
        VriRayTracingPipelineDesc rtpd{}; rtpd.pipelineLayout = traceLayout; rtpd.shaders = sh; rtpd.shaderNum = 3; rtpd.groups = groups; rtpd.groupNum = 3; rtpd.maxRecursionDepth = 1;
        if (rt.CreateRayTracingPipeline(app.dev, &rtpd, &rtPipeline) != VriResult_Success) app.Fail("CreateRayTracingPipeline failed");

        const uint32_t handleSize = dd->rtShaderGroupHandleSize; baseAlign = dd->rtShaderGroupBaseAlignment;
        std::vector<uint8_t> handles(size_t(3) * handleSize);
        if (rt.GetShaderGroupHandles(rtPipeline, 0, 3, handles.size(), handles.data()) != VriResult_Success) app.Fail("GetShaderGroupHandles failed");
        VriBufferDesc sbtd{}; sbtd.size = uint64_t(baseAlign) * 3; sbtd.usage = VriBufferUsage_ShaderBindingTable; sbtd.memoryLocation = VriMemoryLocation_HostUpload;
        c.CreateBuffer(app.dev, &sbtd, &sbt);
        uint8_t* p = static_cast<uint8_t*>(c.MapBuffer(sbt, 0, sbtd.size)); std::memset(p, 0, sbtd.size);
        for (uint32_t i = 0; i < 3; ++i) std::memcpy(p + uint64_t(i) * baseAlign, handles.data() + size_t(i) * handleSize, handleSize);
        c.UnmapBuffer(sbt);
    }
    else // compute: Metal-style inline ray query, or the software BVH fallback (no RT hardware)
    {
        VriComputePipelineDesc cpd{}; cpd.pipelineLayout = traceLayout; cpd.shader.stage = VriShaderStage_Compute; cpd.shader.entryPointName = "computeMain";
#if defined(_WIN32)
        if (app.useDxbc)
        {
            if (useSoftware) { cpd.shader.bytecode = g_rtGltfSoftwareDxilCS; cpd.shader.bytecodeSize = sizeof(g_rtGltfSoftwareDxilCS); }
            else             { cpd.shader.bytecode = g_rtGltfRayqueryDxilCS; cpd.shader.bytecodeSize = sizeof(g_rtGltfRayqueryDxilCS); }
        }
        else
#endif
        if (useSoftware) { cpd.shader.bytecode = app.useWgsl ? static_cast<const void*>(g_rtGltfSoftwareWgsl) : static_cast<const void*>(g_rtGltfSoftwareSpv); cpd.shader.bytecodeSize = app.useWgsl ? sizeof(g_rtGltfSoftwareWgsl) : sizeof(g_rtGltfSoftwareSpv); }
        else             { cpd.shader.bytecode = g_rtGltfRayquerySpv;  cpd.shader.bytecodeSize = sizeof(g_rtGltfRayquerySpv); } // ray query never runs on WebGPU
        if (c.CreateComputePipeline(app.dev, &cpd, &computePipeline) != VriResult_Success) app.Fail("CreateComputePipeline (compute trace) failed");
    }

    // ================= display pipeline (fullscreen show_tex), shared =================
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
    else { dsh[0].bytecode = dsh[1].bytecode = app.useWgsl ? static_cast<const void*>(g_showTexWgsl) : static_cast<const void*>(g_showTexSpv); dsh[0].bytecodeSize = dsh[1].bytecodeSize = app.useWgsl ? sizeof(g_showTexWgsl) : sizeof(g_showTexSpv); }
    VriColorAttachmentDesc ca{}; ca.format = app.swapFormat; ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd{}; pd.pipelineLayout = displayLayout; pd.shaders = dsh; pd.shaderNum = 2;
    pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList; pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
    pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
    VriPipeline* displayPipeline = nullptr; if (c.CreateGraphicsPipeline(app.dev, &pd, &displayPipeline) != VriResult_Success) app.Fail("display CreateGraphicsPipeline failed");
    VriDescriptorSet* displaySet = nullptr; c.AllocateDescriptorSets(pool, displayLayout, 0, &displaySet, 1);
    { const VriDescriptor* a[1] = {outView}; const VriDescriptor* b[1] = {dispSampler}; VriDescriptorRangeUpdateDesc u[2]{}; u[0].descriptors = a; u[0].descriptorNum = 1; u[1].descriptors = b; u[1].descriptorNum = 1; c.UpdateDescriptorRanges(displaySet, 0, 2, u); }

    // camera framing from model bounds
    const V3 center = {(model.boundsMin[0] + model.boundsMax[0]) * 0.5f, (model.boundsMin[1] + model.boundsMax[1]) * 0.5f, (model.boundsMin[2] + model.boundsMax[2]) * 0.5f};
    const V3 ext = {model.boundsMax[0] - model.boundsMin[0], model.boundsMax[1] - model.boundsMin[1], model.boundsMax[2] - model.boundsMin[2]};
    const float radius = 0.5f * std::sqrt(ext.x * ext.x + ext.y * ext.y + ext.z * ext.z);

    app.onUpdate = [=](uint64_t) {
        orbit += app.dt * orbitSpeed;
        Camera cam{};
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
        const float ce = std::cos(lightElev), se = std::sin(lightElev);
        set4(cam.lightDir, norm3({std::cos(lightAzimuth) * ce, se, std::sin(lightAzimuth) * ce}));
        cam.params[0] = ambient;
        std::memcpy(c.MapBuffer(camStg, 0, sizeof(Camera)), &cam, sizeof(Camera)); c.UnmapBuffer(camStg);
    };

    static bool g_imgInit = false;
    const VriPipelineStageFlags traceStageBit = hasRtPipeline ? VriPipelineStage_RayTracingShader : VriPipelineStage_ComputeShader;
    app.onPreRender = [=](VriCommandBuffer* cmd) {
        VriBufferCopyDesc ucp{}; ucp.size = sizeof(Camera); c.CmdCopyBuffer(cmd, camBuf, camStg, &ucp);
        VriBufferBarrierDesc ub{}; ub.buffer = camBuf; ub.before.access = VriAccess_CopyDestinationWrite; ub.before.stages = VriPipelineStage_Transfer;
        ub.after.access = VriAccess_ConstantBufferRead; ub.after.stages = traceStageBit;
        VriBarrierGroupDesc ubg{}; ubg.buffers = &ub; ubg.bufferNum = 1; c.CmdBarrier(cmd, &ubg);

        VriTextureBarrierDesc tw{}; tw.texture = outImg; tw.before.layout = g_imgInit ? VriLayout_ShaderResource : VriLayout_Undefined; tw.before.stages = g_imgInit ? VriPipelineStage_FragmentShader : VriPipelineStage_None;
        tw.after.access = VriAccess_ShaderResourceStorageWrite; tw.after.layout = VriLayout_ShaderResourceStorage; tw.after.stages = traceStageBit; tw.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc gw{}; gw.textures = &tw; gw.textureNum = 1; c.CmdBarrier(cmd, &gw);
        g_imgInit = true;

        c.CmdSetPipeline(cmd, hasRtPipeline ? rtPipeline : computePipeline);
        c.CmdSetPipelineLayout(cmd, traceLayout);
        c.CmdSetDescriptorSet(cmd, 0, traceSet);
        if (hasRtPipeline)
        {
            VriDispatchRaysDesc trace{};
            trace.raygen.buffer = sbt; trace.raygen.offset = 0u * baseAlign; trace.raygen.stride = baseAlign; trace.raygen.size = baseAlign;
            trace.miss.buffer   = sbt; trace.miss.offset   = 1u * baseAlign; trace.miss.stride   = baseAlign; trace.miss.size   = baseAlign;
            trace.hit.buffer    = sbt; trace.hit.offset    = 2u * baseAlign; trace.hit.stride    = baseAlign; trace.hit.size    = baseAlign;
            trace.width = kWidth; trace.height = kHeight; trace.depth = 1;
            rt.CmdTraceRays(cmd, &trace);
        }
        else
        {
            VriDispatchDesc disp{}; disp.x = (kWidth + 7) / 8; disp.y = (kHeight + 7) / 8; disp.z = 1; c.CmdDispatch(cmd, &disp);
        }

        VriTextureBarrierDesc tr{}; tr.texture = outImg; tr.before.access = VriAccess_ShaderResourceStorageWrite; tr.before.layout = VriLayout_ShaderResourceStorage; tr.before.stages = traceStageBit;
        tr.after.access = VriAccess_ShaderResourceRead; tr.after.layout = VriLayout_ShaderResource; tr.after.stages = VriPipelineStage_FragmentShader; tr.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc gr{}; gr.textures = &tr; gr.textureNum = 1; c.CmdBarrier(cmd, &gr);
    };
    app.onRecord = [=](VriCommandBuffer* cmd) {
        c.CmdSetPipeline(cmd, displayPipeline);
        c.CmdSetPipelineLayout(cmd, displayLayout);
        c.CmdSetDescriptorSet(cmd, 0, displaySet);
        VriDrawDesc d{}; d.vertexNum = 3; d.instanceNum = 1; c.CmdDraw(cmd, &d);
    };

    std::printf("[raytracing] %s tracing '%s' (%zu primitives) into %ux%u\n",
                hasRtPipeline ? "RT pipeline" : "inline ray query (compute)", modelPath, model.primitives.size(), kWidth, kHeight); std::fflush(stdout);
    app.SetupCapture();
    app.Run();

    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(displayPipeline); c.DestroyPipelineLayout(displayLayout);
    if (rtPipeline) c.DestroyPipeline(rtPipeline);
    if (computePipeline) c.DestroyPipeline(computePipeline);
    c.DestroyPipelineLayout(traceLayout);
    if (sbt) c.DestroyBuffer(sbt);
    c.DestroyDescriptor(dispSampler); c.DestroyDescriptor(modelSampler);
    c.DestroyDescriptor(outView); c.DestroyTexture(outImg);
    c.DestroyDescriptor(camView); c.DestroyBuffer(camStg); c.DestroyBuffer(camBuf);
    c.DestroyDescriptor(vtxView); c.DestroyDescriptor(idxView); c.DestroyDescriptor(gnView);
    c.DestroyBuffer(geomNodeBuf);
    if (bvhNodeView) c.DestroyDescriptor(bvhNodeView);
    if (triRefView) c.DestroyDescriptor(triRefView);
    if (bvhNodeBuf) c.DestroyBuffer(bvhNodeBuf);
    if (triRefBuf) c.DestroyBuffer(triRefBuf);
    if (!useSoftware) // the ray-tracing interface (and the AS objects) only exist on the hardware path
    {
        c.DestroyDescriptor(tlasDescriptor);
        rt.DestroyAccelerationStructure(tlas); rt.DestroyAccelerationStructure(blas);
        c.DestroyBuffer(instBuf);
    }
    model.Destroy(app);
    app.Shutdown();
    return 0;
}
