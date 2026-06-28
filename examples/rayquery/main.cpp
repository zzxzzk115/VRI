// Ray query (inline ray tracing): a compute shader ray-traces a small scene (ground + cubes) with
// hard, ray-traced shadows into a storage image, which a fullscreen pass then displays. This is the
// inline path - no RT pipeline / shader binding table (see examples/raytracing for that). Two paths
// render the SAME image:
//   - HARDWARE ray query (Vulkan, D3D12): an inline RayQuery against a BLAS/TLAS (rt_rayquery.slang).
//   - SOFTWARE (WebGPU, desktop OpenGL): brute-force Moeller-Trumbore over the triangles in a
//     compute shader (rt_software.slang) - no acceleration structure.
// WebGL2 / OpenGL ES have no compute, so the example degrades EXPLICITLY (a message, no silent
// no-op). The whole scene rides in one constant buffer (camera basis + light + <=64 triangles with
// per-triangle normal & albedo), shared by both shader paths; the hardware path reads the hit
// triangle's normal/albedo from it by committed primitive index, so the two paths match pixel-wise.
#include "../common/example_app.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "shaders/examples/rt_software_spv.h"  // g_rtSoftwareSpv  (compute; desktop GL via SPIR-V)
#include "shaders/examples/rt_software_wgsl.h" // g_rtSoftwareWgsl (WebGPU)
#include "shaders/examples/show_tex_dxbc.h"    // g_showTexDxbcVS / PS
#include "shaders/examples/show_tex_spv.h"     // g_showTexSpv     (display: sample the result)
#include "shaders/examples/show_tex_wgsl.h"    // g_showTexWgsl
#if !defined(__EMSCRIPTEN__)
#include "shaders/examples/rt_rayquery_dxil.h" // g_rtRayqueryDxilCS (D3D12 hardware ray query)
#include "shaders/examples/rt_rayquery_spv.h"  // g_rtRayquerySpv    (Vulkan hardware ray query)
#include "shaders/examples/rt_software_dxbc.h" // g_rtSoftwareDxbcCS (D3D12 software fallback)
#endif

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480;
    constexpr int      kMaxTris = 64; // must match rt_software.slang / rt_rayquery.slang

    struct Tri
    {
        float v0[4];
        float v1[4];
        float v2[4];
        float nrm[4];
        float albedo[4];
    };
    struct Scene
    {
        float eye[4];
        float camRight[4];
        float camUp[4];
        float camFwd[4];
        float lightDir[4];
        float params[4]; // x: triangle count  y: ambient
        Tri   tris[kMaxTris];
    };

    struct V3
    {
        float x, y, z;
    };
    V3    operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
    float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    V3    cross3(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
    V3    norm3(V3 a)
    {
        float l = std::sqrt(dot3(a, a));
        return l > 0 ? V3 {a.x / l, a.y / l, a.z / l} : a;
    }
    void set4(float* d, V3 v, float w = 0.0f)
    {
        d[0] = v.x;
        d[1] = v.y;
        d[2] = v.z;
        d[3] = w;
    }

    std::vector<Tri> g_tris;

    void addTri(V3 a, V3 b, V3 c, V3 albedo, V3 inside)
    {
        V3 n        = norm3(cross3(b - a, c - a));
        V3 centroid = {(a.x + b.x + c.x) / 3, (a.y + b.y + c.y) / 3, (a.z + b.z + c.z) / 3};
        if (dot3(n, centroid - inside) < 0.0f)
            n = {-n.x, -n.y, -n.z};
        Tri t {};
        set4(t.v0, a);
        set4(t.v1, b);
        set4(t.v2, c);
        set4(t.nrm, n);
        set4(t.albedo, albedo);
        g_tris.push_back(t);
    }
    void addQuad(V3 a, V3 b, V3 c, V3 d, V3 albedo, V3 inside)
    {
        addTri(a, b, c, albedo, inside);
        addTri(a, c, d, albedo, inside);
    }
    void addCube(V3 ctr, float h, V3 albedo)
    {
        const V3 p[8] = {
            {ctr.x - h, ctr.y - h, ctr.z - h},
            {ctr.x + h, ctr.y - h, ctr.z - h},
            {ctr.x + h, ctr.y + h, ctr.z - h},
            {ctr.x - h, ctr.y + h, ctr.z - h},
            {ctr.x - h, ctr.y - h, ctr.z + h},
            {ctr.x + h, ctr.y - h, ctr.z + h},
            {ctr.x + h, ctr.y + h, ctr.z + h},
            {ctr.x - h, ctr.y + h, ctr.z + h},
        };
        addQuad(p[0], p[1], p[2], p[3], albedo, ctr);
        addQuad(p[4], p[5], p[6], p[7], albedo, ctr);
        addQuad(p[0], p[4], p[7], p[3], albedo, ctr);
        addQuad(p[1], p[5], p[6], p[2], albedo, ctr);
        addQuad(p[3], p[2], p[6], p[7], albedo, ctr);
        addQuad(p[0], p[1], p[5], p[4], albedo, ctr);
    }
    void buildScene()
    {
        addQuad({-8, 0, -8}, {8, 0, -8}, {8, 0, 8}, {-8, 0, 8}, {0.62f, 0.62f, 0.66f}, {0, -1, 0}); // ground (+Y)
        addCube({-1.6f, 0.8f, 0.0f}, 0.8f, {0.85f, 0.30f, 0.25f});                                  // red
        addCube({1.3f, 0.6f, -0.9f}, 0.6f, {0.35f, 0.75f, 0.40f});                                  // green
        addCube({0.3f, 1.25f, 1.3f}, 0.5f, {0.35f, 0.55f, 0.90f});                                  // blue
    }
} // namespace

int main(int, char**)
{
    buildScene();

    static vriex::ExampleApp app;
    app.requestFeatures = VriFeature_RayTracing | VriFeature_RayQuery; // granted only where supported (best-effort)
    app.Init("rayquery", kWidth, kHeight, /*hasDepth*/ false);
    app.SetClearColor(0.20f, 0.04f, 0.05f); // shows through only on the unsupported path
    VriCoreInterface&    c          = app.c;
    const VriDeviceDesc* dd         = c.GetDeviceDesc(app.dev);
    const bool           hasCompute = dd->hasComputeShader != VRI_FALSE;

    bool useHw = false;
#if !defined(__EMSCRIPTEN__)
    VriRayTracingInterface rt {};
    useHw = dd->hasRayQuery != VRI_FALSE &&
            vriGetInterface(app.dev, VRI_INTERFACE_RAYTRACING, sizeof(rt), &rt) == VriResult_Success;
#endif

    static float orbitSpeed = 0.25f, lightAzimuth = 0.9f, lightElev = 0.95f, ambient = 0.18f;
    static float t = 0.0f;

    // scene constant buffer (camera + light + triangles), refreshed each frame
    VriBufferDesc ubd {};
    ubd.size           = sizeof(Scene);
    ubd.usage          = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst;
    ubd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* ubo     = nullptr;
    c.CreateBuffer(app.dev, &ubd, &ubo);
    VriBufferDesc usd {};
    usd.size           = sizeof(Scene);
    usd.usage          = VriBufferUsage_TransferSrc;
    usd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* ustg    = nullptr;
    c.CreateBuffer(app.dev, &usd, &ustg);
    VriBufferViewDesc ubv {};
    ubv.buffer             = ubo;
    ubv.viewType           = VriDescriptorType_ConstantBuffer;
    ubv.offset             = 0;
    ubv.size               = sizeof(Scene);
    VriDescriptor* uboView = nullptr;
    c.CreateBufferView(app.dev, &ubv, &uboView);

    app.onUpdate = [ustg](uint64_t) {
        t += app.dt;
        Scene       s {};
        const float ang = t * orbitSpeed;
        const float R = 6.0f, camY = 3.4f;
        const V3    eye = {std::cos(ang) * R, camY, std::sin(ang) * R};
        const V3    ctr = {0.0f, 0.8f, 0.0f}, wup = {0.0f, 1.0f, 0.0f};
        const V3    fwd     = norm3(ctr - eye);
        const V3    right   = norm3(cross3(fwd, wup));
        const V3    up      = cross3(right, fwd);
        const float tanHalf = std::tan(0.7f * 0.5f), aspect = float(kWidth) / float(kHeight);
        set4(s.eye, eye);
        set4(s.camRight, {right.x * tanHalf * aspect, right.y * tanHalf * aspect, right.z * tanHalf * aspect});
        set4(s.camUp, {up.x * tanHalf, up.y * tanHalf, up.z * tanHalf});
        set4(s.camFwd, fwd);
        const float ce = std::cos(lightElev), se = std::sin(lightElev);
        set4(s.lightDir, norm3({std::cos(lightAzimuth) * ce, se, std::sin(lightAzimuth) * ce}));
        s.params[0] = float(g_tris.size());
        s.params[1] = ambient;
        for (size_t i = 0; i < g_tris.size() && i < kMaxTris; ++i)
            s.tris[i] = g_tris[i];
        std::memcpy(app.c.MapBuffer(ustg, 0, sizeof(Scene)), &s, sizeof(Scene));
        app.c.UnmapBuffer(ustg);
    };
    app.onGui = [hasCompute, useHw] {
        if (!hasCompute)
        {
            ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "no compute on this backend");
            ImGui::Text("ray tracing unsupported (WebGL2/GLES)");
            return;
        }
        ImGui::Text("%s", useHw ? "hardware ray query (BLAS/TLAS)" : "software (compute brute-force)");
        ImGui::SliderFloat("orbit speed", &orbitSpeed, 0.0f, 1.5f);
        ImGui::SliderFloat("light azimuth", &lightAzimuth, 0.0f, 6.28f);
        ImGui::SliderFloat("light elevation", &lightElev, 0.2f, 1.5f);
        ImGui::SliderFloat("ambient", &ambient, 0.0f, 0.6f);
    };

    if (!hasCompute)
    {
        // WebGL2 / OpenGL ES: no compute -> ray tracing genuinely cannot run. Degrade explicitly.
        std::printf("[rayquery] compute unavailable (WebGL2/GLES) - ray tracing unsupported on this backend\n");
        std::fflush(stdout);
        app.SetupCapture();
        app.Run();
        return 0;
    }

    // storage image: written by the ray-tracing compute pass, sampled by the display pass
    VriTextureDesc td {};
    td.type           = VriTextureType_2D;
    td.format         = VriFormat_RGBA8_UNORM;
    td.width          = kWidth;
    td.height         = kHeight;
    td.depth          = 1;
    td.mipNum         = 1;
    td.layerNum       = 1;
    td.sampleNum      = 1;
    td.usage          = VriTextureUsage_ShaderResourceStorage | VriTextureUsage_ShaderResource;
    td.memoryLocation = VriMemoryLocation_Device;
    VriTexture* img   = nullptr;
    if (c.CreateTexture(app.dev, &td, &img) != VriResult_Success)
        app.Fail("CreateTexture (storage) failed");
    VriTextureViewDesc vd {};
    vd.texture             = img;
    vd.viewType            = VriTextureViewType_2D;
    vd.format              = VriFormat_Unknown;
    vd.aspect              = VriImageAspect_Color;
    VriDescriptor* imgView = nullptr;
    c.CreateTextureView(app.dev, &vd, &imgView);
    VriSamplerDesc smp {};
    smp.magFilter          = VriFilter_Linear;
    smp.minFilter          = VriFilter_Linear;
    smp.mipmapMode         = VriMipmapMode_Nearest;
    smp.addressModeU       = VriAddressMode_ClampToEdge;
    smp.addressModeV       = VriAddressMode_ClampToEdge;
    smp.addressModeW       = VriAddressMode_ClampToEdge;
    smp.maxLod             = 1.0f;
    VriDescriptor* sampler = nullptr;
    c.CreateSampler(app.dev, &smp, &sampler);

    // the compute path's pipeline/layout/descriptor set - filled by the hardware or software branch
    VriPipelineLayout* computeLayout   = nullptr;
    VriPipeline*       computePipeline = nullptr;
    VriDescriptorSet*  computeSet      = nullptr;
    VriDescriptorPool* pool            = nullptr;

#if !defined(__EMSCRIPTEN__)
    VriAccelerationStructure* blas           = nullptr;
    VriAccelerationStructure* tlas           = nullptr;
    VriBuffer*                asVbuf         = nullptr;
    VriBuffer*                asIbuf         = nullptr;
    VriDescriptor*            tlasDescriptor = nullptr;
#endif

    if (useHw)
    {
#if !defined(__EMSCRIPTEN__)
        std::printf("[rayquery] hardware ray query over a BLAS/TLAS (%zu triangles)\n", g_tris.size());
        std::fflush(stdout);

        // BLAS vertex buffer: the scene triangles' vertices, tightly packed (3 verts x 3 floats / tri),
        // in scene order so the committed primitive index equals the triangle index used for shading.
        std::vector<float> vtx;
        vtx.reserve(g_tris.size() * 9);
        for (const Tri& tr : g_tris)
        {
            for (int k = 0; k < 3; ++k)
                vtx.push_back(tr.v0[k]);
            for (int k = 0; k < 3; ++k)
                vtx.push_back(tr.v1[k]);
            for (int k = 0; k < 3; ++k)
                vtx.push_back(tr.v2[k]);
        }
        VriBufferDesc vbd2 {};
        vbd2.size           = vtx.size() * sizeof(float);
        vbd2.usage          = VriBufferUsage_AccelerationBuildInput;
        vbd2.memoryLocation = VriMemoryLocation_HostUpload;
        c.CreateBuffer(app.dev, &vbd2, &asVbuf);
        std::memcpy(c.MapBuffer(asVbuf, 0, vbd2.size), vtx.data(), vbd2.size);
        c.UnmapBuffer(asVbuf);

        VriAsGeometryDesc blasGeom {};
        blasGeom.type                   = VriAsGeometryType_Triangles;
        blasGeom.flags                  = VriAsGeometry_Opaque;
        blasGeom.triangles.vertexBuffer = asVbuf;
        blasGeom.triangles.vertexCount  = uint32_t(g_tris.size()) * 3;
        blasGeom.triangles.vertexStride = 3 * sizeof(float);
        blasGeom.triangles.vertexFormat = VriFormat_RGB32_SFLOAT;
        VriAccelerationStructureDesc blasDesc {};
        blasDesc.type          = VriAccelerationStructureType_BottomLevel;
        blasDesc.flags         = VriAccelerationStructureBuild_PreferFastTrace;
        blasDesc.geometryCount = 1;
        blasDesc.geometries    = &blasGeom;
        if (rt.CreateAccelerationStructure(app.dev, &blasDesc, &blas) != VriResult_Success)
            app.Fail("CreateAccelerationStructure (BLAS) failed");

        // one instance of the BLAS, identity transform (so world == local == the CB's stored normals)
        struct AsInstance
        {
            float    transform[12];
            uint32_t instanceIdAndMask;
            uint32_t sbtOffsetAndFlags;
            uint64_t blasReference;
        };
        AsInstance inst {};
        inst.transform[0]      = 1.0f;
        inst.transform[5]      = 1.0f;
        inst.transform[10]     = 1.0f;
        inst.instanceIdAndMask = 0xFFu << 24;
        inst.sbtOffsetAndFlags = 0x1u << 24; // cull-disable
        inst.blasReference     = rt.GetAccelerationStructureDeviceAddress(blas);
        VriBufferDesc ibd2 {};
        ibd2.size           = sizeof(AsInstance);
        ibd2.usage          = VriBufferUsage_AccelerationBuildInput;
        ibd2.memoryLocation = VriMemoryLocation_HostUpload;
        c.CreateBuffer(app.dev, &ibd2, &asIbuf);
        std::memcpy(c.MapBuffer(asIbuf, 0, sizeof(inst)), &inst, sizeof(inst));
        c.UnmapBuffer(asIbuf);

        VriAsGeometryDesc tlasGeom {};
        tlasGeom.type                     = VriAsGeometryType_Instances;
        tlasGeom.instances.instanceBuffer = asIbuf;
        tlasGeom.instances.instanceCount  = 1;
        VriAccelerationStructureDesc tlasDesc {};
        tlasDesc.type          = VriAccelerationStructureType_TopLevel;
        tlasDesc.flags         = VriAccelerationStructureBuild_PreferFastTrace;
        tlasDesc.geometryCount = 1;
        tlasDesc.geometries    = &tlasGeom;
        if (rt.CreateAccelerationStructure(app.dev, &tlasDesc, &tlas) != VriResult_Success)
            app.Fail("CreateAccelerationStructure (TLAS) failed");

        // build both once (the scene is static)
        c.BeginCommandBuffer(app.cmd);
        VriBuildAccelerationStructureDesc bb {};
        bb.dst      = blas;
        bb.geometry = &blasDesc;
        rt.CmdBuildAccelerationStructure(app.cmd, &bb);
        VriBuildAccelerationStructureDesc tb {};
        tb.dst      = tlas;
        tb.geometry = &tlasDesc;
        rt.CmdBuildAccelerationStructure(app.cmd, &tb);
        c.EndCommandBuffer(app.cmd);
        VriFenceSubmitDesc sig {};
        sig.fence = app.fence;
        sig.value = 1;
        VriQueueSubmitDesc sub {};
        sub.commandBuffers   = &app.cmd;
        sub.commandBufferNum = 1;
        sub.signalFences     = &sig;
        sub.signalFenceNum   = 1;
        c.QueueSubmit(app.queue, &sub);
        c.Wait(app.fence, 1);

        if (rt.CreateAccelerationStructureDescriptor(app.dev, tlas, &tlasDescriptor) != VriResult_Success)
            app.Fail("CreateAccelerationStructureDescriptor failed");

        // compute layout: scene CB @0 (root CBV at b0), TLAS @1 (SRV t0), storage image @2 (UAV u0)
        VriDescriptorRangeDesc cr[3] {};
        cr[0].baseRegister   = 0;
        cr[0].descriptorNum  = 1;
        cr[0].descriptorType = VriDescriptorType_ConstantBuffer;
        cr[0].shaderStages   = VriShaderStage_Compute;
        cr[1].baseRegister   = 1;
        cr[1].descriptorNum  = 1;
        cr[1].descriptorType = VriDescriptorType_AccelerationStructure;
        cr[1].shaderStages   = VriShaderStage_Compute;
        cr[2].baseRegister   = 2;
        cr[2].descriptorNum  = 1;
        cr[2].descriptorType = VriDescriptorType_StorageTexture;
        cr[2].shaderStages   = VriShaderStage_Compute;
        VriDescriptorSetDesc csd {};
        csd.registerSpace = 0;
        csd.ranges        = cr;
        csd.rangeNum      = 3;
        VriPipelineLayoutDesc cld {};
        cld.descriptorSets   = &csd;
        cld.descriptorSetNum = 1;
        c.CreatePipelineLayout(app.dev, &cld, &computeLayout);

        VriComputePipelineDesc cpd {};
        cpd.pipelineLayout        = computeLayout;
        cpd.shader.stage          = VriShaderStage_Compute;
        cpd.shader.entryPointName = "computeMain";
        if (app.useDxbc)
        {
            cpd.shader.bytecode     = g_rtRayqueryDxilCS;
            cpd.shader.bytecodeSize = sizeof(g_rtRayqueryDxilCS);
        }
        else
        {
            cpd.shader.bytecode     = g_rtRayquerySpv;
            cpd.shader.bytecodeSize = sizeof(g_rtRayquerySpv);
        }
        if (c.CreateComputePipeline(app.dev, &cpd, &computePipeline) != VriResult_Success)
            app.Fail("CreateComputePipeline (rayquery) failed");

        VriDescriptorPoolDesc pdsc {};
        pdsc.descriptorSetMaxNum         = 2;
        pdsc.accelerationStructureMaxNum = 1;
        pdsc.storageTextureMaxNum        = 1;
        pdsc.constantBufferMaxNum        = 1;
        pdsc.textureMaxNum               = 1;
        pdsc.samplerMaxNum               = 1;
        c.CreateDescriptorPool(app.dev, &pdsc, &pool);
        c.AllocateDescriptorSets(pool, computeLayout, 0, &computeSet, 1);
        const VriDescriptor*         a0[1] = {uboView};
        const VriDescriptor*         a1[1] = {tlasDescriptor};
        const VriDescriptor*         a2[1] = {imgView};
        VriDescriptorRangeUpdateDesc u[3] {};
        u[0].descriptors   = a0;
        u[0].descriptorNum = 1;
        u[1].descriptors   = a1;
        u[1].descriptorNum = 1;
        u[2].descriptors   = a2;
        u[2].descriptorNum = 1;
        c.UpdateDescriptorRanges(computeSet, 0, 3, u);
#endif
    }
    else
    {
        std::printf("[rayquery] software ray tracer (compute brute-force over %zu triangles)\n", g_tris.size());
        std::fflush(stdout);

        // compute layout: scene CB @0, storage image @1
        VriDescriptorRangeDesc cr[2] {};
        cr[0].baseRegister   = 0;
        cr[0].descriptorNum  = 1;
        cr[0].descriptorType = VriDescriptorType_ConstantBuffer;
        cr[0].shaderStages   = VriShaderStage_Compute;
        cr[1].baseRegister   = 1;
        cr[1].descriptorNum  = 1;
        cr[1].descriptorType = VriDescriptorType_StorageTexture;
        cr[1].shaderStages   = VriShaderStage_Compute;
        VriDescriptorSetDesc csd {};
        csd.registerSpace = 0;
        csd.ranges        = cr;
        csd.rangeNum      = 2;
        VriPipelineLayoutDesc cld {};
        cld.descriptorSets   = &csd;
        cld.descriptorSetNum = 1;
        c.CreatePipelineLayout(app.dev, &cld, &computeLayout);

        VriComputePipelineDesc cpd {};
        cpd.pipelineLayout        = computeLayout;
        cpd.shader.stage          = VriShaderStage_Compute;
        cpd.shader.entryPointName = "computeMain";
#if !defined(__EMSCRIPTEN__)
        if (app.useDxbc)
        {
            cpd.shader.bytecode     = g_rtSoftwareDxbcCS;
            cpd.shader.bytecodeSize = sizeof(g_rtSoftwareDxbcCS);
        }
        else
#endif
        {
            cpd.shader.bytecode =
                app.useWgsl ? static_cast<const void*>(g_rtSoftwareWgsl) : static_cast<const void*>(g_rtSoftwareSpv);
            cpd.shader.bytecodeSize = app.useWgsl ? sizeof(g_rtSoftwareWgsl) : sizeof(g_rtSoftwareSpv);
        }
        if (c.CreateComputePipeline(app.dev, &cpd, &computePipeline) != VriResult_Success)
            app.Fail("CreateComputePipeline (software) failed");

        VriDescriptorPoolDesc pdsc {};
        pdsc.descriptorSetMaxNum  = 2;
        pdsc.constantBufferMaxNum = 1;
        pdsc.storageTextureMaxNum = 1;
        pdsc.textureMaxNum        = 1;
        pdsc.samplerMaxNum        = 1;
        c.CreateDescriptorPool(app.dev, &pdsc, &pool);
        c.AllocateDescriptorSets(pool, computeLayout, 0, &computeSet, 1);
        const VriDescriptor*         a0[1] = {uboView};
        const VriDescriptor*         a1[1] = {imgView};
        VriDescriptorRangeUpdateDesc u[2] {};
        u[0].descriptors   = a0;
        u[0].descriptorNum = 1;
        u[1].descriptors   = a1;
        u[1].descriptorNum = 1;
        c.UpdateDescriptorRanges(computeSet, 0, 2, u);
    }

    // display layout: sampled texture @0, sampler @1
    VriDescriptorRangeDesc dr[2] {};
    dr[0].baseRegister   = 0;
    dr[0].descriptorNum  = 1;
    dr[0].descriptorType = VriDescriptorType_Texture;
    dr[0].shaderStages   = VriShaderStage_Fragment;
    dr[1].baseRegister   = 1;
    dr[1].descriptorNum  = 1;
    dr[1].descriptorType = VriDescriptorType_Sampler;
    dr[1].shaderStages   = VriShaderStage_Fragment;
    VriDescriptorSetDesc dsd {};
    dsd.registerSpace = 0;
    dsd.ranges        = dr;
    dsd.rangeNum      = 2;
    VriPipelineLayoutDesc dld {};
    dld.descriptorSets               = &dsd;
    dld.descriptorSetNum             = 1;
    VriPipelineLayout* displayLayout = nullptr;
    c.CreatePipelineLayout(app.dev, &dld, &displayLayout);

    VriShaderDesc sh[2] {};
    sh[0].stage          = VriShaderStage_Vertex;
    sh[0].entryPointName = "vertexMain";
    sh[1].stage          = VriShaderStage_Fragment;
    sh[1].entryPointName = "fragmentMain";
    if (app.useDxbc)
    {
        sh[0].bytecode     = g_showTexDxbcVS;
        sh[0].bytecodeSize = sizeof(g_showTexDxbcVS);
        sh[1].bytecode     = g_showTexDxbcPS;
        sh[1].bytecodeSize = sizeof(g_showTexDxbcPS);
    }
    else
    {
        sh[0].bytecode = sh[1].bytecode =
            app.useWgsl ? static_cast<const void*>(g_showTexWgsl) : static_cast<const void*>(g_showTexSpv);
        sh[0].bytecodeSize = sh[1].bytecodeSize = app.useWgsl ? sizeof(g_showTexWgsl) : sizeof(g_showTexSpv);
    }
    VriColorAttachmentDesc ca {};
    ca.format         = app.swapFormat;
    ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd {};
    pd.pipelineLayout            = displayLayout;
    pd.shaders                   = sh;
    pd.shaderNum                 = 2;
    pd.inputAssembly.topology    = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode    = VriCullMode_None;
    pd.rasterization.lineWidth   = 1.0f;
    pd.multisample.sampleNum     = 1;
    pd.outputMerger.colors       = &ca;
    pd.outputMerger.colorNum     = 1;
    VriPipeline* displayPipeline = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &pd, &displayPipeline) != VriResult_Success)
        app.Fail("display CreateGraphicsPipeline failed");

    VriDescriptorSet* displaySet = nullptr;
    c.AllocateDescriptorSets(pool, displayLayout, 0, &displaySet, 1);
    {
        const VriDescriptor*         a[1] = {imgView};
        const VriDescriptor*         b[1] = {sampler};
        VriDescriptorRangeUpdateDesc u[2] {};
        u[0].descriptors   = a;
        u[0].descriptorNum = 1;
        u[1].descriptors   = b;
        u[1].descriptorNum = 1;
        c.UpdateDescriptorRanges(displaySet, 0, 2, u);
    }

    static bool g_imgInit = false;
    app.onPreRender       = [=](VriCommandBuffer* cmd) {
        VriBufferCopyDesc ucp {};
        ucp.size = sizeof(Scene);
        c.CmdCopyBuffer(cmd, ubo, ustg, &ucp);
        VriBufferBarrierDesc ub {};
        ub.buffer        = ubo;
        ub.before.access = VriAccess_CopyDestinationWrite;
        ub.before.stages = VriPipelineStage_Transfer;
        ub.after.access  = VriAccess_ConstantBufferRead;
        ub.after.stages  = VriPipelineStage_ComputeShader;
        VriBarrierGroupDesc ubg {};
        ubg.buffers   = &ub;
        ubg.bufferNum = 1;
        c.CmdBarrier(cmd, &ubg);

        VriTextureBarrierDesc tw {};
        tw.texture       = img;
        tw.before.layout = g_imgInit ? VriLayout_ShaderResource : VriLayout_Undefined;
        tw.before.stages = g_imgInit ? VriPipelineStage_FragmentShader : VriPipelineStage_None;
        tw.after.access  = VriAccess_ShaderResourceStorageWrite;
        tw.after.layout  = VriLayout_ShaderResourceStorage;
        tw.after.stages  = VriPipelineStage_ComputeShader;
        tw.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc gw {};
        gw.textures   = &tw;
        gw.textureNum = 1;
        c.CmdBarrier(cmd, &gw);
        g_imgInit = true;

        c.CmdSetPipeline(cmd, computePipeline);
        c.CmdSetPipelineLayout(cmd, computeLayout);
        c.CmdSetDescriptorSet(cmd, 0, computeSet);
        VriDispatchDesc disp {};
        disp.x = (kWidth + 7) / 8;
        disp.y = (kHeight + 7) / 8;
        disp.z = 1;
        c.CmdDispatch(cmd, &disp);

        VriTextureBarrierDesc tr {};
        tr.texture       = img;
        tr.before.access = VriAccess_ShaderResourceStorageWrite;
        tr.before.layout = VriLayout_ShaderResourceStorage;
        tr.before.stages = VriPipelineStage_ComputeShader;
        tr.after.access  = VriAccess_ShaderResourceRead;
        tr.after.layout  = VriLayout_ShaderResource;
        tr.after.stages  = VriPipelineStage_FragmentShader;
        tr.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc gr {};
        gr.textures   = &tr;
        gr.textureNum = 1;
        c.CmdBarrier(cmd, &gr);
    };
    app.onRecord = [=](VriCommandBuffer* cmd) {
        c.CmdSetPipeline(cmd, displayPipeline);
        c.CmdSetPipelineLayout(cmd, displayLayout);
        c.CmdSetDescriptorSet(cmd, 0, displaySet);
        VriDrawDesc d {};
        d.vertexNum   = 3;
        d.instanceNum = 1;
        c.CmdDraw(cmd, &d);
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(displayPipeline);
    c.DestroyPipelineLayout(displayLayout);
    c.DestroyPipeline(computePipeline);
    c.DestroyPipelineLayout(computeLayout);
    if (useHw)
    {
        c.DestroyDescriptor(tlasDescriptor);
        rt.DestroyAccelerationStructure(tlas);
        rt.DestroyAccelerationStructure(blas);
        c.DestroyBuffer(asIbuf);
        c.DestroyBuffer(asVbuf);
    }
    c.DestroyDescriptor(sampler);
    c.DestroyDescriptor(imgView);
    c.DestroyTexture(img);
    c.DestroyDescriptor(uboView);
    c.DestroyBuffer(ustg);
    c.DestroyBuffer(ubo);
    app.Shutdown();
#endif
    return 0;
}
