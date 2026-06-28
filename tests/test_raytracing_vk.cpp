// Ray tracing (Vulkan, KHR acceleration structure + ray tracing pipeline): request
// the feature, confirm the interface is queryable iff the capability is reported,
// then build a BLAS from a triangle + a TLAS with one instance, create an RT
// pipeline (raygen/miss/closesthit), assemble the SBT from the group handles, and
// CmdTraceRays into a storage image. Orthographic rays: the center pixel hits the
// triangle -> red; corners miss -> black. Skips gracefully without RT support.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>

#include "shaders/tests/rt_triangle_spv.h" // g_rtTriangleSpv (raygen/miss/closesthit)

namespace
{
    constexpr uint32_t kWidth  = 64;
    constexpr uint32_t kHeight = 64;

    struct Vk
    {
        VriDevice*       device = nullptr;
        VriCoreInterface core {};
        ~Vk()
        {
            if (device)
                vriDestroyDevice(device);
        }
    };

    bool InitVk(Vk& vk)
    {
        VriDeviceCreationDesc desc {};
        desc.graphicsAPI      = VriGraphicsAPI_Vulkan;
        desc.enableValidation = VRI_TRUE;
        desc.bestEffort       = VRI_TRUE;
        desc.enabledFeatures  = VriFeature_RayTracing;
        if (vriCreateDevice(&desc, &vk.device) != VriResult_Success)
            return false;
        return vriGetInterface(vk.device, VRI_INTERFACE_CORE, sizeof(vk.core), &vk.core) == VriResult_Success;
    }

    // Matches VkAccelerationStructureInstanceKHR (64 bytes).
    struct AsInstance
    {
        float    transform[12];              // 3x4 row-major
        uint32_t instanceCustomIndexAndMask; // [0:24) index, [24:32) mask
        uint32_t sbtOffsetAndFlags;          // [0:24) offset, [24:32) flags
        uint64_t blasReference;              // BLAS device address
    };
} // namespace

TEST_CASE("Vulkan RT: interface availability tracks the reported capability")
{
    Vk vk;
    if (!InitVk(vk))
    {
        MESSAGE("Vulkan unavailable - skipping RT test");
        return;
    }

    const VriDeviceDesc*   d = vk.core.GetDeviceDesc(vk.device);
    VriRayTracingInterface rt {};
    const bool queryable = vriGetInterface(vk.device, VRI_INTERFACE_RAYTRACING, sizeof(rt), &rt) == VriResult_Success;
    CHECK(queryable == (d->hasRayTracing != VRI_FALSE));
    if (queryable)
    {
        CHECK(rt.CmdTraceRays != nullptr);
        CHECK(d->rtShaderGroupHandleSize > 0);
        CHECK(d->rtShaderGroupBaseAlignment > 0);
    }
}

TEST_CASE("Vulkan RT: trace a triangle into a storage image")
{
    Vk vk;
    if (!InitVk(vk))
    {
        MESSAGE("Vulkan unavailable - skipping RT test");
        return;
    }

    const VriCoreInterface& c   = vk.core;
    VriDevice*              dev = vk.device;
    const VriDeviceDesc*    dd  = c.GetDeviceDesc(dev);
    if (dd->hasRayTracing == VRI_FALSE)
    {
        MESSAGE("adapter lacks ray tracing - skipping");
        return;
    }

    VriRayTracingInterface rt {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_RAYTRACING, sizeof(rt), &rt) == VriResult_Success);

    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    // ---- geometry buffers (host-visible, AS build input) ----
    const float   verts[9] = {0.0f, -0.5f, 0.0f, 0.5f, 0.5f, 0.0f, -0.5f, 0.5f, 0.0f};
    VriBufferDesc vbDesc {};
    vbDesc.size           = sizeof(verts);
    vbDesc.usage          = VriBufferUsage_AccelerationBuildInput;
    vbDesc.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* vbuf       = nullptr;
    REQUIRE(c.CreateBuffer(dev, &vbDesc, &vbuf) == VriResult_Success);
    std::memcpy(c.MapBuffer(vbuf, 0, sizeof(verts)), verts, sizeof(verts));
    c.UnmapBuffer(vbuf);

    VriBufferDesc ibDesc {};
    ibDesc.size           = sizeof(AsInstance);
    ibDesc.usage          = VriBufferUsage_AccelerationBuildInput;
    ibDesc.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* ibuf       = nullptr;
    REQUIRE(c.CreateBuffer(dev, &ibDesc, &ibuf) == VriResult_Success);

    // ---- BLAS (triangle) ----
    VriAsGeometryDesc blasGeom {};
    blasGeom.type                   = VriAsGeometryType_Triangles;
    blasGeom.flags                  = VriAsGeometry_Opaque;
    blasGeom.triangles.vertexBuffer = vbuf;
    blasGeom.triangles.vertexCount  = 3;
    blasGeom.triangles.vertexStride = 3 * sizeof(float);
    blasGeom.triangles.vertexFormat = VriFormat_RGB32_SFLOAT;
    VriAccelerationStructureDesc blasDesc {};
    blasDesc.type                  = VriAccelerationStructureType_BottomLevel;
    blasDesc.flags                 = VriAccelerationStructureBuild_PreferFastTrace;
    blasDesc.geometryCount         = 1;
    blasDesc.geometries            = &blasGeom;
    VriAccelerationStructure* blas = nullptr;
    REQUIRE(rt.CreateAccelerationStructure(dev, &blasDesc, &blas) == VriResult_Success);

    // instance referencing the BLAS (identity transform, cull disabled)
    AsInstance inst {};
    inst.transform[0]               = 1.0f;
    inst.transform[5]               = 1.0f;
    inst.transform[10]              = 1.0f;
    inst.instanceCustomIndexAndMask = 0xFFu << 24;
    inst.sbtOffsetAndFlags          = 0x1u << 24; // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT
    inst.blasReference              = rt.GetAccelerationStructureDeviceAddress(blas);
    std::memcpy(c.MapBuffer(ibuf, 0, sizeof(inst)), &inst, sizeof(inst));
    c.UnmapBuffer(ibuf);

    // ---- TLAS (one instance) ----
    VriAsGeometryDesc tlasGeom {};
    tlasGeom.type                     = VriAsGeometryType_Instances;
    tlasGeom.flags                    = VriAsGeometry_Opaque;
    tlasGeom.instances.instanceBuffer = ibuf;
    tlasGeom.instances.instanceCount  = 1;
    VriAccelerationStructureDesc tlasDesc {};
    tlasDesc.type                  = VriAccelerationStructureType_TopLevel;
    tlasDesc.flags                 = VriAccelerationStructureBuild_PreferFastTrace;
    tlasDesc.geometryCount         = 1;
    tlasDesc.geometries            = &tlasGeom;
    VriAccelerationStructure* tlas = nullptr;
    REQUIRE(rt.CreateAccelerationStructure(dev, &tlasDesc, &tlas) == VriResult_Success);

    // ---- output storage image + readback ----
    VriTextureDesc texDesc {};
    texDesc.type           = VriTextureType_2D;
    texDesc.format         = VriFormat_RGBA8_UNORM;
    texDesc.width          = kWidth;
    texDesc.height         = kHeight;
    texDesc.depth          = 1;
    texDesc.mipNum         = 1;
    texDesc.layerNum       = 1;
    texDesc.sampleNum      = 1;
    texDesc.usage          = VriTextureUsage_ShaderResourceStorage | VriTextureUsage_TransferSrc;
    texDesc.memoryLocation = VriMemoryLocation_Device;
    VriTexture* outTex     = nullptr;
    REQUIRE(c.CreateTexture(dev, &texDesc, &outTex) == VriResult_Success);

    VriTextureViewDesc viewDesc {};
    viewDesc.texture       = outTex;
    viewDesc.viewType      = VriTextureViewType_2D;
    viewDesc.format        = VriFormat_Unknown;
    viewDesc.aspect        = VriImageAspect_Color;
    VriDescriptor* outView = nullptr;
    REQUIRE(c.CreateTextureView(dev, &viewDesc, &outView) == VriResult_Success);

    VriBufferDesc rbDesc {};
    rbDesc.size           = static_cast<uint64_t>(kWidth) * kHeight * 4;
    rbDesc.usage          = VriBufferUsage_TransferDst;
    rbDesc.memoryLocation = VriMemoryLocation_HostReadback;
    VriBuffer* readback   = nullptr;
    REQUIRE(c.CreateBuffer(dev, &rbDesc, &readback) == VriResult_Success);

    // ---- pipeline layout: TLAS @ b0, storage image @ b1 (raygen) ----
    VriDescriptorRangeDesc ranges[2] {};
    ranges[0].baseRegister   = 0;
    ranges[0].descriptorNum  = 1;
    ranges[0].descriptorType = VriDescriptorType_AccelerationStructure;
    ranges[0].shaderStages   = VriShaderStage_RayGen;
    ranges[1].baseRegister   = 1;
    ranges[1].descriptorNum  = 1;
    ranges[1].descriptorType = VriDescriptorType_StorageTexture;
    ranges[1].shaderStages   = VriShaderStage_RayGen;
    VriDescriptorSetDesc setDesc {};
    setDesc.registerSpace = 0;
    setDesc.ranges        = ranges;
    setDesc.rangeNum      = 2;
    VriPipelineLayoutDesc ld {};
    ld.descriptorSets         = &setDesc;
    ld.descriptorSetNum       = 1;
    VriPipelineLayout* layout = nullptr;
    REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

    // ---- RT pipeline (3 shaders, 3 groups) ----
    VriShaderDesc sh[3] {};
    sh[0].stage          = VriShaderStage_RayGen;
    sh[0].bytecode       = g_rtTriangleSpv;
    sh[0].bytecodeSize   = sizeof(g_rtTriangleSpv);
    sh[0].entryPointName = "rayGenMain";
    sh[1].stage          = VriShaderStage_Miss;
    sh[1].bytecode       = g_rtTriangleSpv;
    sh[1].bytecodeSize   = sizeof(g_rtTriangleSpv);
    sh[1].entryPointName = "missMain";
    sh[2].stage          = VriShaderStage_ClosestHit;
    sh[2].bytecode       = g_rtTriangleSpv;
    sh[2].bytecodeSize   = sizeof(g_rtTriangleSpv);
    sh[2].entryPointName = "closestHitMain";
    VriShaderGroupDesc groups[3] {};
    groups[0].type               = VriShaderGroupType_General;
    groups[0].generalShader      = 0;
    groups[0].closestHitShader   = VRI_SHADER_UNUSED;
    groups[0].anyHitShader       = VRI_SHADER_UNUSED;
    groups[0].intersectionShader = VRI_SHADER_UNUSED;
    groups[1].type               = VriShaderGroupType_General;
    groups[1].generalShader      = 1;
    groups[1].closestHitShader   = VRI_SHADER_UNUSED;
    groups[1].anyHitShader       = VRI_SHADER_UNUSED;
    groups[1].intersectionShader = VRI_SHADER_UNUSED;
    groups[2].type               = VriShaderGroupType_TrianglesHitGroup;
    groups[2].generalShader      = VRI_SHADER_UNUSED;
    groups[2].closestHitShader   = 2;
    groups[2].anyHitShader       = VRI_SHADER_UNUSED;
    groups[2].intersectionShader = VRI_SHADER_UNUSED;
    VriRayTracingPipelineDesc rtpd {};
    rtpd.pipelineLayout    = layout;
    rtpd.shaders           = sh;
    rtpd.shaderNum         = 3;
    rtpd.groups            = groups;
    rtpd.groupNum          = 3;
    rtpd.maxRecursionDepth = 1;
    VriPipeline* pipeline  = nullptr;
    REQUIRE(rt.CreateRayTracingPipeline(dev, &rtpd, &pipeline) == VriResult_Success);

    // ---- SBT: one record per region, each at base alignment ----
    const uint32_t handleSize      = dd->rtShaderGroupHandleSize;
    const uint32_t baseAlign       = dd->rtShaderGroupBaseAlignment;
    uint8_t        handles[3 * 64] = {}; // handleSize <= 64 in practice
    REQUIRE(handleSize <= 64);
    REQUIRE(rt.GetShaderGroupHandles(pipeline, 0, 3, sizeof(handles), handles) == VriResult_Success);

    VriBufferDesc sbtDesc {};
    sbtDesc.size           = static_cast<uint64_t>(baseAlign) * 3;
    sbtDesc.usage          = VriBufferUsage_ShaderBindingTable;
    sbtDesc.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* sbt         = nullptr;
    REQUIRE(c.CreateBuffer(dev, &sbtDesc, &sbt) == VriResult_Success);
    {
        uint8_t* p = static_cast<uint8_t*>(c.MapBuffer(sbt, 0, sbtDesc.size));
        std::memset(p, 0, sbtDesc.size);
        std::memcpy(p + 0u * baseAlign, handles + 0u * handleSize, handleSize); // raygen
        std::memcpy(p + 1u * baseAlign, handles + 1u * handleSize, handleSize); // miss
        std::memcpy(p + 2u * baseAlign, handles + 2u * handleSize, handleSize); // hit
        c.UnmapBuffer(sbt);
    }

    // ---- descriptors ----
    VriDescriptorPoolDesc pdsc {};
    pdsc.descriptorSetMaxNum         = 1;
    pdsc.accelerationStructureMaxNum = 1;
    pdsc.storageTextureMaxNum        = 1;
    VriDescriptorPool* pool          = nullptr;
    REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
    VriDescriptorSet* set = nullptr;
    REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);

    VriDescriptor* tlasDescriptor = nullptr;
    REQUIRE(rt.CreateAccelerationStructureDescriptor(dev, tlas, &tlasDescriptor) == VriResult_Success);
    const VriDescriptor*         asArr[1]  = {tlasDescriptor};
    const VriDescriptor*         imgArr[1] = {outView};
    VriDescriptorRangeUpdateDesc updates[2] {};
    updates[0].descriptors    = asArr;
    updates[0].descriptorNum  = 1;
    updates[0].baseDescriptor = 0;
    updates[1].descriptors    = imgArr;
    updates[1].descriptorNum  = 1;
    updates[1].baseDescriptor = 0;
    c.UpdateDescriptorRanges(set, 0, 2, updates);

    // ---- record ----
    VriCommandAllocator* alloc = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);

    VriBuildAccelerationStructureDesc blasBuild {};
    blasBuild.dst      = blas;
    blasBuild.geometry = &blasDesc;
    rt.CmdBuildAccelerationStructure(cmd, &blasBuild);
    VriBuildAccelerationStructureDesc tlasBuild {};
    tlasBuild.dst      = tlas;
    tlasBuild.geometry = &tlasDesc;
    rt.CmdBuildAccelerationStructure(cmd, &tlasBuild);

    // output image Undefined -> General (storage write by the raygen shader)
    {
        VriTextureBarrierDesc b {};
        b.texture       = outTex;
        b.before.layout = VriLayout_Undefined;
        b.before.stages = VriPipelineStage_None;
        b.after.access  = VriAccess_ShaderResourceStorageWrite;
        b.after.layout  = VriLayout_ShaderResourceStorage;
        b.after.stages  = VriPipelineStage_RayTracingShader;
        b.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc g {};
        g.textures   = &b;
        g.textureNum = 1;
        c.CmdBarrier(cmd, &g);
    }

    c.CmdSetPipeline(cmd, pipeline);
    c.CmdSetPipelineLayout(cmd, layout);
    c.CmdSetDescriptorSet(cmd, 0, set);

    VriDispatchRaysDesc trace {};
    trace.raygen.buffer = sbt;
    trace.raygen.offset = 0u * baseAlign;
    trace.raygen.stride = baseAlign;
    trace.raygen.size   = baseAlign;
    trace.miss.buffer   = sbt;
    trace.miss.offset   = 1u * baseAlign;
    trace.miss.stride   = baseAlign;
    trace.miss.size     = baseAlign;
    trace.hit.buffer    = sbt;
    trace.hit.offset    = 2u * baseAlign;
    trace.hit.stride    = baseAlign;
    trace.hit.size      = baseAlign;
    trace.width         = kWidth;
    trace.height        = kHeight;
    trace.depth         = 1;
    rt.CmdTraceRays(cmd, &trace);

    // output image General -> CopySource, then read back
    {
        VriTextureBarrierDesc b {};
        b.texture       = outTex;
        b.before.access = VriAccess_ShaderResourceStorageWrite;
        b.before.layout = VriLayout_ShaderResourceStorage;
        b.before.stages = VriPipelineStage_RayTracingShader;
        b.after.access  = VriAccess_CopySourceRead;
        b.after.layout  = VriLayout_CopySource;
        b.after.stages  = VriPipelineStage_Transfer;
        b.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc g {};
        g.textures   = &b;
        g.textureNum = 1;
        c.CmdBarrier(cmd, &g);
    }

    VriBufferTextureCopyDesc copy {};
    copy.texture.aspect   = VriImageAspect_Color;
    copy.texture.layerNum = 1;
    c.CmdReadbackTextureToBuffer(cmd, readback, outTex, &copy);

    REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

    VriFenceSubmitDesc signal {};
    signal.fence  = fence;
    signal.value  = 1;
    signal.stages = VriPipelineStage_AllCommands;
    VriQueueSubmitDesc submit {};
    submit.commandBuffers   = &cmd;
    submit.commandBufferNum = 1;
    submit.signalFences     = &signal;
    submit.signalFenceNum   = 1;
    c.QueueSubmit(queue, &submit);
    c.Wait(fence, 1);

    const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbDesc.size));
    REQUIRE(px != nullptr);
    const uint32_t center = ((kHeight / 2) * kWidth + (kWidth / 2)) * 4;
    CHECK(px[center + 0] == 255); // R - ray hit the triangle
    CHECK(px[center + 1] == 0);
    CHECK(px[center + 2] == 0);
    CHECK(px[center + 3] == 255);
    const uint32_t corner = 0; // top-left misses -> black background
    CHECK(px[corner + 0] == 0);
    CHECK(px[corner + 1] == 0);
    CHECK(px[corner + 2] == 0);
    c.UnmapBuffer(readback);

    c.DeviceWaitIdle(dev);
    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    c.DestroyDescriptorPool(pool);
    c.DestroyDescriptor(tlasDescriptor);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyBuffer(sbt);
    c.DestroyBuffer(readback);
    c.DestroyDescriptor(outView);
    c.DestroyTexture(outTex);
    rt.DestroyAccelerationStructure(tlas);
    rt.DestroyAccelerationStructure(blas);
    c.DestroyBuffer(ibuf);
    c.DestroyBuffer(vbuf);
}
