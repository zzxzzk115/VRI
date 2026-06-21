// Ray query (inline ray tracing) on Vulkan (VK_KHR_ray_query) AND D3D12 (DXR 1.1
// inline). The SAME VRI calls drive both: build a BLAS+TLAS from a triangle (the RT
// interface - shared with ray-tracing pipelines), then a COMPUTE shader uses RayQuery
// against the TLAS and writes hit (red) / miss (black) to a storage image. No RT
// pipeline, no shader binding table - ray query reuses acceleration structures +
// compute + the AS/UAV descriptors. Self-skips where ray query is unavailable.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>

#include "shaders/rayquery_spv.h"  // g_rayquerySpv     (Vulkan)
#include "shaders/rayquery_dxil.h" // g_rayqueryDxilCS  (D3D12)

namespace
{
    constexpr uint32_t kW = 64, kH = 64;

    struct AsInstance // D3D12_RAYTRACING_INSTANCE_DESC == VkAccelerationStructureInstanceKHR (64 bytes)
    {
        float    transform[12];
        uint32_t instanceIdAndMask;
        uint32_t sbtOffsetAndFlags;
        uint64_t blasReference;
    };

    void RunRayQuery(VriGraphicsAPI api, const void* cs, size_t csSize)
    {
        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = api; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
        dc.enabledFeatures = VriFeature_RayTracing | VriFeature_RayQuery; // RT -> AS creation interface
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success) { MESSAGE("device unavailable - skipping"); return; }
        struct Guard { VriDevice* d; ~Guard() { vriDestroyDevice(d); } } guard{dev};

        VriCoreInterface c{};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        const VriDeviceDesc* dd = c.GetDeviceDesc(dev);

        VriRayTracingInterface rt{};
        const bool hasRq = dd->hasRayQuery != VRI_FALSE &&
                           vriGetInterface(dev, VRI_INTERFACE_RAYTRACING, sizeof(rt), &rt) == VriResult_Success;
        if (!hasRq) { MESSAGE("ray query unavailable on this adapter - skipping"); return; }

        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        auto hostBuf = [&](uint64_t size, VriBufferUsageFlags usage, const void* data) {
            VriBufferDesc bd{}; bd.size = size; bd.usage = usage; bd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* b = nullptr; REQUIRE(c.CreateBuffer(dev, &bd, &b) == VriResult_Success);
            if (data) { std::memcpy(c.MapBuffer(b, 0, size), data, size); c.UnmapBuffer(b); }
            return b;
        };

        const float verts[9] = { 0.0f, -0.5f, 0.0f,  0.5f, 0.5f, 0.0f,  -0.5f, 0.5f, 0.0f };
        VriBuffer* vbuf = hostBuf(sizeof(verts), VriBufferUsage_AccelerationBuildInput, verts);
        VriBuffer* ibuf = hostBuf(sizeof(AsInstance), VriBufferUsage_AccelerationBuildInput, nullptr);

        VriAsGeometryDesc blasGeom{};
        blasGeom.type = VriAsGeometryType_Triangles; blasGeom.flags = VriAsGeometry_Opaque;
        blasGeom.triangles.vertexBuffer = vbuf; blasGeom.triangles.vertexCount = 3;
        blasGeom.triangles.vertexStride = 3 * sizeof(float); blasGeom.triangles.vertexFormat = VriFormat_RGB32_SFLOAT;
        VriAccelerationStructureDesc blasDesc{};
        blasDesc.type = VriAccelerationStructureType_BottomLevel; blasDesc.flags = VriAccelerationStructureBuild_PreferFastTrace;
        blasDesc.geometryCount = 1; blasDesc.geometries = &blasGeom;
        VriAccelerationStructure* blas = nullptr;
        REQUIRE(rt.CreateAccelerationStructure(dev, &blasDesc, &blas) == VriResult_Success);

        AsInstance inst{};
        inst.transform[0] = 1.0f; inst.transform[5] = 1.0f; inst.transform[10] = 1.0f;
        inst.instanceIdAndMask = 0xFFu << 24;
        inst.sbtOffsetAndFlags = 0x1u << 24; // cull-disable
        inst.blasReference = rt.GetAccelerationStructureDeviceAddress(blas);
        std::memcpy(c.MapBuffer(ibuf, 0, sizeof(inst)), &inst, sizeof(inst)); c.UnmapBuffer(ibuf);

        VriAsGeometryDesc tlasGeom{};
        tlasGeom.type = VriAsGeometryType_Instances; tlasGeom.instances.instanceBuffer = ibuf; tlasGeom.instances.instanceCount = 1;
        VriAccelerationStructureDesc tlasDesc{};
        tlasDesc.type = VriAccelerationStructureType_TopLevel; tlasDesc.flags = VriAccelerationStructureBuild_PreferFastTrace;
        tlasDesc.geometryCount = 1; tlasDesc.geometries = &tlasGeom;
        VriAccelerationStructure* tlas = nullptr;
        REQUIRE(rt.CreateAccelerationStructure(dev, &tlasDesc, &tlas) == VriResult_Success);

        // output storage image + readback
        VriTextureDesc texDesc{};
        texDesc.type = VriTextureType_2D; texDesc.format = VriFormat_RGBA8_UNORM;
        texDesc.width = kW; texDesc.height = kH; texDesc.depth = 1; texDesc.mipNum = 1; texDesc.layerNum = 1; texDesc.sampleNum = 1;
        texDesc.usage = VriTextureUsage_ShaderResourceStorage | VriTextureUsage_TransferSrc; texDesc.memoryLocation = VriMemoryLocation_Device;
        VriTexture* outTex = nullptr; REQUIRE(c.CreateTexture(dev, &texDesc, &outTex) == VriResult_Success);
        VriTextureViewDesc viewDesc{}; viewDesc.texture = outTex; viewDesc.viewType = VriTextureViewType_2D; viewDesc.format = VriFormat_Unknown; viewDesc.aspect = VriImageAspect_Color;
        VriDescriptor* outView = nullptr; REQUIRE(c.CreateTextureView(dev, &viewDesc, &outView) == VriResult_Success);
        VriBufferDesc rbDesc{}; rbDesc.size = static_cast<uint64_t>(kW) * kH * 4; rbDesc.usage = VriBufferUsage_TransferDst; rbDesc.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr; REQUIRE(c.CreateBuffer(dev, &rbDesc, &readback) == VriResult_Success);

        // compute pipeline layout: TLAS @ t0, storage image @ u0 (compute)
        VriDescriptorRangeDesc ranges[2]{};
        // VK bindings share one namespace (AS@0, image@1, matching the shader's
        // vk::binding); D3D12 ignores baseRegister and flattens per type (t0 + u0).
        ranges[0].baseRegister = 0; ranges[0].descriptorNum = 1; ranges[0].descriptorType = VriDescriptorType_AccelerationStructure; ranges[0].shaderStages = VriShaderStage_Compute;
        ranges[1].baseRegister = 1; ranges[1].descriptorNum = 1; ranges[1].descriptorType = VriDescriptorType_StorageTexture;        ranges[1].shaderStages = VriShaderStage_Compute;
        VriDescriptorSetDesc setDesc{}; setDesc.registerSpace = 0; setDesc.ranges = ranges; setDesc.rangeNum = 2;
        VriPipelineLayoutDesc ld{}; ld.descriptorSets = &setDesc; ld.descriptorSetNum = 1;
        VriPipelineLayout* layout = nullptr; REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriComputePipelineDesc cpd{};
        cpd.pipelineLayout = layout;
        cpd.shader.stage = VriShaderStage_Compute; cpd.shader.bytecode = cs; cpd.shader.bytecodeSize = csSize; cpd.shader.entryPointName = "computeMain";
        VriPipeline* pipeline = nullptr;
        REQUIRE(c.CreateComputePipeline(dev, &cpd, &pipeline) == VriResult_Success);

        VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 1; pdsc.accelerationStructureMaxNum = 1; pdsc.storageTextureMaxNum = 1;
        VriDescriptorPool* pool = nullptr; REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
        VriDescriptorSet* set = nullptr; REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);
        VriDescriptor* tlasDescriptor = nullptr; REQUIRE(rt.CreateAccelerationStructureDescriptor(dev, tlas, &tlasDescriptor) == VriResult_Success);
        const VriDescriptor* asArr[1] = {tlasDescriptor};
        const VriDescriptor* imgArr[1] = {outView};
        VriDescriptorRangeUpdateDesc updates[2]{};
        updates[0].descriptors = asArr; updates[0].descriptorNum = 1; updates[0].baseDescriptor = 0;
        updates[1].descriptors = imgArr; updates[1].descriptorNum = 1; updates[1].baseDescriptor = 0;
        c.UpdateDescriptorRanges(set, 0, 2, updates);

        VriCommandAllocator* alloc = nullptr; REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr; REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr; REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        VriBuildAccelerationStructureDesc blasBuild{}; blasBuild.dst = blas; blasBuild.geometry = &blasDesc;
        rt.CmdBuildAccelerationStructure(cmd, &blasBuild);
        VriBuildAccelerationStructureDesc tlasBuild{}; tlasBuild.dst = tlas; tlasBuild.geometry = &tlasDesc;
        rt.CmdBuildAccelerationStructure(cmd, &tlasBuild);
        {
            VriTextureBarrierDesc b{};
            b.texture = outTex; b.before.layout = VriLayout_Undefined; b.before.stages = VriPipelineStage_None;
            b.after.access = VriAccess_ShaderResourceStorageWrite; b.after.layout = VriLayout_ShaderResourceStorage; b.after.stages = VriPipelineStage_ComputeShader;
            b.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc g{}; g.textures = &b; g.textureNum = 1; c.CmdBarrier(cmd, &g);
        }
        c.CmdSetPipeline(cmd, pipeline);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, set);
        VriDispatchDesc disp{}; disp.x = (kW + 7) / 8; disp.y = (kH + 7) / 8; disp.z = 1;
        c.CmdDispatch(cmd, &disp);
        {
            VriTextureBarrierDesc b{};
            b.texture = outTex; b.before.access = VriAccess_ShaderResourceStorageWrite; b.before.layout = VriLayout_ShaderResourceStorage; b.before.stages = VriPipelineStage_ComputeShader;
            b.after.access = VriAccess_CopySourceRead; b.after.layout = VriLayout_CopySource; b.after.stages = VriPipelineStage_Transfer;
            b.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc g{}; g.textures = &b; g.textureNum = 1; c.CmdBarrier(cmd, &g);
        }
        VriBufferTextureCopyDesc copy{}; copy.texture.aspect = VriImageAspect_Color; copy.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, outTex, &copy);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

        VriFenceSubmitDesc signal{}; signal.fence = fence; signal.value = 1; signal.stages = VriPipelineStage_AllCommands;
        VriQueueSubmitDesc submit{}; submit.commandBuffers = &cmd; submit.commandBufferNum = 1; submit.signalFences = &signal; submit.signalFenceNum = 1;
        c.QueueSubmit(queue, &submit);
        c.Wait(fence, 1);

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbDesc.size));
        REQUIRE(px != nullptr);
        const uint32_t center = ((kH / 2) * kW + (kW / 2)) * 4;
        CHECK(px[center + 0] == 255); // ray query hit the triangle
        CHECK(px[center + 1] == 0);
        CHECK(px[center + 2] == 0);
        const uint32_t corner = 0;
        CHECK(px[corner + 0] == 0);   // corner misses -> black
        CHECK(px[corner + 1] == 0);
        CHECK(px[corner + 2] == 0);
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyDescriptorPool(pool);
        c.DestroyDescriptor(tlasDescriptor); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(readback); c.DestroyDescriptor(outView); c.DestroyTexture(outTex);
        rt.DestroyAccelerationStructure(tlas); rt.DestroyAccelerationStructure(blas);
        c.DestroyBuffer(ibuf); c.DestroyBuffer(vbuf);
    }
} // namespace

// Both cases self-skip when their backend isn't compiled (vriCreateDevice -> Unsupported)
// or the adapter lacks ray query - so this one file covers Vulkan + D3D12.
TEST_CASE("Vulkan ray query: inline trace in a compute shader") { RunRayQuery(VriGraphicsAPI_Vulkan, g_rayquerySpv, sizeof(g_rayquerySpv)); }
TEST_CASE("D3D12 ray query: inline trace in a compute shader") { RunRayQuery(VriGraphicsAPI_D3D12, g_rayqueryDxilCS, sizeof(g_rayqueryDxilCS)); }
