// Mesh shaders (Metal, MTLMeshRenderPipeline): request the feature, confirm the
// interface is queryable iff the capability is reported, build a mesh+fragment
// graphics pipeline (no vertex stage / no vertex input), and dispatch one mesh
// task that emits a triangle. The center pixel must read back red - proving the
// mesh pipeline + CmdDrawMeshTasks (drawMeshThreadgroups) records and executes.
// The SPIR-V mesh shader is transpiled to MSL object/mesh functions by SPIRV-Cross.
// Skips gracefully when the adapter lacks mesh shaders. Mirrors test_meshshader_vk.cpp.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/triangle_mesh_spv.h" // g_triangleMeshSpv (mesh entry only)
#include "shaders/triangle_spv.h"      // g_triangleSpv     (reuse its fragmentMain)

namespace
{
    constexpr uint32_t kWidth = 64;
    constexpr uint32_t kHeight = 64;

    struct Mtl
    {
        VriDevice* device = nullptr;
        VriCoreInterface core{};
        ~Mtl() { if (device) vriDestroyDevice(device); }
    };

    bool InitMtl(Mtl& mtl)
    {
        VriDeviceCreationDesc desc{};
        desc.graphicsAPI = VriGraphicsAPI_Metal;
        desc.enableValidation = VRI_TRUE;
        desc.bestEffort = VRI_TRUE;
        desc.enabledFeatures = VriFeature_MeshShader;
        if (vriCreateDevice(&desc, &mtl.device) != VriResult_Success)
            return false;
        return vriGetInterface(mtl.device, VRI_INTERFACE_CORE, sizeof(mtl.core), &mtl.core) == VriResult_Success;
    }
} // namespace

TEST_CASE("Metal mesh shader: interface availability tracks the reported capability")
{
    Mtl mtl;
    if (!InitMtl(mtl)) { MESSAGE("Metal unavailable - skipping mesh shader test"); return; }

    const VriDeviceDesc* d = mtl.core.GetDeviceDesc(mtl.device);
    VriMeshShaderInterface mesh{};
    const bool queryable = vriGetInterface(mtl.device, VRI_INTERFACE_MESHSHADER, sizeof(mesh), &mesh) == VriResult_Success;
    CHECK(queryable == (d->hasMeshShader != VRI_FALSE));
    if (queryable)
        CHECK(mesh.CmdDrawMeshTasks != nullptr);
}

TEST_CASE("Metal mesh shader: dispatch one task that emits a triangle")
{
    Mtl mtl;
    if (!InitMtl(mtl)) { MESSAGE("Metal unavailable - skipping mesh shader test"); return; }

    const VriCoreInterface& c = mtl.core;
    VriDevice* dev = mtl.device;

    if (c.GetDeviceDesc(dev)->hasMeshShader == VRI_FALSE)
    {
        MESSAGE("adapter lacks mesh shaders - skipping");
        return;
    }

    VriMeshShaderInterface mesh{};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_MESHSHADER, sizeof(mesh), &mesh) == VriResult_Success);

    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    VriTextureDesc texDesc{};
    texDesc.type = VriTextureType_2D;
    texDesc.format = VriFormat_RGBA8_UNORM;
    texDesc.width = kWidth; texDesc.height = kHeight; texDesc.depth = 1;
    texDesc.mipNum = 1; texDesc.layerNum = 1; texDesc.sampleNum = 1;
    texDesc.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
    texDesc.memoryLocation = VriMemoryLocation_Device;
    VriTexture* color = nullptr;
    REQUIRE(c.CreateTexture(dev, &texDesc, &color) == VriResult_Success);

    VriTextureViewDesc viewDesc{};
    viewDesc.texture = color;
    viewDesc.viewType = VriTextureViewType_2D;
    viewDesc.format = VriFormat_Unknown;
    viewDesc.aspect = VriImageAspect_Color;
    VriDescriptor* colorView = nullptr;
    REQUIRE(c.CreateTextureView(dev, &viewDesc, &colorView) == VriResult_Success);

    VriBufferDesc bufDesc{};
    bufDesc.size = static_cast<uint64_t>(kWidth) * kHeight * 4;
    bufDesc.usage = VriBufferUsage_TransferDst;
    bufDesc.memoryLocation = VriMemoryLocation_HostReadback;
    VriBuffer* readback = nullptr;
    REQUIRE(c.CreateBuffer(dev, &bufDesc, &readback) == VriResult_Success);

    VriPipelineLayoutDesc layoutDesc{};
    VriPipelineLayout* layout = nullptr;
    REQUIRE(c.CreatePipelineLayout(dev, &layoutDesc, &layout) == VriResult_Success);

    // Mesh + fragment - no vertex stage, no vertex input (ignored for mesh pipelines).
    VriShaderDesc shaders[2]{};
    shaders[0].stage = VriShaderStage_Mesh;
    shaders[0].bytecode = g_triangleMeshSpv; shaders[0].bytecodeSize = sizeof(g_triangleMeshSpv);
    shaders[0].entryPointName = "meshMain";
    shaders[1].stage = VriShaderStage_Fragment;
    shaders[1].bytecode = g_triangleSpv; shaders[1].bytecodeSize = sizeof(g_triangleSpv);
    shaders[1].entryPointName = "fragmentMain";

    VriColorAttachmentDesc colorAttach{};
    colorAttach.format = VriFormat_RGBA8_UNORM;
    colorAttach.colorWriteMask = VriColorWrite_RGBA;

    VriGraphicsPipelineDesc pipeDesc{};
    pipeDesc.pipelineLayout = layout;
    pipeDesc.shaders = shaders;
    pipeDesc.shaderNum = 2;
    pipeDesc.rasterization.polygonMode = VriPolygonMode_Fill;
    pipeDesc.rasterization.cullMode = VriCullMode_None;
    pipeDesc.rasterization.frontFace = VriFrontFace_CounterClockwise;
    pipeDesc.rasterization.lineWidth = 1.0f;
    pipeDesc.multisample.sampleNum = 1;
    pipeDesc.outputMerger.colors = &colorAttach;
    pipeDesc.outputMerger.colorNum = 1;
    VriPipeline* pipeline = nullptr;
    REQUIRE(c.CreateGraphicsPipeline(dev, &pipeDesc, &pipeline) == VriResult_Success);

    VriCommandAllocator* allocator = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &allocator) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(allocator, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
    {
        VriTextureBarrierDesc b{};
        b.texture = color;
        b.before.layout = VriLayout_Undefined;
        b.before.stages = VriPipelineStage_None;
        b.after.access = VriAccess_ColorAttachmentWrite;
        b.after.layout = VriLayout_ColorAttachment;
        b.after.stages = VriPipelineStage_ColorAttachmentOutput;
        b.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc g{};
        g.textures = &b; g.textureNum = 1;
        c.CmdBarrier(cmd, &g);
    }

    VriAttachmentDesc colorRT{};
    colorRT.view = colorView;
    colorRT.loadOp = VriAttachmentLoadOp_Clear;
    colorRT.storeOp = VriAttachmentStoreOp_Store;
    colorRT.clearValue.color.f32[3] = 1.0f;

    VriAttachmentsDesc attachments{};
    attachments.colors = &colorRT;
    attachments.colorNum = 1;
    attachments.renderArea.width = kWidth;
    attachments.renderArea.height = kHeight;
    attachments.layerNum = 1;

    c.CmdBeginRendering(cmd, &attachments);
    VriViewport vp{0.0f, 0.0f, static_cast<float>(kWidth), static_cast<float>(kHeight), 0.0f, 1.0f};
    c.CmdSetViewports(cmd, &vp, 1);
    VriRect scissor{0, 0, kWidth, kHeight};
    c.CmdSetScissors(cmd, &scissor, 1);
    c.CmdSetPipeline(cmd, pipeline);
    mesh.CmdDrawMeshTasks(cmd, 1, 1, 1); // one workgroup -> one triangle
    c.CmdEndRendering(cmd);

    {
        VriTextureBarrierDesc b{};
        b.texture = color;
        b.before.access = VriAccess_ColorAttachmentWrite;
        b.before.layout = VriLayout_ColorAttachment;
        b.before.stages = VriPipelineStage_ColorAttachmentOutput;
        b.after.access = VriAccess_CopySourceRead;
        b.after.layout = VriLayout_CopySource;
        b.after.stages = VriPipelineStage_Transfer;
        b.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc g{};
        g.textures = &b; g.textureNum = 1;
        c.CmdBarrier(cmd, &g);
    }

    VriBufferTextureCopyDesc copy{};
    copy.texture.aspect = VriImageAspect_Color;
    copy.texture.layerNum = 1;
    c.CmdReadbackTextureToBuffer(cmd, readback, color, &copy);

    REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

    VriFenceSubmitDesc signal{};
    signal.fence = fence;
    signal.value = 1;
    signal.stages = VriPipelineStage_AllCommands;
    VriQueueSubmitDesc submit{};
    submit.commandBuffers = &cmd;
    submit.commandBufferNum = 1;
    submit.signalFences = &signal;
    submit.signalFenceNum = 1;
    c.QueueSubmit(queue, &submit);
    c.Wait(fence, 1);

    const uint8_t* pixels = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bufDesc.size));
    REQUIRE(pixels != nullptr);
    const uint32_t centerOffset = ((kHeight / 2) * kWidth + (kWidth / 2)) * 4;
    CHECK(pixels[centerOffset + 0] == 255); // R
    CHECK(pixels[centerOffset + 1] == 0);   // G
    CHECK(pixels[centerOffset + 2] == 0);   // B
    CHECK(pixels[centerOffset + 3] == 255); // A
    c.UnmapBuffer(readback);

    c.DeviceWaitIdle(dev);
    c.DestroyFence(fence);
    c.DestroyCommandAllocator(allocator);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyBuffer(readback);
    c.DestroyDescriptor(colorView);
    c.DestroyTexture(color);
}
