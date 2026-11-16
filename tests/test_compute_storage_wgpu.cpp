// WebGPU compute with push constants and typed storage textures: one dispatch reads and writes an r32float
// image (read_write, core WebGPU) and writes an rgba16float image (write-only), with a push-constant value,
// through a layout that names every stage as layouts written for the other backends do. Both images are read
// back and checked. The WGSL is what Slang emits: an undecorated `var<uniform>` push block.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>

namespace
{
    constexpr uint32_t kW = 64; // 64 r32 texels and 32 rgba16 texels fill one 256-byte row each.
    constexpr uint32_t kH = 2;

    constexpr char kShader[] = R"(struct Push
{
    value : f32,
};
var<uniform> push : Push;
@group(0) @binding(0) var depth : texture_storage_2d<r32float, read_write>;
@group(0) @binding(1) var color : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 2, 1)
fn computeMain(@builtin(global_invocation_id) id : vec3<u32>)
{
    let previous = textureLoad(depth, id.xy).x;
    textureStore(depth, id.xy, vec4<f32>(previous + push.value, 0.0, 0.0, 0.0));
    textureStore(color, id.xy, vec4<f32>(push.value, f32(id.x), f32(id.y), 1.0));
}
)";

    // IEEE half to float for the readback of the rgba16float image.
    float HalfToFloat(uint16_t h)
    {
        const uint32_t sign     = (h >> 15) & 1u;
        const uint32_t exponent = (h >> 10) & 0x1fu;
        const uint32_t mantissa = h & 0x3ffu;
        uint32_t       bits     = sign << 31;
        if (exponent != 0)
            bits |= ((exponent + 112u) << 23) | (mantissa << 13);
        float result = 0;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
} // namespace

TEST_CASE("WebGPU compute: push constants and typed storage textures")
{
    VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
    dc.graphicsAPI           = VriGraphicsAPI_WebGPU;
    dc.enableValidation      = VRI_TRUE;
    VriDevice* dev           = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("WebGPU unavailable - skipped");
        return;
    }
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    REQUIRE(c.GetDeviceDesc(dev)->hasComputeShader);
    CHECK((c.GetFormatSupport(dev, VriFormat_RGBA16_SFLOAT) & VriFormatSupport_StorageTexture) != 0);
    CHECK((c.GetFormatSupport(dev, VriFormat_R32_SFLOAT) & VriFormatSupport_StorageTexture) != 0);
    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    const VriFormat formats[2] {VriFormat_R32_SFLOAT, VriFormat_RGBA16_SFLOAT};
    VriTexture*     textures[2] {};
    VriDescriptor*  views[2] {};
    VriBuffer*      readbacks[2] {};
    for (int i = 0; i < 2; ++i)
    {
        VriTextureDesc td {};
        td.type           = VriTextureType_2D;
        td.format         = formats[i];
        td.width          = kW;
        td.height         = kH;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = 1;
        td.sampleNum      = 1;
        td.usage          = VriTextureUsage_ShaderResourceStorage | VriTextureUsage_TransferSrc;
        td.memoryLocation = VriMemoryLocation_Device;
        REQUIRE(c.CreateTexture(dev, &td, &textures[i]) == VriResult_Success);
        VriTextureViewDesc vd {};
        vd.texture  = textures[i];
        vd.viewType = VriTextureViewType_2D;
        vd.aspect   = VriImageAspect_Color;
        REQUIRE(c.CreateTextureView(dev, &vd, &views[i]) == VriResult_Success);
        VriBufferDesc rbd {};
        rbd.size           = 256u * kH * (i == 0 ? 1u : 2u);
        rbd.usage          = VriBufferUsage_TransferDst;
        rbd.memoryLocation = VriMemoryLocation_HostReadback;
        REQUIRE(c.CreateBuffer(dev, &rbd, &readbacks[i]) == VriResult_Success);
    }

    // Every stage named, as a Vulkan-style layout would; the backend drops vertex for writable storage.
    const VriShaderStageFlags stages = VriShaderStage_Vertex | VriShaderStage_Fragment | VriShaderStage_Compute;
    VriDescriptorRangeDesc    ranges[2] {};
    ranges[0] = {0,
                 1,
                 VriDescriptorType_StorageTexture,
                 stages,
                 VriDescriptorRange_None,
                 VriTextureViewType_2D,
                 VriFormat_R32_SFLOAT};
    ranges[1] = {1,
                 1,
                 VriDescriptorType_StorageTexture,
                 stages,
                 VriDescriptorRange_WriteOnly,
                 VriTextureViewType_2D,
                 VriFormat_RGBA16_SFLOAT};
    VriDescriptorSetDesc  set {0, ranges, 2};
    VriPushConstantDesc   push {0, sizeof(float), stages};
    VriPipelineLayoutDesc ld {};
    ld.descriptorSets         = &set;
    ld.descriptorSetNum       = 1;
    ld.pushConstants          = &push;
    ld.pushConstantNum        = 1;
    ld.shaderStages           = stages;
    VriPipelineLayout* layout = nullptr;
    REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);
    VriComputePipelineDesc cpd {};
    cpd.pipelineLayout        = layout;
    cpd.shader.stage          = VriShaderStage_Compute;
    cpd.shader.bytecode       = kShader;
    cpd.shader.bytecodeSize   = sizeof(kShader);
    cpd.shader.entryPointName = "computeMain";
    VriPipeline* pipeline     = nullptr;
    REQUIRE(c.CreateComputePipeline(dev, &cpd, &pipeline) == VriResult_Success);

    VriDescriptorPoolDesc pd {};
    pd.descriptorSetMaxNum  = 1;
    pd.storageTextureMaxNum = 2;
    VriDescriptorPool* pool = nullptr;
    REQUIRE(c.CreateDescriptorPool(dev, &pd, &pool) == VriResult_Success);
    VriDescriptorSet* descriptorSet = nullptr;
    REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &descriptorSet, 1) == VriResult_Success);
    const VriDescriptor*         first[1] {views[0]};
    const VriDescriptor*         second[1] {views[1]};
    VriDescriptorRangeUpdateDesc updates[2] {{first, 1, 0}, {second, 1, 0}};
    c.UpdateDescriptorRanges(descriptorSet, 0, 2, updates);

    VriCommandAllocator* alloc = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
    c.CmdSetPipelineLayout(cmd, layout);
    c.CmdSetPipeline(cmd, pipeline);
    c.CmdSetDescriptorSet(cmd, 0, descriptorSet);
    const float value = 2.5f;
    c.CmdSetConstants(cmd, 0, &value, sizeof(value));
    VriDispatchDesc dispatch {kW / 8, 1, 1};
    c.CmdDispatch(cmd, &dispatch);
    for (int i = 0; i < 2; ++i)
    {
        VriTextureBarrierDesc tb {};
        tb.texture       = textures[i];
        tb.before.access = VriAccess_ShaderResourceStorageWrite;
        tb.before.layout = VriLayout_General;
        tb.before.stages = VriPipelineStage_ComputeShader;
        tb.after.access  = VriAccess_CopySourceRead;
        tb.after.layout  = VriLayout_CopySource;
        tb.after.stages  = VriPipelineStage_Transfer;
        tb.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc g {};
        g.textures   = &tb;
        g.textureNum = 1;
        c.CmdBarrier(cmd, &g);
        VriBufferTextureCopyDesc rc {};
        rc.texture.aspect   = VriImageAspect_Color;
        rc.texture.layerNum = 1;
        rc.texture.width    = kW;
        rc.texture.height   = kH;
        c.CmdReadbackTextureToBuffer(cmd, readbacks[i], textures[i], &rc);
    }
    REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
    VriFenceSubmitDesc signal {};
    signal.fence = fence;
    signal.value = 1;
    VriQueueSubmitDesc submit {};
    submit.commandBuffers   = &cmd;
    submit.commandBufferNum = 1;
    submit.signalFences     = &signal;
    submit.signalFenceNum   = 1;
    c.QueueSubmit(queue, &submit);
    c.Wait(fence, 1);

    const auto* depth = static_cast<const float*>(c.MapBuffer(readbacks[0], 0, 256u * kH));
    REQUIRE(depth != nullptr);
    CHECK(depth[0] == 2.5f);
    CHECK(depth[64 + 63] == 2.5f);
    c.UnmapBuffer(readbacks[0]);
    const auto* color = static_cast<const uint16_t*>(c.MapBuffer(readbacks[1], 0, 512u * kH));
    REQUIRE(color != nullptr);
    const uint16_t* texel = color + (1 * kW + 5) * 4; // (x 5, y 1)
    CHECK(HalfToFloat(texel[0]) == 2.5f);
    CHECK(HalfToFloat(texel[1]) == 5.0f);
    CHECK(HalfToFloat(texel[2]) == 1.0f);
    CHECK(HalfToFloat(texel[3]) == 1.0f);
    c.UnmapBuffer(readbacks[1]);

    c.DeviceWaitIdle(dev);
    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    for (int i = 0; i < 2; ++i)
    {
        c.DestroyDescriptor(views[i]);
        c.DestroyTexture(textures[i]);
        c.DestroyBuffer(readbacks[i]);
    }
    vriDestroyDevice(dev);
}

namespace
{
    constexpr char kDepthShader[] = R"(@group(0) @binding(0) var source : texture_2d<f32>;
@group(0) @binding(1) var copied : texture_storage_2d<r32float, read_write>;

@compute @workgroup_size(8, 2, 1)
fn computeMain(@builtin(global_invocation_id) id : vec3<u32>)
{
    textureStore(copied, id.xy, vec4<f32>(textureLoad(source, id.xy, 0).x, 0.0, 0.0, 0.0));
}
)";
    constexpr char kSceneShader[] = R"(struct Push
{
    depth : f32,
};
var<uniform> push : Push;

@vertex
fn vertexMain(@builtin(vertex_index) id : u32) -> @builtin(position) vec4<f32>
{
    let corner = vec2<f32>(f32((id << 1u) & 2u), f32(id & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragmentMain() -> @builtin(frag_depth) f32
{
    return push.depth;
}
)";
} // namespace

// A depth buffer cleared and then drawn by a depth-only render pass, with the depth from a push constant, and read
// by a compute shader as an unfilterable float, as a depth pyramid does.
TEST_CASE("WebGPU compute reads a depth buffer as an unfilterable float")
{
    VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
    dc.graphicsAPI           = VriGraphicsAPI_WebGPU;
    dc.enableValidation      = VRI_TRUE;
    VriDevice* dev           = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("WebGPU unavailable - skipped");
        return;
    }
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    VriTextureDesc td {};
    td.type           = VriTextureType_2D;
    td.format         = VriFormat_D32_SFLOAT;
    td.width          = kW;
    td.height         = kH;
    td.depth          = 1;
    td.mipNum         = 1;
    td.layerNum       = 1;
    td.sampleNum      = 1;
    td.usage          = VriTextureUsage_DepthStencilAttachment | VriTextureUsage_ShaderResource;
    td.memoryLocation = VriMemoryLocation_Device;
    VriTexture* depth = nullptr;
    REQUIRE(c.CreateTexture(dev, &td, &depth) == VriResult_Success);
    td.format        = VriFormat_R32_SFLOAT;
    td.usage         = VriTextureUsage_ShaderResourceStorage | VriTextureUsage_TransferSrc;
    VriTexture* copy = nullptr;
    REQUIRE(c.CreateTexture(dev, &td, &copy) == VriResult_Success);
    VriTextureViewDesc vd {};
    vd.texture               = depth;
    vd.viewType              = VriTextureViewType_2D;
    vd.aspect                = VriImageAspect_Depth;
    VriDescriptor* depthView = nullptr;
    REQUIRE(c.CreateTextureView(dev, &vd, &depthView) == VriResult_Success);
    vd.texture              = copy;
    vd.aspect               = VriImageAspect_Color;
    VriDescriptor* copyView = nullptr;
    REQUIRE(c.CreateTextureView(dev, &vd, &copyView) == VriResult_Success);
    VriBufferDesc rbd {};
    rbd.size            = 256u * kH;
    rbd.usage           = VriBufferUsage_TransferDst;
    rbd.memoryLocation  = VriMemoryLocation_HostReadback;
    VriBuffer* readback = nullptr;
    REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

    VriDescriptorRangeDesc ranges[2] {};
    ranges[0] = {0,
                 1,
                 VriDescriptorType_Texture,
                 VriShaderStage_Compute,
                 VriDescriptorRange_None,
                 VriTextureViewType_2D,
                 VriFormat_D32_SFLOAT};
    ranges[1] = {1,
                 1,
                 VriDescriptorType_StorageTexture,
                 VriShaderStage_Compute,
                 VriDescriptorRange_None,
                 VriTextureViewType_2D,
                 VriFormat_R32_SFLOAT};
    VriDescriptorSetDesc  set {0, ranges, 2};
    VriPipelineLayoutDesc ld {};
    ld.descriptorSets         = &set;
    ld.descriptorSetNum       = 1;
    ld.shaderStages           = VriShaderStage_Compute;
    VriPipelineLayout* layout = nullptr;
    REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);
    VriComputePipelineDesc cpd {};
    cpd.pipelineLayout        = layout;
    cpd.shader.stage          = VriShaderStage_Compute;
    cpd.shader.bytecode       = kDepthShader;
    cpd.shader.bytecodeSize   = sizeof(kDepthShader);
    cpd.shader.entryPointName = "computeMain";
    VriPipeline* pipeline     = nullptr;
    REQUIRE(c.CreateComputePipeline(dev, &cpd, &pipeline) == VriResult_Success);
    VriDescriptorPoolDesc pd {};
    pd.descriptorSetMaxNum  = 1;
    pd.textureMaxNum        = 1;
    pd.storageTextureMaxNum = 1;
    VriDescriptorPool* pool = nullptr;
    REQUIRE(c.CreateDescriptorPool(dev, &pd, &pool) == VriResult_Success);
    VriDescriptorSet* descriptorSet = nullptr;
    REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &descriptorSet, 1) == VriResult_Success);
    const VriDescriptor*         first[1] {depthView};
    const VriDescriptor*         second[1] {copyView};
    VriDescriptorRangeUpdateDesc updates[2] {{first, 1, 0}, {second, 1, 0}};
    c.UpdateDescriptorRanges(descriptorSet, 0, 2, updates);

    VriPushConstantDesc   push {0, sizeof(float), VriShaderStage_Vertex | VriShaderStage_Fragment};
    VriPipelineLayoutDesc sceneLd {};
    sceneLd.pushConstants          = &push;
    sceneLd.pushConstantNum        = 1;
    sceneLd.shaderStages           = VriShaderStage_Vertex | VriShaderStage_Fragment;
    VriPipelineLayout* sceneLayout = nullptr;
    REQUIRE(c.CreatePipelineLayout(dev, &sceneLd, &sceneLayout) == VriResult_Success);
    VriShaderDesc sceneShaders[2] {};
    sceneShaders[0].stage          = VriShaderStage_Vertex;
    sceneShaders[0].bytecode       = kSceneShader;
    sceneShaders[0].bytecodeSize   = sizeof(kSceneShader);
    sceneShaders[0].entryPointName = "vertexMain";
    sceneShaders[1]                = sceneShaders[0];
    sceneShaders[1].stage          = VriShaderStage_Fragment;
    sceneShaders[1].entryPointName = "fragmentMain";
    VriGraphicsPipelineDesc gpd {};
    gpd.pipelineLayout                  = sceneLayout;
    gpd.shaders                         = sceneShaders;
    gpd.shaderNum                       = 2;
    gpd.inputAssembly.topology          = VriPrimitiveTopology_TriangleList;
    gpd.rasterization.cullMode          = VriCullMode_None;
    gpd.rasterization.lineWidth         = 1;
    gpd.multisample.sampleNum           = 1;
    gpd.depthStencil.depthTest          = VRI_TRUE;
    gpd.depthStencil.depthWrite         = VRI_TRUE;
    gpd.depthStencil.depthCompareOp     = VriCompareOp_Always;
    gpd.outputMerger.depthStencilFormat = VriFormat_D32_SFLOAT;
    VriPipeline* scene                  = nullptr;
    REQUIRE(c.CreateGraphicsPipeline(dev, &gpd, &scene) == VriResult_Success);

    VriCommandAllocator* alloc = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);
    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
    VriAttachmentDesc depthAttachment {};
    depthAttachment.view                          = depthView;
    depthAttachment.loadOp                        = VriAttachmentLoadOp_Clear;
    depthAttachment.storeOp                       = VriAttachmentStoreOp_Store;
    depthAttachment.clearValue.depthStencil.depth = 0.75f;
    VriAttachmentsDesc attachments {};
    attachments.depth      = &depthAttachment;
    attachments.renderArea = {0, 0, kW, kH};
    attachments.layerNum   = 1;
    c.CmdBeginRendering(cmd, &attachments);
    c.CmdSetPipelineLayout(cmd, sceneLayout);
    c.CmdSetPipeline(cmd, scene);
    const VriViewport viewport {0, 0, float(kW), float(kH), 0, 1};
    const VriRect     scissor {0, 0, kW, kH};
    c.CmdSetViewports(cmd, &viewport, 1);
    c.CmdSetScissors(cmd, &scissor, 1);
    const float sceneDepth = 0.5f;
    c.CmdSetConstants(cmd, 0, &sceneDepth, sizeof(sceneDepth));
    const VriDrawDesc draw {3, 1, 0, 0};
    c.CmdDraw(cmd, &draw);
    c.CmdEndRendering(cmd);
    c.CmdSetPipelineLayout(cmd, layout);
    c.CmdSetPipeline(cmd, pipeline);
    c.CmdSetDescriptorSet(cmd, 0, descriptorSet);
    VriDispatchDesc dispatch {kW / 8, 1, 1};
    c.CmdDispatch(cmd, &dispatch);
    VriBufferTextureCopyDesc rc {};
    rc.texture.aspect   = VriImageAspect_Color;
    rc.texture.layerNum = 1;
    rc.texture.width    = kW;
    rc.texture.height   = kH;
    c.CmdReadbackTextureToBuffer(cmd, readback, copy, &rc);
    REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
    VriFenceSubmitDesc signal {};
    signal.fence = fence;
    signal.value = 1;
    VriQueueSubmitDesc submit {};
    submit.commandBuffers   = &cmd;
    submit.commandBufferNum = 1;
    submit.signalFences     = &signal;
    submit.signalFenceNum   = 1;
    c.QueueSubmit(queue, &submit);
    c.Wait(fence, 1);
    const auto* values = static_cast<const float*>(c.MapBuffer(readback, 0, rbd.size));
    REQUIRE(values != nullptr);
    CHECK(values[0] == 0.5f);
    CHECK(values[64 + 10] == 0.5f);
    c.UnmapBuffer(readback);

    c.DeviceWaitIdle(dev);
    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(scene);
    c.DestroyPipelineLayout(sceneLayout);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyDescriptor(depthView);
    c.DestroyDescriptor(copyView);
    c.DestroyTexture(depth);
    c.DestroyTexture(copy);
    c.DestroyBuffer(readback);
    vriDestroyDevice(dev);
}
