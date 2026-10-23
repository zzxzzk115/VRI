#include "shaders/tests/conventions_spv.h"
#include "shaders/tests/conventions_wgsl.h"
#if defined(_WIN32)
#include "shaders/tests/conventions_dxil.h"
#endif

#include <vri/vri.h>

#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

extern "C" int vriTestCoordinatesC(VriDevice* device, size_t creationSize, size_t conventionsSize);

namespace
{
    constexpr uint32_t kSize = 64;
    using Pixel              = std::array<uint8_t, 4>;

    struct Push
    {
        std::array<float, 4> clip;
        std::array<float, 4> authoring;
        std::array<float, 4> uv;
        std::array<float, 4> tint;
    };
    static_assert(sizeof(Push) == 64);

    void VRI_CALL Diagnostic(void* user, VriMessageSeverity severity, const char* message)
    {
        if (severity == VriMessageSeverity_Error)
        {
            ++*static_cast<uint32_t*>(user);
            std::fprintf(stderr, "[coordinate test] %s\n", message);
        }
    }

    void Transition(const VriCoreInterface& c,
                    VriCommandBuffer*       cmd,
                    VriTexture*             texture,
                    VriLayout               before,
                    VriLayout               after,
                    VriAccessFlags          access,
                    VriPipelineStageFlags   stages,
                    VriImageAspectFlags     aspect)
    {
        VriTextureBarrierDesc barrier {};
        barrier.texture       = texture;
        barrier.before.layout = before;
        barrier.before.access = before == VriLayout_ColorAttachment ? VriAccess_ColorAttachmentWrite :
                                before == VriLayout_CopyDestination ? VriAccess_CopyDestinationWrite :
                                                                      VriAccess_None;
        barrier.before.stages = before == VriLayout_ColorAttachment ? VriPipelineStage_ColorAttachmentOutput :
                                before == VriLayout_CopyDestination ? VriPipelineStage_Transfer :
                                                                      VriPipelineStage_None;
        barrier.after         = {access, after, stages};
        barrier.aspect        = aspect;
        VriBarrierGroupDesc group {};
        group.textures   = &barrier;
        group.textureNum = 1;
        c.CmdBarrier(cmd, &group);
    }

    void Capture(VriGraphicsAPI api, uint32_t variant, const std::vector<uint8_t>& image)
    {
        const char* directory = std::getenv("VRI_COORDINATE_CAPTURE_DIR");
        if (!directory)
        {
            return;
        }
        std::filesystem::create_directories(directory);
        const auto path = std::filesystem::path(directory) /
                          ("coordinates-" + std::to_string(api) + "-" + std::to_string(variant) + ".ppm");
        std::ofstream file(path, std::ios::binary);
        REQUIRE(file.good());
        file << "P6\n64 64\n255\n";
        for (size_t i = 0; i < image.size(); i += 4)
        {
            file.write(reinterpret_cast<const char*>(image.data() + i), 3);
        }
        REQUIRE(file.good());
    }

    std::vector<uint8_t> Render(VriGraphicsAPI api, uint32_t variant)
    {
        uint32_t              errors = 0;
        VriCallbackInterface  callback {&errors, Diagnostic};
        VriDeviceCreationDesc creation  = vriDefaultDeviceCreationDesc();
        creation.graphicsAPI            = api;
        creation.enableValidation       = VRI_TRUE;
        creation.callbackInterface      = &callback;
        creation.coordinates.clipSpaceY = variant & 1 ? VriClipSpaceY_Up : VriClipSpaceY_Down;
        creation.coordinates.clipDepthRange =
            variant & 2 ? VriClipDepthRange_NegativeOneToOne : VriClipDepthRange_ZeroToOne;
        creation.coordinates.textureOrigin = variant & 4 ? VriTextureOrigin_BottomLeft : VriTextureOrigin_TopLeft;
        VriDevice*      device             = nullptr;
        const VriResult result             = vriCreateDevice(&creation, &device);
        if (result == VriResult_Unsupported)
        {
            MESSAGE("Coordinate rendering backend unavailable: ", api);
            return {};
        }
        REQUIRE(result == VriResult_Success);
        REQUIRE(vriTestCoordinatesC(device, sizeof(creation), sizeof(creation.coordinates)) == 0);
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriCoordinateTransform transform {};
        REQUIRE(vriGetCoordinateTransform(device, nullptr, &transform) == VriResult_Success);

        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(device, VriQueueType_Graphics, 0, &queue) == VriResult_Success);
        VriTextureDesc textureDesc {};
        textureDesc.type                    = VriTextureType_2D;
        textureDesc.format                  = VriFormat_RGBA8_UNORM;
        textureDesc.width                   = kSize;
        textureDesc.height                  = kSize;
        textureDesc.depth                   = 1;
        textureDesc.mipNum                  = 1;
        textureDesc.layerNum                = 1;
        textureDesc.sampleNum               = 1;
        textureDesc.usage                   = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
        textureDesc.clearValue.color.f32[3] = 1;
        VriTexture* color                   = nullptr;
        REQUIRE(c.CreateTexture(device, &textureDesc, &color) == VriResult_Success);
        VriTextureViewDesc viewDesc {};
        viewDesc.texture         = color;
        viewDesc.viewType        = VriTextureViewType_2D;
        viewDesc.aspect          = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr;
        REQUIRE(c.CreateTextureView(device, &viewDesc, &colorView) == VriResult_Success);

        textureDesc.format                        = VriFormat_D32_SFLOAT;
        textureDesc.usage                         = VriTextureUsage_DepthStencilAttachment;
        textureDesc.clearValue.depthStencil.depth = 1;
        VriTexture* depth                         = nullptr;
        REQUIRE(c.CreateTexture(device, &textureDesc, &depth) == VriResult_Success);
        viewDesc.texture         = depth;
        viewDesc.aspect          = VriImageAspect_Depth;
        VriDescriptor* depthView = nullptr;
        REQUIRE(c.CreateTextureView(device, &viewDesc, &depthView) == VriResult_Success);

        textureDesc.format  = VriFormat_RGBA8_UNORM;
        textureDesc.usage   = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst;
        VriTexture* sampled = nullptr;
        REQUIRE(c.CreateTexture(device, &textureDesc, &sampled) == VriResult_Success);
        viewDesc.texture           = sampled;
        viewDesc.aspect            = VriImageAspect_Color;
        VriDescriptor* sampledView = nullptr;
        REQUIRE(c.CreateTextureView(device, &viewDesc, &sampledView) == VriResult_Success);
        VriSamplerDesc samplerDesc {};
        samplerDesc.maxLod       = 1;
        samplerDesc.addressModeU = VriAddressMode_ClampToEdge;
        samplerDesc.addressModeV = VriAddressMode_ClampToEdge;
        samplerDesc.addressModeW = VriAddressMode_ClampToEdge;
        VriDescriptor* sampler   = nullptr;
        REQUIRE(c.CreateSampler(device, &samplerDesc, &sampler) == VriResult_Success);
        constexpr uint64_t bytes = kSize * kSize * 4;
        VriBufferDesc      bufferDesc {};
        bufferDesc.size           = bytes;
        bufferDesc.usage          = VriBufferUsage_TransferSrc;
        bufferDesc.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* staging        = nullptr;
        REQUIRE(c.CreateBuffer(device, &bufferDesc, &staging) == VriResult_Success);
        auto* upload = static_cast<Pixel*>(c.MapBuffer(staging, 0, bytes));
        REQUIRE(upload != nullptr);
        constexpr Pixel colors[4] {{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}, {255, 255, 0, 255}};
        for (uint32_t y = 0; y < kSize; ++y)
        {
            for (uint32_t x = 0; x < kSize; ++x)
            {
                upload[y * kSize + x] = colors[(y >= kSize / 2 ? 2 : 0) + (x >= kSize / 2 ? 1 : 0)];
            }
        }
        c.UnmapBuffer(staging);
        bufferDesc.usage          = VriBufferUsage_TransferDst;
        bufferDesc.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback       = nullptr;
        REQUIRE(c.CreateBuffer(device, &bufferDesc, &readback) == VriResult_Success);

        VriDescriptorRangeDesc ranges[2] {};
        ranges[0] = {0, 1, VriDescriptorType_Texture, VriShaderStage_Fragment, 0, VriTextureViewType_2D};
        ranges[1] = {1, 1, VriDescriptorType_Sampler, VriShaderStage_Fragment, 0, VriTextureViewType_2D};
        VriDescriptorSetDesc  setDesc {0, ranges, 2};
        VriPushConstantDesc   pushDesc {0, sizeof(Push), VriShaderStage_Vertex | VriShaderStage_Fragment};
        VriPipelineLayoutDesc layoutDesc {&setDesc, 1, &pushDesc, 1, VriShaderStage_Vertex | VriShaderStage_Fragment};
        VriPipelineLayout*    layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(device, &layoutDesc, &layout) == VriResult_Success);
        const void* shader =
            api == VriGraphicsAPI_WebGPU ? static_cast<const void*>(g_conventionsWgsl) : g_conventionsSpv;
        const size_t  shaderSize = api == VriGraphicsAPI_WebGPU ? sizeof(g_conventionsWgsl) : sizeof(g_conventionsSpv);
        VriShaderDesc shaders[2] {{VriShaderStage_Vertex, shader, shaderSize, "vertexMain"},
                                  {VriShaderStage_Fragment, shader, shaderSize, "fragmentMain"}};
#if defined(_WIN32)
        if (api == VriGraphicsAPI_D3D12)
        {
            shaders[0].bytecode     = g_conventionsDxilVS;
            shaders[0].bytecodeSize = sizeof(g_conventionsDxilVS);
            shaders[1].bytecode     = g_conventionsDxilPS;
            shaders[1].bytecodeSize = sizeof(g_conventionsDxilPS);
        }
#endif
        VriColorAttachmentDesc attachmentDesc {};
        attachmentDesc.format         = VriFormat_RGBA8_UNORM;
        attachmentDesc.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pipelineDesc {};
        pipelineDesc.pipelineLayout                  = layout;
        pipelineDesc.shaders                         = shaders;
        pipelineDesc.shaderNum                       = 2;
        pipelineDesc.inputAssembly.topology          = VriPrimitiveTopology_TriangleList;
        pipelineDesc.rasterization.cullMode          = VriCullMode_Back;
        pipelineDesc.rasterization.frontFace         = VriFrontFace_CounterClockwise;
        pipelineDesc.rasterization.lineWidth         = 1;
        pipelineDesc.multisample.sampleNum           = 1;
        pipelineDesc.depthStencil.depthTest          = VRI_TRUE;
        pipelineDesc.depthStencil.depthWrite         = VRI_TRUE;
        pipelineDesc.depthStencil.depthCompareOp     = VriCompareOp_Less;
        pipelineDesc.outputMerger.colors             = &attachmentDesc;
        pipelineDesc.outputMerger.colorNum           = 1;
        pipelineDesc.outputMerger.depthStencilFormat = VriFormat_D32_SFLOAT;
        VriPipeline* pipeline                        = nullptr;
        REQUIRE(c.CreateGraphicsPipeline(device, &pipelineDesc, &pipeline) == VriResult_Success);
        VriDescriptorPoolDesc poolDesc {};
        poolDesc.descriptorSetMaxNum = 1;
        poolDesc.textureMaxNum       = 1;
        poolDesc.samplerMaxNum       = 1;
        VriDescriptorPool* pool      = nullptr;
        REQUIRE(c.CreateDescriptorPool(device, &poolDesc, &pool) == VriResult_Success);
        VriDescriptorSet* set = nullptr;
        REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);
        const VriDescriptor*         descriptors[2] {sampledView, sampler};
        VriDescriptorRangeUpdateDesc updates[2] {{descriptors, 1, 0}, {descriptors + 1, 1, 0}};
        c.UpdateDescriptorRanges(set, 0, 2, updates);

        VriCommandAllocator* allocator = nullptr;
        REQUIRE(c.CreateCommandAllocator(device, VriQueueType_Graphics, &allocator) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(allocator, &cmd) == VriResult_Success);
        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        Transition(c,
                   cmd,
                   sampled,
                   VriLayout_Undefined,
                   VriLayout_CopyDestination,
                   VriAccess_CopyDestinationWrite,
                   VriPipelineStage_Transfer,
                   VriImageAspect_Color);
        VriBufferTextureCopyDesc copy {};
        copy.texture.aspect   = VriImageAspect_Color;
        copy.texture.layerNum = 1;
        c.CmdUploadBufferToTexture(cmd, sampled, staging, &copy);
        Transition(c,
                   cmd,
                   sampled,
                   VriLayout_CopyDestination,
                   VriLayout_ShaderResource,
                   VriAccess_ShaderResourceRead,
                   VriPipelineStage_FragmentShader,
                   VriImageAspect_Color);
        Transition(c,
                   cmd,
                   color,
                   VriLayout_Undefined,
                   VriLayout_ColorAttachment,
                   VriAccess_ColorAttachmentWrite,
                   VriPipelineStage_ColorAttachmentOutput,
                   VriImageAspect_Color);
        Transition(c,
                   cmd,
                   depth,
                   VriLayout_Undefined,
                   VriLayout_DepthStencilAttachment,
                   VriAccess_DepthStencilAttachmentWrite,
                   VriPipelineStage_EarlyFragmentTests,
                   VriImageAspect_Depth);
        VriAttachmentDesc colorAttachment {};
        colorAttachment.view                    = colorView;
        colorAttachment.loadOp                  = VriAttachmentLoadOp_Clear;
        colorAttachment.clearValue.color.f32[3] = 1;
        VriAttachmentDesc depthAttachment {};
        depthAttachment.view                          = depthView;
        depthAttachment.loadOp                        = VriAttachmentLoadOp_Clear;
        depthAttachment.clearValue.depthStencil.depth = 1;
        VriAttachmentsDesc attachments {};
        attachments.colors     = &colorAttachment;
        attachments.colorNum   = 1;
        attachments.depth      = &depthAttachment;
        attachments.renderArea = {0, 0, kSize, kSize};
        attachments.layerNum   = 1;
        c.CmdBeginRendering(cmd, &attachments);
        VriViewport viewport {0, 0, kSize, kSize, 0, 1};
        c.CmdSetViewports(cmd, &viewport, 1);
        VriRect scissor {0, 0, kSize, kSize};
        c.CmdSetScissors(cmd, &scissor, 1);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetPipeline(cmd, pipeline);
        c.CmdSetDescriptorSet(cmd, 0, set);
        const bool  negativeDepth = creation.coordinates.clipDepthRange == VriClipDepthRange_NegativeOneToOne;
        const bool  bottomUv      = creation.coordinates.textureOrigin == VriTextureOrigin_BottomLeft;
        Push        push {{transform.clipYScale, transform.clipZScale, transform.clipZBias, 0},
                          {creation.coordinates.clipSpaceY == VriClipSpaceY_Up ? 1.0f : -1.0f,
                    bottomUv ? -1.0f : 1.0f,
                    bottomUv ? 1.0f : 0.0f,
                    negativeDepth ? -0.6f : 0.2f},
                          {transform.uvYScale, transform.uvYBias, 0, 0},
                          {1, 1, 1, 1}};
        VriDrawDesc draw {3, 1, 0, 0};
        c.CmdSetConstants(cmd, 0, &push, sizeof(push));
        c.CmdDraw(cmd, &draw);
        // A farther black triangle must fail depth testing even when drawn last.
        push.authoring[3] = negativeDepth ? 0.6f : 0.8f;
        push.tint         = {0, 0, 0, 1};
        c.CmdSetConstants(cmd, 0, &push, sizeof(push));
        c.CmdDraw(cmd, &draw);
        c.CmdEndRendering(cmd);
        Transition(c,
                   cmd,
                   color,
                   VriLayout_ColorAttachment,
                   VriLayout_CopySource,
                   VriAccess_CopySourceRead,
                   VriPipelineStage_Transfer,
                   VriImageAspect_Color);
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &copy);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        c.QueueSubmit(queue, &submit);
        c.DeviceWaitIdle(device);
        const auto* pixels = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bytes));
        REQUIRE(pixels != nullptr);
        std::vector<uint8_t> image(pixels, pixels + bytes);
        c.UnmapBuffer(readback);
        c.DestroyCommandAllocator(allocator);
        c.DestroyDescriptorPool(pool);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyDescriptor(sampler);
        c.DestroyDescriptor(sampledView);
        c.DestroyDescriptor(colorView);
        c.DestroyDescriptor(depthView);
        c.DestroyTexture(sampled);
        c.DestroyTexture(color);
        c.DestroyTexture(depth);
        c.DestroyBuffer(staging);
        c.DestroyBuffer(readback);
        vriDestroyDevice(device);
        CHECK(errors == 0);
        Capture(api, variant, image);
        return image;
    }
} // namespace

TEST_CASE("device coordinate configuration stays independent across live contexts")
{
    // Configuration queries do not require either context to record GPU work.
    // OpenGL GPU calls still require its owning native context to be current.
    for (VriGraphicsAPI api : {VriGraphicsAPI_Vulkan,
                               VriGraphicsAPI_OpenGL,
                               VriGraphicsAPI_WebGPU,
                               VriGraphicsAPI_D3D12,
                               VriGraphicsAPI_Metal})
    {
        CAPTURE(api);
        VriDeviceCreationDesc creation = vriDefaultDeviceCreationDesc();
        creation.graphicsAPI           = api;
        creation.enableValidation      = VRI_TRUE;
        VriDevice*      device         = nullptr;
        const VriResult result         = vriCreateDevice(&creation, &device);
        if (result == VriResult_Unsupported)
        {
            MESSAGE("Coordinate configuration backend unavailable: ", api);
            continue;
        }
        REQUIRE(result == VriResult_Success);
        VriDeviceCreationDesc otherCreation  = creation;
        otherCreation.coordinates.clipSpaceY = VriClipSpaceY_Up;
        VriDevice* other                     = nullptr;
        REQUIRE(vriCreateDevice(&otherCreation, &other) == VriResult_Success);
        VriCoordinateConventions copy {};
        REQUIRE(vriGetCoordinateConventions(device, &copy) == VriResult_Success);
        CHECK(copy.clipSpaceY == VriClipSpaceY_Down);
        REQUIRE(vriGetCoordinateConventions(other, &copy) == VriResult_Success);
        CHECK(copy.clipSpaceY == VriClipSpaceY_Up);
        otherCreation.coordinates.clipSpaceY = VriClipSpaceY_Down;
        REQUIRE(vriGetCoordinateConventions(other, &copy) == VriResult_Success);
        CHECK(copy.clipSpaceY == VriClipSpaceY_Up);
        vriDestroyDevice(other);
        vriDestroyDevice(device);
    }
}

TEST_CASE("coordinate rendering preserves asymmetric texture depth and culling across conventions")
{
    std::vector<uint8_t> reference;
    for (VriGraphicsAPI api : {VriGraphicsAPI_Vulkan,
                               VriGraphicsAPI_OpenGL,
                               VriGraphicsAPI_WebGPU,
                               VriGraphicsAPI_D3D12,
                               VriGraphicsAPI_Metal})
    {
        for (uint32_t variant = 0; variant < 8; ++variant)
        {
            CAPTURE(api);
            CAPTURE(variant);
            const auto image = Render(api, variant);
            if (image.empty())
            {
                break;
            }
            auto pixel = [&](uint32_t x, uint32_t y) {
                const uint32_t offset = (y * kSize + x) * 4;
                return Pixel {image[offset], image[offset + 1], image[offset + 2], image[offset + 3]};
            };
            CHECK(pixel(20, 16) == Pixel {255, 0, 0, 255});
            CHECK(pixel(44, 16) == Pixel {0, 255, 0, 255});
            CHECK(pixel(28, 40) == Pixel {0, 0, 255, 255});
            CHECK(pixel(36, 40) == Pixel {255, 255, 0, 255});
            CHECK(pixel(4, 4) == Pixel {0, 0, 0, 255});
            if (reference.empty())
            {
                reference = image;
            }
            else
            {
                CHECK(image == reference);
            }
        }
    }
}
