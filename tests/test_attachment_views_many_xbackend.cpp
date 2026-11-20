// Views of attachment-capable textures that are only sampled hold no attachment descriptor: far more
// of them than D3D12's RTV (4096) and DSV (1024) heaps hold can be alive at once, and a few of them
// then render. Each rendered view is cleared to its own color, read back, and checked.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <string>
#include <vector>

namespace
{
    constexpr uint32_t kViews = 5000; // more than either D3D12 attachment heap holds
    constexpr uint32_t kSize  = 4;

    struct Errors
    {
        int count = 0;
    };
    void VRI_CALL OnMessage(void* userArg, VriMessageSeverity severity, const char*)
    {
        if (severity == VriMessageSeverity_Error)
            ++static_cast<Errors*>(userArg)->count;
    }

    VriTexture* CreateTarget(const VriCoreInterface& c, VriDevice* dev, VriFormat format, VriTextureUsageFlags usage)
    {
        VriTextureDesc td {};
        td.type             = VriTextureType_2D;
        td.format           = format;
        td.width            = kSize;
        td.height           = kSize;
        td.depth            = 1;
        td.mipNum           = 1;
        td.layerNum         = 1;
        td.sampleNum        = 1;
        td.usage            = usage | VriTextureUsage_ShaderResource;
        td.memoryLocation   = VriMemoryLocation_Device;
        VriTexture* texture = nullptr;
        return c.CreateTexture(dev, &td, &texture) == VriResult_Success ? texture : nullptr;
    }

    void Check(VriGraphicsAPI api, const char* name)
    {
        Errors                errors;
        VriCallbackInterface  callbacks = {&errors, OnMessage};
        VriDeviceCreationDesc dc        = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI                  = api;
        dc.bestEffort                   = VRI_TRUE;
        dc.callbackInterface            = &callbacks;
        VriDevice* dev                  = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << name << "] unavailable - skipped");
            return;
        }
        INFO(std::string(name));
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        VriTexture* color =
            CreateTarget(c, dev, VriFormat_RGBA8_UNORM, VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc);
        VriTexture* depth = CreateTarget(c, dev, VriFormat_D32_SFLOAT, VriTextureUsage_DepthStencilAttachment);
        REQUIRE(color != nullptr);
        REQUIRE(depth != nullptr);

        std::vector<VriDescriptor*> colorViews(kViews, nullptr);
        std::vector<VriDescriptor*> depthViews(kViews, nullptr);
        for (uint32_t i = 0; i < kViews; ++i)
        {
            VriTextureViewDesc vd {};
            vd.texture  = color;
            vd.viewType = VriTextureViewType_2D;
            vd.aspect   = VriImageAspect_Color;
            REQUIRE(c.CreateTextureView(dev, &vd, &colorViews[i]) == VriResult_Success);
            vd.texture = depth;
            vd.aspect  = VriImageAspect_Depth;
            REQUIRE(c.CreateTextureView(dev, &vd, &depthViews[i]) == VriResult_Success);
        }

        VriBufferDesc rd {};
        rd.size             = 256 * kSize; // rows padded to 256 bytes for every backend
        rd.usage            = VriBufferUsage_TransferDst;
        rd.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rd, &readback) == VriResult_Success);
        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        const auto barrier =
            [&](VriTexture* texture, VriAccessFlags ba, VriLayout bl, VriAccessFlags aa, VriLayout al) {
                VriTextureBarrierDesc tb {};
                tb.texture       = texture;
                tb.before.access = ba;
                tb.before.layout = bl;
                tb.before.stages = VriPipelineStage_AllCommands;
                tb.after.access  = aa;
                tb.after.layout  = al;
                tb.after.stages  = VriPipelineStage_AllCommands;
                tb.aspect        = texture == depth ? VriImageAspect_Depth : VriImageAspect_Color;
                VriBarrierGroupDesc g {};
                g.textures   = &tb;
                g.textureNum = 1;
                c.CmdBarrier(cmd, &g);
            };

        // Render through three far-apart views, each to its own clear color, and read each back.
        const uint32_t picks[]     = {0, kViews / 2, kViews - 1};
        uint64_t       signalled   = 0;
        VriLayout      layout      = VriLayout_Undefined;
        VriLayout      depthLayout = VriLayout_Undefined;
        for (uint32_t n = 0; n < 3; ++n)
        {
            const uint32_t pick = picks[n];
            const float    red  = 0.25f * static_cast<float>(n + 1);
            c.ResetCommandAllocator(alloc);
            REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
            barrier(color, VriAccess_None, layout, VriAccess_ColorAttachmentWrite, VriLayout_ColorAttachment);
            barrier(depth,
                    VriAccess_None,
                    depthLayout,
                    VriAccess_DepthStencilAttachmentWrite,
                    VriLayout_DepthStencilAttachment);
            VriAttachmentDesc colorAttachment {};
            colorAttachment.view                    = colorViews[pick];
            colorAttachment.loadOp                  = VriAttachmentLoadOp_Clear;
            colorAttachment.storeOp                 = VriAttachmentStoreOp_Store;
            colorAttachment.clearValue.color.f32[0] = red;
            colorAttachment.clearValue.color.f32[3] = 1.0f;
            VriAttachmentDesc depthAttachment {};
            depthAttachment.view                          = depthViews[pick];
            depthAttachment.loadOp                        = VriAttachmentLoadOp_Clear;
            depthAttachment.storeOp                       = VriAttachmentStoreOp_Store;
            depthAttachment.clearValue.depthStencil.depth = 1.0f;
            VriAttachmentsDesc attachments {};
            attachments.colors     = &colorAttachment;
            attachments.colorNum   = 1;
            attachments.depth      = &depthAttachment;
            attachments.renderArea = {0, 0, kSize, kSize};
            attachments.layerNum   = 1;
            c.CmdBeginRendering(cmd, &attachments);
            c.CmdEndRendering(cmd);
            barrier(color,
                    VriAccess_ColorAttachmentWrite,
                    VriLayout_ColorAttachment,
                    VriAccess_CopySourceRead,
                    VriLayout_CopySource);
            layout      = VriLayout_CopySource;
            depthLayout = VriLayout_DepthStencilAttachment;
            VriBufferTextureCopyDesc copy {};
            copy.bufferRowLength  = 256 / 4;
            copy.texture.aspect   = VriImageAspect_Color;
            copy.texture.layerNum = 1;
            c.CmdReadbackTextureToBuffer(cmd, readback, color, &copy);
            REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
            VriFenceSubmitDesc signal {};
            signal.fence = fence;
            signal.value = ++signalled;
            VriQueueSubmitDesc submit {};
            submit.commandBuffers   = &cmd;
            submit.commandBufferNum = 1;
            submit.signalFences     = &signal;
            submit.signalFenceNum   = 1;
            REQUIRE(c.QueueSubmit(queue, &submit) == VriResult_Success);
            c.Wait(fence, signalled);

            const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rd.size));
            REQUIRE(px != nullptr);
            const int expected = static_cast<int>(red * 255.0f + 0.5f);
            INFO("view " << pick);
            CHECK(px[0] >= expected - 1);
            CHECK(px[0] <= expected + 1);
            c.UnmapBuffer(readback);
        }
        CHECK(errors.count == 0);

        c.DeviceWaitIdle(dev);
        for (uint32_t i = 0; i < kViews; ++i)
        {
            c.DestroyDescriptor(colorViews[i]);
            c.DestroyDescriptor(depthViews[i]);
        }
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        c.DestroyTexture(depth);
        c.DestroyTexture(color);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("Attachment views: many sampled views, a few of them rendered")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
    Check(VriGraphicsAPI_Metal, "Metal");
}
