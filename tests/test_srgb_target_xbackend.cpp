// Rendering into an sRGB texture stores display-encoded bytes on every backend: a clear to linear 0.5 reads back
// as 188 (0.7354), alpha as 128. Desktop GL encodes only with GL_FRAMEBUFFER_SRGB on, which the device enables.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <string>

namespace
{
    constexpr uint32_t kWidth = 64; // 256-byte rows, the readback row pitch D3D12 requires

    void CheckSrgbClear(VriGraphicsAPI api, const std::string& name)
    {
        CAPTURE(name);
        VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI           = api;
        VriDevice* dev           = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE(name << " unavailable - skipping");
            return;
        }
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        VriTextureDesc td {};
        td.type                    = VriTextureType_2D;
        td.format                  = VriFormat_RGBA8_SRGB;
        td.width                   = kWidth;
        td.height                  = 1;
        td.depth                   = 1;
        td.mipNum                  = 1;
        td.layerNum                = 1;
        td.sampleNum               = 1;
        td.usage                   = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
        td.memoryLocation          = VriMemoryLocation_Device;
        td.clearValue.color.f32[0] = 0.5f;
        td.clearValue.color.f32[1] = 0.5f;
        td.clearValue.color.f32[2] = 0.5f;
        td.clearValue.color.f32[3] = 0.5f;
        VriTexture* texture        = nullptr;
        REQUIRE(c.CreateTexture(dev, &td, &texture) == VriResult_Success);
        VriTextureViewDesc vd {};
        vd.texture          = texture;
        vd.viewType         = VriTextureViewType_2D;
        vd.aspect           = VriImageAspect_Color;
        VriDescriptor* view = nullptr;
        REQUIRE(c.CreateTextureView(dev, &vd, &view) == VriResult_Success);
        VriBufferDesc bd {};
        bd.size             = kWidth * 4;
        bd.usage            = VriBufferUsage_TransferDst;
        bd.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &readback) == VriResult_Success);
        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        VriTextureBarrierDesc tb {};
        tb.texture       = texture;
        tb.before.layout = VriLayout_Undefined;
        tb.after.layout  = VriLayout_ColorAttachment;
        tb.after.access  = VriAccess_ColorAttachmentWrite;
        tb.after.stages  = VriPipelineStage_ColorAttachmentOutput;
        tb.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc g {};
        g.textures   = &tb;
        g.textureNum = 1;
        c.CmdBarrier(cmd, &g);
        VriAttachmentDesc rt {};
        rt.view       = view;
        rt.loadOp     = VriAttachmentLoadOp_Clear;
        rt.storeOp    = VriAttachmentStoreOp_Store;
        rt.clearValue = td.clearValue;
        VriAttachmentsDesc att {};
        att.colors            = &rt;
        att.colorNum          = 1;
        att.renderArea.width  = kWidth;
        att.renderArea.height = 1;
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        c.CmdEndRendering(cmd);
        tb.before       = tb.after;
        tb.after.layout = VriLayout_CopySource;
        tb.after.access = VriAccess_CopySourceRead;
        tb.after.stages = VriPipelineStage_Transfer;
        c.CmdBarrier(cmd, &g);
        VriBufferTextureCopyDesc cp {};
        cp.texture.aspect   = VriImageAspect_Color;
        cp.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, texture, &cp);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
        VriFenceSubmitDesc signal {};
        signal.fence = fence;
        signal.value = 1;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        REQUIRE(c.QueueSubmit(queue, &submit) == VriResult_Success);
        c.Wait(fence, 1);

        const auto* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bd.size));
        REQUIRE(px != nullptr);
        CHECK(px[0] >= 187);
        CHECK(px[0] <= 189);
        CHECK(px[3] >= 127);
        CHECK(px[3] <= 128);
        c.UnmapBuffer(readback);

        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyDescriptor(view);
        c.DestroyBuffer(readback);
        c.DestroyTexture(texture);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("sRGB render targets store encoded bytes on every backend")
{
    CheckSrgbClear(VriGraphicsAPI_Vulkan, "Vulkan");
    CheckSrgbClear(VriGraphicsAPI_D3D12, "D3D12");
    CheckSrgbClear(VriGraphicsAPI_Metal, "Metal");
    CheckSrgbClear(VriGraphicsAPI_WebGPU, "WebGPU");
    CheckSrgbClear(VriGraphicsAPI_OpenGL, "OpenGL");
}
