// sRGB render targets on every backend:
// - rendering into an sRGB texture stores display-encoded bytes: a clear to linear 0.2158605 reads back as 128,
//   as a UNORM target cleared to 128/255 does. Desktop GL encodes only with GL_FRAMEBUFFER_SRGB on, which the
//   device enables;
// - copying an sRGB or a UNORM texture moves its bytes unchanged, every one of the 256 values in every channel.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace
{
    constexpr uint32_t kWidth  = 64; // 256-byte rows, the readback row pitch D3D12 requires
    constexpr uint32_t kHeight = 4;  // 64 x 4 texels x 4 channels: every byte value once per channel
    constexpr uint64_t kBytes  = static_cast<uint64_t>(kWidth) * kHeight * 4;

    struct Gpu
    {
        VriDevice*       dev = nullptr;
        VriCoreInterface c {};
        VriQueue*        queue = nullptr;
    };

    bool Open(VriGraphicsAPI api, const std::string& name, Gpu& gpu)
    {
        VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI           = api;
        if (vriCreateDevice(&dc, &gpu.dev) != VriResult_Success)
        {
            MESSAGE(name << " unavailable - skipping");
            return false;
        }
        REQUIRE(vriGetInterface(gpu.dev, VRI_INTERFACE_CORE, sizeof(gpu.c), &gpu.c) == VriResult_Success);
        REQUIRE(gpu.c.GetQueue(gpu.dev, VriQueueType_Graphics, 0, &gpu.queue) == VriResult_Success);
        return true;
    }

    // Records with `record`, submits and waits for completion.
    void Run(Gpu& gpu, const std::function<void(VriCommandBuffer*)>& record)
    {
        VriCommandAllocator* alloc = nullptr;
        REQUIRE(gpu.c.CreateCommandAllocator(gpu.dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(gpu.c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(gpu.c.CreateFence(gpu.dev, 0, &fence) == VriResult_Success);
        REQUIRE(gpu.c.BeginCommandBuffer(cmd) == VriResult_Success);
        record(cmd);
        REQUIRE(gpu.c.EndCommandBuffer(cmd) == VriResult_Success);
        VriFenceSubmitDesc signal {};
        signal.fence = fence;
        signal.value = 1;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        REQUIRE(gpu.c.QueueSubmit(gpu.queue, &submit) == VriResult_Success);
        gpu.c.Wait(fence, 1);
        gpu.c.DestroyFence(fence);
        gpu.c.DestroyCommandAllocator(alloc);
    }

    VriTexture* CreateTexture(Gpu& gpu, VriFormat format, VriTextureUsageFlags usage, float clear)
    {
        VriTextureDesc td {};
        td.type           = VriTextureType_2D;
        td.format         = format;
        td.width          = kWidth;
        td.height         = kHeight;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = 1;
        td.sampleNum      = 1;
        td.usage          = usage;
        td.memoryLocation = VriMemoryLocation_Device;
        for (float& v : td.clearValue.color.f32)
            v = clear;
        VriTexture* texture = nullptr;
        REQUIRE(gpu.c.CreateTexture(gpu.dev, &td, &texture) == VriResult_Success);
        return texture;
    }

    VriBuffer* CreateBuffer(Gpu& gpu, VriBufferUsageFlags usage, VriMemoryLocation location)
    {
        VriBufferDesc bd {};
        bd.size           = kBytes;
        bd.usage          = usage;
        bd.memoryLocation = location;
        VriBuffer* buffer = nullptr;
        REQUIRE(gpu.c.CreateBuffer(gpu.dev, &bd, &buffer) == VriResult_Success);
        return buffer;
    }

    // The stages and access that use a texture in `layout`; both sides of a barrier take them, so the earlier
    // write is made visible to the next operation.
    VriAccessLayoutStage Scope(VriLayout layout)
    {
        VriAccessLayoutStage scope {};
        scope.layout = layout;
        switch (layout)
        {
            case VriLayout_ColorAttachment:
                scope.access = VriAccess_ColorAttachmentWrite;
                scope.stages = VriPipelineStage_ColorAttachmentOutput;
                break;
            case VriLayout_CopySource:
                scope.access = VriAccess_CopySourceRead;
                scope.stages = VriPipelineStage_Transfer;
                break;
            case VriLayout_CopyDestination:
                scope.access = VriAccess_CopyDestinationWrite;
                scope.stages = VriPipelineStage_Transfer;
                break;
            default:
                break; // Undefined: nothing to wait for
        }
        return scope;
    }

    void Barrier(Gpu& gpu, VriCommandBuffer* cmd, VriTexture* texture, VriLayout before, VriLayout after)
    {
        VriTextureBarrierDesc tb {};
        tb.texture = texture;
        tb.before  = Scope(before);
        tb.after   = Scope(after);
        tb.aspect  = VriImageAspect_Color;
        VriBarrierGroupDesc g {};
        g.textures   = &tb;
        g.textureNum = 1;
        gpu.c.CmdBarrier(cmd, &g);
    }

    VriBufferTextureCopyDesc WholeTexture()
    {
        VriBufferTextureCopyDesc cp {};
        cp.texture.aspect   = VriImageAspect_Color;
        cp.texture.layerNum = 1;
        return cp;
    }

    std::vector<uint8_t> Read(Gpu& gpu, VriBuffer* readback)
    {
        const auto* px = static_cast<const uint8_t*>(gpu.c.MapBuffer(readback, 0, kBytes));
        REQUIRE(px != nullptr);
        std::vector<uint8_t> bytes(px, px + kBytes);
        gpu.c.UnmapBuffer(readback);
        return bytes;
    }

    // Clears a `format` target to `value` in every channel and returns its first texel.
    std::vector<uint8_t> ClearAndRead(Gpu& gpu, VriFormat format, float value)
    {
        VriTexture* texture =
            CreateTexture(gpu, format, VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc, value);
        VriTextureViewDesc vd {};
        vd.texture          = texture;
        vd.viewType         = VriTextureViewType_2D;
        vd.aspect           = VriImageAspect_Color;
        VriDescriptor* view = nullptr;
        REQUIRE(gpu.c.CreateTextureView(gpu.dev, &vd, &view) == VriResult_Success);
        VriBuffer* readback = CreateBuffer(gpu, VriBufferUsage_TransferDst, VriMemoryLocation_HostReadback);
        Run(gpu, [&](VriCommandBuffer* cmd) {
            Barrier(gpu, cmd, texture, VriLayout_Undefined, VriLayout_ColorAttachment);
            VriAttachmentDesc rt {};
            rt.view    = view;
            rt.loadOp  = VriAttachmentLoadOp_Clear;
            rt.storeOp = VriAttachmentStoreOp_Store;
            for (float& v : rt.clearValue.color.f32)
                v = value;
            VriAttachmentsDesc att {};
            att.colors            = &rt;
            att.colorNum          = 1;
            att.renderArea.width  = kWidth;
            att.renderArea.height = kHeight;
            att.layerNum          = 1;
            gpu.c.CmdBeginRendering(cmd, &att);
            gpu.c.CmdEndRendering(cmd);
            Barrier(gpu, cmd, texture, VriLayout_ColorAttachment, VriLayout_CopySource);
            const VriBufferTextureCopyDesc cp = WholeTexture();
            gpu.c.CmdReadbackTextureToBuffer(cmd, readback, texture, &cp);
        });
        std::vector<uint8_t> bytes = Read(gpu, readback);
        bytes.resize(4);
        gpu.c.DestroyDescriptor(view);
        gpu.c.DestroyBuffer(readback);
        gpu.c.DestroyTexture(texture);
        return bytes;
    }

    // Uploads every byte value to a `format` texture, copies it to a second one, reads that back.
    std::vector<uint8_t> CopyAndRead(Gpu& gpu, VriFormat format, const std::vector<uint8_t>& source)
    {
        constexpr VriTextureUsageFlags usage =
            VriTextureUsage_TransferSrc | VriTextureUsage_TransferDst | VriTextureUsage_ColorAttachment;
        VriTexture* src    = CreateTexture(gpu, format, usage, 0.0f);
        VriTexture* dst    = CreateTexture(gpu, format, usage, 0.0f);
        VriBuffer*  upload = CreateBuffer(gpu, VriBufferUsage_TransferSrc, VriMemoryLocation_HostUpload);
        void*       mapped = gpu.c.MapBuffer(upload, 0, kBytes);
        REQUIRE(mapped != nullptr);
        std::memcpy(mapped, source.data(), kBytes);
        gpu.c.UnmapBuffer(upload);
        VriBuffer* readback = CreateBuffer(gpu, VriBufferUsage_TransferDst, VriMemoryLocation_HostReadback);
        Run(gpu, [&](VriCommandBuffer* cmd) {
            const VriBufferTextureCopyDesc whole = WholeTexture();
            Barrier(gpu, cmd, src, VriLayout_Undefined, VriLayout_CopyDestination);
            gpu.c.CmdUploadBufferToTexture(cmd, src, upload, &whole);
            Barrier(gpu, cmd, src, VriLayout_CopyDestination, VriLayout_CopySource);
            Barrier(gpu, cmd, dst, VriLayout_Undefined, VriLayout_CopyDestination);
            VriTextureCopyDesc copy {};
            copy.src.aspect   = VriImageAspect_Color;
            copy.src.layerNum = 1;
            copy.dst          = copy.src;
            gpu.c.CmdCopyTexture(cmd, dst, src, &copy);
            Barrier(gpu, cmd, dst, VriLayout_CopyDestination, VriLayout_CopySource);
            gpu.c.CmdReadbackTextureToBuffer(cmd, readback, dst, &whole);
        });
        std::vector<uint8_t> bytes = Read(gpu, readback);
        gpu.c.DestroyBuffer(readback);
        gpu.c.DestroyBuffer(upload);
        gpu.c.DestroyTexture(dst);
        gpu.c.DestroyTexture(src);
        return bytes;
    }

    void Check(VriGraphicsAPI api, const std::string& name)
    {
        CAPTURE(name);
        Gpu gpu;
        if (!Open(api, name, gpu))
            return;

        // Linear 0.2158605 encodes to sRGB 128; alpha is stored linearly.
        const std::vector<uint8_t> srgb = ClearAndRead(gpu, VriFormat_RGBA8_SRGB, 0.2158605f);
        CHECK(srgb[0] == 128);
        CHECK(srgb[1] == 128);
        CHECK(srgb[2] == 128);
        CHECK(srgb[3] == 55);
        const std::vector<uint8_t> unorm = ClearAndRead(gpu, VriFormat_RGBA8_UNORM, 128.0f / 255.0f);
        CHECK(unorm[0] == 128);
        CHECK(unorm[3] == 128);

        std::vector<uint8_t> source(kBytes);
        for (uint64_t i = 0; i < kBytes; ++i)
            source[i] = static_cast<uint8_t>(i / 4 + i % 4 * 64); // each channel walks all 256 values
        for (const VriFormat format : {VriFormat_RGBA8_SRGB, VriFormat_RGBA8_UNORM})
        {
            CAPTURE(format);
            CHECK(CopyAndRead(gpu, format, source) == source);
        }
        vriDestroyDevice(gpu.dev);
    }
} // namespace

TEST_CASE("sRGB render targets store encoded bytes and copy unchanged on every backend")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_Metal, "Metal");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
}
