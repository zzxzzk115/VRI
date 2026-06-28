// Clear storage texture (VriCoreInterface::CmdClearStorageTexture): clear a storage texture to a
// color outside a render pass, then read it back and confirm the pixels hold that color. Gated on
// VriDeviceDesc::hasClearStorageTexture (Vulkan + D3D12 + desktop OpenGL 4.4; WebGPU/Metal can't
// clear a texture outside a pass, so the test self-skips there). Like the buffer clear it is a
// transfer-domain op (barrier the texture to CopyDestination, then CopySource).
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

namespace
{
    constexpr uint32_t kW = 64; // 64*4 = 256-byte rows (D3D12 readback row-pitch alignment)
    constexpr uint32_t kH = 64;

    void Check(VriGraphicsAPI api, const char* name)
    {
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        VriDevice* dev      = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << name << "] unavailable - skipped");
            return;
        }
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        if (c.GetDeviceDesc(dev)->hasClearStorageTexture == VRI_FALSE)
        {
            MESSAGE("[" << name << "] storage-texture clear unsupported - skipped");
            vriDestroyDevice(dev);
            return;
        }
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        VriTextureDesc td {};
        td.type      = VriTextureType_2D;
        td.format    = VriFormat_RGBA8_UNORM;
        td.width     = kW;
        td.height    = kH;
        td.depth     = 1;
        td.mipNum    = 1;
        td.layerNum  = 1;
        td.sampleNum = 1;
        td.usage = VriTextureUsage_ShaderResourceStorage | VriTextureUsage_TransferSrc | VriTextureUsage_TransferDst;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* tex   = nullptr;
        REQUIRE(c.CreateTexture(dev, &td, &tex) == VriResult_Success);

        VriBufferDesc rbd {};
        rbd.size            = static_cast<uint64_t>(kW) * kH * 4;
        rbd.usage           = VriBufferUsage_TransferDst;
        rbd.memoryLocation  = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        auto barrier = [&](VriAccessFlags        beforeA,
                           VriLayout             beforeL,
                           VriPipelineStageFlags beforeS,
                           VriAccessFlags        afterA,
                           VriLayout             afterL,
                           VriPipelineStageFlags afterS) {
            VriTextureBarrierDesc tb {};
            tb.texture       = tex;
            tb.before.access = beforeA;
            tb.before.layout = beforeL;
            tb.before.stages = beforeS;
            tb.after.access  = afterA;
            tb.after.layout  = afterL;
            tb.after.stages  = afterS;
            tb.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        };

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        barrier(VriAccess_None,
                VriLayout_Undefined,
                VriPipelineStage_None,
                VriAccess_CopyDestinationWrite,
                VriLayout_CopyDestination,
                VriPipelineStage_Transfer);
        VriClearColor red {};
        red.f32[0] = 1.0f;
        red.f32[3] = 1.0f; // opaque red (float/normalized -> f32 member)
        c.CmdClearStorageTexture(cmd, tex, &red);
        barrier(VriAccess_CopyDestinationWrite,
                VriLayout_CopyDestination,
                VriPipelineStage_Transfer,
                VriAccess_CopySourceRead,
                VriLayout_CopySource,
                VriPipelineStage_Transfer);
        VriBufferTextureCopyDesc rc {};
        rc.texture.aspect   = VriImageAspect_Color;
        rc.texture.layerNum = 1;
        rc.texture.width    = kW;
        rc.texture.height   = kH;
        c.CmdReadbackTextureToBuffer(cmd, readback, tex, &rc);
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

        const uint8_t* px    = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
        const uint8_t* p     = px + (static_cast<size_t>(kH / 2) * kW + kW / 2) * 4;
        const bool     isRed = p[0] > 240 && p[1] < 16 && p[2] < 16 && p[3] > 240;
        c.UnmapBuffer(readback);
        CHECK(isRed);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        c.DestroyTexture(tex);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("Clear storage texture: clear to a color")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
}
