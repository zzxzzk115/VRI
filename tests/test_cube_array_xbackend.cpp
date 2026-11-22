// A CubeArray of two cubes holds twelve distinct layer-faces on every backend: each one is uploaded with its own
// value and read back. GL used to allocate a single cube for any CubeArray, so layer-faces 6 to 11 did not exist
// there. Desktop GL has cube map arrays (GL 4.0+), so it must pass; GLES and WebGL refuse them as Unsupported.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace
{
    constexpr uint32_t kSize       = 64; // 256-byte rows, the readback row pitch D3D12 and WebGPU require
    constexpr uint32_t kLayerFaces = 12; // two cubes
    constexpr uint64_t kFaceBytes  = uint64_t(kSize) * kSize * 4;

    uint8_t FaceValue(uint32_t layerFace) { return static_cast<uint8_t>(0x10 + layerFace * 0x11); }

    void Check(VriGraphicsAPI api, const char* name)
    {
        VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI           = api;
        dc.enableValidation      = VRI_TRUE;
        dc.bestEffort            = VRI_TRUE;
        VriDevice* dev           = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << std::string(name) << "] unavailable - skipped");
            return;
        }
        INFO(std::string(name));
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        VriTextureDesc td {};
        td.type           = VriTextureType_CubeArray;
        td.format         = VriFormat_RGBA8_UNORM;
        td.width          = kSize;
        td.height         = kSize;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = kLayerFaces;
        td.sampleNum      = 1;
        td.usage          = VriTextureUsage_ShaderResource | VriTextureUsage_TransferSrc | VriTextureUsage_TransferDst;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture*     tex = nullptr;
        const VriResult r   = c.CreateTexture(dev, &td, &tex);
        // Android's GL device is an ES context, so ask the device which API it runs.
        if (c.GetDeviceDesc(dev)->graphicsAPI == VriGraphicsAPI_OpenGLES && r == VriResult_Unsupported)
        {
            MESSAGE("[" << std::string(name) << "] no cube map arrays on GLES - skipped");
            vriDestroyDevice(dev);
            return;
        }
        REQUIRE(r == VriResult_Success);

        VriBufferDesc ud {};
        ud.size              = kFaceBytes * kLayerFaces;
        ud.usage             = VriBufferUsage_TransferSrc;
        ud.memoryLocation    = VriMemoryLocation_HostUpload;
        VriBuffer*    upload = nullptr;
        VriBufferDesc rd     = ud;
        rd.usage             = VriBufferUsage_TransferDst;
        rd.memoryLocation    = VriMemoryLocation_HostReadback;
        VriBuffer* readback  = nullptr;
        REQUIRE(c.CreateBuffer(dev, &ud, &upload) == VriResult_Success);
        REQUIRE(c.CreateBuffer(dev, &rd, &readback) == VriResult_Success);
        auto* mapped = static_cast<uint8_t*>(c.MapBuffer(upload, 0, ud.size));
        REQUIRE(mapped != nullptr);
        for (uint32_t f = 0; f < kLayerFaces; ++f)
            std::memset(mapped + kFaceBytes * f, FaceValue(f), static_cast<size_t>(kFaceBytes));
        c.UnmapBuffer(upload);

        VriCommandAllocator* alloc = nullptr;
        VriCommandBuffer*    cmd   = nullptr;
        VriFence*            fence = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);
        const auto barrier = [&](VriAccessFlags ba, VriLayout bl, VriAccessFlags aa, VriLayout al) {
            VriTextureBarrierDesc tb {};
            tb.texture       = tex;
            tb.before.access = ba;
            tb.before.layout = bl;
            tb.before.stages = ba == VriAccess_None ? VriPipelineStage_None : VriPipelineStage_Transfer;
            tb.after.access  = aa;
            tb.after.layout  = al;
            tb.after.stages  = VriPipelineStage_Transfer;
            tb.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        };
        const auto region = [](uint32_t layerFace) {
            VriBufferTextureCopyDesc copy {};
            copy.bufferOffset      = kFaceBytes * layerFace;
            copy.bufferRowLength   = kSize;
            copy.bufferImageHeight = kSize;
            copy.texture.aspect    = VriImageAspect_Color;
            copy.texture.baseLayer = layerFace;
            copy.texture.layerNum  = 1;
            copy.texture.width     = kSize;
            copy.texture.height    = kSize;
            copy.texture.depth     = 1;
            return copy;
        };

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        barrier(VriAccess_None, VriLayout_Undefined, VriAccess_CopyDestinationWrite, VriLayout_CopyDestination);
        for (uint32_t f = 0; f < kLayerFaces; ++f)
        {
            const VriBufferTextureCopyDesc copy = region(f);
            c.CmdUploadBufferToTexture(cmd, tex, upload, &copy);
        }
        barrier(
            VriAccess_CopyDestinationWrite, VriLayout_CopyDestination, VriAccess_CopySourceRead, VriLayout_CopySource);
        for (uint32_t f = 0; f < kLayerFaces; ++f)
        {
            const VriBufferTextureCopyDesc copy = region(f);
            c.CmdReadbackTextureToBuffer(cmd, readback, tex, &copy);
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
        REQUIRE(c.QueueSubmit(queue, &submit) == VriResult_Success);
        c.Wait(fence, 1);

        const auto* result = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rd.size));
        REQUIRE(result != nullptr);
        for (uint32_t f = 0; f < kLayerFaces; ++f)
        {
            CAPTURE(f);
            const uint8_t* face  = result + kFaceBytes * f;
            uint64_t       wrong = 0;
            for (uint64_t i = 0; i < kFaceBytes; ++i)
                wrong += face[i] != FaceValue(f);
            CHECK(wrong == 0);
        }
        c.UnmapBuffer(readback);

        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(upload);
        c.DestroyTexture(tex);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("CubeArray: two cubes hold twelve distinct layer-faces")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan");
    Check(VriGraphicsAPI_D3D12, "D3D12");
    Check(VriGraphicsAPI_OpenGL, "OpenGL");
    Check(VriGraphicsAPI_WebGPU, "WebGPU");
    Check(VriGraphicsAPI_Metal, "Metal");
}
