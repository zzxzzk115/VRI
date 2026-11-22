// Compressed textures on WebGPU upload and read back in blocks: rows and pitches count 4x4 blocks, for every layer
// of an array. Stored blocks are compared byte for byte, so the test does not depend on an encoder or on sampling.
// A 64-texel row is 256 bytes of BC7 and copies directly; a 32-texel row is 128 bytes, so the upload repacks it
// and the readback names a 256-byte pitch through bufferRowLength.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>

TEST_CASE("WebGPU: BC7 array layers upload and read back in blocks")
{
    std::atomic<unsigned> errors {0};
    VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
    dc.graphicsAPI           = VriGraphicsAPI_WebGPU;
    dc.enableValidation      = VRI_TRUE;
    VriCallbackInterface callbacks {};
    callbacks.userArg         = &errors;
    callbacks.MessageCallback = [](void* user, VriMessageSeverity severity, const char* message) {
        if (severity == VriMessageSeverity_Error)
        {
            ++*static_cast<std::atomic<unsigned>*>(user);
            std::fprintf(stderr, "%s\n", message);
        }
    };
    dc.callbackInterface = &callbacks;
    VriDevice* dev       = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("WebGPU unavailable - skipped");
        return;
    }
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    if ((c.GetFormatSupport(dev, VriFormat_BC7_UNORM) & VriFormatSupport_Texture) == 0)
    {
        MESSAGE("WebGPU adapter without texture-compression-bc - skipped");
        vriDestroyDevice(dev);
        return;
    }
    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    constexpr unsigned kBlockBytes = 16;
    constexpr unsigned kHeight     = 8; // two block rows
    constexpr unsigned kLayers     = 2;
    constexpr unsigned kPitch      = 256;
    for (unsigned width : {64u, 32u})
    {
        INFO("width = ", width);
        VriTextureDesc td {};
        td.type           = VriTextureType_2DArray;
        td.format         = VriFormat_BC7_UNORM;
        td.width          = width;
        td.height         = kHeight;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = kLayers;
        td.sampleNum      = 1;
        td.usage          = VriTextureUsage_ShaderResource | VriTextureUsage_TransferSrc | VriTextureUsage_TransferDst;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* texture = nullptr;
        REQUIRE(c.CreateTexture(dev, &td, &texture) == VriResult_Success);

        const unsigned             rowBytes  = width / 4 * kBlockBytes;
        const unsigned             blockRows = kHeight / 4;
        std::vector<unsigned char> source(kLayers * blockRows * rowBytes);
        for (size_t i = 0; i < source.size(); ++i)
            source[i] = static_cast<unsigned char>((i * 37 + width) & 255);

        VriBufferDesc bd {};
        bd.size           = source.size();
        bd.usage          = VriBufferUsage_TransferSrc;
        bd.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* upload = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &upload) == VriResult_Success);
        void* mapped = c.MapBuffer(upload, 0, bd.size);
        REQUIRE(mapped != nullptr);
        std::memcpy(mapped, source.data(), source.size());
        c.UnmapBuffer(upload);
        bd.size             = kLayers * blockRows * kPitch;
        bd.usage            = VriBufferUsage_TransferDst;
        bd.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &readback) == VriResult_Success);

        VriCommandAllocator* allocator = nullptr;
        VriCommandBuffer*    cmd       = nullptr;
        VriFence*            fence     = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &allocator) == VriResult_Success);
        REQUIRE(c.CreateCommandBuffer(allocator, &cmd) == VriResult_Success);
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);
        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        VriBufferTextureCopyDesc copy {};
        copy.texture.aspect   = VriImageAspect_Color;
        copy.texture.layerNum = kLayers;
        copy.texture.width    = width;
        copy.texture.height   = kHeight;
        c.CmdUploadBufferToTexture(cmd, texture, upload, &copy);
        copy.bufferRowLength = kPitch / kBlockBytes * 4;
        c.CmdReadbackTextureToBuffer(cmd, readback, texture, &copy);
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

        const auto* result = static_cast<const unsigned char*>(c.MapBuffer(readback, 0, bd.size));
        REQUIRE(result != nullptr);
        for (unsigned row = 0; row < kLayers * blockRows; ++row)
            CHECK(std::memcmp(result + row * kPitch, source.data() + row * rowBytes, rowBytes) == 0);
        c.UnmapBuffer(readback);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(allocator);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(upload);
        c.DestroyTexture(texture);
    }
    CHECK(errors.load() == 0);
    vriDestroyDevice(dev);
}
