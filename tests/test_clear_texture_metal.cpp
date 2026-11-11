#include <doctest/doctest.h>
#include <vri/vri.h>

#include <array>
#include <cstring>
#include <vector>

TEST_CASE("Metal storage texture clear covers typed layers and preserves other mips")
{
    VriDeviceCreationDesc dc {};
    dc.graphicsAPI      = VriGraphicsAPI_Metal;
    dc.enableValidation = VRI_TRUE;
    VriDevice* dev      = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("Metal unavailable");
        return;
    }
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    REQUIRE(c.GetDeviceDesc(dev)->hasClearStorageTexture);
    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);
    VriCommandAllocator* allocator = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &allocator) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(allocator, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);
    VriBufferDesc bd {};
    bd.size           = 256 * 32;
    bd.usage          = VriBufferUsage_TransferSrc;
    bd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* upload = nullptr;
    REQUIRE(c.CreateBuffer(dev, &bd, &upload) == VriResult_Success);
    auto* input = c.MapBuffer(upload, 0, bd.size);
    REQUIRE(input);
    std::memset(input, 0x55, bd.size);
    c.UnmapBuffer(upload);
    bd.usage            = VriBufferUsage_TransferDst;
    bd.memoryLocation   = VriMemoryLocation_HostReadback;
    VriBuffer* readback = nullptr;
    REQUIRE(c.CreateBuffer(dev, &bd, &readback) == VriResult_Success);
    uint64_t         serial = 0;
    const std::array types {VriTextureType_1D,
                            VriTextureType_1DArray,
                            VriTextureType_2D,
                            VriTextureType_2DArray,
                            VriTextureType_3D,
                            VriTextureType_Cube,
                            VriTextureType_CubeArray};
    const std::array formats {
        VriFormat_R32_SFLOAT, VriFormat_R32_UINT, VriFormat_R32_SINT, VriFormat_RGBA8_UNORM, VriFormat_RGBA16_SFLOAT};
    for (auto type : types)
    {
        for (auto format : formats)
        {
            CAPTURE(type);
            CAPTURE(format);
            VriTextureDesc td {};
            td.type      = type;
            td.format    = format;
            td.width     = 16;
            td.height    = type == VriTextureType_1D || type == VriTextureType_1DArray ? 1 : 16;
            td.depth     = type == VriTextureType_3D ? 3 : 1;
            td.layerNum  = type == VriTextureType_Cube                                      ? 6 :
                           type == VriTextureType_CubeArray                                 ? 12 :
                           type == VriTextureType_1DArray || type == VriTextureType_2DArray ? 3 :
                                                                                              1;
            td.mipNum    = td.height == 1 ? 1 : 2;
            td.sampleNum = 1;
            td.usage =
                VriTextureUsage_ShaderResourceStorage | VriTextureUsage_TransferSrc | VriTextureUsage_TransferDst;
            td.memoryLocation   = VriMemoryLocation_Device;
            VriTexture* texture = nullptr;
            REQUIRE(c.CreateTexture(dev, &td, &texture) == VriResult_Success);
            const uint32_t         bytes = format == VriFormat_RGBA16_SFLOAT ? 8 : 4;
            VriClearColor          color {};
            std::array<uint8_t, 8> expected {};
            if (format == VriFormat_R32_UINT)
            {
                color.u32[0] = 0xf1234567u;
                std::memcpy(expected.data(), color.u32, bytes);
            }
            else if (format == VriFormat_R32_SINT)
            {
                color.i32[0] = -1234567;
                std::memcpy(expected.data(), color.i32, bytes);
            }
            else if (format == VriFormat_R32_SFLOAT)
            {
                color.f32[0] = -1.25f;
                std::memcpy(expected.data(), color.f32, bytes);
            }
            else if (format == VriFormat_RGBA8_UNORM)
            {
                color.f32[0] = .2f;
                color.f32[1] = .4f;
                color.f32[2] = .6f;
                color.f32[3] = 1;
                expected     = {51, 102, 153, 255};
            }
            else
            {
                color.f32[0] = .5f;
                color.f32[1] = -2;
                color.f32[2] = 1;
                color.f32[3] = 0;
                const std::array<uint16_t, 4> half {0x3800, 0xc000, 0x3c00, 0};
                std::memcpy(expected.data(), half.data(), bytes);
            }
            const uint32_t slices = td.type == VriTextureType_3D ? td.depth : td.layerNum;
            for (uint32_t slice = 0; slice < slices; ++slice)
            {
                c.ResetCommandAllocator(allocator);
                REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
                VriTextureBarrierDesc tb {};
                tb.texture = texture;
                tb.after   = {VriAccess_CopyDestinationWrite, VriLayout_CopyDestination, VriPipelineStage_Transfer};
                tb.aspect  = VriImageAspect_Color;
                VriBarrierGroupDesc barriers {};
                barriers.textures   = &tb;
                barriers.textureNum = 1;
                c.CmdBarrier(cmd, &barriers);
                // Set mip one on the first slice; clearing mip zero must leave it untouched.
                VriBufferTextureCopyDesc copy {};
                copy.bufferRowLength  = 256 / bytes;
                copy.texture.aspect   = VriImageAspect_Color;
                copy.texture.layerNum = 1;
                copy.texture.mip      = td.mipNum > 1 ? 1 : 0;
                copy.texture.width    = 8;
                copy.texture.height   = td.height == 1 ? 1 : 8;
                copy.texture.depth    = 1;
                if (td.mipNum > 1)
                    c.CmdUploadBufferToTexture(cmd, texture, upload, &copy);
                c.CmdClearStorageTexture(cmd, texture, &color);
                tb.before = tb.after;
                tb.after  = {VriAccess_CopySourceRead, VriLayout_CopySource, VriPipelineStage_Transfer};
                c.CmdBarrier(cmd, &barriers);
                if (td.mipNum > 1)
                    c.CmdReadbackTextureToBuffer(cmd, readback, texture, &copy);
                copy.bufferOffset      = 256 * 8;
                copy.texture.mip       = 0;
                copy.texture.width     = td.width;
                copy.texture.height    = td.height;
                copy.texture.baseLayer = td.type == VriTextureType_3D ? 0 : slice;
                copy.texture.z         = td.type == VriTextureType_3D ? slice : 0;
                c.CmdReadbackTextureToBuffer(cmd, readback, texture, &copy);
                REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
                VriFenceSubmitDesc signal {fence, ++serial};
                VriQueueSubmitDesc submit {};
                submit.commandBuffers   = &cmd;
                submit.commandBufferNum = 1;
                submit.signalFences     = &signal;
                submit.signalFenceNum   = 1;
                c.QueueSubmit(queue, &submit);
                c.Wait(fence, serial);
                const auto* result = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bd.size));
                REQUIRE(result);
                bool preserved = true, cleared = true;
                for (uint32_t y = 0; td.mipNum > 1 && y < 8u; ++y)
                    for (uint32_t x = 0; x < 8 * bytes; ++x)
                        preserved &= result[y * 256 + x] == 0x55;
                for (uint32_t y = 0; y < td.height; ++y)
                    for (uint32_t x = 0; x < td.width; ++x)
                        cleared &= std::memcmp(result + 256 * 8 + y * 256 + x * bytes, expected.data(), bytes) == 0;
                c.UnmapBuffer(readback);
                CHECK(preserved);
                CHECK(cleared);
            }
            c.DestroyTexture(texture);
        }
    }
    c.DeviceWaitIdle(dev);
    c.DestroyBuffer(upload);
    c.DestroyBuffer(readback);
    c.DestroyFence(fence);
    c.DestroyCommandAllocator(allocator);
    vriDestroyDevice(dev);
}
