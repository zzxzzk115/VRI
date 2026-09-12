// BC uploads must retain the compressed DXGI format, including tiny mip levels
// which need a row-pitch/placement-aligned bounce buffer. Compare stored blocks,
// so this test does not depend on encoder or sampling precision.
#include <doctest/doctest.h>
#include <vri/vri.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>

// CI can enable the D3D12 option on non-Windows hosts, where no backend is built.
// Keep device creation mandatory on Windows so missing coverage cannot pass silently.
#ifdef _WIN32
TEST_CASE("D3D12: BC formats round-trip aligned and tiny mip uploads")
#else
TEST_CASE("D3D12: BC formats round-trip aligned and tiny mip uploads" * doctest::skip())
#endif
{
    std::atomic<unsigned> errors {0};
    VriDeviceCreationDesc dc {};
    dc.graphicsAPI      = VriGraphicsAPI_D3D12;
    dc.enableValidation = VRI_TRUE;
    dc.bestEffort       = VRI_TRUE;
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
    VriDevice* device    = nullptr;
    REQUIRE(vriCreateDevice(&dc, &device) == VriResult_Success);
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(device, VriQueueType_Graphics, 0, &queue) == VriResult_Success);
    for (VriFormat format : {VriFormat_BC1_UNORM,
                             VriFormat_BC2_UNORM,
                             VriFormat_BC3_UNORM,
                             VriFormat_BC4_UNORM,
                             VriFormat_BC5_UNORM,
                             VriFormat_BC6H_UFLOAT,
                             VriFormat_BC7_UNORM})
    {
        INFO("format = ", static_cast<int>(format));
        REQUIRE((c.GetFormatSupport(device, format) & VriFormatSupport_Texture) != 0);
        const unsigned blockBytes = (format == VriFormat_BC1_UNORM || format == VriFormat_BC4_UNORM) ? 8 : 16;
        VriTextureDesc td {};
        td.type           = VriTextureType_2D;
        td.format         = format;
        td.width          = 128;
        td.height         = 128;
        td.depth          = 1;
        td.mipNum         = 8;
        td.layerNum       = 3;
        td.sampleNum      = 1;
        td.usage          = VriTextureUsage_ShaderResource | VriTextureUsage_TransferSrc | VriTextureUsage_TransferDst;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* texture = nullptr;
        REQUIRE(c.CreateTexture(device, &td, &texture) == VriResult_Success);
        for (unsigned mip : {0u, 5u, 7u})
        {
            INFO("mip = ", mip);
            const unsigned             width         = td.width >> mip;
            const unsigned             rows          = (width + 3) / 4;
            const unsigned             rowBytes      = rows * blockBytes;
            const unsigned             sourceOffset  = mip ? blockBytes : 0;
            const unsigned             rowPitch      = (rowBytes + 255) & ~255u;
            const unsigned             readbackSlice = (rows * rowPitch + 511) & ~511u;
            std::vector<unsigned char> source(sourceOffset + 2 * rows * rowBytes);
            for (size_t i = 0; i < source.size(); ++i)
                source[i] = static_cast<unsigned char>((i * 37 + mip * 13) & 255);
            VriBufferDesc bd {};
            bd.size           = source.size();
            bd.usage          = VriBufferUsage_TransferSrc;
            bd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* upload = nullptr;
            REQUIRE(c.CreateBuffer(device, &bd, &upload) == VriResult_Success);
            void* mapped = c.MapBuffer(upload, 0, bd.size);
            REQUIRE(mapped != nullptr);
            std::memcpy(mapped, source.data(), source.size());
            c.UnmapBuffer(upload);
            bd.size             = 2 * readbackSlice;
            bd.usage            = VriBufferUsage_TransferDst;
            bd.memoryLocation   = VriMemoryLocation_HostReadback;
            VriBuffer* readback = nullptr;
            REQUIRE(c.CreateBuffer(device, &bd, &readback) == VriResult_Success);
            VriCommandAllocator* allocator = nullptr;
            VriCommandBuffer*    cmd       = nullptr;
            VriFence*            fence     = nullptr;
            REQUIRE(c.CreateCommandAllocator(device, VriQueueType_Graphics, &allocator) == VriResult_Success);
            REQUIRE(c.CreateCommandBuffer(allocator, &cmd) == VriResult_Success);
            REQUIRE(c.CreateFence(device, 0, &fence) == VriResult_Success);
            REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
            VriBufferTextureCopyDesc copy {};
            copy.bufferOffset      = sourceOffset;
            copy.texture.aspect    = VriImageAspect_Color;
            copy.texture.mip       = mip;
            copy.texture.baseLayer = 1;
            copy.texture.layerNum  = 2;
            copy.texture.width     = width;
            copy.texture.height    = width;
            c.CmdUploadBufferToTexture(cmd, texture, upload, &copy);
            std::vector<VriBuffer*> patches;
            if (mip == 0)
            {
                // Patch only three blocks per row, leaving sentinel blocks around the region.
                // Exercise tight, padded, and directly usable D3D12 source pitches.
                for (unsigned mode : {0u, 1u, 2u})
                {
                    const unsigned             patchPitch  = mode == 2 ? 256 : (mode == 1 ? 5 : 3) * blockBytes;
                    const unsigned             patchOffset = mode == 2 ? 512 : blockBytes;
                    const unsigned             patchRows   = mode == 1 ? 3 : 2;
                    std::vector<unsigned char> patch(patchOffset + (patchRows + 1) * patchPitch + 3 * blockBytes, 0xe7);
                    for (unsigned layer = 0; layer < 2; ++layer)
                        for (unsigned row = 0; row < 2; ++row)
                            for (unsigned byte = 0; byte < 3 * blockBytes; ++byte)
                                patch[patchOffset + (layer * patchRows + row) * patchPitch + byte] =
                                    static_cast<unsigned char>(layer * 17 + mode * 61 + row * 23 + byte);
                    VriBufferDesc patchDesc {};
                    patchDesc.size           = patch.size();
                    patchDesc.usage          = VriBufferUsage_TransferSrc;
                    patchDesc.memoryLocation = VriMemoryLocation_HostUpload;
                    VriBuffer* patchBuffer   = nullptr;
                    REQUIRE(c.CreateBuffer(device, &patchDesc, &patchBuffer) == VriResult_Success);
                    void* patchData = c.MapBuffer(patchBuffer, 0, patchDesc.size);
                    REQUIRE(patchData != nullptr);
                    std::memcpy(patchData, patch.data(), patch.size());
                    c.UnmapBuffer(patchBuffer);
                    patches.push_back(patchBuffer);
                    VriBufferTextureCopyDesc patchCopy {};
                    patchCopy.bufferOffset      = patchOffset;
                    patchCopy.bufferRowLength   = mode == 0 ? 0 : patchPitch / blockBytes * 4;
                    patchCopy.bufferImageHeight = mode == 1 ? 12 : 0;
                    patchCopy.texture.aspect    = VriImageAspect_Color;
                    patchCopy.texture.baseLayer = 1;
                    patchCopy.texture.layerNum  = 2;
                    patchCopy.texture.x         = 4 + mode * 16;
                    patchCopy.texture.y         = 8;
                    patchCopy.texture.width     = 12;
                    patchCopy.texture.height    = 8;
                    c.CmdUploadBufferToTexture(cmd, texture, patchBuffer, &patchCopy);
                    for (unsigned layer = 0; layer < 2; ++layer)
                        for (unsigned row = 0; row < 2; ++row)
                            std::memcpy(source.data() + (layer * rows + row + 2) * rowBytes +
                                            (1 + mode * 4) * blockBytes,
                                        patch.data() + patchOffset + (layer * patchRows + row) * patchPitch,
                                        3 * blockBytes);
                }
            }
            copy.texture.layerNum = 1;
            for (unsigned layer = 0; layer < 2; ++layer)
            {
                copy.texture.baseLayer = layer + 1;
                copy.bufferOffset      = layer * readbackSlice;
                c.CmdReadbackTextureToBuffer(cmd, readback, texture, &copy);
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
            const auto* result = static_cast<const unsigned char*>(c.MapBuffer(readback, 0, bd.size));
            REQUIRE(result != nullptr);
            for (unsigned layer = 0; layer < 2; ++layer)
                for (unsigned row = 0; row < rows; ++row)
                    CHECK(std::memcmp(result + layer * readbackSlice + row * rowPitch,
                                      source.data() + sourceOffset + (layer * rows + row) * rowBytes,
                                      rowBytes) == 0);
            c.UnmapBuffer(readback);
            c.DestroyFence(fence);
            c.DestroyCommandAllocator(allocator);
            c.DestroyBuffer(readback);
            c.DestroyBuffer(upload);
            for (VriBuffer* patch : patches)
                c.DestroyBuffer(patch);
        }
        c.DestroyTexture(texture);
    }
    CHECK(errors.load() == 0);
    vriDestroyDevice(device);
}
