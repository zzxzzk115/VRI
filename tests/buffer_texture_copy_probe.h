// Buffer -> texture -> buffer round trip with padded buffer rows and slices, shared by the desktop
// suite and the WebGL page. Every texel byte must come back where it went; padding is not compared.
#pragma once

#include <vri/vri.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace vri::test
{
    struct CopyCase
    {
        VriTextureType type;
        VriFormat      format;
        uint32_t       texelSize; // bytes
        uint32_t       width;
        uint32_t       height;
        uint32_t       depth; // 1 unless 3D
    };

    // Returns 0 on success, else the failing line.
    inline int CheckBufferTextureRoundTrip(VriDevice* device, const VriCoreInterface& c, const CopyCase& k)
    {
        // Rows padded by three texels and rounded up to the device's row alignment; slices padded by
        // two rows. Neither is the tightly packed layout.
        const uint32_t align     = c.GetDeviceDesc(device)->uploadBufferTextureRowAlignment;
        const uint32_t rowAlign  = align > k.texelSize ? align : k.texelSize;
        const uint32_t rowBytes  = ((k.width + 3) * k.texelSize + rowAlign - 1) / rowAlign * rowAlign;
        const uint32_t rowLength = rowBytes / k.texelSize;
        const uint32_t rows      = k.height + 2;
        const uint64_t bytes     = static_cast<uint64_t>(rowBytes) * rows * k.depth;

        VriQueue* queue = nullptr;
        if (c.GetQueue(device, VriQueueType_Graphics, 0, &queue) != VriResult_Success)
            return __LINE__;
        VriTextureDesc td {};
        td.type           = k.type;
        td.format         = k.format;
        td.width          = k.width;
        td.height         = k.height;
        td.depth          = k.depth;
        td.mipNum         = 1;
        td.layerNum       = 1;
        td.sampleNum      = 1;
        td.usage          = VriTextureUsage_TransferSrc | VriTextureUsage_TransferDst | VriTextureUsage_ShaderResource;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* texture = nullptr;
        if (c.CreateTexture(device, &td, &texture) != VriResult_Success)
            return __LINE__;
        VriBufferDesc ud {};
        ud.size              = bytes;
        ud.usage             = VriBufferUsage_TransferSrc;
        ud.memoryLocation    = VriMemoryLocation_HostUpload;
        VriBuffer*    upload = nullptr;
        VriBufferDesc rd {};
        rd.size             = bytes;
        rd.usage            = VriBufferUsage_TransferDst;
        rd.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        if (c.CreateBuffer(device, &ud, &upload) != VriResult_Success ||
            c.CreateBuffer(device, &rd, &readback) != VriResult_Success)
            return __LINE__;

        // Distinct bytes per texel byte; 0xEE in the padding.
        std::vector<uint8_t> source(static_cast<size_t>(bytes), 0xEE);
        const auto           at = [&](uint32_t x, uint32_t y, uint32_t z, uint32_t b) {
            return static_cast<size_t>(z) * rows * rowBytes + static_cast<size_t>(y) * rowBytes + x * k.texelSize + b;
        };
        for (uint32_t z = 0; z < k.depth; ++z)
            for (uint32_t y = 0; y < k.height; ++y)
                for (uint32_t x = 0; x < k.width; ++x)
                    for (uint32_t b = 0; b < k.texelSize; ++b)
                        // Half-float bytes stay below 0x7C in the high byte: no NaN/Inf to canonicalize.
                        source[at(x, y, z, b)] = static_cast<uint8_t>((x * 7 + y * 31 + z * 61 + b * 3) % 0x7B);
        void* mapped = c.MapBuffer(upload, 0, bytes);
        if (!mapped)
            return __LINE__;
        std::memcpy(mapped, source.data(), source.size());
        c.UnmapBuffer(upload);

        VriCommandAllocator* alloc = nullptr;
        VriCommandBuffer*    cmd   = nullptr;
        VriFence*            fence = nullptr;
        if (c.CreateCommandAllocator(device, VriQueueType_Graphics, &alloc) != VriResult_Success ||
            c.CreateCommandBuffer(alloc, &cmd) != VriResult_Success ||
            c.CreateFence(device, 0, &fence) != VriResult_Success)
            return __LINE__;
        const auto barrier = [&](VriAccessFlags ba, VriLayout bl, VriAccessFlags aa, VriLayout al) {
            VriTextureBarrierDesc tb {};
            tb.texture       = texture;
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
        VriBufferTextureCopyDesc copy {};
        copy.bufferRowLength   = rowLength;
        copy.bufferImageHeight = rows;
        copy.texture.aspect    = VriImageAspect_Color;
        copy.texture.layerNum  = 1;
        copy.texture.width     = k.width;
        copy.texture.height    = k.height;
        copy.texture.depth     = k.depth;

        c.BeginCommandBuffer(cmd);
        barrier(VriAccess_None, VriLayout_Undefined, VriAccess_CopyDestinationWrite, VriLayout_CopyDestination);
        c.CmdUploadBufferToTexture(cmd, texture, upload, &copy);
        barrier(
            VriAccess_CopyDestinationWrite, VriLayout_CopyDestination, VriAccess_CopySourceRead, VriLayout_CopySource);
        c.CmdReadbackTextureToBuffer(cmd, readback, texture, &copy);
        c.EndCommandBuffer(cmd);
        VriFenceSubmitDesc signal {};
        signal.fence = fence;
        signal.value = 1;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        if (c.QueueSubmit(queue, &submit) != VriResult_Success)
            return __LINE__;
        c.Wait(fence, 1);

        int            failure = 0;
        const uint8_t* result  = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bytes));
        if (!result)
            failure = __LINE__;
        for (uint32_t z = 0; z < k.depth && !failure; ++z)
            for (uint32_t y = 0; y < k.height && !failure; ++y)
                if (std::memcmp(result + at(0, y, z, 0), source.data() + at(0, y, z, 0), k.width * k.texelSize) != 0)
                    failure = __LINE__;
        if (result)
            c.UnmapBuffer(readback);

        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(upload);
        c.DestroyTexture(texture);
        return failure;
    }

    // An odd-width two-byte format (13-texel rows are 26 bytes, not a multiple of four) and a 3D texture.
    inline const CopyCase kCopyCases[] = {
        {VriTextureType_2D, VriFormat_R16_SFLOAT, 2, 13, 7, 1},
        {VriTextureType_3D, VriFormat_RGBA8_UNORM, 4, 5, 3, 4},
    };
} // namespace vri::test
