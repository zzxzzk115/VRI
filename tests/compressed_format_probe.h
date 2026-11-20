// Upload/readback round trip of a block-compressed format, shared by the desktop suite and the
// WebGPU page: an 8x8 texture (2x2 blocks) goes up from 256-byte buffer rows and comes back into the
// same layout; the block bytes must match exactly.
#pragma once

#include <vri/vri.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace vri::test
{
    struct CompressedFormat
    {
        VriFormat   format;
        uint32_t    blockBytes;
        const char* name;
    };

    // The formats appended after ASTC_4x4_UNORM.
    inline const CompressedFormat kAddedCompressedFormats[] = {
        {VriFormat_BC1_SRGB, 8, "BC1_SRGB"},
        {VriFormat_BC2_SRGB, 16, "BC2_SRGB"},
        {VriFormat_BC3_SRGB, 16, "BC3_SRGB"},
        {VriFormat_BC7_SRGB, 16, "BC7_SRGB"},
        {VriFormat_BC6H_SFLOAT, 16, "BC6H_SFLOAT"},
        {VriFormat_BC4_SNORM, 8, "BC4_SNORM"},
        {VriFormat_BC5_SNORM, 16, "BC5_SNORM"},
        {VriFormat_ETC2_RGBA8_SRGB, 16, "ETC2_RGBA8_SRGB"},
        {VriFormat_ASTC_4x4_SRGB, 16, "ASTC_4x4_SRGB"},
    };

    // Returns 0 on success, else the failing line.
    inline int CheckCompressedRoundTrip(VriDevice* device, const VriCoreInterface& c, const CompressedFormat& f)
    {
        constexpr uint32_t kSize      = 8; // texels: 2x2 blocks
        constexpr uint32_t kBlocks    = kSize / 4;
        constexpr uint32_t kRowBytes  = 256; // one block row per 256 bytes: every backend's row alignment
        constexpr uint64_t kBytes     = uint64_t(kRowBytes) * kBlocks;
        const uint32_t     rowTexels  = kRowBytes / f.blockBytes * 4;
        const uint32_t     blockBytes = f.blockBytes;

        VriQueue* queue = nullptr;
        if (c.GetQueue(device, VriQueueType_Graphics, 0, &queue) != VriResult_Success)
            return __LINE__;
        VriTextureDesc td {};
        td.type           = VriTextureType_2D;
        td.format         = f.format;
        td.width          = kSize;
        td.height         = kSize;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = 1;
        td.sampleNum      = 1;
        td.usage          = VriTextureUsage_TransferSrc | VriTextureUsage_TransferDst | VriTextureUsage_ShaderResource;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* texture = nullptr;
        if (c.CreateTexture(device, &td, &texture) != VriResult_Success)
            return __LINE__;
        VriBufferDesc ud {};
        ud.size              = kBytes;
        ud.usage             = VriBufferUsage_TransferSrc;
        ud.memoryLocation    = VriMemoryLocation_HostUpload;
        VriBuffer*    upload = nullptr;
        VriBufferDesc rd {};
        rd.size             = kBytes;
        rd.usage            = VriBufferUsage_TransferDst;
        rd.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        if (c.CreateBuffer(device, &ud, &upload) != VriResult_Success ||
            c.CreateBuffer(device, &rd, &readback) != VriResult_Success)
            return __LINE__;

        // Each block's bytes are distinct; the rest of a row is padding.
        std::vector<uint8_t> source(static_cast<size_t>(kBytes), 0);
        for (uint32_t row = 0; row < kBlocks; ++row)
            for (uint32_t i = 0; i < kBlocks * blockBytes; ++i)
                source[row * kRowBytes + i] = static_cast<uint8_t>(row * 97 + i * 13 + 5);
        void* mapped = c.MapBuffer(upload, 0, kBytes);
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
        copy.bufferRowLength  = rowTexels;
        copy.texture.aspect   = VriImageAspect_Color;
        copy.texture.layerNum = 1;
        copy.texture.width    = kSize;
        copy.texture.height   = kSize;
        copy.texture.depth    = 1;
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
        const uint8_t* result  = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, kBytes));
        if (!result)
            failure = __LINE__;
        for (uint32_t row = 0; row < kBlocks && !failure; ++row)
            if (std::memcmp(result + row * kRowBytes, source.data() + row * kRowBytes, kBlocks * blockBytes) != 0)
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
} // namespace vri::test
