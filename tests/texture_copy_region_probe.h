// CmdCopyTexture follows its region on every backend, shared by the desktop suite and the WebGL page. A 64x64
// 2DArray of two mips and two layers is filled with texels that encode their own mip, layer and position, then:
// - a 16x8 block at (4, 8) of mip 1, layer 1 is copied to (12, 12) of the same subresource of another texture;
// - mip 1 of layer 0 is copied with an explicit width of 32 and a height of 0, which is the mip's full height;
// - the whole of mip 0, layer 0 is copied with no region at all.
// Every texel of the destinations is read back: copied texels must match, all others must stay as they were. GL used
// to copy mip 0 of the base layer at the source's full size whatever the region said, D3D12 took a zero height as
// 1, and Metal took a zero size at mip 1 as the size of mip 0.
#pragma once

#include <vri/vri.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace vri::test
{
    // Returns 0 on success, else the failing line.
    inline int CheckTextureCopyRegions(VriDevice* dev, const VriCoreInterface& c)
    {
        constexpr uint32_t kSize      = 64;
        constexpr uint32_t kRowLength = 64; // texels: 256-byte rows at every mip, the pitch D3D12 and WebGPU require
        constexpr uint64_t kSlot      = uint64_t(kRowLength) * kSize * 4; // one subresource in the staging buffers

        const auto texel = [](uint32_t mip, uint32_t layer, uint32_t x, uint32_t y) {
            const uint32_t v = (x & 0xFF) | (y & 0xFF) << 8 | (1 + layer + 2 * mip) << 16 | 0xFFu << 24;
            return v;
        };
        const auto mipSize = [](uint32_t mip) { return kSize >> mip; };

        VriQueue* queue = nullptr;
        if (c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) != VriResult_Success)
            return __LINE__;
        const auto makeTexture = [&](uint32_t mips, uint32_t layers) -> VriTexture* {
            VriTextureDesc td {};
            td.type      = layers > 1 ? VriTextureType_2DArray : VriTextureType_2D;
            td.format    = VriFormat_RGBA8_UNORM;
            td.width     = kSize;
            td.height    = kSize;
            td.depth     = 1;
            td.mipNum    = mips;
            td.layerNum  = layers;
            td.sampleNum = 1;
            td.usage     = VriTextureUsage_TransferSrc | VriTextureUsage_TransferDst | VriTextureUsage_ShaderResource;
            td.memoryLocation = VriMemoryLocation_Device;
            VriTexture* t     = nullptr;
            return c.CreateTexture(dev, &td, &t) == VriResult_Success ? t : nullptr;
        };
        VriTexture* source = makeTexture(2, 2);
        VriTexture* target = makeTexture(2, 2);
        VriTexture* whole  = makeTexture(1, 1);
        if (!source || !target || !whole)
            return __LINE__;

        // Staging: slot (mip * 2 + layer) holds the source subresource; slot 4 holds zeros for the targets.
        VriBufferDesc ud {};
        ud.size              = kSlot * 5;
        ud.usage             = VriBufferUsage_TransferSrc;
        ud.memoryLocation    = VriMemoryLocation_HostUpload;
        VriBuffer*    upload = nullptr;
        VriBufferDesc rd     = ud;
        rd.size              = kSlot * 4;
        rd.usage             = VriBufferUsage_TransferDst;
        rd.memoryLocation    = VriMemoryLocation_HostReadback;
        VriBuffer* readback  = nullptr;
        if (c.CreateBuffer(dev, &ud, &upload) != VriResult_Success ||
            c.CreateBuffer(dev, &rd, &readback) != VriResult_Success)
            return __LINE__;
        auto* staging = static_cast<uint32_t*>(c.MapBuffer(upload, 0, ud.size));
        if (!staging)
            return __LINE__;
        for (uint32_t i = 0; i < ud.size / 4; ++i)
            staging[i] = 0;
        for (uint32_t mip = 0; mip < 2; ++mip)
            for (uint32_t layer = 0; layer < 2; ++layer)
                for (uint32_t y = 0; y < mipSize(mip); ++y)
                    for (uint32_t x = 0; x < mipSize(mip); ++x)
                        staging[(mip * 2 + layer) * (kSlot / 4) + y * kRowLength + x] = texel(mip, layer, x, y);
        c.UnmapBuffer(upload);

        VriCommandAllocator* alloc = nullptr;
        VriCommandBuffer*    cmd   = nullptr;
        VriFence*            fence = nullptr;
        if (c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) != VriResult_Success ||
            c.CreateCommandBuffer(alloc, &cmd) != VriResult_Success ||
            c.CreateFence(dev, 0, &fence) != VriResult_Success)
            return __LINE__;
        const auto barrier = [&](VriTexture* t, VriAccessFlags ba, VriLayout bl, VriAccessFlags aa, VriLayout al) {
            VriTextureBarrierDesc tb {};
            tb.texture       = t;
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
        const auto bufferRegion = [&](uint64_t offset, uint32_t mip, uint32_t layer) {
            VriBufferTextureCopyDesc r {};
            r.bufferOffset      = offset;
            r.bufferRowLength   = kRowLength;
            r.bufferImageHeight = mipSize(mip);
            r.texture.aspect    = VriImageAspect_Color;
            r.texture.mip       = mip;
            r.texture.baseLayer = layer;
            r.texture.layerNum  = 1;
            r.texture.width     = mipSize(mip);
            r.texture.height    = mipSize(mip);
            r.texture.depth     = 1;
            return r;
        };

        if (c.BeginCommandBuffer(cmd) != VriResult_Success)
            return __LINE__;
        for (VriTexture* t : {source, target, whole})
            barrier(t, VriAccess_None, VriLayout_Undefined, VriAccess_CopyDestinationWrite, VriLayout_CopyDestination);
        for (uint32_t mip = 0; mip < 2; ++mip)
            for (uint32_t layer = 0; layer < 2; ++layer)
            {
                const VriBufferTextureCopyDesc fill = bufferRegion((mip * 2 + layer) * kSlot, mip, layer);
                c.CmdUploadBufferToTexture(cmd, source, upload, &fill);
                const VriBufferTextureCopyDesc zero = bufferRegion(4 * kSlot, mip, layer);
                c.CmdUploadBufferToTexture(cmd, target, upload, &zero);
            }
        const VriBufferTextureCopyDesc zeroWhole = bufferRegion(4 * kSlot, 0, 0);
        c.CmdUploadBufferToTexture(cmd, whole, upload, &zeroWhole);
        barrier(source,
                VriAccess_CopyDestinationWrite,
                VriLayout_CopyDestination,
                VriAccess_CopySourceRead,
                VriLayout_CopySource);
        barrier(target,
                VriAccess_CopyDestinationWrite,
                VriLayout_CopyDestination,
                VriAccess_CopyDestinationWrite,
                VriLayout_CopyDestination);
        barrier(whole,
                VriAccess_CopyDestinationWrite,
                VriLayout_CopyDestination,
                VriAccess_CopyDestinationWrite,
                VriLayout_CopyDestination);

        VriTextureCopyDesc block {}; // mip 1, layer 1: 16x8 from (4, 8) to (12, 12)
        block.src.aspect    = VriImageAspect_Color;
        block.src.mip       = 1;
        block.src.baseLayer = 1;
        block.src.layerNum  = 1;
        block.src.x         = 4;
        block.src.y         = 8;
        block.src.width     = 16;
        block.src.height    = 8;
        block.src.depth     = 1;
        block.dst           = block.src;
        block.dst.x         = 12;
        block.dst.y         = 12;
        c.CmdCopyTexture(cmd, target, source, &block);
        VriTextureCopyDesc fullHeight {}; // mip 1, layer 0: width 32, height 0 = the mip's full height
        fullHeight.src.aspect   = VriImageAspect_Color;
        fullHeight.src.mip      = 1;
        fullHeight.src.layerNum = 1;
        fullHeight.src.width    = mipSize(1);
        fullHeight.dst          = fullHeight.src;
        c.CmdCopyTexture(cmd, target, source, &fullHeight);
        c.CmdCopyTexture(cmd, whole, source, nullptr); // mip 0, layer 0, whole

        for (VriTexture* t : {target, whole})
            barrier(t,
                    VriAccess_CopyDestinationWrite,
                    VriLayout_CopyDestination,
                    VriAccess_CopySourceRead,
                    VriLayout_CopySource);
        struct Check
        {
            VriTexture* texture;
            uint32_t    mip;
            uint32_t    layer;
        };
        const Check checks[] = {{target, 1, 1}, {target, 1, 0}, {target, 0, 0}, {whole, 0, 0}};
        for (uint32_t i = 0; i < 4; ++i)
        {
            const VriBufferTextureCopyDesc rb = bufferRegion(i * kSlot, checks[i].mip, checks[i].layer);
            c.CmdReadbackTextureToBuffer(cmd, readback, checks[i].texture, &rb);
        }
        if (c.EndCommandBuffer(cmd) != VriResult_Success)
            return __LINE__;
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

        // What each read-back subresource must hold at (x, y).
        const auto expected = [&](uint32_t check, uint32_t x, uint32_t y) -> uint32_t {
            switch (check)
            {
                case 0: // the copied block, zeros around it
                    if (x >= 12 && x < 28 && y >= 12 && y < 20)
                        return texel(1, 1, x - 8, y - 4);
                    return 0;
                case 1: // mip 1 of layer 0, whole
                    return texel(1, 0, x, y);
                case 2: // mip 0 of the target: no copy touched it
                    return 0;
                default: // the region-less copy of mip 0, layer 0
                    return texel(0, 0, x, y);
            }
        };
        int        failure = 0;
        const auto result  = static_cast<const uint32_t*>(c.MapBuffer(readback, 0, rd.size));
        if (!result)
            failure = __LINE__;
        for (uint32_t i = 0; i < 4 && !failure; ++i)
            for (uint32_t y = 0; y < mipSize(checks[i].mip) && !failure; ++y)
                for (uint32_t x = 0; x < mipSize(checks[i].mip) && !failure; ++x)
                {
                    const uint32_t got  = result[i * (kSlot / 4) + y * kRowLength + x];
                    const uint32_t want = expected(i, x, y);
                    if (got != want)
                    {
                        std::printf(
                            "texture copy region: check %u (mip %u, layer %u) at (%u, %u): got %08x, want %08x\n",
                            i,
                            checks[i].mip,
                            checks[i].layer,
                            x,
                            y,
                            got,
                            want);
                        failure = __LINE__;
                    }
                }
        if (result)
            c.UnmapBuffer(readback);

        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(upload);
        c.DestroyTexture(whole);
        c.DestroyTexture(target);
        c.DestroyTexture(source);
        return failure;
    }

    // A depth texture copied whole: a depth-only pass clears a D32 texture to 0.25, a copy with no region takes it to
    // another D32 texture, and both are read back (the source too, so a wrong result points at the copy and not at
    // the readback). Needs depth readback, which GLES and WebGL do not have. Returns 0 on success, else the failing
    // line.
    inline int CheckDepthTextureCopy(VriDevice* dev, const VriCoreInterface& c)
    {
        constexpr uint32_t kSize      = 64;
        constexpr uint64_t kLayerSize = uint64_t(kSize) * kSize * 4; // 256-byte rows of floats
        constexpr float    kDepth     = 0.25f;

        VriQueue* queue = nullptr;
        if (c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) != VriResult_Success)
            return __LINE__;
        const auto makeDepth = [&]() -> VriTexture* {
            VriTextureDesc td {};
            td.type      = VriTextureType_2D;
            td.format    = VriFormat_D32_SFLOAT;
            td.width     = kSize;
            td.height    = kSize;
            td.depth     = 1;
            td.mipNum    = 1;
            td.layerNum  = 1;
            td.sampleNum = 1;
            td.usage =
                VriTextureUsage_DepthStencilAttachment | VriTextureUsage_TransferSrc | VriTextureUsage_TransferDst;
            td.memoryLocation                = VriMemoryLocation_Device;
            td.clearValue.depthStencil.depth = kDepth;
            VriTexture* t                    = nullptr;
            return c.CreateTexture(dev, &td, &t) == VriResult_Success ? t : nullptr;
        };
        VriTexture* source = makeDepth();
        VriTexture* target = makeDepth();
        if (!source || !target)
            return __LINE__;
        VriTextureViewDesc vd {};
        vd.texture          = source;
        vd.viewType         = VriTextureViewType_2D;
        vd.aspect           = VriImageAspect_Depth;
        vd.mipNum           = 1;
        vd.layerNum         = 1;
        VriDescriptor* view = nullptr;
        if (c.CreateTextureView(dev, &vd, &view) != VriResult_Success)
            return __LINE__;
        VriBufferDesc rd {};
        rd.size             = kLayerSize * 2;
        rd.usage            = VriBufferUsage_TransferDst;
        rd.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        if (c.CreateBuffer(dev, &rd, &readback) != VriResult_Success)
            return __LINE__;
        VriCommandAllocator* alloc = nullptr;
        VriCommandBuffer*    cmd   = nullptr;
        VriFence*            fence = nullptr;
        if (c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) != VriResult_Success ||
            c.CreateCommandBuffer(alloc, &cmd) != VriResult_Success ||
            c.CreateFence(dev, 0, &fence) != VriResult_Success)
            return __LINE__;
        const auto barrier = [&](VriTexture* t, VriAccessFlags ba, VriLayout bl, VriAccessFlags aa, VriLayout al) {
            VriTextureBarrierDesc tb {};
            tb.texture       = t;
            tb.before.access = ba;
            tb.before.layout = bl;
            tb.before.stages = VriPipelineStage_AllCommands;
            tb.after.access  = aa;
            tb.after.layout  = al;
            tb.after.stages  = VriPipelineStage_AllCommands;
            tb.aspect        = VriImageAspect_Depth;
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        };

        if (c.BeginCommandBuffer(cmd) != VriResult_Success)
            return __LINE__;
        barrier(source,
                VriAccess_None,
                VriLayout_Undefined,
                VriAccess_DepthStencilAttachmentWrite,
                VriLayout_DepthStencilAttachment);
        barrier(target, VriAccess_None, VriLayout_Undefined, VriAccess_CopyDestinationWrite, VriLayout_CopyDestination);
        VriAttachmentDesc depth {};
        depth.view                          = view;
        depth.loadOp                        = VriAttachmentLoadOp_Clear;
        depth.storeOp                       = VriAttachmentStoreOp_Store;
        depth.clearValue.depthStencil.depth = kDepth;
        VriAttachmentsDesc att {};
        att.depth             = &depth;
        att.renderArea.width  = kSize;
        att.renderArea.height = kSize;
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        c.CmdEndRendering(cmd);
        barrier(source,
                VriAccess_DepthStencilAttachmentWrite,
                VriLayout_DepthStencilAttachment,
                VriAccess_CopySourceRead,
                VriLayout_CopySource);
        c.CmdCopyTexture(cmd, target, source, nullptr);
        barrier(target,
                VriAccess_CopyDestinationWrite,
                VriLayout_CopyDestination,
                VriAccess_CopySourceRead,
                VriLayout_CopySource);
        VriTexture* const read[2] = {source, target};
        for (uint32_t i = 0; i < 2; ++i)
        {
            VriBufferTextureCopyDesc rb {};
            rb.bufferOffset     = kLayerSize * i;
            rb.texture.aspect   = VriImageAspect_Depth;
            rb.texture.layerNum = 1;
            rb.texture.width    = kSize;
            rb.texture.height   = kSize;
            rb.texture.depth    = 1;
            c.CmdReadbackTextureToBuffer(cmd, readback, read[i], &rb);
        }
        if (c.EndCommandBuffer(cmd) != VriResult_Success)
            return __LINE__;
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

        int        failure = 0;
        const auto result  = static_cast<const float*>(c.MapBuffer(readback, 0, rd.size));
        if (!result)
            failure = __LINE__;
        for (uint32_t i = 0; i < 2 && !failure; ++i)
            for (uint32_t p = 0; p < kSize * kSize; ++p)
                if (result[i * (kLayerSize / 4) + p] != kDepth)
                {
                    std::printf("depth texture copy: %s texel %u reads %f, want %f\n",
                                i == 0 ? "source" : "copy",
                                p,
                                static_cast<double>(result[i * (kLayerSize / 4) + p]),
                                static_cast<double>(kDepth));
                    failure = __LINE__; // the message says whether the source or the copy is wrong
                    break;
                }
        if (result)
            c.UnmapBuffer(readback);

        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        c.DestroyDescriptor(view);
        c.DestroyTexture(target);
        c.DestroyTexture(source);
        return failure;
    }
} // namespace vri::test
