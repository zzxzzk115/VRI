// One layer of an array or cube texture as a render target and as a copy end, shared by the desktop suite and the
// WebGL page. Each pass clears a 2D view of one layer to its own color; the layers are read back and compared:
// - a 2DArray of one layer, cleared;
// - a 2DArray of two layers, layer 0 then layer 1 cleared to different colors (a pass that ignored the view's layer
//   would leave layer 0 with layer 1's color);
// - a 2DArray of one layer copied into another, with a region and with none (the whole texture);
// - each face of a Cube cleared to its own color.
// GL attached these with glFramebufferTexture2D and the texture's own target, which an array target or a cube is
// not, and its DSA path attached layer 0 whatever the view said.
#pragma once

#include <vri/vri.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace vri::test
{
    // Returns 0 on success, else the failing line.
    inline int CheckLayerAttachments(VriDevice* dev, const VriCoreInterface& c)
    {
        constexpr uint32_t kSize      = 64; // 256-byte rows, the readback row pitch D3D12 and WebGPU require
        constexpr uint64_t kLayerSize = uint64_t(kSize) * kSize * 4;
        struct Rgba
        {
            uint8_t r, g, b, a;
        };
        struct Clear
        {
            VriTexture* texture;
            uint32_t    layer;
            Rgba        color;
        };
        struct Expect
        {
            VriTexture* texture;
            uint32_t    layer;
            Rgba        color;
        };

        VriQueue* queue = nullptr;
        if (c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) != VriResult_Success)
            return __LINE__;
        std::vector<VriTexture*>    textures;
        std::vector<VriDescriptor*> views;
        const auto                  makeTexture = [&](VriTextureType type, uint32_t layers) -> VriTexture* {
            VriTextureDesc td {};
            td.type      = type;
            td.format    = VriFormat_RGBA8_UNORM;
            td.width     = kSize;
            td.height    = kSize;
            td.depth     = 1;
            td.mipNum    = 1;
            td.layerNum  = layers;
            td.sampleNum = 1;
            td.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc | VriTextureUsage_TransferDst;
            td.memoryLocation = VriMemoryLocation_Device;
            VriTexture* t     = nullptr;
            if (c.CreateTexture(dev, &td, &t) != VriResult_Success)
                return nullptr;
            textures.push_back(t);
            return t;
        };
        VriTexture* single = makeTexture(VriTextureType_2DArray, 1);
        VriTexture* pair   = makeTexture(VriTextureType_2DArray, 2);
        VriTexture* source = makeTexture(VriTextureType_2DArray, 1);
        VriTexture* copy   = makeTexture(VriTextureType_2DArray, 1);
        VriTexture* whole  = makeTexture(VriTextureType_2DArray, 1); // copied with no region
        VriTexture* cube   = makeTexture(VriTextureType_Cube, 6);
        if (!single || !pair || !source || !copy || !whole || !cube)
            return __LINE__;

        std::vector<Clear> clears = {
            {single, 0, {0, 255, 0, 255}},
            {pair, 0, {0, 0, 255, 255}},
            {pair, 1, {255, 0, 0, 255}},
            {source, 0, {255, 255, 0, 255}},
        };
        for (uint32_t face = 0; face < 6; ++face)
            clears.push_back({cube, face, {static_cast<uint8_t>(40 * (face + 1)), 0, 255, 255}});
        std::vector<Expect> expects = {
            {single, 0, {0, 255, 0, 255}},
            {pair, 0, {0, 0, 255, 255}},
            {pair, 1, {255, 0, 0, 255}},
            {copy, 0, {255, 255, 0, 255}},
            {whole, 0, {255, 255, 0, 255}},
        };
        for (uint32_t face = 0; face < 6; ++face)
            expects.push_back({cube, face, {static_cast<uint8_t>(40 * (face + 1)), 0, 255, 255}});

        VriBufferDesc rd {};
        rd.size             = kLayerSize * expects.size();
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

        const auto stagesOf = [](VriAccessFlags access) -> VriPipelineStageFlags {
            if (access == VriAccess_None)
                return VriPipelineStage_None;
            if (access == VriAccess_ColorAttachmentWrite)
                return VriPipelineStage_ColorAttachmentOutput;
            return VriPipelineStage_Transfer;
        };
        const auto barrier = [&](VriTexture* t, VriAccessFlags ba, VriLayout bl, VriAccessFlags aa, VriLayout al) {
            VriTextureBarrierDesc tb {};
            tb.texture       = t;
            tb.before.access = ba;
            tb.before.layout = bl;
            tb.before.stages = stagesOf(ba);
            tb.after.access  = aa;
            tb.after.layout  = al;
            tb.after.stages  = stagesOf(aa);
            tb.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        };

        if (c.BeginCommandBuffer(cmd) != VriResult_Success)
            return __LINE__;
        for (VriTexture* t : {single, pair, source, cube})
            barrier(t, VriAccess_None, VriLayout_Undefined, VriAccess_ColorAttachmentWrite, VriLayout_ColorAttachment);
        for (VriTexture* t : {copy, whole})
            barrier(t, VriAccess_None, VriLayout_Undefined, VriAccess_CopyDestinationWrite, VriLayout_CopyDestination);
        for (const Clear& cl : clears)
        {
            VriTextureViewDesc vd {};
            vd.texture          = cl.texture;
            vd.viewType         = VriTextureViewType_2D;
            vd.aspect           = VriImageAspect_Color;
            vd.baseLayer        = cl.layer;
            vd.layerNum         = 1;
            vd.mipNum           = 1;
            VriDescriptor* view = nullptr;
            if (c.CreateTextureView(dev, &vd, &view) != VriResult_Success)
                return __LINE__;
            views.push_back(view);
            VriAttachmentDesc rt {};
            rt.view                    = view;
            rt.loadOp                  = VriAttachmentLoadOp_Clear;
            rt.storeOp                 = VriAttachmentStoreOp_Store;
            rt.clearValue.color.f32[0] = cl.color.r / 255.0f;
            rt.clearValue.color.f32[1] = cl.color.g / 255.0f;
            rt.clearValue.color.f32[2] = cl.color.b / 255.0f;
            rt.clearValue.color.f32[3] = cl.color.a / 255.0f;
            VriAttachmentsDesc att {};
            att.colors            = &rt;
            att.colorNum          = 1;
            att.renderArea.width  = kSize;
            att.renderArea.height = kSize;
            att.layerNum          = 1;
            c.CmdBeginRendering(cmd, &att);
            c.CmdEndRendering(cmd);
        }
        for (VriTexture* t : {single, pair, source, cube})
            barrier(t,
                    VriAccess_ColorAttachmentWrite,
                    VriLayout_ColorAttachment,
                    VriAccess_CopySourceRead,
                    VriLayout_CopySource);
        VriTextureCopyDesc region {};
        region.src.aspect   = VriImageAspect_Color;
        region.src.layerNum = 1;
        region.dst.aspect   = VriImageAspect_Color;
        region.dst.layerNum = 1;
        c.CmdCopyTexture(cmd, copy, source, &region);
        c.CmdCopyTexture(cmd, whole, source, nullptr); // no region: the whole texture
        for (VriTexture* t : {copy, whole})
            barrier(t,
                    VriAccess_CopyDestinationWrite,
                    VriLayout_CopyDestination,
                    VriAccess_CopySourceRead,
                    VriLayout_CopySource);
        for (size_t i = 0; i < expects.size(); ++i)
        {
            VriBufferTextureCopyDesc rb {};
            rb.bufferOffset      = kLayerSize * i;
            rb.bufferRowLength   = kSize;
            rb.bufferImageHeight = kSize;
            rb.texture.aspect    = VriImageAspect_Color;
            rb.texture.baseLayer = expects[i].layer;
            rb.texture.layerNum  = 1;
            rb.texture.width     = kSize;
            rb.texture.height    = kSize;
            rb.texture.depth     = 1;
            c.CmdReadbackTextureToBuffer(cmd, readback, expects[i].texture, &rb);
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

        int            failure = 0;
        const uint8_t* result  = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rd.size));
        if (!result)
            failure = __LINE__;
        for (size_t i = 0; i < expects.size() && !failure; ++i)
        {
            const uint8_t* layer = result + kLayerSize * i;
            const Rgba&    want  = expects[i].color;
            for (uint64_t p = 0; p < kLayerSize; p += 4)
                if (layer[p] != want.r || layer[p + 1] != want.g || layer[p + 2] != want.b || layer[p + 3] != want.a)
                {
                    std::printf("layer attachment mismatch in check %zu (layer %u) at byte %llu: got %u %u %u %u, want "
                                "%u %u %u %u\n",
                                i,
                                expects[i].layer,
                                static_cast<unsigned long long>(p),
                                layer[p],
                                layer[p + 1],
                                layer[p + 2],
                                layer[p + 3],
                                want.r,
                                want.g,
                                want.b,
                                want.a);
                    failure = __LINE__;
                    break;
                }
        }
        if (result)
            c.UnmapBuffer(readback);

        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        for (VriDescriptor* v : views)
            c.DestroyDescriptor(v);
        for (VriTexture* t : textures)
            c.DestroyTexture(t);
        return failure;
    }
} // namespace vri::test
