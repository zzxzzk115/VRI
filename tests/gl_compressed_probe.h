// GL block-compressed textures, shared by the desktop suite, the WebGL page and the Android probe. Each format
// gets one solid 4x4 block whose decode is known exactly from its specification; the block is uploaded,
// sampled onto an RGBA8 target and read back from there. A compressed texture itself cannot be read back on
// GL, and that refusal is checked too.
#pragma once

#include <vri/vri.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "shaders/tests/triangle_tex_spv.h" // g_triangleTexSpv: the triangle samples a texture

namespace vri::test
{
    struct CompressedSample
    {
        VriFormat   format;
        uint32_t    blockBytes;
        const char* name;
        uint8_t     block[16];
        uint8_t     expected[4]; // RGBA8 the decoded block lands as
    };

    // Writes `count` bits of `value` from bit `pos` on, least significant bit first, as BC6H and BC7 lay out.
    inline void PutBits(uint8_t* block, uint32_t& pos, uint32_t value, uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i, ++pos)
            if (value & (1u << i))
                block[pos / 8] |= static_cast<uint8_t>(1u << (pos % 8));
    }

    // One solid block per format the GL backend maps. Every texel uses index 0, so the decode is the first
    // endpoint (or the constant colour) as the format's specification defines it.
    inline std::vector<CompressedSample> CompressedSamples()
    {
        std::vector<CompressedSample> s;
        // BC1: both endpoints pure red (RGB565 0xF800).
        s.push_back({VriFormat_BC1_UNORM, 8, "BC1_UNORM", {0x00, 0xF8, 0x00, 0xF8}, {255, 0, 0, 255}});
        // BC2: explicit alpha 15 everywhere, then the BC1 colour block.
        s.push_back({VriFormat_BC2_UNORM,
                     16,
                     "BC2_UNORM",
                     {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xF8, 0x00, 0xF8},
                     {255, 0, 0, 255}});
        // BC3: alpha endpoints 255, then the BC1 colour block.
        s.push_back({VriFormat_BC3_UNORM,
                     16,
                     "BC3_UNORM",
                     {0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0x00, 0xF8, 0x00, 0xF8},
                     {255, 0, 0, 255}});
        // BC4: red endpoints 255.
        s.push_back({VriFormat_BC4_UNORM, 8, "BC4_UNORM", {0xFF, 0xFF}, {255, 0, 0, 255}});
        // BC5: red endpoints 255, green endpoints 128.
        s.push_back(
            {VriFormat_BC5_UNORM, 16, "BC5_UNORM", {0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0x80, 0x80}, {255, 128, 0, 255}});

        // BC6H mode 11 (00011): one region, 10-bit endpoints. Red 462 unquantizes to half 0x3801 (0.5005),
        // green and blue 0; BC6H has no alpha, so it samples as 1.
        CompressedSample bc6 {VriFormat_BC6H_UFLOAT, 16, "BC6H_UFLOAT", {}, {128, 0, 0, 255}};
        uint32_t         pos = 0;
        PutBits(bc6.block, pos, 0x03, 5);
        const uint32_t endpoint[3] = {462, 0, 0};
        for (uint32_t e = 0; e < 2; ++e)
            for (uint32_t ch = 0; ch < 3; ++ch)
                PutBits(bc6.block, pos, endpoint[ch], 10);
        s.push_back(bc6);

        // BC7 mode 6: 7-bit RGBA endpoints plus a p-bit each. R 127, G 64, B 0, A 127 with p 1 decode as
        // (255, 129, 1, 255).
        CompressedSample bc7 {VriFormat_BC7_UNORM, 16, "BC7_UNORM", {}, {255, 129, 1, 255}};
        pos = 0;
        PutBits(bc7.block, pos, 1u << 6, 7);
        const uint32_t channel[4] = {127, 64, 0, 127};
        for (uint32_t ch = 0; ch < 4; ++ch)
        {
            PutBits(bc7.block, pos, channel[ch], 7);
            PutBits(bc7.block, pos, channel[ch], 7);
        }
        PutBits(bc7.block, pos, 1, 1);
        PutBits(bc7.block, pos, 1, 1);
        s.push_back(bc7);

        // ETC2 RGBA8: EAC alpha base 255 with multiplier 0, then an individual-mode colour block with red 15/15
        // and green/blue 0, table 0. Index 0 adds 2 to every channel: (255, 2, 2).
        s.push_back({VriFormat_ETC2_RGBA8_UNORM,
                     16,
                     "ETC2_RGBA8_UNORM",
                     {0xFF, 0x00, 0, 0, 0, 0, 0, 0, 0xFF, 0x00, 0x00, 0x00},
                     {255, 2, 2, 255}});
        // ASTC 4x4: an LDR void-extent block, the constant colour as UNORM16 RGBA (red, opaque).
        s.push_back({VriFormat_ASTC_4x4_UNORM,
                     16,
                     "ASTC_4x4_UNORM",
                     {0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF},
                     {255, 0, 0, 255}});
        return s;
    }

    // An empty 4x4 texture of `format`, or a 3D one 4x4x4. Null on failure.
    inline VriTexture* CreateCompressedTexture(VriDevice*              device,
                                               const VriCoreInterface& c,
                                               VriFormat               format,
                                               VriTextureType          type = VriTextureType_2D)
    {
        VriTextureDesc td {};
        td.type           = type;
        td.format         = format;
        td.width          = 4;
        td.height         = 4;
        td.depth          = type == VriTextureType_3D ? 4 : 1;
        td.mipNum         = 1;
        td.layerNum       = 1;
        td.sampleNum      = 1;
        td.usage          = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst | VriTextureUsage_TransferSrc;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* texture = nullptr;
        return c.CreateTexture(device, &td, &texture) == VriResult_Success ? texture : nullptr;
    }

    // A 4x4 texture of `sample`'s block, uploaded from a buffer. Null on failure.
    inline VriTexture* UploadCompressedBlock(VriDevice*              device,
                                             const VriCoreInterface& c,
                                             VriCommandBuffer*       cmd,
                                             const CompressedSample& sample,
                                             VriBuffer*&             upload)
    {
        VriTexture* texture = CreateCompressedTexture(device, c, sample.format);
        if (!texture)
            return nullptr;
        VriBufferDesc bd {};
        bd.size           = sample.blockBytes;
        bd.usage          = VriBufferUsage_TransferSrc;
        bd.memoryLocation = VriMemoryLocation_HostUpload;
        if (c.CreateBuffer(device, &bd, &upload) != VriResult_Success)
            return nullptr;
        std::memcpy(c.MapBuffer(upload, 0, sample.blockBytes), sample.block, sample.blockBytes);
        c.UnmapBuffer(upload);
        VriBufferTextureCopyDesc copy {};
        copy.texture.aspect   = VriImageAspect_Color;
        copy.texture.layerNum = 1;
        copy.texture.width    = 4;
        copy.texture.height   = 4;
        c.CmdUploadBufferToTexture(cmd, texture, upload, &copy);
        return texture;
    }

    // Uploads `sample`, samples it onto an RGBA8 target and reads the target's centre back into `rgba`. With
    // `viaCopy` the uploaded texture is first copied (CmdCopyTexture) into a second one, which is sampled.
    // Returns 0 on success, else the failing line.
    inline int SampleCompressed(VriDevice*              device,
                                const VriCoreInterface& c,
                                const CompressedSample& sample,
                                uint8_t                 rgba[4],
                                bool                    viaCopy = false)
    {
        constexpr uint32_t kW    = 64;
        constexpr uint32_t kH    = 64;
        VriQueue*          queue = nullptr;
        if (c.GetQueue(device, VriQueueType_Graphics, 0, &queue) != VriResult_Success)
            return __LINE__;

        VriTextureDesc td {};
        td.type           = VriTextureType_2D;
        td.format         = VriFormat_RGBA8_UNORM;
        td.width          = kW;
        td.height         = kH;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = 1;
        td.sampleNum      = 1;
        td.usage          = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* color = nullptr;
        if (c.CreateTexture(device, &td, &color) != VriResult_Success)
            return __LINE__;
        VriTextureViewDesc vd {};
        vd.texture               = color;
        vd.viewType              = VriTextureViewType_2D;
        vd.aspect                = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr;
        c.CreateTextureView(device, &vd, &colorView);

        VriCommandAllocator* alloc = nullptr;
        VriCommandBuffer*    cmd   = nullptr;
        VriFence*            fence = nullptr;
        c.CreateCommandAllocator(device, VriQueueType_Graphics, &alloc);
        c.CreateCommandBuffer(alloc, &cmd);
        c.CreateFence(device, 0, &fence);
        c.BeginCommandBuffer(cmd);
        VriBuffer*  upload   = nullptr;
        VriTexture* uploaded = UploadCompressedBlock(device, c, cmd, sample, upload);
        if (!uploaded)
            return __LINE__;
        VriTexture* sampled = uploaded;
        if (viaCopy)
        {
            sampled = CreateCompressedTexture(device, c, sample.format);
            if (!sampled)
                return __LINE__;
            c.CmdCopyTexture(cmd, sampled, uploaded, nullptr);
        }
        vd.texture             = sampled;
        VriDescriptor* texView = nullptr;
        c.CreateTextureView(device, &vd, &texView);
        VriSamplerDesc sd {};
        sd.magFilter           = VriFilter_Nearest;
        sd.minFilter           = VriFilter_Nearest;
        sd.addressModeU        = VriAddressMode_ClampToEdge;
        sd.addressModeV        = VriAddressMode_ClampToEdge;
        sd.addressModeW        = VriAddressMode_ClampToEdge;
        sd.maxLod              = 1.0f;
        VriDescriptor* sampler = nullptr;
        c.CreateSampler(device, &sd, &sampler);

        VriDescriptorRangeDesc ranges[2] {};
        ranges[0].baseRegister   = 0;
        ranges[0].descriptorNum  = 1;
        ranges[0].descriptorType = VriDescriptorType_Texture;
        ranges[0].shaderStages   = VriShaderStage_Fragment;
        ranges[1].baseRegister   = 1;
        ranges[1].descriptorNum  = 1;
        ranges[1].descriptorType = VriDescriptorType_Sampler;
        ranges[1].shaderStages   = VriShaderStage_Fragment;
        VriDescriptorSetDesc setDesc {};
        setDesc.ranges   = ranges;
        setDesc.rangeNum = 2;
        VriPipelineLayoutDesc ld {};
        ld.descriptorSets         = &setDesc;
        ld.descriptorSetNum       = 1;
        VriPipelineLayout* layout = nullptr;
        c.CreatePipelineLayout(device, &ld, &layout);
        VriShaderDesc sh[2] {};
        sh[0].stage          = VriShaderStage_Vertex;
        sh[0].bytecode       = g_triangleTexSpv;
        sh[0].bytecodeSize   = sizeof(g_triangleTexSpv);
        sh[0].entryPointName = "vertexMain";
        sh[1].stage          = VriShaderStage_Fragment;
        sh[1].bytecode       = g_triangleTexSpv;
        sh[1].bytecodeSize   = sizeof(g_triangleTexSpv);
        sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca {};
        ca.format         = VriFormat_RGBA8_UNORM;
        ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd {};
        pd.pipelineLayout          = layout;
        pd.shaders                 = sh;
        pd.shaderNum               = 2;
        pd.inputAssembly.topology  = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode  = VriCullMode_None;
        pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum   = 1;
        pd.outputMerger.colors     = &ca;
        pd.outputMerger.colorNum   = 1;
        VriPipeline* pipeline      = nullptr;
        if (c.CreateGraphicsPipeline(device, &pd, &pipeline) != VriResult_Success)
            return __LINE__;
        VriDescriptorPoolDesc pool {};
        pool.descriptorSetMaxNum   = 1;
        pool.textureMaxNum         = 1;
        pool.samplerMaxNum         = 1;
        VriDescriptorPool* setPool = nullptr;
        c.CreateDescriptorPool(device, &pool, &setPool);
        VriDescriptorSet* set = nullptr;
        c.AllocateDescriptorSets(setPool, layout, 0, &set, 1);
        const VriDescriptor*         textures[1] = {texView};
        const VriDescriptor*         samplers[1] = {sampler};
        VriDescriptorRangeUpdateDesc upd[2] {};
        upd[0].descriptors   = textures;
        upd[0].descriptorNum = 1;
        upd[1].descriptors   = samplers;
        upd[1].descriptorNum = 1;
        c.UpdateDescriptorRanges(set, 0, 2, upd);

        VriBufferDesc rb {};
        rb.size             = uint64_t(kW) * kH * 4;
        rb.usage            = VriBufferUsage_TransferDst;
        rb.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        c.CreateBuffer(device, &rb, &readback);

        VriAttachmentDesc rt {};
        rt.view                    = colorView;
        rt.loadOp                  = VriAttachmentLoadOp_Clear;
        rt.storeOp                 = VriAttachmentStoreOp_Store;
        rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att {};
        att.colors            = &rt;
        att.colorNum          = 1;
        att.renderArea.width  = kW;
        att.renderArea.height = kH;
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport viewport {0, 0, float(kW), float(kH), 0, 1};
        c.CmdSetViewports(cmd, &viewport, 1);
        VriRect scissor {0, 0, kW, kH};
        c.CmdSetScissors(cmd, &scissor, 1);
        c.CmdSetPipeline(cmd, pipeline);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, set);
        VriDrawDesc draw {};
        draw.vertexNum   = 3;
        draw.instanceNum = 1;
        c.CmdDraw(cmd, &draw);
        c.CmdEndRendering(cmd);
        VriBufferTextureCopyDesc copy {};
        copy.texture.aspect   = VriImageAspect_Color;
        copy.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &copy);
        c.EndCommandBuffer(cmd);
        VriFenceSubmitDesc signal {};
        signal.fence = fence;
        signal.value = 1;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        int failure             = c.QueueSubmit(queue, &submit) == VriResult_Success ? 0 : __LINE__;
        c.Wait(fence, 1);
        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rb.size));
        if (px)
        {
            std::memcpy(rgba, px + (uint64_t(kH / 2) * kW + kW / 2) * 4, 4);
            c.UnmapBuffer(readback);
        }
        else if (!failure)
            failure = __LINE__;

        c.DeviceWaitIdle(device);
        c.DestroyBuffer(readback);
        c.DestroyDescriptorPool(setPool);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyDescriptor(sampler);
        c.DestroyDescriptor(texView);
        c.DestroyDescriptor(colorView);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(upload);
        if (sampled != uploaded)
            c.DestroyTexture(sampled);
        c.DestroyTexture(uploaded);
        c.DestroyTexture(color);
        return failure;
    }

    // Whether every channel is within `tolerance` of the expected decode.
    inline bool MatchesDecode(const CompressedSample& sample, const uint8_t rgba[4], int tolerance = 1)
    {
        for (int i = 0; i < 4; ++i)
            if (std::abs(int(rgba[i]) - int(sample.expected[i])) > tolerance)
                return false;
        return true;
    }

    // A copy between compressed textures is refused with one error where GL has no copy for them (GLES, WebGL).
    // `errors` counts the device's error messages. Returns 0 on success, else the failing line.
    inline int CheckCompressedCopyRefused(VriDevice*              device,
                                          const VriCoreInterface& c,
                                          const CompressedSample& sample,
                                          const int&              errors)
    {
        VriTexture* from = CreateCompressedTexture(device, c, sample.format);
        VriTexture* to   = CreateCompressedTexture(device, c, sample.format);
        if (!from || !to)
            return __LINE__;
        VriCommandAllocator* alloc = nullptr;
        VriCommandBuffer*    cmd   = nullptr;
        c.CreateCommandAllocator(device, VriQueueType_Graphics, &alloc);
        c.CreateCommandBuffer(alloc, &cmd);
        c.BeginCommandBuffer(cmd);
        const int before = errors;
        c.CmdCopyTexture(cmd, to, from, nullptr);
        const int failure = errors == before + 1 ? 0 : __LINE__;
        c.EndCommandBuffer(cmd);
        c.DestroyCommandAllocator(alloc);
        c.DestroyTexture(to);
        c.DestroyTexture(from);
        return failure;
    }

    // Reading a compressed texture back is refused with one error and leaves the buffer untouched. `errors`
    // counts the device's error messages. Returns 0 on success, else the failing line.
    inline int CheckCompressedReadbackRefused(VriDevice*              device,
                                              const VriCoreInterface& c,
                                              const CompressedSample& sample,
                                              const int&              errors)
    {
        VriQueue* queue = nullptr;
        c.GetQueue(device, VriQueueType_Graphics, 0, &queue);
        VriCommandAllocator* alloc = nullptr;
        VriCommandBuffer*    cmd   = nullptr;
        VriFence*            fence = nullptr;
        c.CreateCommandAllocator(device, VriQueueType_Graphics, &alloc);
        c.CreateCommandBuffer(alloc, &cmd);
        c.CreateFence(device, 0, &fence);
        VriBufferDesc rb {};
        rb.size             = sample.blockBytes;
        rb.usage            = VriBufferUsage_TransferDst;
        rb.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        c.CreateBuffer(device, &rb, &readback);
        c.BeginCommandBuffer(cmd);
        VriBuffer*  upload  = nullptr;
        VriTexture* texture = UploadCompressedBlock(device, c, cmd, sample, upload);
        if (!texture)
            return __LINE__;
        const int                before = errors;
        VriBufferTextureCopyDesc copy {};
        copy.texture.aspect   = VriImageAspect_Color;
        copy.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, texture, &copy);
        const int failure = errors == before + 1 ? 0 : __LINE__;
        c.EndCommandBuffer(cmd);
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
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(upload);
        c.DestroyTexture(texture);
        return failure;
    }
} // namespace vri::test
