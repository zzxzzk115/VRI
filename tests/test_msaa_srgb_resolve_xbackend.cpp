// An MSAA resolve averages samples in linear space for an sRGB target, as on every backend; it is not a byte copy.
// The same red triangle is rendered into a 4x RGBA8_UNORM and a 4x RGBA8_SRGB target and resolved. An edge pixel
// with k of 4 samples covered reads 255 * k / 4 from the UNORM resolve, and the sRGB encoding of k / 4 from the sRGB
// one: 137, 188 and 225 for k = 1, 2, 3. Averaging the encoded bytes would give the UNORM values instead.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "shaders/common/triangle_dxbc.h" // g_triangleDxbcVS / g_triangleDxbcPS (D3D12)
#include "shaders/common/triangle_spv.h"  // g_triangleSpv (Vulkan, OpenGL)
#include "shaders/common/triangle_wgsl.h" // g_triangleWgsl (WebGPU)

namespace
{
    constexpr uint32_t kW       = 64; // 256-byte rows, the readback row pitch D3D12 requires
    constexpr uint32_t kH       = 64;
    constexpr uint32_t kSamples = 4;

    struct Shaders
    {
        const void* vs;
        size_t      vsSize;
        const void* ps;
        size_t      psSize;
    };

    VriTexture* CreateTarget(VriDevice* dev, VriCoreInterface& c, VriFormat format, uint32_t samples)
    {
        VriTextureDesc td {};
        td.type                    = VriTextureType_2D;
        td.format                  = format;
        td.width                   = kW;
        td.height                  = kH;
        td.depth                   = 1;
        td.mipNum                  = 1;
        td.layerNum                = 1;
        td.sampleNum               = samples;
        td.usage                   = VriTextureUsage_ColorAttachment;
        td.memoryLocation          = VriMemoryLocation_Device;
        td.clearValue.color.f32[3] = 1.0f;
        if (samples == 1)
            td.usage |= VriTextureUsage_TransferSrc;
        VriTexture* texture = nullptr;
        REQUIRE(c.CreateTexture(dev, &td, &texture) == VriResult_Success);
        return texture;
    }

    VriDescriptor* CreateView(VriDevice* dev, VriCoreInterface& c, VriTexture* texture)
    {
        VriTextureViewDesc vd {};
        vd.texture          = texture;
        vd.viewType         = VriTextureViewType_2D;
        vd.aspect           = VriImageAspect_Color;
        VriDescriptor* view = nullptr;
        REQUIRE(c.CreateTextureView(dev, &vd, &view) == VriResult_Success);
        return view;
    }

    // Renders the triangle into a 4x `format` target, resolves it and returns the resolved red channel.
    std::vector<uint8_t> RenderResolved(VriDevice* dev, VriCoreInterface& c, const Shaders& shaders, VriFormat format)
    {
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);
        VriTexture*    msaa        = CreateTarget(dev, c, format, kSamples);
        VriTexture*    resolve     = CreateTarget(dev, c, format, 1);
        VriDescriptor* msaaView    = CreateView(dev, c, msaa);
        VriDescriptor* resolveView = CreateView(dev, c, resolve);
        VriBufferDesc  bd {};
        bd.size             = static_cast<uint64_t>(kW) * kH * 4;
        bd.usage            = VriBufferUsage_TransferDst;
        bd.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &readback) == VriResult_Success);

        VriPipelineLayoutDesc ld {};
        VriPipelineLayout*    layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);
        VriShaderDesc sh[2] {};
        sh[0].stage          = VriShaderStage_Vertex;
        sh[0].bytecode       = shaders.vs;
        sh[0].bytecodeSize   = shaders.vsSize;
        sh[0].entryPointName = "vertexMain";
        sh[1].stage          = VriShaderStage_Fragment;
        sh[1].bytecode       = shaders.ps;
        sh[1].bytecodeSize   = shaders.psSize;
        sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca {};
        ca.format         = format;
        ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd {};
        pd.pipelineLayout          = layout;
        pd.shaders                 = sh;
        pd.shaderNum               = 2;
        pd.inputAssembly.topology  = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode  = VriCullMode_None;
        pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum   = kSamples;
        pd.outputMerger.colors     = &ca;
        pd.outputMerger.colorNum   = 1;
        VriPipeline* pipeline      = nullptr;
        REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);
        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        VriTextureBarrierDesc tb[2] {};
        for (int i = 0; i < 2; ++i)
        {
            tb[i].texture       = i == 0 ? msaa : resolve;
            tb[i].before.layout = VriLayout_Undefined;
            tb[i].after.access  = VriAccess_ColorAttachmentWrite;
            tb[i].after.layout  = VriLayout_ColorAttachment;
            tb[i].after.stages  = VriPipelineStage_ColorAttachmentOutput;
            tb[i].aspect        = VriImageAspect_Color;
        }
        VriBarrierGroupDesc g {};
        g.textures   = tb;
        g.textureNum = 2;
        c.CmdBarrier(cmd, &g);
        VriAttachmentDesc att {};
        att.view                    = msaaView;
        att.loadOp                  = VriAttachmentLoadOp_Clear;
        att.storeOp                 = VriAttachmentStoreOp_Store;
        att.clearValue.color.f32[3] = 1.0f;
        att.resolveView             = resolveView;
        VriAttachmentsDesc rp {};
        rp.colors            = &att;
        rp.colorNum          = 1;
        rp.renderArea.width  = kW;
        rp.renderArea.height = kH;
        rp.layerNum          = 1;
        c.CmdBeginRendering(cmd, &rp);
        VriViewport vp {0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1};
        c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc {0, 0, kW, kH};
        c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        VriDrawDesc draw {};
        draw.vertexNum   = 3;
        draw.instanceNum = 1;
        c.CmdDraw(cmd, &draw);
        c.CmdEndRendering(cmd);
        VriTextureBarrierDesc rb {};
        rb.texture       = resolve;
        rb.before.access = VriAccess_ColorAttachmentWrite;
        rb.before.layout = VriLayout_ColorAttachment;
        rb.before.stages = VriPipelineStage_ColorAttachmentOutput;
        rb.after.access  = VriAccess_CopySourceRead;
        rb.after.layout  = VriLayout_CopySource;
        rb.after.stages  = VriPipelineStage_Transfer;
        rb.aspect        = VriImageAspect_Color;
        g.textures       = &rb;
        g.textureNum     = 1;
        c.CmdBarrier(cmd, &g);
        VriBufferTextureCopyDesc cp {};
        cp.texture.aspect   = VriImageAspect_Color;
        cp.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, resolve, &cp);
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

        const auto* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bd.size));
        REQUIRE(px != nullptr);
        std::vector<uint8_t> red(kW * kH);
        for (uint32_t p = 0; p < kW * kH; ++p)
            red[p] = px[p * 4];
        c.UnmapBuffer(readback);

        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(readback);
        c.DestroyDescriptor(msaaView);
        c.DestroyDescriptor(resolveView);
        c.DestroyTexture(msaa);
        c.DestroyTexture(resolve);
        return red;
    }

    int EncodedByte(double linear)
    {
        const double srgb = linear <= 0.0031308 ? 12.92 * linear : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
        return static_cast<int>(std::lround(srgb * 255.0));
    }

    void Check(VriGraphicsAPI api, const std::string& name, const Shaders& shaders)
    {
        CAPTURE(name);
        VriDeviceCreationDesc dc  = vriDefaultDeviceCreationDesc();
        dc.coordinates.clipSpaceY = VriClipSpaceY_Up;
        dc.graphicsAPI            = api;
        VriDevice* dev            = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE(name << " unavailable - skipping");
            return;
        }
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        const std::vector<uint8_t> unorm = RenderResolved(dev, c, shaders, VriFormat_RGBA8_UNORM);
        const std::vector<uint8_t> srgb  = RenderResolved(dev, c, shaders, VriFormat_RGBA8_SRGB);
        vriDestroyDevice(dev);

        int checked[4]  = {};
        int srgbMin[4]  = {255, 255, 255, 255};
        int srgbMax[4]  = {};
        int unormMin[4] = {255, 255, 255, 255};
        int unormMax[4] = {};
        for (uint32_t p = 0; p < kW * kH; ++p)
        {
            if (unorm[p] == 0 || unorm[p] == 255)
                continue;
            const int coverage = static_cast<int>(std::lround(unorm[p] * 4.0 / 255.0));
            CAPTURE(p);
            CAPTURE(static_cast<int>(unorm[p]));
            CAPTURE(static_cast<int>(srgb[p]));
            REQUIRE(coverage >= 1);
            REQUIRE(coverage <= 3);
            CHECK(std::abs(unorm[p] - static_cast<int>(std::lround(255.0 * coverage / 4.0))) <= 1);
            CHECK(std::abs(srgb[p] - EncodedByte(coverage / 4.0)) <= 1);
            ++checked[coverage];
            srgbMin[coverage]  = std::min(srgbMin[coverage], static_cast<int>(srgb[p]));
            srgbMax[coverage]  = std::max(srgbMax[coverage], static_cast<int>(srgb[p]));
            unormMin[coverage] = std::min(unormMin[coverage], static_cast<int>(unorm[p]));
            unormMax[coverage] = std::max(unormMax[coverage], static_cast<int>(unorm[p]));
        }
        for (int k = 1; k <= 3; ++k)
            MESSAGE(name << ": " << checked[k] << " pixels with " << k << " of 4 samples covered: UNORM " << unormMin[k]
                         << ".." << unormMax[k] << ", sRGB " << srgbMin[k] << ".." << srgbMax[k] << " (expected "
                         << EncodedByte(k / 4.0) << ")");
        CHECK(checked[2] > 0); // at least one half-covered pixel
    }
} // namespace

TEST_CASE("MSAA resolve of an sRGB target averages in linear space")
{
    const Shaders spv {g_triangleSpv, sizeof(g_triangleSpv), g_triangleSpv, sizeof(g_triangleSpv)};
    const Shaders wgsl {g_triangleWgsl, sizeof(g_triangleWgsl), g_triangleWgsl, sizeof(g_triangleWgsl)};
    const Shaders dxbc {g_triangleDxbcVS, sizeof(g_triangleDxbcVS), g_triangleDxbcPS, sizeof(g_triangleDxbcPS)};
    Check(VriGraphicsAPI_Vulkan, "Vulkan", spv);
    Check(VriGraphicsAPI_D3D12, "D3D12", dxbc);
    Check(VriGraphicsAPI_WebGPU, "WebGPU", wgsl);
    Check(VriGraphicsAPI_OpenGL, "OpenGL", spv);
}
