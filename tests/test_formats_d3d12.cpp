// D3D12 format mapping: every VriFormat with a DXGI equivalent is mapped, and a format without
// one (S8_UINT: D3D12 has no stencil-only format) is reported as unsupported everywhere instead
// of being replaced by RGBA8. A BGRA8_SRGB target cleared to linear 0.5 must read back sRGB-encoded,
// which RGBA8_UNORM (the old substitute) would not.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/common/triangle_dxbc.h" // g_triangleDxbcVS + g_triangleDxbcPS

namespace
{
    struct D3D12Device
    {
        VriDevice*       dev = nullptr;
        VriCoreInterface c {};

        D3D12Device()
        {
            VriDeviceCreationDesc dc  = vriDefaultDeviceCreationDesc();
            dc.coordinates.clipSpaceY = VriClipSpaceY_Up;
            dc.graphicsAPI            = VriGraphicsAPI_D3D12;
            dc.enableValidation       = VRI_TRUE;
            dc.bestEffort             = VRI_TRUE;
            if (vriCreateDevice(&dc, &dev) == VriResult_Success)
                vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c);
            else
                dev = nullptr;
        }
        ~D3D12Device()
        {
            if (dev)
                vriDestroyDevice(dev);
        }
    };

    VriTextureDesc Texture2D(VriFormat format, VriTextureUsageFlags usage)
    {
        VriTextureDesc td {};
        td.type           = VriTextureType_2D;
        td.format         = format;
        td.width          = 16;
        td.height         = 16;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = 1;
        td.sampleNum      = 1;
        td.usage          = usage;
        td.memoryLocation = VriMemoryLocation_Device;
        return td;
    }
} // namespace

TEST_CASE("D3D12: formats with a DXGI equivalent are mapped; S8_UINT is unsupported")
{
    D3D12Device d;
    if (!d.dev)
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    const VriFormat colors[] = {
        VriFormat_R8_SNORM,      VriFormat_R8_UINT,     VriFormat_R8_SINT,     VriFormat_RG8_SNORM,
        VriFormat_RG8_UINT,      VriFormat_RG8_SINT,    VriFormat_RGBA8_UINT,  VriFormat_RGBA8_SINT,
        VriFormat_BGRA8_SRGB,    VriFormat_R16_UNORM,   VriFormat_R16_SNORM,   VriFormat_R16_UINT,
        VriFormat_R16_SINT,      VriFormat_R16_SFLOAT,  VriFormat_RG16_UNORM,  VriFormat_RG16_SNORM,
        VriFormat_RG16_UINT,     VriFormat_RG16_SINT,   VriFormat_RG16_SFLOAT, VriFormat_RGBA16_UNORM,
        VriFormat_RGBA16_SNORM,  VriFormat_RGBA16_UINT, VriFormat_RGBA16_SINT, VriFormat_RGB10A2_UNORM,
        VriFormat_RG11B10_UFLOAT};
    for (const VriFormat f : colors)
    {
        CAPTURE(f);
        CHECK((d.c.GetFormatSupport(d.dev, f) & VriFormatSupport_Texture) != 0);
        const VriTextureDesc td = Texture2D(f, VriTextureUsage_ShaderResource);
        VriTexture*          t  = nullptr;
        CHECK(d.c.CreateTexture(d.dev, &td, &t) == VriResult_Success);
        d.c.DestroyTexture(t);
    }
    const VriFormat depths[] = {VriFormat_D16_UNORM, VriFormat_D32_SFLOAT_S8_UINT};
    for (const VriFormat f : depths)
    {
        CAPTURE(f);
        CHECK((d.c.GetFormatSupport(d.dev, f) & VriFormatSupport_DepthStencil) != 0);
        // Sampled depth takes the typeless resource path.
        const VriTextureDesc td = Texture2D(f, VriTextureUsage_DepthStencilAttachment | VriTextureUsage_ShaderResource);
        VriTexture*          t  = nullptr;
        CHECK(d.c.CreateTexture(d.dev, &td, &t) == VriResult_Success);
        d.c.DestroyTexture(t);
    }

    CHECK(d.c.GetFormatSupport(d.dev, VriFormat_S8_UINT) == VriFormatSupport_None);
    const VriTextureDesc td = Texture2D(VriFormat_S8_UINT, VriTextureUsage_DepthStencilAttachment);
    VriTexture*          t  = nullptr;
    CHECK(d.c.CreateTexture(d.dev, &td, &t) == VriResult_Unsupported);
    CHECK(t == nullptr);

    // A pipeline with an unmapped attachment format must fail, not render into RGBA8.
    VriPipelineLayoutDesc ld {};
    VriPipelineLayout*    layout = nullptr;
    REQUIRE(d.c.CreatePipelineLayout(d.dev, &ld, &layout) == VriResult_Success);
    VriShaderDesc sh[2] {};
    sh[0].stage          = VriShaderStage_Vertex;
    sh[0].bytecode       = g_triangleDxbcVS;
    sh[0].bytecodeSize   = sizeof(g_triangleDxbcVS);
    sh[0].entryPointName = "vertexMain";
    sh[1].stage          = VriShaderStage_Fragment;
    sh[1].bytecode       = g_triangleDxbcPS;
    sh[1].bytecodeSize   = sizeof(g_triangleDxbcPS);
    sh[1].entryPointName = "fragmentMain";
    VriColorAttachmentDesc ca {};
    ca.format         = VriFormat_S8_UINT;
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
    CHECK(d.c.CreateGraphicsPipeline(d.dev, &pd, &pipeline) == VriResult_Unsupported);
    d.c.DestroyPipelineLayout(layout);
}

TEST_CASE("D3D12: a BGRA8_SRGB target stores sRGB-encoded values")
{
    D3D12Device d;
    if (!d.dev)
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    VriCoreInterface& c     = d.c;
    VriQueue*         queue = nullptr;
    REQUIRE(c.GetQueue(d.dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);
    VriTextureDesc td = Texture2D(VriFormat_BGRA8_SRGB, VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc);
    td.width          = 64; // 256-byte rows: the readback row pitch D3D12 requires
    for (float& v : td.clearValue.color.f32)
        v = 0.5f;
    VriTexture* target = nullptr;
    REQUIRE(c.CreateTexture(d.dev, &td, &target) == VriResult_Success);
    VriTextureViewDesc vd {};
    vd.texture          = target;
    vd.viewType         = VriTextureViewType_2D;
    vd.aspect           = VriImageAspect_Color;
    VriDescriptor* view = nullptr;
    REQUIRE(c.CreateTextureView(d.dev, &vd, &view) == VriResult_Success);
    VriBufferDesc bd {};
    bd.size             = uint64_t(td.width) * td.height * 4;
    bd.usage            = VriBufferUsage_TransferDst;
    bd.memoryLocation   = VriMemoryLocation_HostReadback;
    VriBuffer* readback = nullptr;
    REQUIRE(c.CreateBuffer(d.dev, &bd, &readback) == VriResult_Success);
    VriCommandAllocator* alloc = nullptr;
    REQUIRE(c.CreateCommandAllocator(d.dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriCommandBuffer* cmd = nullptr;
    REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(d.dev, 0, &fence) == VriResult_Success);

    REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
    VriTextureBarrierDesc tb {};
    tb.texture      = target;
    tb.after.layout = VriLayout_ColorAttachment;
    tb.after.access = VriAccess_ColorAttachmentWrite;
    tb.after.stages = VriPipelineStage_ColorAttachmentOutput;
    tb.aspect       = VriImageAspect_Color;
    VriBarrierGroupDesc g {};
    g.textures   = &tb;
    g.textureNum = 1;
    c.CmdBarrier(cmd, &g);
    VriAttachmentDesc rt {};
    rt.view       = view;
    rt.loadOp     = VriAttachmentLoadOp_Clear;
    rt.storeOp    = VriAttachmentStoreOp_Store;
    rt.clearValue = td.clearValue;
    VriAttachmentsDesc att {};
    att.colors            = &rt;
    att.colorNum          = 1;
    att.renderArea.width  = td.width;
    att.renderArea.height = td.height;
    att.layerNum          = 1;
    c.CmdBeginRendering(cmd, &att);
    c.CmdEndRendering(cmd);
    tb.before       = tb.after;
    tb.after.layout = VriLayout_CopySource;
    tb.after.access = VriAccess_CopySourceRead;
    tb.after.stages = VriPipelineStage_Transfer;
    c.CmdBarrier(cmd, &g);
    VriBufferTextureCopyDesc cp {};
    cp.texture.aspect   = VriImageAspect_Color;
    cp.texture.layerNum = 1;
    c.CmdReadbackTextureToBuffer(cmd, readback, target, &cp);
    REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
    VriFenceSubmitDesc signal {fence, 1};
    VriQueueSubmitDesc submit {};
    submit.commandBuffers   = &cmd;
    submit.commandBufferNum = 1;
    submit.signalFences     = &signal;
    submit.signalFenceNum   = 1;
    REQUIRE(c.QueueSubmit(queue, &submit) == VriResult_Success);
    c.Wait(fence, 1);

    const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bd.size));
    REQUIRE(px != nullptr);
    // Linear 0.5 encodes to sRGB 188 (0.7354); alpha is stored linearly (128).
    CHECK(px[0] >= 187);
    CHECK(px[0] <= 189);
    CHECK(px[3] >= 127);
    CHECK(px[3] <= 128);
    c.UnmapBuffer(readback);

    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    c.DestroyDescriptor(view);
    c.DestroyBuffer(readback);
    c.DestroyTexture(target);
}
