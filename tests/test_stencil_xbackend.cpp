// Cross-backend stencil parity. Two passes into one depth-stencil (D24S8) target:
//  1) a small triangle writes stencil=1 (compareOp=Always, passOp=Replace, ref=1);
//  2) a large green triangle draws only where stencil==1 (compareOp=Equal, ref=1).
// The large triangle covers the small one's area plus more, so:
//  - center (inside both): green  (stencil==1 -> pass 2 draws)
//  - off-center (inside large only): black (stencil==0 -> pass 2 rejected)
// This exercises front/back stencil ops, compare/write masks, and the reference -
// all of which the backends previously ignored. Runs on Vulkan, D3D12, WebGPU, OpenGL
// (D3D12 stencil: PSO stencil state + dynamic OMSetStencilRef + a stencil-plane clear).
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <utility>

#include "shaders/triangle_vbuf_spv.h"  // g_triangleVbufSpv  (position.xyz + color)
#include "shaders/triangle_vbuf_wgsl.h" // g_triangleVbufWgsl
#include "shaders/triangle_vbuf_dxbc.h" // g_triangleVbufDxbcVS / PS  (D3D12)

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    // Pass 1 writes stencil=1; pass 2 tests stencil==1. front==back (cullMode None).
    VriPipeline* MakePipeline(const VriCoreInterface& c, VriDevice* dev, VriPipelineLayout* layout,
                              const void* vs, size_t vsSize, const void* ps, size_t psSize, bool write)
    {
        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = vs; sh[0].bytecodeSize = vsSize; sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = ps; sh[1].bytecodeSize = psSize; sh[1].entryPointName = "fragmentMain";
        VriVertexStreamDesc stream{}; stream.stride = 6 * sizeof(float); stream.bindingSlot = 0; stream.stepRate = VriVertexStepRate_PerVertex;
        VriVertexAttributeDesc attrs[2]{};
        attrs[0].format = VriFormat_RGB32_SFLOAT; attrs[0].offset = 0;                 attrs[0].streamIndex = 0;
        attrs[1].format = VriFormat_RGB32_SFLOAT; attrs[1].offset = 3 * sizeof(float); attrs[1].streamIndex = 0;
        VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;

        VriStencilOpDesc face{};
        face.compareMask = 0xFFu;
        face.reference = 1u;
        if (write) { face.compareOp = VriCompareOp_Always; face.passOp = VriStencilOp_Replace; face.writeMask = 0xFFu; }
        else       { face.compareOp = VriCompareOp_Equal;  face.passOp = VriStencilOp_Keep;    face.writeMask = 0x00u; }
        face.failOp = VriStencilOp_Keep; face.depthFailOp = VriStencilOp_Keep;

        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.vertexInput.streams = &stream; pd.vertexInput.streamNum = 1;
        pd.vertexInput.attributes = attrs; pd.vertexInput.attributeNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1;
        pd.depthStencil.stencilTest = VRI_TRUE; pd.depthStencil.front = face; pd.depthStencil.back = face;
        pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
        pd.outputMerger.depthStencilFormat = VriFormat_D24_UNORM_S8_UINT;
        VriPipeline* p = nullptr;
        REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &p) == VriResult_Success);
        return p;
    }

    bool RunStencil(VriGraphicsAPI api, const void* vs, size_t vsSize, const void* ps, size_t psSize, bool& ran)
    {
        ran = false;
        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = api; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;
        ran = true;

        VriCoreInterface c{};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        VriTextureDesc td{};
        td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM;
        td.width = kW; td.height = kH; td.depth = 1; td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1;
        td.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc; td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* color = nullptr; REQUIRE(c.CreateTexture(dev, &td, &color) == VriResult_Success);
        VriTextureViewDesc cvd{}; cvd.texture = color; cvd.viewType = VriTextureViewType_2D; cvd.format = VriFormat_Unknown; cvd.aspect = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr; REQUIRE(c.CreateTextureView(dev, &cvd, &colorView) == VriResult_Success);

        VriTextureDesc dsd{};
        dsd.type = VriTextureType_2D; dsd.format = VriFormat_D24_UNORM_S8_UINT;
        dsd.width = kW; dsd.height = kH; dsd.depth = 1; dsd.mipNum = 1; dsd.layerNum = 1; dsd.sampleNum = 1;
        dsd.usage = VriTextureUsage_DepthStencilAttachment; dsd.memoryLocation = VriMemoryLocation_Device;
        VriTexture* depthStencil = nullptr; REQUIRE(c.CreateTexture(dev, &dsd, &depthStencil) == VriResult_Success);
        VriTextureViewDesc dvd{}; dvd.texture = depthStencil; dvd.viewType = VriTextureViewType_2D; dvd.format = VriFormat_Unknown; dvd.aspect = VriImageAspect_Depth | VriImageAspect_Stencil;
        VriDescriptor* dsView = nullptr; REQUIRE(c.CreateTextureView(dev, &dvd, &dsView) == VriResult_Success);

        VriBufferDesc rbd{}; rbd.size = static_cast<uint64_t>(kW) * kH * 4; rbd.usage = VriBufferUsage_TransferDst; rbd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr; REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

        // small (writes stencil), large green (tested) - same apex-down shape, scaled.
        const float small_[] = {
            0.0f, -0.3f, 0.0f, 1.0f, 0.0f, 0.0f,
            0.3f,  0.3f, 0.0f, 1.0f, 0.0f, 0.0f,
           -0.3f,  0.3f, 0.0f, 1.0f, 0.0f, 0.0f,
        };
        const float large[] = {
            0.0f, -0.9f, 0.0f, 0.0f, 1.0f, 0.0f,
            0.9f,  0.9f, 0.0f, 0.0f, 1.0f, 0.0f,
           -0.9f,  0.9f, 0.0f, 0.0f, 1.0f, 0.0f,
        };
        auto makeVbuf = [&](const float* data) {
            const uint64_t sz = 18 * sizeof(float);
            VriBufferDesc vd{}; vd.size = sz; vd.usage = VriBufferUsage_VertexBuffer | VriBufferUsage_TransferDst; vd.memoryLocation = VriMemoryLocation_Device;
            VriBuffer* vb = nullptr; REQUIRE(c.CreateBuffer(dev, &vd, &vb) == VriResult_Success);
            VriBufferDesc sd{}; sd.size = sz; sd.usage = VriBufferUsage_TransferSrc; sd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* st = nullptr; REQUIRE(c.CreateBuffer(dev, &sd, &st) == VriResult_Success);
            void* m = c.MapBuffer(st, 0, sz); REQUIRE(m != nullptr);
            for (uint64_t i = 0; i < sz; ++i) static_cast<uint8_t*>(m)[i] = reinterpret_cast<const uint8_t*>(data)[i];
            c.UnmapBuffer(st);
            return std::pair<VriBuffer*, VriBuffer*>{vb, st};
        };
        auto [vbSmall, stSmall] = makeVbuf(small_);
        auto [vbLarge, stLarge] = makeVbuf(large);

        VriPipelineLayoutDesc ld{}; VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);
        VriPipeline* writePipe = MakePipeline(c, dev, layout, vs, vsSize, ps, psSize, true);
        VriPipeline* testPipe = MakePipeline(c, dev, layout, vs, vsSize, ps, psSize, false);

        VriCommandAllocator* alloc = nullptr; REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr; REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr; REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        VriBufferCopyDesc cp{}; cp.size = 18 * sizeof(float);
        c.CmdCopyBuffer(cmd, vbSmall, stSmall, &cp);
        c.CmdCopyBuffer(cmd, vbLarge, stLarge, &cp);
        {
            // copy-dest -> vertex buffer (D3D12 validates this transition; the others are lenient)
            VriBufferBarrierDesc bb[2]{};
            for (int i = 0; i < 2; ++i)
            {
                bb[i].buffer = i == 0 ? vbSmall : vbLarge;
                bb[i].before.access = VriAccess_CopyDestinationWrite; bb[i].before.stages = VriPipelineStage_Transfer;
                bb[i].after.access = VriAccess_VertexBufferRead; bb[i].after.stages = VriPipelineStage_VertexInput;
            }
            VriBarrierGroupDesc gb{}; gb.buffers = bb; gb.bufferNum = 2; c.CmdBarrier(cmd, &gb);
        }
        {
            VriTextureBarrierDesc tb[2]{};
            tb[0].texture = color; tb[0].before.layout = VriLayout_Undefined; tb[0].before.stages = VriPipelineStage_None;
            tb[0].after.access = VriAccess_ColorAttachmentWrite; tb[0].after.layout = VriLayout_ColorAttachment; tb[0].after.stages = VriPipelineStage_ColorAttachmentOutput;
            tb[0].aspect = VriImageAspect_Color;
            tb[1].texture = depthStencil; tb[1].before.layout = VriLayout_Undefined; tb[1].before.stages = VriPipelineStage_None;
            tb[1].after.access = VriAccess_DepthStencilAttachmentWrite; tb[1].after.layout = VriLayout_DepthStencilAttachment; tb[1].after.stages = VriPipelineStage_EarlyFragmentTests;
            tb[1].aspect = VriImageAspect_Depth | VriImageAspect_Stencil;
            VriBarrierGroupDesc g{}; g.textures = tb; g.textureNum = 2; c.CmdBarrier(cmd, &g);
        }

        VriAttachmentDesc rt{}; rt.view = colorView; rt.loadOp = VriAttachmentLoadOp_Clear; rt.storeOp = VriAttachmentStoreOp_Store; rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentDesc dsAtt{}; dsAtt.view = dsView; dsAtt.loadOp = VriAttachmentLoadOp_Clear; dsAtt.storeOp = VriAttachmentStoreOp_Store;
        dsAtt.clearValue.depthStencil.depth = 1.0f; dsAtt.clearValue.depthStencil.stencil = 0u;
        VriAttachmentsDesc att{}; att.colors = &rt; att.colorNum = 1; att.depth = &dsAtt; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
        VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 1;
        { VriVertexBufferBinding vbb{}; vbb.buffer = vbSmall; vbb.offset = 0; c.CmdSetPipeline(cmd, writePipe); c.CmdSetVertexBuffers(cmd, 0, &vbb, 1); c.CmdDraw(cmd, &draw); }
        { VriVertexBufferBinding vbb{}; vbb.buffer = vbLarge; vbb.offset = 0; c.CmdSetPipeline(cmd, testPipe);  c.CmdSetVertexBuffers(cmd, 0, &vbb, 1); c.CmdDraw(cmd, &draw); }
        c.CmdEndRendering(cmd);

        {
            VriTextureBarrierDesc tb{};
            tb.texture = color; tb.before.access = VriAccess_ColorAttachmentWrite; tb.before.layout = VriLayout_ColorAttachment; tb.before.stages = VriPipelineStage_ColorAttachmentOutput;
            tb.after.access = VriAccess_CopySourceRead; tb.after.layout = VriLayout_CopySource; tb.after.stages = VriPipelineStage_Transfer;
            tb.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc g{}; g.textures = &tb; g.textureNum = 1; c.CmdBarrier(cmd, &g);
        }
        VriBufferTextureCopyDesc tc{}; tc.texture.aspect = VriImageAspect_Color; tc.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &tc);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

        VriFenceSubmitDesc signal{}; signal.fence = fence; signal.value = 1;
        VriQueueSubmitDesc submit{}; submit.commandBuffers = &cmd; submit.commandBufferNum = 1; submit.signalFences = &signal; submit.signalFenceNum = 1;
        c.QueueSubmit(queue, &submit);
        c.Wait(fence, 1);

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
        REQUIRE(px != nullptr);
        const uint32_t oc = ((kH / 2) * kW + (kW / 2)) * 4;       // center: inside both -> green
        const uint32_t oo = ((kH / 2) * kW + 42) * 4;             // off-center: inside large only -> black (stencil rejected)
        const bool centerGreen = px[oc + 0] == 0 && px[oc + 1] == 255 && px[oc + 2] == 0;
        const bool offBlack = px[oo + 0] == 0 && px[oo + 1] == 0 && px[oo + 2] == 0;
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(writePipe); c.DestroyPipeline(testPipe); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(vbSmall); c.DestroyBuffer(vbLarge); c.DestroyBuffer(stSmall); c.DestroyBuffer(stLarge);
        c.DestroyBuffer(readback); c.DestroyDescriptor(colorView); c.DestroyDescriptor(dsView); c.DestroyTexture(color); c.DestroyTexture(depthStencil);
        vriDestroyDevice(dev);
        return centerGreen && offBlack;
    }
} // namespace

TEST_CASE("stencil parity: write stencil then test (center green, off-center stencil-rejected)")
{
    bool ran = false;
    const bool vk = RunStencil(VriGraphicsAPI_Vulkan, g_triangleVbufSpv, sizeof(g_triangleVbufSpv), g_triangleVbufSpv, sizeof(g_triangleVbufSpv), ran);
    if (ran) { CHECK(vk); } else { MESSAGE("Vulkan unavailable - skipped"); }

    const bool d3d12 = RunStencil(VriGraphicsAPI_D3D12, g_triangleVbufDxbcVS, sizeof(g_triangleVbufDxbcVS), g_triangleVbufDxbcPS, sizeof(g_triangleVbufDxbcPS), ran);
    if (ran) { CHECK(d3d12); } else { MESSAGE("D3D12 unavailable - skipped"); }

    const bool wgpu = RunStencil(VriGraphicsAPI_WebGPU, g_triangleVbufWgsl, sizeof(g_triangleVbufWgsl), g_triangleVbufWgsl, sizeof(g_triangleVbufWgsl), ran);
    if (ran) { CHECK(wgpu); } else { MESSAGE("WebGPU unavailable - skipped"); }

    const bool gl = RunStencil(VriGraphicsAPI_OpenGL, g_triangleVbufSpv, sizeof(g_triangleVbufSpv), g_triangleVbufSpv, sizeof(g_triangleVbufSpv), ran);
    if (ran) { CHECK(gl); } else { MESSAGE("OpenGL unavailable - skipped"); }
}
