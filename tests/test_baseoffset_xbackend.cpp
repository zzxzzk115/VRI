// Cross-backend base-vertex parity: CmdDrawIndexed with vertexOffset must offset the
// vertex fetch. The vertex buffer holds two triangles at the same screen position - a
// RED decoy at vertices 0-2 and the real GREEN triangle at vertices 3-5. Drawing index
// list [0,1,2] with vertexOffset=3 must fetch vertices 3-5, painting the center GREEN.
// A backend that silently drops vertexOffset would fetch 0-2 and paint it RED - which
// this test rejects.
//
// VK/WebGPU honor vertexOffset/baseVertex natively; the GL backend now honors it via
// glDrawElements(Instanced)BaseVertex (GL 3.2, all desktop incl. macOS 4.1). The
// shader's SV_VertexID is local (D3D semantics, base excluded), so the base offset is
// observable through attribute fetching here, not through the vertex-id builtin.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/triangle_vbuf_spv.h"  // g_triangleVbufSpv  (Vulkan + OpenGL via SPIRV-Cross)
#include "shaders/triangle_vbuf_wgsl.h" // g_triangleVbufWgsl (WebGPU)

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    // Interleaved [pos.xyz, color.rgb]. Same apex-down triangle twice: vertices 0-2 RED
    // (decoy), vertices 3-5 GREEN (real). vertexOffset=3 selects the green set.
    const float kVertices[] = {
        0.0f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.5f,  0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
       -0.5f,  0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.5f,  0.5f, 0.0f, 0.0f, 1.0f, 0.0f,
       -0.5f,  0.5f, 0.0f, 0.0f, 1.0f, 0.0f,
    };
    const uint32_t kIndices[] = {0, 1, 2};

    // Returns true when vertexOffset=3 fetched the green vertices (center is green).
    bool RunBaseVertex(VriGraphicsAPI api, const void* shader, size_t shaderSize, bool& ran)
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
        td.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* color = nullptr;
        REQUIRE(c.CreateTexture(dev, &td, &color) == VriResult_Success);
        VriTextureViewDesc cvd{};
        cvd.texture = color; cvd.viewType = VriTextureViewType_2D; cvd.format = VriFormat_Unknown; cvd.aspect = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr;
        REQUIRE(c.CreateTextureView(dev, &cvd, &colorView) == VriResult_Success);

        VriBufferDesc rbd{};
        rbd.size = static_cast<uint64_t>(kW) * kH * 4; rbd.usage = VriBufferUsage_TransferDst; rbd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

        auto makeDeviceBuffer = [&](uint64_t size, VriBufferUsageFlags usage) {
            VriBufferDesc bd{}; bd.size = size; bd.usage = usage | VriBufferUsage_TransferDst; bd.memoryLocation = VriMemoryLocation_Device;
            VriBuffer* b = nullptr; REQUIRE(c.CreateBuffer(dev, &bd, &b) == VriResult_Success); return b;
        };
        auto makeStaging = [&](const void* data, uint64_t size) {
            VriBufferDesc bd{}; bd.size = size; bd.usage = VriBufferUsage_TransferSrc; bd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* b = nullptr; REQUIRE(c.CreateBuffer(dev, &bd, &b) == VriResult_Success);
            void* m = c.MapBuffer(b, 0, size); REQUIRE(m != nullptr);
            for (uint64_t i = 0; i < size; ++i) static_cast<uint8_t*>(m)[i] = static_cast<const uint8_t*>(data)[i];
            c.UnmapBuffer(b); return b;
        };
        VriBuffer* vbuf = makeDeviceBuffer(sizeof(kVertices), VriBufferUsage_VertexBuffer);
        VriBuffer* ibuf = makeDeviceBuffer(sizeof(kIndices), VriBufferUsage_IndexBuffer);
        VriBuffer* vStaging = makeStaging(kVertices, sizeof(kVertices));
        VriBuffer* iStaging = makeStaging(kIndices, sizeof(kIndices));

        VriPipelineLayoutDesc ld{};
        VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = shader; sh[0].bytecodeSize = shaderSize; sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = shader; sh[1].bytecodeSize = shaderSize; sh[1].entryPointName = "fragmentMain";

        VriVertexStreamDesc stream{};
        stream.stride = 6 * sizeof(float); stream.bindingSlot = 0; stream.stepRate = VriVertexStepRate_PerVertex;
        VriVertexAttributeDesc attrs[2]{};
        attrs[0].format = VriFormat_RGB32_SFLOAT; attrs[0].offset = 0;                attrs[0].streamIndex = 0; // POSITION -> location 0
        attrs[1].format = VriFormat_RGB32_SFLOAT; attrs[1].offset = 3 * sizeof(float); attrs[1].streamIndex = 0; // COLOR0  -> location 1

        VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.vertexInput.streams = &stream; pd.vertexInput.streamNum = 1;
        pd.vertexInput.attributes = attrs; pd.vertexInput.attributeNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
        VriPipeline* pipeline = nullptr;
        REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr; REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr; REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr; REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        VriBufferCopyDesc vcopy{}; vcopy.size = sizeof(kVertices); c.CmdCopyBuffer(cmd, vbuf, vStaging, &vcopy);
        VriBufferCopyDesc icopy{}; icopy.size = sizeof(kIndices); c.CmdCopyBuffer(cmd, ibuf, iStaging, &icopy);
        {
            VriBufferBarrierDesc bb[2]{};
            bb[0].buffer = vbuf; bb[0].before.access = VriAccess_CopyDestinationWrite; bb[0].before.stages = VriPipelineStage_Transfer;
            bb[0].after.access = VriAccess_VertexBufferRead; bb[0].after.stages = VriPipelineStage_VertexInput;
            bb[1].buffer = ibuf; bb[1].before.access = VriAccess_CopyDestinationWrite; bb[1].before.stages = VriPipelineStage_Transfer;
            bb[1].after.access = VriAccess_IndexBufferRead; bb[1].after.stages = VriPipelineStage_VertexInput;
            VriBarrierGroupDesc g{}; g.buffers = bb; g.bufferNum = 2; c.CmdBarrier(cmd, &g);
        }
        {
            VriTextureBarrierDesc tb{};
            tb.texture = color; tb.before.layout = VriLayout_Undefined; tb.before.stages = VriPipelineStage_None;
            tb.after.access = VriAccess_ColorAttachmentWrite; tb.after.layout = VriLayout_ColorAttachment; tb.after.stages = VriPipelineStage_ColorAttachmentOutput;
            tb.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc g{}; g.textures = &tb; g.textureNum = 1; c.CmdBarrier(cmd, &g);
        }

        VriAttachmentDesc rt{}; rt.view = colorView; rt.loadOp = VriAttachmentLoadOp_Clear; rt.storeOp = VriAttachmentStoreOp_Store; rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att{}; att.colors = &rt; att.colorNum = 1; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        VriVertexBufferBinding vbb{}; vbb.buffer = vbuf; vbb.offset = 0;
        c.CmdSetVertexBuffers(cmd, 0, &vbb, 1);
        c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt32);
        VriDrawIndexedDesc di{}; di.indexNum = 3; di.instanceNum = 1; di.vertexOffset = 3; // fetch the GREEN vertices 3-5
        c.CmdDrawIndexed(cmd, &di);
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
        const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
        const bool green = (px[o + 0] == 0 && px[o + 1] == 255 && px[o + 2] == 0);
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(vbuf); c.DestroyBuffer(ibuf); c.DestroyBuffer(vStaging); c.DestroyBuffer(iStaging);
        c.DestroyBuffer(readback); c.DestroyDescriptor(colorView); c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return green;
    }
} // namespace

TEST_CASE("base-vertex parity: CmdDrawIndexed vertexOffset=3 fetches the green vertices (not the red decoy)")
{
    bool ran = false;
    const bool vk = RunBaseVertex(VriGraphicsAPI_Vulkan, g_triangleVbufSpv, sizeof(g_triangleVbufSpv), ran);
    if (ran) { CHECK(vk); } else { MESSAGE("Vulkan unavailable - skipped"); }

    const bool wgpu = RunBaseVertex(VriGraphicsAPI_WebGPU, g_triangleVbufWgsl, sizeof(g_triangleVbufWgsl), ran);
    if (ran) { CHECK(wgpu); } else { MESSAGE("WebGPU unavailable - skipped"); }

    const bool gl = RunBaseVertex(VriGraphicsAPI_OpenGL, g_triangleVbufSpv, sizeof(g_triangleVbufSpv), ran);
    if (ran) { CHECK(gl); } else { MESSAGE("OpenGL unavailable - skipped"); }
}
