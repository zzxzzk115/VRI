// webgpu_triangle.cpp - WebGPU-on-Emscripten (browser WebGPU) end-to-end check.
// Renders the solid-red triangle to an offscreen texture via the WebGPU backend,
// reads it back, and prints a pass/fail line for a headless-browser grade. Device
// creation + buffer mapping are async in the browser; the backend yields via
// emscripten_sleep (ASYNCIFY) so they resolve.
//
// Build: xmake f -p wasm --vri_backend_wgpu=y; xmake build vri-webgpu-triangle
// Run:   emrun (headless Chrome with --enable-unsafe-webgpu) over the .html

#include <vri/vri.h>

#include <cstdint>
#include <cstdio>

#include "shaders/triangle_wgsl.h" // g_triangleWgsl (solid red)

#if defined(__EMSCRIPTEN__)
#    include <emscripten/emscripten.h>
#endif

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    bool RenderAndProbe(bool& centerRed)
    {
        centerRed = false;
        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = VriGraphicsAPI_WebGPU;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            std::printf("VRI_WEBGPU: device creation FAILED\n");
            return false;
        }
        std::printf("VRI_WEBGPU_STEP: device-created\n"); std::fflush(stdout);

        VriCoreInterface c{};
        if (vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) != VriResult_Success)
            return false;
        VriQueue* queue = nullptr;
        c.GetQueue(dev, VriQueueType_Graphics, 0, &queue);

        VriTextureDesc td{};
        td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM;
        td.width = kW; td.height = kH; td.depth = 1; td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1;
        td.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* color = nullptr; c.CreateTexture(dev, &td, &color);
        VriTextureViewDesc vdsc{}; vdsc.texture = color; vdsc.viewType = VriTextureViewType_2D; vdsc.format = VriFormat_Unknown; vdsc.aspect = VriImageAspect_Color;
        VriDescriptor* view = nullptr; c.CreateTextureView(dev, &vdsc, &view);

        VriBufferDesc bd{};
        bd.size = static_cast<uint64_t>(kW) * kH * 4; bd.usage = VriBufferUsage_TransferDst; bd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr; c.CreateBuffer(dev, &bd, &readback);

        VriPipelineLayoutDesc ld{}; VriPipelineLayout* layout = nullptr; c.CreatePipelineLayout(dev, &ld, &layout);
        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = g_triangleWgsl; sh[0].bytecodeSize = sizeof(g_triangleWgsl); sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = g_triangleWgsl; sh[1].bytecodeSize = sizeof(g_triangleWgsl); sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
        VriPipeline* pipeline = nullptr;
        if (c.CreateGraphicsPipeline(dev, &pd, &pipeline) != VriResult_Success)
        {
            std::printf("VRI_WEBGPU: pipeline creation FAILED\n");
            return false;
        }
        std::printf("VRI_WEBGPU_STEP: pipeline-created\n"); std::fflush(stdout);

        VriCommandAllocator* alloc = nullptr; c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
        VriCommandBuffer* cmd = nullptr; c.CreateCommandBuffer(alloc, &cmd);
        VriFence* fence = nullptr; c.CreateFence(dev, 0, &fence);

        c.BeginCommandBuffer(cmd);
        {
            VriTextureBarrierDesc tb{};
            tb.texture = color; tb.before.layout = VriLayout_Undefined; tb.before.stages = VriPipelineStage_None;
            tb.after.access = VriAccess_ColorAttachmentWrite; tb.after.layout = VriLayout_ColorAttachment; tb.after.stages = VriPipelineStage_ColorAttachmentOutput;
            tb.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc g{}; g.textures = &tb; g.textureNum = 1; c.CmdBarrier(cmd, &g);
        }
        VriAttachmentDesc rt{}; rt.view = view; rt.loadOp = VriAttachmentLoadOp_Clear; rt.storeOp = VriAttachmentStoreOp_Store; rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att{}; att.colors = &rt; att.colorNum = 1; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 1; c.CmdDraw(cmd, &draw);
        c.CmdEndRendering(cmd);
        {
            VriTextureBarrierDesc tb{};
            tb.texture = color; tb.before.access = VriAccess_ColorAttachmentWrite; tb.before.layout = VriLayout_ColorAttachment; tb.before.stages = VriPipelineStage_ColorAttachmentOutput;
            tb.after.access = VriAccess_CopySourceRead; tb.after.layout = VriLayout_CopySource; tb.after.stages = VriPipelineStage_Transfer;
            tb.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc g{}; g.textures = &tb; g.textureNum = 1; c.CmdBarrier(cmd, &g);
        }
        VriBufferTextureCopyDesc copy{}; copy.texture.aspect = VriImageAspect_Color; copy.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &copy);
        c.EndCommandBuffer(cmd);

        VriFenceSubmitDesc sig{}; sig.fence = fence; sig.value = 1;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(queue, &sub);
        c.Wait(fence, 1);
        std::printf("VRI_WEBGPU_STEP: submitted\n"); std::fflush(stdout);

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bd.size));
        const bool ok = px != nullptr;
        if (ok)
        {
            const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
            centerRed = px[o + 0] == 255 && px[o + 1] == 0 && px[o + 2] == 0;
            c.UnmapBuffer(readback);
        }

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(readback); c.DestroyDescriptor(view); c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return ok;
    }
} // namespace

int main()
{
    std::printf("VRI_WEBGPU_STEP: main\n"); std::fflush(stdout);
    bool centerRed = false;
    const bool ran = RenderAndProbe(centerRed);
    const bool pass = ran && centerRed;
    std::printf("VRI_WEBGPU_RESULT: %s (ran=%d centerRed=%d)\n", pass ? "PASS" : "FAIL", ran ? 1 : 0, centerRed ? 1 : 0);
    std::fflush(stdout);
#if defined(__EMSCRIPTEN__)
    emscripten_force_exit(pass ? 0 : 1);
#endif
    return pass ? 0 : 1;
}
