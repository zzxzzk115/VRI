// webgl_triangle.cpp - WebGL (Emscripten) end-to-end check for the VRI GL backend.
//
// Creates a VRI device on the OpenGL ES / WebGL2 backend, renders the solid-red
// apex-down triangle to an offscreen texture, reads it back, and verifies the
// Y-up orientation (same probe pixels as tests/test_coordsys_xbackend.cpp). Prints
// a single machine-parseable result line so a headless browser run can grade it.
//
// Build: xmake f -p wasm; xmake build vri-webgl-triangle
// Run:   emrun (headless Chrome) over the generated .html

#include <vri/vri.h>

#include <cstdint>
#include <cstdio>

#include "shaders/triangle_spv.h" // g_triangleSpv (consumed by GL via SPIRV-Cross)

#if defined(__EMSCRIPTEN__)
#    include <emscripten/emscripten.h>
#endif

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    // Returns true on success; sets topRed/botRed probes. Mirrors the cross-backend
    // coordinate parity test: topProbe (col44,row20) is inside the wide base near the
    // top -> red iff Y-up; botProbe (col44,row44) near the narrow apex -> red iff flipped.
    void Step(const char* s) { std::printf("VRI_WEBGL_STEP: %s\n", s); std::fflush(stdout); }

    bool RenderAndProbe(bool& topRed, bool& botRed)
    {
        topRed = botRed = false;
        Step("enter");

        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = VriGraphicsAPI_OpenGLES;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            std::printf("VRI_WEBGL: device creation FAILED\n");
            return false;
        }
        Step("device-created");

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
        VriTexture* color = nullptr;
        c.CreateTexture(dev, &td, &color);
        VriTextureViewDesc vdsc{};
        vdsc.texture = color; vdsc.viewType = VriTextureViewType_2D; vdsc.format = VriFormat_Unknown; vdsc.aspect = VriImageAspect_Color;
        VriDescriptor* view = nullptr;
        c.CreateTextureView(dev, &vdsc, &view);

        VriBufferDesc bd{};
        bd.size = static_cast<uint64_t>(kW) * kH * 4; bd.usage = VriBufferUsage_TransferDst; bd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        c.CreateBuffer(dev, &bd, &readback);

        VriPipelineLayoutDesc ld{};
        VriPipelineLayout* layout = nullptr;
        c.CreatePipelineLayout(dev, &ld, &layout);

        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = g_triangleSpv; sh[0].bytecodeSize = sizeof(g_triangleSpv); sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = g_triangleSpv; sh[1].bytecodeSize = sizeof(g_triangleSpv); sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
        Step("resources-created");
        VriPipeline* pipeline = nullptr;
        if (c.CreateGraphicsPipeline(dev, &pd, &pipeline) != VriResult_Success)
        {
            std::printf("VRI_WEBGL: pipeline creation FAILED\n");
            return false;
        }
        Step("pipeline-created");

        VriCommandAllocator* alloc = nullptr; c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
        VriCommandBuffer* cmd = nullptr; c.CreateCommandBuffer(alloc, &cmd);
        VriFence* fence = nullptr; c.CreateFence(dev, 0, &fence);

        c.BeginCommandBuffer(cmd);
        VriAttachmentDesc rt{}; rt.view = view; rt.loadOp = VriAttachmentLoadOp_Clear; rt.storeOp = VriAttachmentStoreOp_Store; rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att{}; att.colors = &rt; att.colorNum = 1; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 1; c.CmdDraw(cmd, &draw);
        c.CmdEndRendering(cmd);
        VriBufferTextureCopyDesc copy{}; copy.texture.aspect = VriImageAspect_Color; copy.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &copy);
        c.EndCommandBuffer(cmd);

        VriFenceSubmitDesc sig{}; sig.fence = fence; sig.value = 1;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(queue, &sub);
        c.Wait(fence, 1);
        Step("submitted");

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bd.size));
        bool ok = px != nullptr;
        if (ok)
        {
            auto isRed = [&](uint32_t col, uint32_t row) {
                const uint32_t o = (row * kW + col) * 4;
                return px[o + 0] == 255 && px[o + 1] == 0 && px[o + 2] == 0;
            };
            topRed = isRed(44, 20);
            botRed = isRed(44, 44);
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
    std::printf("VRI_WEBGL_STEP: main\n"); std::fflush(stdout);
    bool topRed = false, botRed = false;
    const bool ran = RenderAndProbe(topRed, botRed);
    const bool pass = ran && topRed && !botRed;
    std::printf("VRI_WEBGL_RESULT: %s (ran=%d topRed=%d botRed=%d)\n",
                pass ? "PASS" : "FAIL", ran ? 1 : 0, topRed ? 1 : 0, botRed ? 1 : 0);
    std::fflush(stdout);
#if defined(__EMSCRIPTEN__)
    // Make emrun return the process exit code so a headless run can be graded.
    emscripten_force_exit(pass ? 0 : 1);
#endif
    return pass ? 0 : 1;
}
