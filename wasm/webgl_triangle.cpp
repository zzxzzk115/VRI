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

#include "shaders/triangle_spv.h"     // g_triangleSpv     (solid red, vertex-id only)
#include "shaders/triangle_ubo_spv.h" // g_triangleUboSpv  (color from a UBO descriptor)

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

    // Renders the UBO-tinted triangle and returns whether the center pixel is the
    // UBO color (green) -- exercises the GL descriptor-set flattening on WebGL.
    bool RenderUboAndProbe(bool& centerGreen)
    {
        centerGreen = false;

        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = VriGraphicsAPI_OpenGLES;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;

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
        VriTextureViewDesc vd{}; vd.texture = color; vd.viewType = VriTextureViewType_2D; vd.format = VriFormat_Unknown; vd.aspect = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr; c.CreateTextureView(dev, &vd, &colorView);

        VriBufferDesc rb{}; rb.size = static_cast<uint64_t>(kW) * kH * 4; rb.usage = VriBufferUsage_TransferDst; rb.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr; c.CreateBuffer(dev, &rb, &readback);

        VriBufferDesc sb{}; sb.size = 16; sb.usage = VriBufferUsage_TransferSrc; sb.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* staging = nullptr; c.CreateBuffer(dev, &sb, &staging);
        {
            float* m = static_cast<float*>(c.MapBuffer(staging, 0, 16));
            if (!m) return false;
            m[0] = 0.0f; m[1] = 1.0f; m[2] = 0.0f; m[3] = 1.0f; // green
            c.UnmapBuffer(staging);
        }
        VriBufferDesc ub{}; ub.size = 16; ub.usage = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst; ub.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* ubo = nullptr; c.CreateBuffer(dev, &ub, &ubo);

        VriDescriptorRangeDesc range{}; range.baseRegister = 0; range.descriptorNum = 1; range.descriptorType = VriDescriptorType_ConstantBuffer; range.shaderStages = VriShaderStage_Fragment;
        VriDescriptorSetDesc setDesc{}; setDesc.registerSpace = 0; setDesc.ranges = &range; setDesc.rangeNum = 1;
        VriPipelineLayoutDesc ld{}; ld.descriptorSets = &setDesc; ld.descriptorSetNum = 1;
        VriPipelineLayout* layout = nullptr; c.CreatePipelineLayout(dev, &ld, &layout);

        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = g_triangleUboSpv; sh[0].bytecodeSize = sizeof(g_triangleUboSpv); sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = g_triangleUboSpv; sh[1].bytecodeSize = sizeof(g_triangleUboSpv); sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
        VriPipeline* pipeline = nullptr;
        if (c.CreateGraphicsPipeline(dev, &pd, &pipeline) != VriResult_Success)
        {
            std::printf("VRI_WEBGL: UBO pipeline creation FAILED\n");
            return false;
        }

        VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 1; pdsc.constantBufferMaxNum = 1;
        VriDescriptorPool* pool = nullptr; c.CreateDescriptorPool(dev, &pdsc, &pool);
        VriDescriptorSet* set = nullptr; c.AllocateDescriptorSets(pool, layout, 0, &set, 1);
        VriBufferViewDesc uboView{}; uboView.buffer = ubo; uboView.viewType = VriDescriptorType_ConstantBuffer; uboView.offset = 0; uboView.size = 16;
        VriDescriptor* uboDescriptor = nullptr; c.CreateBufferView(dev, &uboView, &uboDescriptor);
        VriDescriptorRangeUpdateDesc upd{}; const VriDescriptor* descs[1] = {uboDescriptor}; upd.descriptors = descs; upd.descriptorNum = 1; upd.baseDescriptor = 0;
        c.UpdateDescriptorRanges(set, 0, 1, &upd);

        VriCommandAllocator* alloc = nullptr; c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
        VriCommandBuffer* cmd = nullptr; c.CreateCommandBuffer(alloc, &cmd);
        VriFence* fence = nullptr; c.CreateFence(dev, 0, &fence);

        c.BeginCommandBuffer(cmd);
        VriBufferCopyDesc copy{}; copy.size = 16; c.CmdCopyBuffer(cmd, ubo, staging, &copy);
        VriAttachmentDesc rt{}; rt.view = colorView; rt.loadOp = VriAttachmentLoadOp_Clear; rt.storeOp = VriAttachmentStoreOp_Store; rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att{}; att.colors = &rt; att.colorNum = 1; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, set);
        VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 1; c.CmdDraw(cmd, &draw);
        c.CmdEndRendering(cmd);
        VriBufferTextureCopyDesc tc{}; tc.texture.aspect = VriImageAspect_Color; tc.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &tc);
        c.EndCommandBuffer(cmd);

        VriFenceSubmitDesc sig{}; sig.fence = fence; sig.value = 1;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(queue, &sub);
        c.Wait(fence, 1);

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rb.size));
        bool ok = px != nullptr;
        if (ok)
        {
            const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
            centerGreen = (px[o + 0] == 0 && px[o + 1] == 255 && px[o + 2] == 0);
            c.UnmapBuffer(readback);
        }

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyDescriptor(uboDescriptor); c.DestroyDescriptorPool(pool);
        c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(ubo); c.DestroyBuffer(staging); c.DestroyBuffer(readback);
        c.DestroyDescriptor(colorView); c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return ok;
    }
} // namespace

int main()
{
    std::printf("VRI_WEBGL_STEP: main\n"); std::fflush(stdout);
    bool topRed = false, botRed = false;
    const bool ranTri = RenderAndProbe(topRed, botRed);
    const bool triPass = ranTri && topRed && !botRed;

    bool centerGreen = false;
    const bool ranUbo = RenderUboAndProbe(centerGreen);
    const bool uboPass = ranUbo && centerGreen;

    const bool pass = triPass && uboPass;
    std::printf("VRI_WEBGL_RESULT: %s (triangle: ran=%d topRed=%d botRed=%d | ubo: ran=%d green=%d)\n",
                pass ? "PASS" : "FAIL", ranTri, topRed, botRed, ranUbo, centerGreen);
    std::fflush(stdout);
#if defined(__EMSCRIPTEN__)
    // Make emrun return the process exit code so a headless run can be graded.
    emscripten_force_exit(pass ? 0 : 1);
#endif
    return pass ? 0 : 1;
}
