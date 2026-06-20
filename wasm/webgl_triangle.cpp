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
#include "shaders/triangle_tex_spv.h"  // g_triangleTexSpv  (color sampled from a texture)
#include "shaders/triangle_vbuf_spv.h" // g_triangleVbufSpv (vertex buffer + indexed draw)
#include "shaders/triangle_mrt_spv.h"  // g_triangleMrtSpv  (two render targets)
#include "shaders/triangle_inst_spv.h" // g_triangleInstSpv (instanced draw)

#include <cstdlib>

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

    // Renders a triangle sampling a solid-blue texture (separate Texture + Sampler
    // descriptors) and returns whether the center pixel is blue -- exercises the GL
    // combined-sampler path on WebGL.
    bool RenderTexAndProbe(bool& centerBlue)
    {
        centerBlue = false;

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
        VriTextureViewDesc cvd{}; cvd.texture = color; cvd.viewType = VriTextureViewType_2D; cvd.format = VriFormat_Unknown; cvd.aspect = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr; c.CreateTextureView(dev, &cvd, &colorView);

        constexpr uint32_t kTexW = 64, kTexH = 64;
        VriTextureDesc std_{};
        std_.type = VriTextureType_2D; std_.format = VriFormat_RGBA8_UNORM;
        std_.width = kTexW; std_.height = kTexH; std_.depth = 1; std_.mipNum = 1; std_.layerNum = 1; std_.sampleNum = 1;
        std_.usage = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst;
        std_.memoryLocation = VriMemoryLocation_Device;
        VriTexture* sampled = nullptr; c.CreateTexture(dev, &std_, &sampled);
        VriTextureViewDesc svd{}; svd.texture = sampled; svd.viewType = VriTextureViewType_2D; svd.format = VriFormat_Unknown; svd.aspect = VriImageAspect_Color;
        VriDescriptor* texView = nullptr; c.CreateTextureView(dev, &svd, &texView);
        VriSamplerDesc smpDesc{}; smpDesc.magFilter = VriFilter_Nearest; smpDesc.minFilter = VriFilter_Nearest;
        smpDesc.addressModeU = VriAddressMode_ClampToEdge; smpDesc.addressModeV = VriAddressMode_ClampToEdge; smpDesc.addressModeW = VriAddressMode_ClampToEdge; smpDesc.maxLod = 1.0f;
        VriDescriptor* sampler = nullptr; c.CreateSampler(dev, &smpDesc, &sampler);

        const uint64_t texBytes = static_cast<uint64_t>(kTexW) * kTexH * 4;
        VriBufferDesc sb{}; sb.size = texBytes; sb.usage = VriBufferUsage_TransferSrc; sb.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* staging = nullptr; c.CreateBuffer(dev, &sb, &staging);
        {
            uint8_t* m = static_cast<uint8_t*>(c.MapBuffer(staging, 0, texBytes));
            if (!m) return false;
            for (uint64_t i = 0; i < texBytes; i += 4) { m[i + 0] = 0; m[i + 1] = 0; m[i + 2] = 255; m[i + 3] = 255; }
            c.UnmapBuffer(staging);
        }
        VriBufferDesc rb{}; rb.size = static_cast<uint64_t>(kW) * kH * 4; rb.usage = VriBufferUsage_TransferDst; rb.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr; c.CreateBuffer(dev, &rb, &readback);

        VriDescriptorRangeDesc ranges[2]{};
        ranges[0].baseRegister = 0; ranges[0].descriptorNum = 1; ranges[0].descriptorType = VriDescriptorType_Texture; ranges[0].shaderStages = VriShaderStage_Fragment;
        ranges[1].baseRegister = 1; ranges[1].descriptorNum = 1; ranges[1].descriptorType = VriDescriptorType_Sampler; ranges[1].shaderStages = VriShaderStage_Fragment;
        VriDescriptorSetDesc setDesc{}; setDesc.registerSpace = 0; setDesc.ranges = ranges; setDesc.rangeNum = 2;
        VriPipelineLayoutDesc ld{}; ld.descriptorSets = &setDesc; ld.descriptorSetNum = 1;
        VriPipelineLayout* layout = nullptr; c.CreatePipelineLayout(dev, &ld, &layout);

        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = g_triangleTexSpv; sh[0].bytecodeSize = sizeof(g_triangleTexSpv); sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = g_triangleTexSpv; sh[1].bytecodeSize = sizeof(g_triangleTexSpv); sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
        VriPipeline* pipeline = nullptr;
        if (c.CreateGraphicsPipeline(dev, &pd, &pipeline) != VriResult_Success)
        {
            std::printf("VRI_WEBGL: tex pipeline creation FAILED\n");
            return false;
        }

        VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 1; pdsc.textureMaxNum = 1; pdsc.samplerMaxNum = 1;
        VriDescriptorPool* pool = nullptr; c.CreateDescriptorPool(dev, &pdsc, &pool);
        VriDescriptorSet* set = nullptr; c.AllocateDescriptorSets(pool, layout, 0, &set, 1);
        const VriDescriptor* texDescs[1] = {texView};
        const VriDescriptor* smpDescs[1] = {sampler};
        VriDescriptorRangeUpdateDesc upd[2]{};
        upd[0].descriptors = texDescs; upd[0].descriptorNum = 1; upd[0].baseDescriptor = 0;
        upd[1].descriptors = smpDescs; upd[1].descriptorNum = 1; upd[1].baseDescriptor = 0;
        c.UpdateDescriptorRanges(set, 0, 2, upd);

        VriCommandAllocator* alloc = nullptr; c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
        VriCommandBuffer* cmd = nullptr; c.CreateCommandBuffer(alloc, &cmd);
        VriFence* fence = nullptr; c.CreateFence(dev, 0, &fence);

        c.BeginCommandBuffer(cmd);
        VriBufferTextureCopyDesc up{}; up.bufferOffset = 0; up.texture.aspect = VriImageAspect_Color; up.texture.layerNum = 1; up.texture.width = kTexW; up.texture.height = kTexH;
        c.CmdUploadBufferToTexture(cmd, sampled, staging, &up);
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
            centerBlue = (px[o + 0] == 0 && px[o + 1] == 0 && px[o + 2] == 255);
            c.UnmapBuffer(readback);
        }

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyDescriptorPool(pool); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyDescriptor(sampler); c.DestroyDescriptor(texView); c.DestroyTexture(sampled);
        c.DestroyBuffer(staging); c.DestroyBuffer(readback); c.DestroyDescriptor(colorView); c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return ok;
    }

    // Renders a triangle from a vertex buffer (position + color attributes) via
    // CmdDrawIndexed and returns whether the center pixel is the vertex color
    // (yellow) -- exercises GL vertex input + indexed draw on WebGL.
    bool RenderVbufAndProbe(bool& centerYellow)
    {
        centerYellow = false;
        const float verts[] = {
            0.0f, -0.5f, 0.0f, 1.0f, 1.0f, 0.0f,
            0.5f,  0.5f, 0.0f, 1.0f, 1.0f, 0.0f,
           -0.5f,  0.5f, 0.0f, 1.0f, 1.0f, 0.0f,
        };
        const uint32_t idx[] = {0, 1, 2};

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
        VriTextureViewDesc cvd{}; cvd.texture = color; cvd.viewType = VriTextureViewType_2D; cvd.format = VriFormat_Unknown; cvd.aspect = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr; c.CreateTextureView(dev, &cvd, &colorView);

        VriBufferDesc rbd{}; rbd.size = static_cast<uint64_t>(kW) * kH * 4; rbd.usage = VriBufferUsage_TransferDst; rbd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr; c.CreateBuffer(dev, &rbd, &readback);

        auto hostBuf = [&](const void* data, uint64_t size, VriBufferUsageFlags usage) {
            VriBufferDesc bd{}; bd.size = size; bd.usage = usage; bd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* b = nullptr; c.CreateBuffer(dev, &bd, &b);
            void* m = c.MapBuffer(b, 0, size);
            for (uint64_t i = 0; i < size; ++i) static_cast<uint8_t*>(m)[i] = static_cast<const uint8_t*>(data)[i];
            c.UnmapBuffer(b); return b;
        };
        VriBuffer* vbuf = hostBuf(verts, sizeof(verts), VriBufferUsage_VertexBuffer);
        VriBuffer* ibuf = hostBuf(idx, sizeof(idx), VriBufferUsage_IndexBuffer);

        VriPipelineLayoutDesc ld{};
        VriPipelineLayout* layout = nullptr; c.CreatePipelineLayout(dev, &ld, &layout);
        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = g_triangleVbufSpv; sh[0].bytecodeSize = sizeof(g_triangleVbufSpv); sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = g_triangleVbufSpv; sh[1].bytecodeSize = sizeof(g_triangleVbufSpv); sh[1].entryPointName = "fragmentMain";
        VriVertexStreamDesc stream{}; stream.stride = 6 * sizeof(float); stream.bindingSlot = 0; stream.stepRate = VriVertexStepRate_PerVertex;
        VriVertexAttributeDesc attrs[2]{};
        attrs[0].format = VriFormat_RGB32_SFLOAT; attrs[0].offset = 0;                 attrs[0].streamIndex = 0;
        attrs[1].format = VriFormat_RGB32_SFLOAT; attrs[1].offset = 3 * sizeof(float); attrs[1].streamIndex = 0;
        VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.vertexInput.streams = &stream; pd.vertexInput.streamNum = 1;
        pd.vertexInput.attributes = attrs; pd.vertexInput.attributeNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
        VriPipeline* pipeline = nullptr;
        if (c.CreateGraphicsPipeline(dev, &pd, &pipeline) != VriResult_Success)
        {
            std::printf("VRI_WEBGL: vbuf pipeline creation FAILED\n");
            return false;
        }

        VriCommandAllocator* alloc = nullptr; c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
        VriCommandBuffer* cmd = nullptr; c.CreateCommandBuffer(alloc, &cmd);
        VriFence* fence = nullptr; c.CreateFence(dev, 0, &fence);

        c.BeginCommandBuffer(cmd);
        VriAttachmentDesc rt{}; rt.view = colorView; rt.loadOp = VriAttachmentLoadOp_Clear; rt.storeOp = VriAttachmentStoreOp_Store; rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att{}; att.colors = &rt; att.colorNum = 1; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        VriVertexBufferBinding vbb{}; vbb.buffer = vbuf; vbb.offset = 0;
        c.CmdSetVertexBuffers(cmd, 0, &vbb, 1);
        c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt32);
        VriDrawIndexedDesc di{}; di.indexNum = 3; di.instanceNum = 1; c.CmdDrawIndexed(cmd, &di);
        c.CmdEndRendering(cmd);
        VriBufferTextureCopyDesc tc{}; tc.texture.aspect = VriImageAspect_Color; tc.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &tc);
        c.EndCommandBuffer(cmd);

        VriFenceSubmitDesc sig{}; sig.fence = fence; sig.value = 1;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(queue, &sub);
        c.Wait(fence, 1);

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
        bool ok = px != nullptr;
        if (ok)
        {
            const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
            centerYellow = (px[o + 0] == 255 && px[o + 1] == 255 && px[o + 2] == 0);
            c.UnmapBuffer(readback);
        }

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(vbuf); c.DestroyBuffer(ibuf);
        c.DestroyBuffer(readback); c.DestroyDescriptor(colorView); c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return ok;
    }

    // Depth test: three overlapping triangles (red far, green near, blue farther)
    // with depthTest=Less -> center must be green. Exercises GL depth attachments.
    bool RenderDepthAndProbe(bool& centerGreen)
    {
        centerGreen = false;
        auto fillTri = [](float* v, float z, float r, float g, float b) {
            const float xy[3][2] = {{0.0f, -0.5f}, {0.5f, 0.5f}, {-0.5f, 0.5f}};
            for (int i = 0; i < 3; ++i) { v[i*6+0]=xy[i][0]; v[i*6+1]=xy[i][1]; v[i*6+2]=z; v[i*6+3]=r; v[i*6+4]=g; v[i*6+5]=b; }
        };
        float tri[3][18];
        fillTri(tri[0], 0.7f, 1, 0, 0); fillTri(tri[1], 0.3f, 0, 1, 0); fillTri(tri[2], 0.9f, 0, 0, 1);

        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = VriGraphicsAPI_OpenGLES; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success) return false;
        VriCoreInterface c{};
        if (vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) != VriResult_Success) return false;
        VriQueue* queue = nullptr; c.GetQueue(dev, VriQueueType_Graphics, 0, &queue);

        VriTextureDesc td{}; td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM;
        td.width = kW; td.height = kH; td.depth = 1; td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1;
        td.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc; td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* color = nullptr; c.CreateTexture(dev, &td, &color);
        VriTextureViewDesc cvd{}; cvd.texture = color; cvd.viewType = VriTextureViewType_2D; cvd.format = VriFormat_Unknown; cvd.aspect = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr; c.CreateTextureView(dev, &cvd, &colorView);

        VriTextureDesc dtd{}; dtd.type = VriTextureType_2D; dtd.format = VriFormat_D32_SFLOAT;
        dtd.width = kW; dtd.height = kH; dtd.depth = 1; dtd.mipNum = 1; dtd.layerNum = 1; dtd.sampleNum = 1;
        dtd.usage = VriTextureUsage_DepthStencilAttachment; dtd.memoryLocation = VriMemoryLocation_Device;
        VriTexture* depth = nullptr; c.CreateTexture(dev, &dtd, &depth);
        VriTextureViewDesc dvd{}; dvd.texture = depth; dvd.viewType = VriTextureViewType_2D; dvd.format = VriFormat_D32_SFLOAT; dvd.aspect = VriImageAspect_Depth;
        VriDescriptor* depthView = nullptr; c.CreateTextureView(dev, &dvd, &depthView);

        VriBufferDesc rbd{}; rbd.size = static_cast<uint64_t>(kW) * kH * 4; rbd.usage = VriBufferUsage_TransferDst; rbd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr; c.CreateBuffer(dev, &rbd, &readback);

        auto hostVbuf = [&](const float* data) {
            const uint64_t sz = 18 * sizeof(float);
            VriBufferDesc bd{}; bd.size = sz; bd.usage = VriBufferUsage_VertexBuffer; bd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* b = nullptr; c.CreateBuffer(dev, &bd, &b);
            void* m = c.MapBuffer(b, 0, sz);
            for (uint64_t i = 0; i < sz; ++i) static_cast<uint8_t*>(m)[i] = reinterpret_cast<const uint8_t*>(data)[i];
            c.UnmapBuffer(b); return b;
        };
        VriBuffer* vb[3] = {hostVbuf(tri[0]), hostVbuf(tri[1]), hostVbuf(tri[2])};

        VriPipelineLayoutDesc ld{}; VriPipelineLayout* layout = nullptr; c.CreatePipelineLayout(dev, &ld, &layout);
        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = g_triangleVbufSpv; sh[0].bytecodeSize = sizeof(g_triangleVbufSpv); sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = g_triangleVbufSpv; sh[1].bytecodeSize = sizeof(g_triangleVbufSpv); sh[1].entryPointName = "fragmentMain";
        VriVertexStreamDesc stream{}; stream.stride = 6 * sizeof(float); stream.bindingSlot = 0; stream.stepRate = VriVertexStepRate_PerVertex;
        VriVertexAttributeDesc attrs[2]{};
        attrs[0].format = VriFormat_RGB32_SFLOAT; attrs[0].offset = 0;                 attrs[0].streamIndex = 0;
        attrs[1].format = VriFormat_RGB32_SFLOAT; attrs[1].offset = 3 * sizeof(float); attrs[1].streamIndex = 0;
        VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.vertexInput.streams = &stream; pd.vertexInput.streamNum = 1; pd.vertexInput.attributes = attrs; pd.vertexInput.attributeNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.depthStencil.depthTest = VRI_TRUE; pd.depthStencil.depthWrite = VRI_TRUE; pd.depthStencil.depthCompareOp = VriCompareOp_Less;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1; pd.outputMerger.depthStencilFormat = VriFormat_D32_SFLOAT;
        VriPipeline* pipeline = nullptr;
        if (c.CreateGraphicsPipeline(dev, &pd, &pipeline) != VriResult_Success) { std::printf("VRI_WEBGL: depth pipeline FAILED\n"); return false; }

        VriCommandAllocator* alloc = nullptr; c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
        VriCommandBuffer* cmd = nullptr; c.CreateCommandBuffer(alloc, &cmd);
        VriFence* fence = nullptr; c.CreateFence(dev, 0, &fence);

        c.BeginCommandBuffer(cmd);
        VriAttachmentDesc rt{}; rt.view = colorView; rt.loadOp = VriAttachmentLoadOp_Clear; rt.storeOp = VriAttachmentStoreOp_Store; rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentDesc dat{}; dat.view = depthView; dat.loadOp = VriAttachmentLoadOp_Clear; dat.storeOp = VriAttachmentStoreOp_Store; dat.clearValue.depthStencil.depth = 1.0f;
        VriAttachmentsDesc att{}; att.colors = &rt; att.colorNum = 1; att.depth = &dat; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        for (int i = 0; i < 3; ++i) { VriVertexBufferBinding vbb{}; vbb.buffer = vb[i]; vbb.offset = 0; c.CmdSetVertexBuffers(cmd, 0, &vbb, 1); VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 1; c.CmdDraw(cmd, &draw); }
        c.CmdEndRendering(cmd);
        VriBufferTextureCopyDesc tc{}; tc.texture.aspect = VriImageAspect_Color; tc.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &tc);
        c.EndCommandBuffer(cmd);

        VriFenceSubmitDesc sig{}; sig.fence = fence; sig.value = 1;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(queue, &sub); c.Wait(fence, 1);

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
        bool ok = px != nullptr;
        if (ok)
        {
            const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
            centerGreen = (px[o + 0] == 0 && px[o + 1] == 255 && px[o + 2] == 0);
            c.UnmapBuffer(readback);
        }

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(vb[0]); c.DestroyBuffer(vb[1]); c.DestroyBuffer(vb[2]);
        c.DestroyBuffer(readback); c.DestroyDescriptor(depthView); c.DestroyTexture(depth); c.DestroyDescriptor(colorView); c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return ok;
    }

    // Alpha blending: opaque blue then red @ alpha 0.5 -> center purple (~128,0,128).
    bool RenderBlendAndProbe(bool& centerPurple)
    {
        centerPurple = false;
        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = VriGraphicsAPI_OpenGLES; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success) return false;
        VriCoreInterface c{};
        if (vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) != VriResult_Success) return false;
        VriQueue* queue = nullptr; c.GetQueue(dev, VriQueueType_Graphics, 0, &queue);

        VriTextureDesc td{}; td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM;
        td.width = kW; td.height = kH; td.depth = 1; td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1;
        td.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc; td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* color = nullptr; c.CreateTexture(dev, &td, &color);
        VriTextureViewDesc cvd{}; cvd.texture = color; cvd.viewType = VriTextureViewType_2D; cvd.format = VriFormat_Unknown; cvd.aspect = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr; c.CreateTextureView(dev, &cvd, &colorView);
        VriBufferDesc rbd{}; rbd.size = static_cast<uint64_t>(kW) * kH * 4; rbd.usage = VriBufferUsage_TransferDst; rbd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr; c.CreateBuffer(dev, &rbd, &readback);

        auto hostUbo = [&](float r, float g, float b, float a) {
            VriBufferDesc ud{}; ud.size = 16; ud.usage = VriBufferUsage_ConstantBuffer; ud.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* u = nullptr; c.CreateBuffer(dev, &ud, &u);
            float* m = static_cast<float*>(c.MapBuffer(u, 0, 16)); m[0]=r; m[1]=g; m[2]=b; m[3]=a; c.UnmapBuffer(u); return u;
        };
        VriBuffer* uboBlue = hostUbo(0, 0, 1, 1.0f);
        VriBuffer* uboRed = hostUbo(1, 0, 0, 0.5f);

        VriDescriptorRangeDesc range{}; range.baseRegister = 0; range.descriptorNum = 1; range.descriptorType = VriDescriptorType_ConstantBuffer; range.shaderStages = VriShaderStage_Fragment;
        VriDescriptorSetDesc setDesc{}; setDesc.registerSpace = 0; setDesc.ranges = &range; setDesc.rangeNum = 1;
        VriPipelineLayoutDesc ld{}; ld.descriptorSets = &setDesc; ld.descriptorSetNum = 1;
        VriPipelineLayout* layout = nullptr; c.CreatePipelineLayout(dev, &ld, &layout);
        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = g_triangleUboSpv; sh[0].bytecodeSize = sizeof(g_triangleUboSpv); sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = g_triangleUboSpv; sh[1].bytecodeSize = sizeof(g_triangleUboSpv); sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
        ca.blend.enable = VRI_TRUE;
        ca.blend.srcColor = VriBlendFactor_SrcAlpha; ca.blend.dstColor = VriBlendFactor_OneMinusSrcAlpha; ca.blend.colorOp = VriBlendOp_Add;
        ca.blend.srcAlpha = VriBlendFactor_One; ca.blend.dstAlpha = VriBlendFactor_OneMinusSrcAlpha; ca.blend.alphaOp = VriBlendOp_Add;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
        VriPipeline* pipeline = nullptr;
        if (c.CreateGraphicsPipeline(dev, &pd, &pipeline) != VriResult_Success) { std::printf("VRI_WEBGL: blend pipeline FAILED\n"); return false; }

        VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 2; pdsc.constantBufferMaxNum = 2;
        VriDescriptorPool* pool = nullptr; c.CreateDescriptorPool(dev, &pdsc, &pool);
        VriDescriptorSet* setBlue = nullptr; c.AllocateDescriptorSets(pool, layout, 0, &setBlue, 1);
        VriDescriptorSet* setRed = nullptr; c.AllocateDescriptorSets(pool, layout, 0, &setRed, 1);
        auto bind = [&](VriDescriptorSet* set, VriBuffer* ubo) {
            VriBufferViewDesc bv{}; bv.buffer = ubo; bv.viewType = VriDescriptorType_ConstantBuffer; bv.offset = 0; bv.size = 16;
            VriDescriptor* d = nullptr; c.CreateBufferView(dev, &bv, &d);
            const VriDescriptor* arr[1] = {d}; VriDescriptorRangeUpdateDesc u{}; u.descriptors = arr; u.descriptorNum = 1; u.baseDescriptor = 0;
            c.UpdateDescriptorRanges(set, 0, 1, &u); return d;
        };
        VriDescriptor* dBlue = bind(setBlue, uboBlue);
        VriDescriptor* dRed = bind(setRed, uboRed);

        VriCommandAllocator* alloc = nullptr; c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
        VriCommandBuffer* cmd = nullptr; c.CreateCommandBuffer(alloc, &cmd);
        VriFence* fence = nullptr; c.CreateFence(dev, 0, &fence);
        c.BeginCommandBuffer(cmd);
        VriAttachmentDesc rt{}; rt.view = colorView; rt.loadOp = VriAttachmentLoadOp_Clear; rt.storeOp = VriAttachmentStoreOp_Store; rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att{}; att.colors = &rt; att.colorNum = 1; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline); c.CmdSetPipelineLayout(cmd, layout);
        VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 1;
        c.CmdSetDescriptorSet(cmd, 0, setBlue); c.CmdDraw(cmd, &draw);
        c.CmdSetDescriptorSet(cmd, 0, setRed);  c.CmdDraw(cmd, &draw);
        c.CmdEndRendering(cmd);
        VriBufferTextureCopyDesc tc{}; tc.texture.aspect = VriImageAspect_Color; tc.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &tc);
        c.EndCommandBuffer(cmd);
        VriFenceSubmitDesc sig{}; sig.fence = fence; sig.value = 1;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(queue, &sub); c.Wait(fence, 1);

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
        bool ok = px != nullptr;
        if (ok)
        {
            const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
            centerPurple = std::abs(int(px[o + 0]) - 128) <= 4 && px[o + 1] == 0 && std::abs(int(px[o + 2]) - 128) <= 4;
            c.UnmapBuffer(readback);
        }
        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyDescriptor(dBlue); c.DestroyDescriptor(dRed); c.DestroyDescriptorPool(pool);
        c.DestroyBuffer(uboBlue); c.DestroyBuffer(uboRed); c.DestroyBuffer(readback); c.DestroyDescriptor(colorView); c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return ok;
    }

    // Back-face culling: CCW green front (drawn) + CW red back (culled, cullMode=Back)
    // -> center green. Verifies GL's flip_vert_y + front-face-swap winding on WebGL.
    bool RenderCullAndProbe(bool& centerGreen)
    {
        centerGreen = false;
        const float frontGreen[] = {
            0.0f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f,
            0.5f,  0.5f, 0.0f, 0.0f, 1.0f, 0.0f,
           -0.5f,  0.5f, 0.0f, 0.0f, 1.0f, 0.0f,
        };
        const float backRed[] = {
            0.0f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
           -0.5f,  0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
            0.5f,  0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
        };

        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = VriGraphicsAPI_OpenGLES; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success) return false;
        VriCoreInterface c{};
        if (vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) != VriResult_Success) return false;
        VriQueue* queue = nullptr; c.GetQueue(dev, VriQueueType_Graphics, 0, &queue);

        VriTextureDesc td{}; td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM;
        td.width = kW; td.height = kH; td.depth = 1; td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1;
        td.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc; td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* color = nullptr; c.CreateTexture(dev, &td, &color);
        VriTextureViewDesc cvd{}; cvd.texture = color; cvd.viewType = VriTextureViewType_2D; cvd.format = VriFormat_Unknown; cvd.aspect = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr; c.CreateTextureView(dev, &cvd, &colorView);
        VriBufferDesc rbd{}; rbd.size = static_cast<uint64_t>(kW) * kH * 4; rbd.usage = VriBufferUsage_TransferDst; rbd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr; c.CreateBuffer(dev, &rbd, &readback);

        auto hostVbuf = [&](const float* data) {
            const uint64_t sz = 18 * sizeof(float);
            VriBufferDesc bd{}; bd.size = sz; bd.usage = VriBufferUsage_VertexBuffer; bd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* b = nullptr; c.CreateBuffer(dev, &bd, &b);
            void* m = c.MapBuffer(b, 0, sz);
            for (uint64_t i = 0; i < sz; ++i) static_cast<uint8_t*>(m)[i] = reinterpret_cast<const uint8_t*>(data)[i];
            c.UnmapBuffer(b); return b;
        };
        VriBuffer* vbFront = hostVbuf(frontGreen);
        VriBuffer* vbBack = hostVbuf(backRed);

        VriPipelineLayoutDesc ld{}; VriPipelineLayout* layout = nullptr; c.CreatePipelineLayout(dev, &ld, &layout);
        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = g_triangleVbufSpv; sh[0].bytecodeSize = sizeof(g_triangleVbufSpv); sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = g_triangleVbufSpv; sh[1].bytecodeSize = sizeof(g_triangleVbufSpv); sh[1].entryPointName = "fragmentMain";
        VriVertexStreamDesc stream{}; stream.stride = 6 * sizeof(float); stream.bindingSlot = 0; stream.stepRate = VriVertexStepRate_PerVertex;
        VriVertexAttributeDesc attrs[2]{};
        attrs[0].format = VriFormat_RGB32_SFLOAT; attrs[0].offset = 0;                 attrs[0].streamIndex = 0;
        attrs[1].format = VriFormat_RGB32_SFLOAT; attrs[1].offset = 3 * sizeof(float); attrs[1].streamIndex = 0;
        VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.vertexInput.streams = &stream; pd.vertexInput.streamNum = 1; pd.vertexInput.attributes = attrs; pd.vertexInput.attributeNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_Back; pd.rasterization.frontFace = VriFrontFace_CounterClockwise; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
        VriPipeline* pipeline = nullptr;
        if (c.CreateGraphicsPipeline(dev, &pd, &pipeline) != VriResult_Success) { std::printf("VRI_WEBGL: cull pipeline FAILED\n"); return false; }

        VriCommandAllocator* alloc = nullptr; c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
        VriCommandBuffer* cmd = nullptr; c.CreateCommandBuffer(alloc, &cmd);
        VriFence* fence = nullptr; c.CreateFence(dev, 0, &fence);
        c.BeginCommandBuffer(cmd);
        VriAttachmentDesc rt{}; rt.view = colorView; rt.loadOp = VriAttachmentLoadOp_Clear; rt.storeOp = VriAttachmentStoreOp_Store; rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att{}; att.colors = &rt; att.colorNum = 1; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 1;
        { VriVertexBufferBinding vbb{}; vbb.buffer = vbFront; vbb.offset = 0; c.CmdSetVertexBuffers(cmd, 0, &vbb, 1); c.CmdDraw(cmd, &draw); }
        { VriVertexBufferBinding vbb{}; vbb.buffer = vbBack;  vbb.offset = 0; c.CmdSetVertexBuffers(cmd, 0, &vbb, 1); c.CmdDraw(cmd, &draw); }
        c.CmdEndRendering(cmd);
        VriBufferTextureCopyDesc tc{}; tc.texture.aspect = VriImageAspect_Color; tc.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &tc);
        c.EndCommandBuffer(cmd);
        VriFenceSubmitDesc sig{}; sig.fence = fence; sig.value = 1;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(queue, &sub); c.Wait(fence, 1);

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
        bool ok = px != nullptr;
        if (ok)
        {
            const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
            centerGreen = (px[o + 0] == 0 && px[o + 1] == 255 && px[o + 2] == 0);
            c.UnmapBuffer(readback);
        }
        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(vbFront); c.DestroyBuffer(vbBack); c.DestroyBuffer(readback); c.DestroyDescriptor(colorView); c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return ok;
    }

    // MRT: one draw writes red to target 0 and blue to target 1 -> both readbacks
    // hold their color. Exercises GL glDrawBuffers + per-target readback on WebGL.
    bool RenderMrtAndProbe(bool& ok2)
    {
        ok2 = false;
        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = VriGraphicsAPI_OpenGLES; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success) return false;
        VriCoreInterface c{};
        if (vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) != VriResult_Success) return false;
        VriQueue* queue = nullptr; c.GetQueue(dev, VriQueueType_Graphics, 0, &queue);

        auto makeTarget = [&](VriTexture*& tex, VriDescriptor*& view) {
            VriTextureDesc td{}; td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM;
            td.width = kW; td.height = kH; td.depth = 1; td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1;
            td.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc; td.memoryLocation = VriMemoryLocation_Device;
            c.CreateTexture(dev, &td, &tex);
            VriTextureViewDesc vd{}; vd.texture = tex; vd.viewType = VriTextureViewType_2D; vd.format = VriFormat_Unknown; vd.aspect = VriImageAspect_Color;
            c.CreateTextureView(dev, &vd, &view);
        };
        VriTexture* tex0 = nullptr; VriDescriptor* view0 = nullptr; makeTarget(tex0, view0);
        VriTexture* tex1 = nullptr; VriDescriptor* view1 = nullptr; makeTarget(tex1, view1);
        VriBufferDesc rbd{}; rbd.size = static_cast<uint64_t>(kW) * kH * 4; rbd.usage = VriBufferUsage_TransferDst; rbd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* rb0 = nullptr; c.CreateBuffer(dev, &rbd, &rb0);
        VriBuffer* rb1 = nullptr; c.CreateBuffer(dev, &rbd, &rb1);

        VriPipelineLayoutDesc ld{}; VriPipelineLayout* layout = nullptr; c.CreatePipelineLayout(dev, &ld, &layout);
        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = g_triangleMrtSpv; sh[0].bytecodeSize = sizeof(g_triangleMrtSpv); sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = g_triangleMrtSpv; sh[1].bytecodeSize = sizeof(g_triangleMrtSpv); sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca[2]{};
        ca[0].format = VriFormat_RGBA8_UNORM; ca[0].colorWriteMask = VriColorWrite_RGBA;
        ca[1].format = VriFormat_RGBA8_UNORM; ca[1].colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = ca; pd.outputMerger.colorNum = 2;
        VriPipeline* pipeline = nullptr;
        if (c.CreateGraphicsPipeline(dev, &pd, &pipeline) != VriResult_Success) { std::printf("VRI_WEBGL: mrt pipeline FAILED\n"); return false; }

        VriCommandAllocator* alloc = nullptr; c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
        VriCommandBuffer* cmd = nullptr; c.CreateCommandBuffer(alloc, &cmd);
        VriFence* fence = nullptr; c.CreateFence(dev, 0, &fence);
        c.BeginCommandBuffer(cmd);
        VriAttachmentDesc rt[2]{};
        rt[0].view = view0; rt[0].loadOp = VriAttachmentLoadOp_Clear; rt[0].storeOp = VriAttachmentStoreOp_Store; rt[0].clearValue.color.f32[3] = 1.0f;
        rt[1].view = view1; rt[1].loadOp = VriAttachmentLoadOp_Clear; rt[1].storeOp = VriAttachmentStoreOp_Store; rt[1].clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att{}; att.colors = rt; att.colorNum = 2; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 1; c.CmdDraw(cmd, &draw);
        c.CmdEndRendering(cmd);
        VriBufferTextureCopyDesc tc{}; tc.texture.aspect = VriImageAspect_Color; tc.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, rb0, tex0, &tc);
        c.CmdReadbackTextureToBuffer(cmd, rb1, tex1, &tc);
        c.EndCommandBuffer(cmd);
        VriFenceSubmitDesc sig{}; sig.fence = fence; sig.value = 1;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(queue, &sub); c.Wait(fence, 1);

        const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
        const uint8_t* p0 = static_cast<const uint8_t*>(c.MapBuffer(rb0, 0, rbd.size));
        const bool red = p0 && p0[o + 0] == 255 && p0[o + 1] == 0 && p0[o + 2] == 0;
        if (p0) c.UnmapBuffer(rb0);
        const uint8_t* p1 = static_cast<const uint8_t*>(c.MapBuffer(rb1, 0, rbd.size));
        const bool blue = p1 && p1[o + 0] == 0 && p1[o + 1] == 0 && p1[o + 2] == 255;
        if (p1) c.UnmapBuffer(rb1);
        ok2 = red && blue;

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(rb0); c.DestroyBuffer(rb1);
        c.DestroyDescriptor(view0); c.DestroyTexture(tex0); c.DestroyDescriptor(view1); c.DestroyTexture(tex1);
        vriDestroyDevice(dev);
        return true;
    }

    // Instanced draw: 2 instances via SV_InstanceID -> red left, green right.
    bool RenderInstanceAndProbe(bool& ok2)
    {
        ok2 = false;
        VriDeviceCreationDesc dc{};
        dc.graphicsAPI = VriGraphicsAPI_OpenGLES; dc.enableValidation = VRI_TRUE; dc.bestEffort = VRI_TRUE;
        VriDevice* dev = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success) return false;
        VriCoreInterface c{};
        if (vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) != VriResult_Success) return false;
        VriQueue* queue = nullptr; c.GetQueue(dev, VriQueueType_Graphics, 0, &queue);

        VriTextureDesc td{}; td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM;
        td.width = kW; td.height = kH; td.depth = 1; td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1;
        td.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc; td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* color = nullptr; c.CreateTexture(dev, &td, &color);
        VriTextureViewDesc cvd{}; cvd.texture = color; cvd.viewType = VriTextureViewType_2D; cvd.format = VriFormat_Unknown; cvd.aspect = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr; c.CreateTextureView(dev, &cvd, &colorView);
        VriBufferDesc rbd{}; rbd.size = static_cast<uint64_t>(kW) * kH * 4; rbd.usage = VriBufferUsage_TransferDst; rbd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr; c.CreateBuffer(dev, &rbd, &readback);

        VriPipelineLayoutDesc ld{}; VriPipelineLayout* layout = nullptr; c.CreatePipelineLayout(dev, &ld, &layout);
        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].bytecode = g_triangleInstSpv; sh[0].bytecodeSize = sizeof(g_triangleInstSpv); sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].bytecode = g_triangleInstSpv; sh[1].bytecodeSize = sizeof(g_triangleInstSpv); sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca{}; ca.format = VriFormat_RGBA8_UNORM; ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
        VriPipeline* pipeline = nullptr;
        if (c.CreateGraphicsPipeline(dev, &pd, &pipeline) != VriResult_Success) { std::printf("VRI_WEBGL: inst pipeline FAILED\n"); return false; }

        VriCommandAllocator* alloc = nullptr; c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
        VriCommandBuffer* cmd = nullptr; c.CreateCommandBuffer(alloc, &cmd);
        VriFence* fence = nullptr; c.CreateFence(dev, 0, &fence);
        c.BeginCommandBuffer(cmd);
        VriAttachmentDesc rt{}; rt.view = colorView; rt.loadOp = VriAttachmentLoadOp_Clear; rt.storeOp = VriAttachmentStoreOp_Store; rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att{}; att.colors = &rt; att.colorNum = 1; att.renderArea.width = kW; att.renderArea.height = kH; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kW, kH}; c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        VriDrawDesc draw{}; draw.vertexNum = 3; draw.instanceNum = 2; c.CmdDraw(cmd, &draw);
        c.CmdEndRendering(cmd);
        VriBufferTextureCopyDesc tc{}; tc.texture.aspect = VriImageAspect_Color; tc.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &tc);
        c.EndCommandBuffer(cmd);
        VriFenceSubmitDesc sig{}; sig.fence = fence; sig.value = 1;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(queue, &sub); c.Wait(fence, 1);

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rbd.size));
        bool ok = px != nullptr;
        if (ok)
        {
            const uint32_t oL = ((kH / 2) * kW + 19) * 4;
            const uint32_t oR = ((kH / 2) * kW + 45) * 4;
            const bool leftRed = px[oL + 0] == 255 && px[oL + 1] == 0 && px[oL + 2] == 0;
            const bool rightGreen = px[oR + 0] == 0 && px[oR + 1] == 255 && px[oR + 2] == 0;
            ok2 = leftRed && rightGreen;
            c.UnmapBuffer(readback);
        }
        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence); c.DestroyCommandAllocator(alloc); c.DestroyPipeline(pipeline); c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(readback); c.DestroyDescriptor(colorView); c.DestroyTexture(color);
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

    bool centerBlue = false;
    const bool ranTex = RenderTexAndProbe(centerBlue);
    const bool texPass = ranTex && centerBlue;

    bool centerYellow = false;
    const bool ranVbuf = RenderVbufAndProbe(centerYellow);
    const bool vbufPass = ranVbuf && centerYellow;

    bool depthGreen = false;
    const bool ranDepth = RenderDepthAndProbe(depthGreen);
    const bool depthPass = ranDepth && depthGreen;

    bool blendPurple = false;
    const bool ranBlend = RenderBlendAndProbe(blendPurple);
    const bool blendPass = ranBlend && blendPurple;

    bool cullGreen = false;
    const bool ranCull = RenderCullAndProbe(cullGreen);
    const bool cullPass = ranCull && cullGreen;

    bool mrtOk = false;
    const bool ranMrt = RenderMrtAndProbe(mrtOk);
    const bool mrtPass = ranMrt && mrtOk;

    bool instOk = false;
    const bool ranInst = RenderInstanceAndProbe(instOk);
    const bool instPass = ranInst && instOk;

    const bool pass = triPass && uboPass && texPass && vbufPass && depthPass && blendPass && cullPass && mrtPass && instPass;
    std::printf("VRI_WEBGL_RESULT: %s (triangle: ran=%d topRed=%d botRed=%d | ubo: ran=%d green=%d | tex: ran=%d blue=%d | vbuf: ran=%d yellow=%d | depth: ran=%d green=%d | blend: ran=%d purple=%d | cull: ran=%d green=%d | mrt: ran=%d ok=%d | inst: ran=%d ok=%d)\n",
                pass ? "PASS" : "FAIL", ranTri, topRed, botRed, ranUbo, centerGreen, ranTex, centerBlue, ranVbuf, centerYellow, ranDepth, depthGreen, ranBlend, blendPurple, ranCull, cullGreen, ranMrt, mrtOk, ranInst, instOk);
    std::fflush(stdout);
#if defined(__EMSCRIPTEN__)
    // Make emrun return the process exit code so a headless run can be graded.
    emscripten_force_exit(pass ? 0 : 1);
#endif
    return pass ? 0 : 1;
}
