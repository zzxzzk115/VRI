// imgui_vri.h - a Dear ImGui renderer drawn through the VRI core interface. One path for every
// backend (VK/D3D12/GL/WebGPU/WebGL2): no per-backend imgui_impl_* needed, because VRI already
// abstracts blend / scissor / dynamic buffers / sampled textures / push constants. Examples-only
// (the VRI core never links ImGui); a real app would wire its own. Input is handled by the
// caller (examples/common/example_app.h: SDL3 on desktop, Emscripten on web).
//
// Usage: Init(...) once; per frame UploadFrame(drawData, cmd) outside the render pass (records
// the vertex/index copies) then Render(drawData, cmd, fbW, fbH) inside it.
#pragma once

#include <vri/vri.h>
#include <imgui.h>

#include <cstring>

#include "tests/shaders/imgui_spv.h"  // g_imguiSpv  (Vulkan + OpenGL)
#include "tests/shaders/imgui_wgsl.h" // g_imguiWgsl (WebGPU)
#include "tests/shaders/imgui_dxbc.h" // g_imguiDxbcVS / PS (D3D12)

namespace vriex
{
    struct ImguiVri
    {
        VriCoreInterface* c = nullptr;
        VriPipelineLayout* layout = nullptr;
        VriPipeline* pipeline = nullptr;
        VriTexture* font = nullptr; VriDescriptor* fontView = nullptr; VriDescriptor* sampler = nullptr;
        VriDescriptorPool* pool = nullptr; VriDescriptorSet* set = nullptr;
        VriBuffer* vbuf = nullptr; VriBuffer* vstg = nullptr; uint32_t vCap = 0; // capacity in vertices
        VriBuffer* ibuf = nullptr; VriBuffer* istg = nullptr; uint32_t iCap = 0; // capacity in indices
        VriDevice* dev = nullptr;

        struct Ortho { float scale[2]; float translate[2]; };

        // depthFormat: the render pass's depth attachment format (VriFormat_Unknown if none).
        // WebGPU requires the pipeline's depth-stencil state to match the pass even when ImGui
        // doesn't test/write depth, so we declare the format with test+write disabled.
        void Init(VriCoreInterface& core, VriDevice* device, VriQueue* queue, VriFormat swapFormat, VriFormat depthFormat, bool useWgsl, bool useDxbc)
        {
            c = &core; dev = device;

            // ---- font atlas -> a sampled texture, uploaded via a self-contained one-shot submit
            ImGuiIO& io = ImGui::GetIO();
            unsigned char* px = nullptr; int fw = 0, fh = 0; io.Fonts->GetTexDataAsRGBA32(&px, &fw, &fh);
            VriTextureDesc td{}; td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM; td.width = uint32_t(fw); td.height = uint32_t(fh); td.depth = 1;
            td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1; td.usage = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst; td.memoryLocation = VriMemoryLocation_Device;
            c->CreateTexture(dev, &td, &font);
            VriTextureViewDesc vd{}; vd.texture = font; vd.viewType = VriTextureViewType_2D; vd.format = VriFormat_Unknown; vd.aspect = VriImageAspect_Color;
            c->CreateTextureView(dev, &vd, &fontView);
            const uint64_t fontBytes = uint64_t(fw) * fh * 4;
            VriBufferDesc fsd{}; fsd.size = fontBytes; fsd.usage = VriBufferUsage_TransferSrc; fsd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* fstg = nullptr; c->CreateBuffer(dev, &fsd, &fstg);
            std::memcpy(c->MapBuffer(fstg, 0, fontBytes), px, size_t(fontBytes)); c->UnmapBuffer(fstg);
            {
                VriCommandAllocator* a = nullptr; c->CreateCommandAllocator(dev, VriQueueType_Graphics, &a);
                VriCommandBuffer* cmd = nullptr; c->CreateCommandBuffer(a, &cmd);
                VriFence* f = nullptr; c->CreateFence(dev, 0, &f);
                c->BeginCommandBuffer(cmd);
                VriTextureBarrierDesc tb{}; tb.texture = font; tb.before.layout = VriLayout_Undefined; tb.before.stages = VriPipelineStage_None;
                tb.after.access = VriAccess_CopyDestinationWrite; tb.after.layout = VriLayout_CopyDestination; tb.after.stages = VriPipelineStage_Transfer; tb.aspect = VriImageAspect_Color;
                VriBarrierGroupDesc g0{}; g0.textures = &tb; g0.textureNum = 1; c->CmdBarrier(cmd, &g0);
                VriBufferTextureCopyDesc up{}; up.texture.aspect = VriImageAspect_Color; up.texture.layerNum = 1; up.texture.width = uint32_t(fw); up.texture.height = uint32_t(fh);
                c->CmdUploadBufferToTexture(cmd, font, fstg, &up);
                VriTextureBarrierDesc tb2{}; tb2.texture = font; tb2.before.access = VriAccess_CopyDestinationWrite; tb2.before.layout = VriLayout_CopyDestination; tb2.before.stages = VriPipelineStage_Transfer;
                tb2.after.access = VriAccess_ShaderResourceRead; tb2.after.layout = VriLayout_ShaderResource; tb2.after.stages = VriPipelineStage_FragmentShader; tb2.aspect = VriImageAspect_Color;
                VriBarrierGroupDesc g1{}; g1.textures = &tb2; g1.textureNum = 1; c->CmdBarrier(cmd, &g1);
                c->EndCommandBuffer(cmd);
                VriFenceSubmitDesc sig{}; sig.fence = f; sig.value = 1;
                VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
                c->QueueSubmit(queue, &sub); c->Wait(f, 1);
                c->DestroyFence(f); c->DestroyCommandAllocator(a); c->DestroyBuffer(fstg);
            }
            io.Fonts->SetTexID(reinterpret_cast<ImTextureID>(font)); // single texture; Render ignores per-cmd TexID

            VriSamplerDesc smp{}; smp.magFilter = VriFilter_Linear; smp.minFilter = VriFilter_Linear; smp.mipmapMode = VriMipmapMode_Nearest;
            smp.addressModeU = VriAddressMode_Repeat; smp.addressModeV = VriAddressMode_Repeat; smp.addressModeW = VriAddressMode_Repeat; smp.maxLod = 1.0f;
            c->CreateSampler(dev, &smp, &sampler);

            // ---- pipeline layout: push constant (ortho, vertex) + Texture@0 + Sampler@1 (fragment)
            VriDescriptorRangeDesc ranges[2]{};
            ranges[0].baseRegister = 0; ranges[0].descriptorNum = 1; ranges[0].descriptorType = VriDescriptorType_Texture; ranges[0].shaderStages = VriShaderStage_Fragment;
            ranges[1].baseRegister = 1; ranges[1].descriptorNum = 1; ranges[1].descriptorType = VriDescriptorType_Sampler; ranges[1].shaderStages = VriShaderStage_Fragment;
            VriDescriptorSetDesc sd{}; sd.registerSpace = 0; sd.ranges = ranges; sd.rangeNum = 2;
            VriPushConstantDesc pc{}; pc.baseRegister = 0; pc.size = sizeof(Ortho); pc.shaderStages = VriShaderStage_Vertex;
            VriPipelineLayoutDesc ld{}; ld.descriptorSets = &sd; ld.descriptorSetNum = 1; ld.pushConstants = &pc; ld.pushConstantNum = 1;
            c->CreatePipelineLayout(dev, &ld, &layout);

            VriShaderDesc sh[2]{};
            sh[0].stage = VriShaderStage_Vertex;   sh[0].entryPointName = "vertexMain";
            sh[1].stage = VriShaderStage_Fragment; sh[1].entryPointName = "fragmentMain";
            if (useDxbc) { sh[0].bytecode = g_imguiDxbcVS; sh[0].bytecodeSize = sizeof(g_imguiDxbcVS); sh[1].bytecode = g_imguiDxbcPS; sh[1].bytecodeSize = sizeof(g_imguiDxbcPS); }
            else { sh[0].bytecode = sh[1].bytecode = useWgsl ? static_cast<const void*>(g_imguiWgsl) : static_cast<const void*>(g_imguiSpv);
                   sh[0].bytecodeSize = sh[1].bytecodeSize = useWgsl ? sizeof(g_imguiWgsl) : sizeof(g_imguiSpv); }

            VriVertexAttributeDesc attrs[3]{};
            attrs[0].format = VriFormat_RG32_SFLOAT; attrs[0].offset = IM_OFFSETOF(ImDrawVert, pos); attrs[0].streamIndex = 0;
            attrs[1].format = VriFormat_RG32_SFLOAT; attrs[1].offset = IM_OFFSETOF(ImDrawVert, uv);  attrs[1].streamIndex = 0;
            attrs[2].format = VriFormat_RGBA8_UNORM; attrs[2].offset = IM_OFFSETOF(ImDrawVert, col); attrs[2].streamIndex = 0;
            VriVertexStreamDesc stream{}; stream.stride = sizeof(ImDrawVert); stream.bindingSlot = 0; stream.stepRate = VriVertexStepRate_PerVertex;

            VriColorAttachmentDesc ca{}; ca.format = swapFormat; ca.colorWriteMask = VriColorWrite_RGBA;
            ca.blend.enable = VRI_TRUE; ca.blend.srcColor = VriBlendFactor_SrcAlpha; ca.blend.dstColor = VriBlendFactor_OneMinusSrcAlpha; ca.blend.colorOp = VriBlendOp_Add;
            ca.blend.srcAlpha = VriBlendFactor_One; ca.blend.dstAlpha = VriBlendFactor_OneMinusSrcAlpha; ca.blend.alphaOp = VriBlendOp_Add;
            VriGraphicsPipelineDesc pd{};
            pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
            pd.vertexInput.attributes = attrs; pd.vertexInput.attributeNum = 3; pd.vertexInput.streams = &stream; pd.vertexInput.streamNum = 1;
            pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
            pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
            pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
            pd.outputMerger.depthStencilFormat = depthFormat; // match the pass's depth attachment (test/write stay off)
            c->CreateGraphicsPipeline(dev, &pd, &pipeline);

            VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 1; pdsc.textureMaxNum = 1; pdsc.samplerMaxNum = 1;
            c->CreateDescriptorPool(dev, &pdsc, &pool);
            c->AllocateDescriptorSets(pool, layout, 0, &set, 1);
            const VriDescriptor* td0[1] = {fontView}; const VriDescriptor* sd0[1] = {sampler};
            VriDescriptorRangeUpdateDesc u[2]{}; u[0].descriptors = td0; u[0].descriptorNum = 1; u[1].descriptors = sd0; u[1].descriptorNum = 1;
            c->UpdateDescriptorRanges(set, 0, 2, u);
        }

        // Grow a device buffer + its host staging to hold at least `count` elements of `stride`.
        void Grow(VriBuffer*& buf, VriBuffer*& stg, uint32_t& cap, uint32_t count, uint32_t stride, VriBufferUsageFlags usage)
        {
            if (count <= cap && buf) return;
            uint32_t next = cap ? cap : 4096;
            while (next < count) next *= 2;
            if (buf) c->DestroyBuffer(buf);
            if (stg) c->DestroyBuffer(stg);
            VriBufferDesc bd{}; bd.size = uint64_t(next) * stride; bd.usage = usage | VriBufferUsage_TransferDst; bd.memoryLocation = VriMemoryLocation_Device;
            c->CreateBuffer(dev, &bd, &buf);
            VriBufferDesc sd{}; sd.size = uint64_t(next) * stride; sd.usage = VriBufferUsage_TransferSrc; sd.memoryLocation = VriMemoryLocation_HostUpload;
            c->CreateBuffer(dev, &sd, &stg);
            cap = next;
        }

        // WriteFrame writes the ImDrawData into host staging (the MapBuffer can ASYNCIFY-yield on
        // WebGPU, so it must run BEFORE the swapchain image is acquired - like ExampleApp::onUpdate).
        // RecordCopy then records the staging->device copy into the command buffer (no yield).
        uint64_t vBytes = 0, iBytes = 0; bool hasFrame = false;
        void WriteFrame(ImDrawData* dd)
        {
            hasFrame = dd && dd->TotalVtxCount > 0;
            if (!hasFrame) return;
            Grow(vbuf, vstg, vCap, uint32_t(dd->TotalVtxCount), sizeof(ImDrawVert), VriBufferUsage_VertexBuffer);
            Grow(ibuf, istg, iCap, uint32_t(dd->TotalIdxCount), sizeof(ImDrawIdx), VriBufferUsage_IndexBuffer);
            auto align4 = [](uint64_t s) { return (s + 3ull) & ~3ull; }; // WebGPU map/copy sizes must be %4
            vBytes = align4(uint64_t(dd->TotalVtxCount) * sizeof(ImDrawVert));
            iBytes = align4(uint64_t(dd->TotalIdxCount) * sizeof(ImDrawIdx));
            ImDrawVert* vmap = static_cast<ImDrawVert*>(c->MapBuffer(vstg, 0, vBytes));
            ImDrawIdx* imap = static_cast<ImDrawIdx*>(c->MapBuffer(istg, 0, iBytes));
            for (int n = 0; n < dd->CmdListsCount; ++n)
            {
                const ImDrawList* cl = dd->CmdLists[n];
                std::memcpy(vmap, cl->VtxBuffer.Data, size_t(cl->VtxBuffer.Size) * sizeof(ImDrawVert)); vmap += cl->VtxBuffer.Size;
                std::memcpy(imap, cl->IdxBuffer.Data, size_t(cl->IdxBuffer.Size) * sizeof(ImDrawIdx)); imap += cl->IdxBuffer.Size;
            }
            c->UnmapBuffer(vstg); c->UnmapBuffer(istg);
        }

        // Record the staging->device copy + barrier. Call OUTSIDE the render pass (onPreRender phase).
        void RecordCopy(VriCommandBuffer* cmd)
        {
            if (!hasFrame) return;
            VriBufferCopyDesc vcp{}; vcp.size = vBytes; c->CmdCopyBuffer(cmd, vbuf, vstg, &vcp);
            VriBufferCopyDesc icp{}; icp.size = iBytes; c->CmdCopyBuffer(cmd, ibuf, istg, &icp);
            VriBufferBarrierDesc gb[2]{};
            gb[0].buffer = vbuf; gb[0].before.access = VriAccess_CopyDestinationWrite; gb[0].before.stages = VriPipelineStage_Transfer; gb[0].after.access = VriAccess_VertexBufferRead; gb[0].after.stages = VriPipelineStage_VertexInput;
            gb[1].buffer = ibuf; gb[1].before.access = VriAccess_CopyDestinationWrite; gb[1].before.stages = VriPipelineStage_Transfer; gb[1].after.access = VriAccess_IndexBufferRead; gb[1].after.stages = VriPipelineStage_VertexInput;
            VriBarrierGroupDesc g{}; g.buffers = gb; g.bufferNum = 2; c->CmdBarrier(cmd, &g);
        }

        // Record the UI draws. Call INSIDE the render pass (after the example's draw -> UI on top).
        void Render(ImDrawData* dd, VriCommandBuffer* cmd, uint32_t fbW, uint32_t fbH)
        {
            if (!dd || dd->TotalVtxCount == 0 || dd->DisplaySize.x <= 0 || dd->DisplaySize.y <= 0) return;
            Ortho o{};
            o.scale[0] = 2.0f / dd->DisplaySize.x; o.scale[1] = -2.0f / dd->DisplaySize.y; // -Y: ImGui origin is top-left
            o.translate[0] = -1.0f - dd->DisplayPos.x * o.scale[0]; o.translate[1] = 1.0f - dd->DisplayPos.y * o.scale[1];
            c->CmdSetPipeline(cmd, pipeline);
            c->CmdSetPipelineLayout(cmd, layout);
            c->CmdSetConstants(cmd, 0, &o, sizeof(o));
            c->CmdSetDescriptorSet(cmd, 0, set);
            VriViewport vp{0, 0, float(fbW), float(fbH), 0, 1}; c->CmdSetViewports(cmd, &vp, 1);
            VriVertexBufferBinding vb{}; vb.buffer = vbuf; vb.offset = 0; c->CmdSetVertexBuffers(cmd, 0, &vb, 1);
            c->CmdSetIndexBuffer(cmd, ibuf, 0, sizeof(ImDrawIdx) == 2 ? VriIndexType_UInt16 : VriIndexType_UInt32);

            const ImVec2 pos = dd->DisplayPos;
            uint32_t vtxOff = 0, idxOff = 0;
            for (int n = 0; n < dd->CmdListsCount; ++n)
            {
                const ImDrawList* cl = dd->CmdLists[n];
                for (int i = 0; i < cl->CmdBuffer.Size; ++i)
                {
                    const ImDrawCmd& dc = cl->CmdBuffer[i];
                    if (dc.UserCallback) { dc.UserCallback(cl, &dc); continue; }
                    // ClipRect is in display coords; convert to framebuffer pixels + clamp.
                    int cx = int(dc.ClipRect.x - pos.x), cy = int(dc.ClipRect.y - pos.y);
                    int cz = int(dc.ClipRect.z - pos.x), cw = int(dc.ClipRect.w - pos.y);
                    if (cx < 0) cx = 0; if (cy < 0) cy = 0;
                    if (cz > int(fbW)) cz = int(fbW); if (cw > int(fbH)) cw = int(fbH);
                    if (cz <= cx || cw <= cy) continue;
                    VriRect sc{cx, cy, uint32_t(cz - cx), uint32_t(cw - cy)}; c->CmdSetScissors(cmd, &sc, 1);
                    VriDrawIndexedDesc di{}; di.indexNum = dc.ElemCount; di.instanceNum = 1;
                    di.baseIndex = dc.IdxOffset + idxOff; di.vertexOffset = int32_t(dc.VtxOffset + vtxOff);
                    c->CmdDrawIndexed(cmd, &di);
                }
                vtxOff += cl->VtxBuffer.Size; idxOff += cl->IdxBuffer.Size;
            }
        }

        void Shutdown()
        {
            if (!c) return;
            if (vbuf) c->DestroyBuffer(vbuf); if (vstg) c->DestroyBuffer(vstg);
            if (ibuf) c->DestroyBuffer(ibuf); if (istg) c->DestroyBuffer(istg);
            if (pool) c->DestroyDescriptorPool(pool);
            if (pipeline) c->DestroyPipeline(pipeline); if (layout) c->DestroyPipelineLayout(layout);
            if (sampler) c->DestroyDescriptor(sampler); if (fontView) c->DestroyDescriptor(fontView); if (font) c->DestroyTexture(font);
        }
    };
} // namespace vriex
