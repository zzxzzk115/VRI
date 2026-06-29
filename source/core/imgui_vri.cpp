// imgui_vri.cpp - VRI's built-in Dear ImGui renderer (VRI_INTERFACE_IMGUI).
//
// One backend-agnostic implementation: it draws entirely through the device's public core
// interface (fetched via vriGetInterface), so the same code renders ImGui on Vulkan, D3D12,
// WebGPU and OpenGL, and it flows through the validation layer transparently. VRI itself never
// includes or links Dear ImGui; the application translates ImDrawData into the neutral
// VriImguiDrawData (see vri_ext_imgui.h) each frame.

#include "imgui_vri.h"

#include <cstddef>
#include <cstring>
#include <unordered_map>
#include <vector>

// Library-internal ImGui shaders (shaders/lib), embedded as byte arrays.
#include "shaders/lib/imgui_dxbc.h" // g_imguiDxbcVS / g_imguiDxbcPS (D3D12)
#include "shaders/lib/imgui_spv.h"  // g_imguiSpv                    (Vulkan + OpenGL)
#include "shaders/lib/imgui_wgsl.h" // g_imguiWgsl                   (WebGPU)

namespace vri::core
{
    namespace
    {
        struct Ortho
        {
            float scale[2];
            float translate[2];
        };

        struct ImguiState;

        // A frame's geometry (vertex+index device buffers + their host staging). One per ImGui
        // viewport so several detached windows (multi-viewport) can have live geometry at once
        // without clobbering each other's staging. ImguiState owns a default one for the main window.
        struct ImguiViewport
        {
            ImguiState* owner = nullptr; // for the device + core table
            VriBuffer*  vbuf = nullptr;
            VriBuffer*  vstg = nullptr;
            uint32_t    vCap = 0; // capacity in vertices
            VriBuffer*  ibuf = nullptr;
            VriBuffer*  istg = nullptr;
            uint32_t    iCap = 0; // capacity in indices
            uint64_t    vBytes   = 0;
            uint64_t    iBytes   = 0;
            bool        hasFrame = false;
        };

        // The renderer state behind the opaque VriImgui* handle: the shared pipeline/textures plus a
        // default viewport's geometry buffers.
        struct ImguiState
        {
            VriCoreInterface c {}; // the device's (possibly validation-wrapped) core table
            VriDevice*       dev = nullptr;

            VriPipelineLayout* layout   = nullptr;
            VriPipeline*       pipeline = nullptr;
            VriTexture*        font     = nullptr;
            VriDescriptor*     fontView = nullptr; // null when the host drives textures (ImGui 1.92)
            VriDescriptor*     sampler  = nullptr;

            // One descriptor set per distinct texture view, cached so a stable texture allocates once.
            // Pools are added in fixed blocks; a freed slot isn't reclaimed until DestroyImgui (texture
            // churn is rare), so FreeImguiTexture only drops the map entry (see its comment).
            std::vector<VriDescriptorPool*>                       pools;
            uint32_t                                              poolUsed = 0; // sets used in pools.back()
            std::unordered_map<VriDescriptor*, VriDescriptorSet*> texSets;

            ImguiViewport mainVp; // the main window's geometry (extra viewports are created explicitly)
        };

        void DestroyImguiState(ImguiState* s); // defined below; used by CreateImgui's error paths

        ImguiState* Self(VriImgui* h) { return reinterpret_cast<ImguiState*>(h); }

        uint64_t Align4(uint64_t s) { return (s + 3ull) & ~3ull; } // WebGPU map/copy sizes must be %4

        constexpr uint32_t kImguiPoolBlock = 64; // descriptor sets (= textures) per pool block

        // The descriptor set that binds `view` (+ the shared sampler), allocating + caching it on first
        // use. A null view falls back to the font atlas. Returns null if there's no texture to bind.
        VriDescriptorSet* SetForView(ImguiState* s, VriDescriptor* view)
        {
            if (!view)
                view = s->fontView;
            if (!view)
                return nullptr;
            auto it = s->texSets.find(view);
            if (it != s->texSets.end())
                return it->second;

            if (s->pools.empty() || s->poolUsed >= kImguiPoolBlock)
            {
                VriDescriptorPoolDesc pd {};
                pd.descriptorSetMaxNum = kImguiPoolBlock;
                pd.textureMaxNum       = kImguiPoolBlock;
                pd.samplerMaxNum       = kImguiPoolBlock;
                VriDescriptorPool* pool = nullptr;
                if (s->c.CreateDescriptorPool(s->dev, &pd, &pool) != VriResult_Success)
                    return nullptr;
                s->pools.push_back(pool);
                s->poolUsed = 0;
            }
            VriDescriptorSet* set = nullptr;
            if (s->c.AllocateDescriptorSets(s->pools.back(), s->layout, 0, &set, 1) != VriResult_Success)
                return nullptr;
            ++s->poolUsed;
            const VriDescriptor*         td0[1] = {view};
            const VriDescriptor*         sd0[1] = {s->sampler};
            VriDescriptorRangeUpdateDesc u[2] {};
            u[0].descriptors   = td0;
            u[0].descriptorNum = 1;
            u[1].descriptors   = sd0;
            u[1].descriptorNum = 1;
            s->c.UpdateDescriptorRanges(set, 0, 2, u);
            s->texSets[view] = set;
            return set;
        }

        // Grow a device buffer + its host staging to hold at least `count` elements of `stride`.
        void Grow(ImguiState*         s,
                  VriBuffer*&         buf,
                  VriBuffer*&         stg,
                  uint32_t&           cap,
                  uint32_t            count,
                  uint32_t            stride,
                  VriBufferUsageFlags usage)
        {
            if (count <= cap && buf)
                return;
            uint32_t next = cap ? cap : 4096;
            while (next < count)
                next *= 2;
            if (buf)
                s->c.DestroyBuffer(buf);
            if (stg)
                s->c.DestroyBuffer(stg);
            VriBufferDesc bd {};
            bd.size           = uint64_t(next) * stride;
            bd.usage          = usage | VriBufferUsage_TransferDst;
            bd.memoryLocation = VriMemoryLocation_Device;
            s->c.CreateBuffer(s->dev, &bd, &buf);
            VriBufferDesc sd {};
            sd.size           = uint64_t(next) * stride;
            sd.usage          = VriBufferUsage_TransferSrc;
            sd.memoryLocation = VriMemoryLocation_HostUpload;
            s->c.CreateBuffer(s->dev, &sd, &stg);
            cap = next;
        }

        // Upload the RGBA8 font atlas into a sampled texture via a self-contained one-shot submit.
        bool UploadFont(ImguiState* s, VriQueue* queue, const VriImguiDesc* desc)
        {
            VriCoreInterface& c = s->c;

            VriTextureDesc td {};
            td.type           = VriTextureType_2D;
            td.format         = VriFormat_RGBA8_UNORM;
            td.width          = desc->fontWidth;
            td.height         = desc->fontHeight;
            td.depth          = 1;
            td.mipNum         = 1;
            td.layerNum       = 1;
            td.sampleNum      = 1;
            td.usage          = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst;
            td.memoryLocation = VriMemoryLocation_Device;
            if (c.CreateTexture(s->dev, &td, &s->font) != VriResult_Success)
                return false;

            VriTextureViewDesc vd {};
            vd.texture  = s->font;
            vd.viewType = VriTextureViewType_2D;
            vd.format   = VriFormat_Unknown;
            vd.aspect   = VriImageAspect_Color;
            if (c.CreateTextureView(s->dev, &vd, &s->fontView) != VriResult_Success)
                return false;

            const uint64_t fontBytes = uint64_t(desc->fontWidth) * desc->fontHeight * 4;
            VriBufferDesc  fsd {};
            fsd.size           = fontBytes;
            fsd.usage          = VriBufferUsage_TransferSrc;
            fsd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* fstg    = nullptr;
            if (c.CreateBuffer(s->dev, &fsd, &fstg) != VriResult_Success)
                return false;
            std::memcpy(c.MapBuffer(fstg, 0, fontBytes), desc->fontAtlas, size_t(fontBytes));
            c.UnmapBuffer(fstg);

            VriCommandAllocator* a = nullptr;
            c.CreateCommandAllocator(s->dev, VriQueueType_Graphics, &a);
            VriCommandBuffer* cmd = nullptr;
            c.CreateCommandBuffer(a, &cmd);
            VriFence* f = nullptr;
            c.CreateFence(s->dev, 0, &f);

            c.BeginCommandBuffer(cmd);
            VriTextureBarrierDesc tb {};
            tb.texture       = s->font;
            tb.before.layout = VriLayout_Undefined;
            tb.before.stages = VriPipelineStage_None;
            tb.after.access  = VriAccess_CopyDestinationWrite;
            tb.after.layout  = VriLayout_CopyDestination;
            tb.after.stages  = VriPipelineStage_Transfer;
            tb.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g0 {};
            g0.textures   = &tb;
            g0.textureNum = 1;
            c.CmdBarrier(cmd, &g0);
            VriBufferTextureCopyDesc up {};
            up.texture.aspect   = VriImageAspect_Color;
            up.texture.layerNum = 1;
            up.texture.width    = desc->fontWidth;
            up.texture.height   = desc->fontHeight;
            c.CmdUploadBufferToTexture(cmd, s->font, fstg, &up);
            VriTextureBarrierDesc tb2 {};
            tb2.texture       = s->font;
            tb2.before.access = VriAccess_CopyDestinationWrite;
            tb2.before.layout = VriLayout_CopyDestination;
            tb2.before.stages = VriPipelineStage_Transfer;
            tb2.after.access  = VriAccess_ShaderResourceRead;
            tb2.after.layout  = VriLayout_ShaderResource;
            tb2.after.stages  = VriPipelineStage_FragmentShader;
            tb2.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g1 {};
            g1.textures   = &tb2;
            g1.textureNum = 1;
            c.CmdBarrier(cmd, &g1);
            c.EndCommandBuffer(cmd);

            VriFenceSubmitDesc sig {};
            sig.fence = f;
            sig.value = 1;
            VriQueueSubmitDesc sub {};
            sub.commandBuffers   = &cmd;
            sub.commandBufferNum = 1;
            sub.signalFences     = &sig;
            sub.signalFenceNum   = 1;
            c.QueueSubmit(queue, &sub);
            c.Wait(f, 1);

            c.DestroyFence(f);
            c.DestroyCommandAllocator(a);
            c.DestroyBuffer(fstg);
            return true;
        }

        VriResult VRI_CALL CreateImgui(VriDevice* device, const VriImguiDesc* desc, VriImgui** outImgui)
        {
            if (!device || !desc || !outImgui)
                return VriResult_InvalidArgument;
            if (desc->fontAtlas && !desc->uploadQueue) // the built-in font atlas needs an upload queue
                return VriResult_InvalidArgument;
            *outImgui = nullptr;

            ImguiState* s = new ImguiState();
            if (vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(s->c), &s->c) != VriResult_Success)
            {
                delete s;
                return VriResult_Failure;
            }
            s->dev              = device;
            s->mainVp.owner     = s; // the main window's geometry buffers belong to this state
            VriCoreInterface& c = s->c;

            // Pick the shader bytecode for the device's backend. The GL backends consume SPIR-V
            // (transpiled at pipeline creation), so they share the Vulkan blob.
            const void* vsCode = nullptr;
            const void* psCode = nullptr;
            size_t      vsSize = 0, psSize = 0;
            switch (c.GetDeviceDesc(device)->graphicsAPI)
            {
                case VriGraphicsAPI_Vulkan:
                case VriGraphicsAPI_OpenGL:
                case VriGraphicsAPI_OpenGLES:
                case VriGraphicsAPI_Metal: // Metal transpiles the SPIR-V to MSL at pipeline creation
                    vsCode = psCode = g_imguiSpv;
                    vsSize = psSize = sizeof(g_imguiSpv);
                    break;
                case VriGraphicsAPI_WebGPU:
                    vsCode = psCode = g_imguiWgsl;
                    vsSize = psSize = sizeof(g_imguiWgsl);
                    break;
                case VriGraphicsAPI_D3D12:
                    vsCode = g_imguiDxbcVS;
                    vsSize = sizeof(g_imguiDxbcVS);
                    psCode = g_imguiDxbcPS;
                    psSize = sizeof(g_imguiDxbcPS);
                    break;
                default: // D3D11: no embedded ImGui shader for this backend yet
                    delete s;
                    return VriResult_Unsupported;
            }

            // Built-in font atlas is optional: with ImGui 1.92 the host drives the texture lifecycle
            // and routes every texture (font included) through per-command textureView instead.
            if (desc->fontAtlas && !UploadFont(s, desc->uploadQueue, desc))
            {
                DestroyImguiState(s); // declared below
                return VriResult_Failure;
            }

            VriSamplerDesc smp {};
            smp.magFilter    = VriFilter_Linear;
            smp.minFilter    = VriFilter_Linear;
            smp.mipmapMode   = VriMipmapMode_Nearest;
            smp.addressModeU = VriAddressMode_Repeat;
            smp.addressModeV = VriAddressMode_Repeat;
            smp.addressModeW = VriAddressMode_Repeat;
            smp.maxLod       = 1.0f;
            c.CreateSampler(s->dev, &smp, &s->sampler);

            // pipeline layout: push constant (ortho, vertex) + Texture@0 + Sampler@1 (fragment)
            VriDescriptorRangeDesc ranges[2] {};
            ranges[0].baseRegister   = 0;
            ranges[0].descriptorNum  = 1;
            ranges[0].descriptorType = VriDescriptorType_Texture;
            ranges[0].shaderStages   = VriShaderStage_Fragment;
            ranges[1].baseRegister   = 1;
            ranges[1].descriptorNum  = 1;
            ranges[1].descriptorType = VriDescriptorType_Sampler;
            ranges[1].shaderStages   = VriShaderStage_Fragment;
            VriDescriptorSetDesc sd {};
            sd.registerSpace = 0;
            sd.ranges        = ranges;
            sd.rangeNum      = 2;
            VriPushConstantDesc pc {};
            pc.baseRegister = 0;
            pc.size         = sizeof(Ortho);
            pc.shaderStages = VriShaderStage_Vertex;
            VriPipelineLayoutDesc ld {};
            ld.descriptorSets   = &sd;
            ld.descriptorSetNum = 1;
            ld.pushConstants    = &pc;
            ld.pushConstantNum  = 1;
            c.CreatePipelineLayout(s->dev, &ld, &s->layout);

            VriShaderDesc sh[2] {};
            sh[0].stage          = VriShaderStage_Vertex;
            sh[0].entryPointName = "vertexMain";
            sh[0].bytecode       = vsCode;
            sh[0].bytecodeSize   = vsSize;
            sh[1].stage          = VriShaderStage_Fragment;
            sh[1].entryPointName = "fragmentMain";
            sh[1].bytecode       = psCode;
            sh[1].bytecodeSize   = psSize;

            VriVertexAttributeDesc attrs[3] {};
            attrs[0].format      = VriFormat_RG32_SFLOAT;
            attrs[0].offset      = uint32_t(offsetof(VriImguiVertex, position));
            attrs[0].streamIndex = 0;
            attrs[1].format      = VriFormat_RG32_SFLOAT;
            attrs[1].offset      = uint32_t(offsetof(VriImguiVertex, uv));
            attrs[1].streamIndex = 0;
            attrs[2].format      = VriFormat_RGBA8_UNORM;
            attrs[2].offset      = uint32_t(offsetof(VriImguiVertex, color));
            attrs[2].streamIndex = 0;
            VriVertexStreamDesc stream {};
            stream.stride      = sizeof(VriImguiVertex);
            stream.bindingSlot = 0;
            stream.stepRate    = VriVertexStepRate_PerVertex;

            VriColorAttachmentDesc ca {};
            ca.format         = desc->colorFormat;
            ca.colorWriteMask = VriColorWrite_RGBA;
            ca.blend.enable   = VRI_TRUE;
            ca.blend.srcColor = VriBlendFactor_SrcAlpha;
            ca.blend.dstColor = VriBlendFactor_OneMinusSrcAlpha;
            ca.blend.colorOp  = VriBlendOp_Add;
            ca.blend.srcAlpha = VriBlendFactor_One;
            ca.blend.dstAlpha = VriBlendFactor_OneMinusSrcAlpha;
            ca.blend.alphaOp  = VriBlendOp_Add;
            VriGraphicsPipelineDesc pd {};
            pd.pipelineLayout                  = s->layout;
            pd.shaders                         = sh;
            pd.shaderNum                       = 2;
            pd.vertexInput.attributes          = attrs;
            pd.vertexInput.attributeNum        = 3;
            pd.vertexInput.streams             = &stream;
            pd.vertexInput.streamNum           = 1;
            pd.inputAssembly.topology          = VriPrimitiveTopology_TriangleList;
            pd.rasterization.cullMode          = VriCullMode_None;
            pd.rasterization.lineWidth         = 1.0f;
            pd.multisample.sampleNum           = 1;
            pd.outputMerger.colors             = &ca;
            pd.outputMerger.colorNum           = 1;
            pd.outputMerger.depthStencilFormat = desc->depthFormat; // match the pass (test/write off)
            if (c.CreateGraphicsPipeline(s->dev, &pd, &s->pipeline) != VriResult_Success)
            {
                DestroyImguiState(s);
                return VriResult_Failure;
            }

            // Pre-warm the font's descriptor set outside any render pass (others are cached lazily as
            // their textures first appear in CmdDrawImgui).
            if (s->fontView)
                SetForView(s, s->fontView);

            *outImgui = reinterpret_cast<VriImgui*>(s);
            return VriResult_Success;
        }

        void FreeViewportBuffers(ImguiViewport* vp)
        {
            VriCoreInterface& c = vp->owner->c;
            if (vp->vbuf)
                c.DestroyBuffer(vp->vbuf);
            if (vp->vstg)
                c.DestroyBuffer(vp->vstg);
            if (vp->ibuf)
                c.DestroyBuffer(vp->ibuf);
            if (vp->istg)
                c.DestroyBuffer(vp->istg);
            *vp = ImguiViewport {vp->owner};
        }

        void DestroyImguiState(ImguiState* s)
        {
            if (!s)
                return;
            VriCoreInterface& c = s->c;
            FreeViewportBuffers(&s->mainVp);
            for (VriDescriptorPool* pool : s->pools)
                c.DestroyDescriptorPool(pool);
            if (s->pipeline)
                c.DestroyPipeline(s->pipeline);
            if (s->layout)
                c.DestroyPipelineLayout(s->layout);
            if (s->sampler)
                c.DestroyDescriptor(s->sampler);
            if (s->fontView)
                c.DestroyDescriptor(s->fontView);
            if (s->font)
                c.DestroyTexture(s->font);
            delete s;
        }

        void VRI_CALL DestroyImgui(VriImgui* imgui) { DestroyImguiState(Self(imgui)); }

        VriDescriptor* VRI_CALL GetImguiFontView(VriImgui* imgui) { return Self(imgui)->fontView; }

        // Stage one viewport's geometry into its own device buffers (+ host staging).
        void UploadTo(ImguiViewport* vp, const VriImguiDrawData* dd)
        {
            ImguiState* s = vp->owner;
            vp->hasFrame  = dd && dd->vertexCount > 0 && dd->indexCount > 0;
            if (!vp->hasFrame)
                return;
            const uint32_t idxStride = dd->indexSize ? dd->indexSize : 2;
            Grow(s, vp->vbuf, vp->vstg, vp->vCap, dd->vertexCount, sizeof(VriImguiVertex), VriBufferUsage_VertexBuffer);
            Grow(s, vp->ibuf, vp->istg, vp->iCap, dd->indexCount, idxStride, VriBufferUsage_IndexBuffer);
            vp->vBytes = Align4(uint64_t(dd->vertexCount) * sizeof(VriImguiVertex));
            vp->iBytes = Align4(uint64_t(dd->indexCount) * idxStride);
            std::memcpy(
                s->c.MapBuffer(vp->vstg, 0, vp->vBytes), dd->vertices, size_t(dd->vertexCount) * sizeof(VriImguiVertex));
            s->c.UnmapBuffer(vp->vstg);
            std::memcpy(s->c.MapBuffer(vp->istg, 0, vp->iBytes), dd->indices, size_t(dd->indexCount) * idxStride);
            s->c.UnmapBuffer(vp->istg);
        }

        void CmdCopyTo(VriCommandBuffer* cmd, ImguiViewport* vp)
        {
            if (!vp->hasFrame)
                return;
            VriCoreInterface& c = vp->owner->c;
            VriBufferCopyDesc vcp {};
            vcp.size = vp->vBytes;
            c.CmdCopyBuffer(cmd, vp->vbuf, vp->vstg, &vcp);
            VriBufferCopyDesc icp {};
            icp.size = vp->iBytes;
            c.CmdCopyBuffer(cmd, vp->ibuf, vp->istg, &icp);
            VriBufferBarrierDesc gb[2] {};
            gb[0].buffer        = vp->vbuf;
            gb[0].before.access = VriAccess_CopyDestinationWrite;
            gb[0].before.stages = VriPipelineStage_Transfer;
            gb[0].after.access  = VriAccess_VertexBufferRead;
            gb[0].after.stages  = VriPipelineStage_VertexInput;
            gb[1].buffer        = vp->ibuf;
            gb[1].before.access = VriAccess_CopyDestinationWrite;
            gb[1].before.stages = VriPipelineStage_Transfer;
            gb[1].after.access  = VriAccess_IndexBufferRead;
            gb[1].after.stages  = VriPipelineStage_VertexInput;
            VriBarrierGroupDesc g {};
            g.buffers   = gb;
            g.bufferNum = 2;
            c.CmdBarrier(cmd, &g);
        }

        void VRI_CALL UploadImguiData(VriImgui* imgui, const VriImguiDrawData* dd) { UploadTo(&Self(imgui)->mainVp, dd); }

        void VRI_CALL CmdCopyImguiData(VriCommandBuffer* cmd, VriImgui* imgui) { CmdCopyTo(cmd, &Self(imgui)->mainVp); }

        void DrawFrom(VriCommandBuffer* cmd, ImguiState* s, ImguiViewport* vpb, const VriImguiDrawData* dd)
        {
            if (!vpb->hasFrame || !dd || dd->displaySize[0] <= 0 || dd->displaySize[1] <= 0)
                return;
            VriCoreInterface& c   = s->c;
            const uint32_t    fbW = dd->framebufferWidth ? dd->framebufferWidth : uint32_t(dd->displaySize[0]);
            const uint32_t    fbH = dd->framebufferHeight ? dd->framebufferHeight : uint32_t(dd->displaySize[1]);

            Ortho o {};
            o.scale[0]     = 2.0f / dd->displaySize[0];
            o.scale[1]     = -2.0f / dd->displaySize[1]; // -Y: ImGui origin is top-left
            o.translate[0] = -1.0f - dd->displayPos[0] * o.scale[0];
            o.translate[1] = 1.0f - dd->displayPos[1] * o.scale[1];

            c.CmdSetPipeline(cmd, s->pipeline);
            c.CmdSetPipelineLayout(cmd, s->layout);
            c.CmdSetConstants(cmd, 0, &o, sizeof(o));
            VriViewport vp {0, 0, float(fbW), float(fbH), 0, 1};
            c.CmdSetViewports(cmd, &vp, 1);
            VriVertexBufferBinding vb {};
            vb.buffer = vpb->vbuf;
            vb.offset = 0;
            c.CmdSetVertexBuffers(cmd, 0, &vb, 1);
            c.CmdSetIndexBuffer(cmd, vpb->ibuf, 0, dd->indexSize == 4 ? VriIndexType_UInt32 : VriIndexType_UInt16);

            VriDescriptorSet* boundSet = nullptr;
            for (uint32_t i = 0; i < dd->commandCount; ++i)
            {
                const VriImguiDrawCommand& dc = dd->commands[i];
                // Bind the command's texture (null -> font atlas), re-binding only when it changes.
                VriDescriptorSet* set = SetForView(s, dc.textureView);
                if (!set)
                    continue; // no texture to sample (e.g. no font and no per-command view)
                if (set != boundSet)
                {
                    c.CmdSetDescriptorSet(cmd, 0, set);
                    boundSet = set;
                }
                // ClipRect is in display coords; convert to framebuffer pixels + clamp.
                int cx = int(dc.clipRect[0] - dd->displayPos[0]);
                int cy = int(dc.clipRect[1] - dd->displayPos[1]);
                int cz = int(dc.clipRect[2] - dd->displayPos[0]);
                int cw = int(dc.clipRect[3] - dd->displayPos[1]);
                if (cx < 0)
                    cx = 0;
                if (cy < 0)
                    cy = 0;
                if (cz > int(fbW))
                    cz = int(fbW);
                if (cw > int(fbH))
                    cw = int(fbH);
                if (cz <= cx || cw <= cy)
                    continue;
                VriRect sc {cx, cy, uint32_t(cz - cx), uint32_t(cw - cy)};
                c.CmdSetScissors(cmd, &sc, 1);
                VriDrawIndexedDesc di {};
                di.indexNum     = dc.indexCount;
                di.instanceNum  = 1;
                di.baseIndex    = dc.indexOffset;
                di.vertexOffset = dc.vertexOffset;
                c.CmdDrawIndexed(cmd, &di);
            }
        }

        void VRI_CALL CmdDrawImgui(VriCommandBuffer* cmd, VriImgui* imgui, const VriImguiDrawData* dd)
        {
            ImguiState* s = Self(imgui);
            DrawFrom(cmd, s, &s->mainVp, dd);
        }

        // ---- extra viewports (multi-viewport / detached OS windows) ----
        // Each detached window gets its own geometry buffers (a VriImguiViewport) but shares the
        // VriImgui's pipeline, font and texture descriptor-set cache. The host renders each window's
        // ImDrawData through these, so their staging buffers never clobber the main window's.
        ImguiViewport* SelfVp(VriImguiViewport* h) { return reinterpret_cast<ImguiViewport*>(h); }

        VriImguiViewport* VRI_CALL CreateImguiViewport(VriImgui* imgui)
        {
            return reinterpret_cast<VriImguiViewport*>(new ImguiViewport {Self(imgui)});
        }

        void VRI_CALL DestroyImguiViewport(VriImguiViewport* vp)
        {
            if (!vp)
                return;
            ImguiViewport* b = SelfVp(vp);
            FreeViewportBuffers(b);
            delete b;
        }

        void VRI_CALL UploadImguiDataTo(VriImguiViewport* vp, const VriImguiDrawData* dd) { UploadTo(SelfVp(vp), dd); }

        void VRI_CALL CmdCopyImguiDataTo(VriCommandBuffer* cmd, VriImguiViewport* vp) { CmdCopyTo(cmd, SelfVp(vp)); }

        void VRI_CALL CmdDrawImguiTo(VriCommandBuffer* cmd, VriImgui* imgui, VriImguiViewport* vp,
                                     const VriImguiDrawData* dd)
        {
            DrawFrom(cmd, Self(imgui), SelfVp(vp), dd);
        }

        // Drop a texture's cached descriptor set before the host destroys the view. The set's pool
        // slot isn't reclaimed (texture churn is rare), but dropping the map entry is what matters:
        // a later texture that reuses the freed pointer address then gets a fresh, correct set.
        void VRI_CALL FreeImguiTexture(VriImgui* imgui, VriDescriptor* view)
        {
            if (imgui && view)
                Self(imgui)->texSets.erase(view);
        }

        const VriImguiInterface g_imgui = {
            CreateImgui,
            DestroyImgui,
            GetImguiFontView,
            UploadImguiData,
            CmdCopyImguiData,
            CmdDrawImgui,
            FreeImguiTexture,
            CreateImguiViewport,
            DestroyImguiViewport,
            UploadImguiDataTo,
            CmdCopyImguiDataTo,
            CmdDrawImguiTo,
        };
    } // namespace

    const VriImguiInterface* GetImguiInterface() { return &g_imgui; }
} // namespace vri::core
