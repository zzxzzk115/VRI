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

        // The renderer state behind the opaque VriImgui* handle.
        struct ImguiState
        {
            VriCoreInterface c {}; // the device's (possibly validation-wrapped) core table
            VriDevice*       dev = nullptr;

            VriPipelineLayout* layout   = nullptr;
            VriPipeline*       pipeline = nullptr;
            VriTexture*        font     = nullptr;
            VriDescriptor*     fontView = nullptr;
            VriDescriptor*     sampler  = nullptr;
            VriDescriptorPool* pool     = nullptr;
            VriDescriptorSet*  set      = nullptr;

            VriBuffer* vbuf = nullptr;
            VriBuffer* vstg = nullptr;
            uint32_t   vCap = 0; // capacity in vertices
            VriBuffer* ibuf = nullptr;
            VriBuffer* istg = nullptr;
            uint32_t   iCap = 0; // capacity in indices

            uint64_t vBytes   = 0;
            uint64_t iBytes   = 0;
            bool     hasFrame = false;
        };

        void DestroyImguiState(ImguiState* s); // defined below; used by CreateImgui's error paths

        ImguiState* Self(VriImgui* h) { return reinterpret_cast<ImguiState*>(h); }

        uint64_t Align4(uint64_t s) { return (s + 3ull) & ~3ull; } // WebGPU map/copy sizes must be %4

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
            if (!device || !desc || !outImgui || !desc->fontAtlas || !desc->uploadQueue)
                return VriResult_InvalidArgument;
            *outImgui = nullptr;

            ImguiState* s = new ImguiState();
            if (vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(s->c), &s->c) != VriResult_Success)
            {
                delete s;
                return VriResult_Failure;
            }
            s->dev              = device;
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
                default: // Metal/D3D11: no embedded ImGui shader for this backend yet
                    delete s;
                    return VriResult_Unsupported;
            }

            if (!UploadFont(s, desc->uploadQueue, desc))
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

            VriDescriptorPoolDesc pdsc {};
            pdsc.descriptorSetMaxNum = 1;
            pdsc.textureMaxNum       = 1;
            pdsc.samplerMaxNum       = 1;
            c.CreateDescriptorPool(s->dev, &pdsc, &s->pool);
            c.AllocateDescriptorSets(s->pool, s->layout, 0, &s->set, 1);
            const VriDescriptor*         td0[1] = {s->fontView};
            const VriDescriptor*         sd0[1] = {s->sampler};
            VriDescriptorRangeUpdateDesc u[2] {};
            u[0].descriptors   = td0;
            u[0].descriptorNum = 1;
            u[1].descriptors   = sd0;
            u[1].descriptorNum = 1;
            c.UpdateDescriptorRanges(s->set, 0, 2, u);

            *outImgui = reinterpret_cast<VriImgui*>(s);
            return VriResult_Success;
        }

        void DestroyImguiState(ImguiState* s)
        {
            if (!s)
                return;
            VriCoreInterface& c = s->c;
            if (s->vbuf)
                c.DestroyBuffer(s->vbuf);
            if (s->vstg)
                c.DestroyBuffer(s->vstg);
            if (s->ibuf)
                c.DestroyBuffer(s->ibuf);
            if (s->istg)
                c.DestroyBuffer(s->istg);
            if (s->pool)
                c.DestroyDescriptorPool(s->pool);
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

        void VRI_CALL UploadImguiData(VriImgui* imgui, const VriImguiDrawData* dd)
        {
            ImguiState* s = Self(imgui);
            s->hasFrame   = dd && dd->vertexCount > 0 && dd->indexCount > 0;
            if (!s->hasFrame)
                return;
            const uint32_t idxStride = dd->indexSize ? dd->indexSize : 2;
            Grow(s, s->vbuf, s->vstg, s->vCap, dd->vertexCount, sizeof(VriImguiVertex), VriBufferUsage_VertexBuffer);
            Grow(s, s->ibuf, s->istg, s->iCap, dd->indexCount, idxStride, VriBufferUsage_IndexBuffer);
            s->vBytes = Align4(uint64_t(dd->vertexCount) * sizeof(VriImguiVertex));
            s->iBytes = Align4(uint64_t(dd->indexCount) * idxStride);
            std::memcpy(
                s->c.MapBuffer(s->vstg, 0, s->vBytes), dd->vertices, size_t(dd->vertexCount) * sizeof(VriImguiVertex));
            s->c.UnmapBuffer(s->vstg);
            std::memcpy(s->c.MapBuffer(s->istg, 0, s->iBytes), dd->indices, size_t(dd->indexCount) * idxStride);
            s->c.UnmapBuffer(s->istg);
        }

        void VRI_CALL CmdCopyImguiData(VriCommandBuffer* cmd, VriImgui* imgui)
        {
            ImguiState* s = Self(imgui);
            if (!s->hasFrame)
                return;
            VriBufferCopyDesc vcp {};
            vcp.size = s->vBytes;
            s->c.CmdCopyBuffer(cmd, s->vbuf, s->vstg, &vcp);
            VriBufferCopyDesc icp {};
            icp.size = s->iBytes;
            s->c.CmdCopyBuffer(cmd, s->ibuf, s->istg, &icp);
            VriBufferBarrierDesc gb[2] {};
            gb[0].buffer        = s->vbuf;
            gb[0].before.access = VriAccess_CopyDestinationWrite;
            gb[0].before.stages = VriPipelineStage_Transfer;
            gb[0].after.access  = VriAccess_VertexBufferRead;
            gb[0].after.stages  = VriPipelineStage_VertexInput;
            gb[1].buffer        = s->ibuf;
            gb[1].before.access = VriAccess_CopyDestinationWrite;
            gb[1].before.stages = VriPipelineStage_Transfer;
            gb[1].after.access  = VriAccess_IndexBufferRead;
            gb[1].after.stages  = VriPipelineStage_VertexInput;
            VriBarrierGroupDesc g {};
            g.buffers   = gb;
            g.bufferNum = 2;
            s->c.CmdBarrier(cmd, &g);
        }

        void VRI_CALL CmdDrawImgui(VriCommandBuffer* cmd, VriImgui* imgui, const VriImguiDrawData* dd)
        {
            ImguiState* s = Self(imgui);
            if (!s->hasFrame || !dd || dd->displaySize[0] <= 0 || dd->displaySize[1] <= 0)
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
            c.CmdSetDescriptorSet(cmd, 0, s->set);
            VriViewport vp {0, 0, float(fbW), float(fbH), 0, 1};
            c.CmdSetViewports(cmd, &vp, 1);
            VriVertexBufferBinding vb {};
            vb.buffer = s->vbuf;
            vb.offset = 0;
            c.CmdSetVertexBuffers(cmd, 0, &vb, 1);
            c.CmdSetIndexBuffer(cmd, s->ibuf, 0, dd->indexSize == 4 ? VriIndexType_UInt32 : VriIndexType_UInt16);

            for (uint32_t i = 0; i < dd->commandCount; ++i)
            {
                const VriImguiDrawCommand& dc = dd->commands[i];
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

        const VriImguiInterface g_imgui = {
            CreateImgui,
            DestroyImgui,
            GetImguiFontView,
            UploadImguiData,
            CmdCopyImguiData,
            CmdDrawImgui,
        };
    } // namespace

    const VriImguiInterface* GetImguiInterface() { return &g_imgui; }
} // namespace vri::core
