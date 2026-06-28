// Regression test for per-instance VERTEX STREAMS (VriVertexStepRate_PerInstance): the
// existing instance test uses SV_InstanceID, which does NOT exercise a per-instance vertex
// buffer. Here a per-instance model matrix lives in stream 1; two instances are translated
// left and right. If a backend treats the instance stream as per-vertex (a real D3D12 bug
// that was caught only by screenshotting examples/instancing), the geometry shears and the
// left/right probes fail. Uses examples/cube/cube_inst.slang + the cube geometry.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>

#include "../examples/cube/mat4.h"

#include "shaders/common/cube_inst_dxbc.h" // g_cubeInstDxbcVS / PS
#include "shaders/common/cube_inst_spv.h"  // g_cubeInstSpv
#include "shaders/common/cube_inst_wgsl.h" // g_cubeInstWgsl

namespace
{
    constexpr uint32_t kW = 128, kH = 64, kT = 64; // width*4 = 512 -> 256-aligned (D3D12 readback)

    struct Vertex
    {
        float px, py, pz;
        float u, v;
    };
    const Vertex kCube[24] = {
        {-0.5f, -0.5f, 0.5f, 0, 1},  {0.5f, -0.5f, 0.5f, 1, 1},  {0.5f, 0.5f, 0.5f, 1, 0},
        {-0.5f, 0.5f, 0.5f, 0, 0},   {0.5f, -0.5f, -0.5f, 0, 1}, {-0.5f, -0.5f, -0.5f, 1, 1},
        {-0.5f, 0.5f, -0.5f, 1, 0},  {0.5f, 0.5f, -0.5f, 0, 0},  {0.5f, -0.5f, 0.5f, 0, 1},
        {0.5f, -0.5f, -0.5f, 1, 1},  {0.5f, 0.5f, -0.5f, 1, 0},  {0.5f, 0.5f, 0.5f, 0, 0},
        {-0.5f, -0.5f, -0.5f, 0, 1}, {-0.5f, -0.5f, 0.5f, 1, 1}, {-0.5f, 0.5f, 0.5f, 1, 0},
        {-0.5f, 0.5f, -0.5f, 0, 0},  {-0.5f, 0.5f, 0.5f, 0, 1},  {0.5f, 0.5f, 0.5f, 1, 1},
        {0.5f, 0.5f, -0.5f, 1, 0},   {-0.5f, 0.5f, -0.5f, 0, 0}, {-0.5f, -0.5f, -0.5f, 0, 1},
        {0.5f, -0.5f, -0.5f, 1, 1},  {0.5f, -0.5f, 0.5f, 1, 0},  {-0.5f, -0.5f, 0.5f, 0, 0},
    };

    void RunInstancing(VriGraphicsAPI api)
    {
        uint16_t idx[36];
        for (uint32_t f = 0; f < 6; ++f)
        {
            uint16_t  b = uint16_t(f * 4);
            uint16_t* o = idx + f * 6;
            o[0]        = b;
            o[1]        = uint16_t(b + 1);
            o[2]        = uint16_t(b + 2);
            o[3]        = b;
            o[4]        = uint16_t(b + 2);
            o[5]        = uint16_t(b + 3);
        }
        // two instances: translated left / right (depth 0.5), scaled small. Stored column-major.
        Mat4 inst[2]  = {Mul(Translate(-0.45f, 0.0f, 0.5f), Scale(0.3f)),
                         Mul(Translate(0.45f, 0.0f, 0.5f), Scale(0.3f))};
        Mat4 viewProj = Transpose(Mat4 {}); // identity -> clip == object space (ortho-ish)

        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        VriDevice* dev      = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("device unavailable - skipping");
            return;
        }
        struct Guard
        {
            VriDevice* d;
            ~Guard() { vriDestroyDevice(d); }
        } guard {dev};

        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        VriShaderDesc sh[2] {};
        sh[0].stage          = VriShaderStage_Vertex;
        sh[0].entryPointName = "vertexMain";
        sh[1].stage          = VriShaderStage_Fragment;
        sh[1].entryPointName = "fragmentMain";
        if (api == VriGraphicsAPI_D3D12)
        {
            sh[0].bytecode     = g_cubeInstDxbcVS;
            sh[0].bytecodeSize = sizeof(g_cubeInstDxbcVS);
            sh[1].bytecode     = g_cubeInstDxbcPS;
            sh[1].bytecodeSize = sizeof(g_cubeInstDxbcPS);
        }
        else if (api == VriGraphicsAPI_WebGPU)
        {
            sh[0].bytecode     = g_cubeInstWgsl;
            sh[0].bytecodeSize = sizeof(g_cubeInstWgsl);
            sh[1].bytecode     = g_cubeInstWgsl;
            sh[1].bytecodeSize = sizeof(g_cubeInstWgsl);
        }
        else
        {
            sh[0].bytecode     = g_cubeInstSpv;
            sh[0].bytecodeSize = sizeof(g_cubeInstSpv);
            sh[1].bytecode     = g_cubeInstSpv;
            sh[1].bytecodeSize = sizeof(g_cubeInstSpv);
        }

        VriTextureDesc td {};
        td.type                    = VriTextureType_2D;
        td.format                  = VriFormat_RGBA8_UNORM;
        td.width                   = kW;
        td.height                  = kH;
        td.depth                   = 1;
        td.mipNum                  = 1;
        td.layerNum                = 1;
        td.sampleNum               = 1;
        td.usage                   = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
        td.memoryLocation          = VriMemoryLocation_Device;
        td.clearValue.color.f32[2] = 1.0f;
        td.clearValue.color.f32[3] = 1.0f; // match the render-pass clear
        VriTexture* color          = nullptr;
        REQUIRE(c.CreateTexture(dev, &td, &color) == VriResult_Success);
        VriTextureViewDesc cvd {};
        cvd.texture              = color;
        cvd.viewType             = VriTextureViewType_2D;
        cvd.format               = VriFormat_Unknown;
        cvd.aspect               = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr;
        REQUIRE(c.CreateTextureView(dev, &cvd, &colorView) == VriResult_Success);
        VriBufferDesc rb {};
        rb.size             = uint64_t(kW) * kH * 4;
        rb.usage            = VriBufferUsage_TransferDst;
        rb.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rb, &readback) == VriResult_Success);

        VriTextureDesc std_ {};
        std_.type           = VriTextureType_2D;
        std_.format         = VriFormat_RGBA8_UNORM;
        std_.width          = kT;
        std_.height         = kT;
        std_.depth          = 1;
        std_.mipNum         = 1;
        std_.layerNum       = 1;
        std_.sampleNum      = 1;
        std_.usage          = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst;
        std_.memoryLocation = VriMemoryLocation_Device;
        VriTexture* sampled = nullptr;
        REQUIRE(c.CreateTexture(dev, &std_, &sampled) == VriResult_Success);
        VriTextureViewDesc svd {};
        svd.texture            = sampled;
        svd.viewType           = VriTextureViewType_2D;
        svd.format             = VriFormat_Unknown;
        svd.aspect             = VriImageAspect_Color;
        VriDescriptor* texView = nullptr;
        REQUIRE(c.CreateTextureView(dev, &svd, &texView) == VriResult_Success);
        const uint64_t texBytes = uint64_t(kT) * kT * 4;
        VriBufferDesc  tsb {};
        tsb.size           = texBytes;
        tsb.usage          = VriBufferUsage_TransferSrc;
        tsb.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* tstg    = nullptr;
        REQUIRE(c.CreateBuffer(dev, &tsb, &tstg) == VriResult_Success);
        {
            uint8_t* m = static_cast<uint8_t*>(c.MapBuffer(tstg, 0, texBytes));
            for (uint64_t i = 0; i < texBytes; i += 4)
            {
                m[i]     = 0;
                m[i + 1] = 255;
                m[i + 2] = 0;
                m[i + 3] = 255;
            }
            c.UnmapBuffer(tstg);
        }
        VriSamplerDesc smp {};
        smp.magFilter          = VriFilter_Nearest;
        smp.minFilter          = VriFilter_Nearest;
        smp.maxLod             = 1.0f;
        smp.addressModeU       = VriAddressMode_Repeat;
        smp.addressModeV       = VriAddressMode_Repeat;
        smp.addressModeW       = VriAddressMode_Repeat;
        VriDescriptor* sampler = nullptr;
        REQUIRE(c.CreateSampler(dev, &smp, &sampler) == VriResult_Success);

        auto devBuf = [&](uint64_t size, VriBufferUsageFlags usage, const void* src, VriBuffer** outStg) {
            VriBufferDesc bd {};
            bd.size           = size;
            bd.usage          = usage | VriBufferUsage_TransferDst;
            bd.memoryLocation = VriMemoryLocation_Device;
            VriBuffer* b      = nullptr;
            c.CreateBuffer(dev, &bd, &b);
            VriBufferDesc sd {};
            sd.size           = size;
            sd.usage          = VriBufferUsage_TransferSrc;
            sd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* s      = nullptr;
            c.CreateBuffer(dev, &sd, &s);
            std::memcpy(c.MapBuffer(s, 0, size), src, size);
            c.UnmapBuffer(s);
            *outStg = s;
            return b;
        };
        VriBuffer *       vstg, *istg, *nstg, *ustg;
        VriBuffer*        vbuf    = devBuf(sizeof(kCube), VriBufferUsage_VertexBuffer, kCube, &vstg);
        VriBuffer*        ibuf    = devBuf(sizeof(idx), VriBufferUsage_IndexBuffer, idx, &istg);
        VriBuffer*        instBuf = devBuf(sizeof(inst), VriBufferUsage_VertexBuffer, inst, &nstg);
        VriBuffer*        ubo     = devBuf(sizeof(Mat4), VriBufferUsage_ConstantBuffer, &viewProj, &ustg);
        VriBufferViewDesc ubv {};
        ubv.buffer             = ubo;
        ubv.viewType           = VriDescriptorType_ConstantBuffer;
        ubv.offset             = 0;
        ubv.size               = sizeof(Mat4);
        VriDescriptor* uboView = nullptr;
        REQUIRE(c.CreateBufferView(dev, &ubv, &uboView) == VriResult_Success);

        VriDescriptorRangeDesc ranges[3] {};
        ranges[0].baseRegister   = 0;
        ranges[0].descriptorNum  = 1;
        ranges[0].descriptorType = VriDescriptorType_ConstantBuffer;
        ranges[0].shaderStages   = VriShaderStage_Vertex;
        ranges[1].baseRegister   = 1;
        ranges[1].descriptorNum  = 1;
        ranges[1].descriptorType = VriDescriptorType_Texture;
        ranges[1].shaderStages   = VriShaderStage_Fragment;
        ranges[2].baseRegister   = 2;
        ranges[2].descriptorNum  = 1;
        ranges[2].descriptorType = VriDescriptorType_Sampler;
        ranges[2].shaderStages   = VriShaderStage_Fragment;
        VriDescriptorSetDesc setDesc {};
        setDesc.registerSpace = 0;
        setDesc.ranges        = ranges;
        setDesc.rangeNum      = 3;
        VriPipelineLayoutDesc ld {};
        ld.descriptorSets         = &setDesc;
        ld.descriptorSetNum       = 1;
        VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriVertexAttributeDesc attrs[6] {};
        attrs[0].format      = VriFormat_RGB32_SFLOAT;
        attrs[0].offset      = 0;
        attrs[0].streamIndex = 0;
        attrs[1].format      = VriFormat_RG32_SFLOAT;
        attrs[1].offset      = 12;
        attrs[1].streamIndex = 0;
        for (int i = 0; i < 4; ++i)
        {
            attrs[2 + i].format      = VriFormat_RGBA32_SFLOAT;
            attrs[2 + i].offset      = uint32_t(i * 16);
            attrs[2 + i].streamIndex = 1;
        }
        VriVertexStreamDesc streams[2] {};
        streams[0].stride      = sizeof(Vertex);
        streams[0].bindingSlot = 0;
        streams[0].stepRate    = VriVertexStepRate_PerVertex;
        streams[1].stride      = sizeof(Mat4);
        streams[1].bindingSlot = 1;
        streams[1].stepRate    = VriVertexStepRate_PerInstance;
        VriColorAttachmentDesc ca {};
        ca.format         = VriFormat_RGBA8_UNORM;
        ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd {};
        pd.pipelineLayout           = layout;
        pd.shaders                  = sh;
        pd.shaderNum                = 2;
        pd.vertexInput.attributes   = attrs;
        pd.vertexInput.attributeNum = 6;
        pd.vertexInput.streams      = streams;
        pd.vertexInput.streamNum    = 2;
        pd.inputAssembly.topology   = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode   = VriCullMode_None;
        pd.rasterization.frontFace  = VriFrontFace_CounterClockwise;
        pd.rasterization.lineWidth  = 1.0f;
        pd.multisample.sampleNum    = 1;
        pd.outputMerger.colors      = &ca;
        pd.outputMerger.colorNum    = 1;
        VriPipeline* pipeline       = nullptr;
        REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

        VriDescriptorPoolDesc pdsc {};
        pdsc.descriptorSetMaxNum  = 1;
        pdsc.constantBufferMaxNum = 1;
        pdsc.textureMaxNum        = 1;
        pdsc.samplerMaxNum        = 1;
        VriDescriptorPool* pool   = nullptr;
        REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
        VriDescriptorSet* set = nullptr;
        REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);
        const VriDescriptor*         d0[1] = {uboView};
        const VriDescriptor*         d1[1] = {texView};
        const VriDescriptor*         d2[1] = {sampler};
        VriDescriptorRangeUpdateDesc upd[3] {};
        upd[0].descriptors   = d0;
        upd[0].descriptorNum = 1;
        upd[1].descriptors   = d1;
        upd[1].descriptorNum = 1;
        upd[2].descriptors   = d2;
        upd[2].descriptorNum = 1;
        c.UpdateDescriptorRanges(set, 0, 3, upd);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        VriBufferCopyDesc cp {};
        cp.size = sizeof(kCube);
        c.CmdCopyBuffer(cmd, vbuf, vstg, &cp);
        cp.size = sizeof(idx);
        c.CmdCopyBuffer(cmd, ibuf, istg, &cp);
        cp.size = sizeof(inst);
        c.CmdCopyBuffer(cmd, instBuf, nstg, &cp);
        cp.size = sizeof(Mat4);
        c.CmdCopyBuffer(cmd, ubo, ustg, &cp);
        {
            VriBufferBarrierDesc bb[4] {};
            bb[0].buffer       = vbuf;
            bb[0].after.access = VriAccess_VertexBufferRead;
            bb[1].buffer       = ibuf;
            bb[1].after.access = VriAccess_IndexBufferRead;
            bb[2].buffer       = instBuf;
            bb[2].after.access = VriAccess_VertexBufferRead;
            bb[3].buffer       = ubo;
            bb[3].after.access = VriAccess_ConstantBufferRead;
            for (auto& b : bb)
            {
                b.before.access = VriAccess_CopyDestinationWrite;
                b.before.stages = VriPipelineStage_Transfer;
                b.after.stages  = VriPipelineStage_VertexInput | VriPipelineStage_VertexShader;
            }
            VriBarrierGroupDesc g {};
            g.buffers   = bb;
            g.bufferNum = 4;
            c.CmdBarrier(cmd, &g);
        }
        {
            VriTextureBarrierDesc tb {};
            tb.texture       = sampled;
            tb.before.layout = VriLayout_Undefined;
            tb.before.stages = VriPipelineStage_None;
            tb.after.access  = VriAccess_CopyDestinationWrite;
            tb.after.layout  = VriLayout_CopyDestination;
            tb.after.stages  = VriPipelineStage_Transfer;
            tb.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        }
        VriBufferTextureCopyDesc tup {};
        tup.texture.aspect   = VriImageAspect_Color;
        tup.texture.layerNum = 1;
        tup.texture.width    = kT;
        tup.texture.height   = kT;
        c.CmdUploadBufferToTexture(cmd, sampled, tstg, &tup);
        {
            VriTextureBarrierDesc tb {};
            tb.texture       = sampled;
            tb.before.access = VriAccess_CopyDestinationWrite;
            tb.before.layout = VriLayout_CopyDestination;
            tb.before.stages = VriPipelineStage_Transfer;
            tb.after.access  = VriAccess_ShaderResourceRead;
            tb.after.layout  = VriLayout_ShaderResource;
            tb.after.stages  = VriPipelineStage_FragmentShader;
            tb.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        }
        {
            VriTextureBarrierDesc tb {};
            tb.texture       = color;
            tb.before.layout = VriLayout_Undefined;
            tb.before.stages = VriPipelineStage_None;
            tb.after.access  = VriAccess_ColorAttachmentWrite;
            tb.after.layout  = VriLayout_ColorAttachment;
            tb.after.stages  = VriPipelineStage_ColorAttachmentOutput;
            tb.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        }
        VriAttachmentDesc rt {};
        rt.view                    = colorView;
        rt.loadOp                  = VriAttachmentLoadOp_Clear;
        rt.storeOp                 = VriAttachmentStoreOp_Store;
        rt.clearValue.color.f32[2] = 1.0f;
        rt.clearValue.color.f32[3] = 1.0f; // blue clear
        VriAttachmentsDesc att {};
        att.colors            = &rt;
        att.colorNum          = 1;
        att.renderArea.width  = kW;
        att.renderArea.height = kH;
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp {0, 0, float(kW), float(kH), 0, 1};
        c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc {0, 0, kW, kH};
        c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, set);
        VriVertexBufferBinding vbs[2] {};
        vbs[0].buffer = vbuf;
        vbs[1].buffer = instBuf;
        c.CmdSetVertexBuffers(cmd, 0, vbs, 2);
        c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);
        VriDrawIndexedDesc di {};
        di.indexNum    = 36;
        di.instanceNum = 2;
        c.CmdDrawIndexed(cmd, &di);
        c.CmdEndRendering(cmd);
        {
            VriTextureBarrierDesc tb {};
            tb.texture       = color;
            tb.before.access = VriAccess_ColorAttachmentWrite;
            tb.before.layout = VriLayout_ColorAttachment;
            tb.before.stages = VriPipelineStage_ColorAttachmentOutput;
            tb.after.access  = VriAccess_CopySourceRead;
            tb.after.layout  = VriLayout_CopySource;
            tb.after.stages  = VriPipelineStage_Transfer;
            tb.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        }
        VriBufferTextureCopyDesc rc {};
        rc.texture.aspect   = VriImageAspect_Color;
        rc.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &rc);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

        VriFenceSubmitDesc signal {};
        signal.fence  = fence;
        signal.value  = 1;
        signal.stages = VriPipelineStage_AllCommands;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        c.QueueSubmit(queue, &submit);
        c.Wait(fence, 1);

        const uint8_t* px    = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, rb.size));
        auto           green = [&](uint32_t x, uint32_t y) {
            const uint8_t* p = px + (uint64_t(y) * kW + x) * 4;
            return p[0] < 64 && p[1] > 192 && p[2] < 64;
        };
        auto blue = [&](uint32_t x, uint32_t y) {
            const uint8_t* p = px + (uint64_t(y) * kW + x) * 4;
            return p[2] > 192 && p[1] < 64;
        };
        CHECK(green(kW / 4, kH / 2));     // left instance
        CHECK(green(3 * kW / 4, kH / 2)); // right instance
        CHECK(blue(kW / 2, kH / 2));      // gap between them (background)
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyDescriptorPool(pool);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyDescriptor(sampler);
        c.DestroyDescriptor(uboView);
        c.DestroyBuffer(ustg);
        c.DestroyBuffer(ubo);
        c.DestroyBuffer(nstg);
        c.DestroyBuffer(instBuf);
        c.DestroyBuffer(istg);
        c.DestroyBuffer(ibuf);
        c.DestroyBuffer(vstg);
        c.DestroyBuffer(vbuf);
        c.DestroyBuffer(tstg);
        c.DestroyDescriptor(texView);
        c.DestroyTexture(sampled);
        c.DestroyBuffer(readback);
        c.DestroyDescriptor(colorView);
        c.DestroyTexture(color);
    }
} // namespace

TEST_CASE("Vulkan: per-instance vertex stream (instanced matrices)") { RunInstancing(VriGraphicsAPI_Vulkan); }
TEST_CASE("D3D12: per-instance vertex stream (instanced matrices)") { RunInstancing(VriGraphicsAPI_D3D12); }
TEST_CASE("OpenGL: per-instance vertex stream (instanced matrices)") { RunInstancing(VriGraphicsAPI_OpenGL); }
TEST_CASE("WebGPU: per-instance vertex stream (instanced matrices)") { RunInstancing(VriGraphicsAPI_WebGPU); }
