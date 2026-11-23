// webgpu_triangle.cpp - WebGPU-on-Emscripten (browser WebGPU) end-to-end check.
// Renders the solid-red triangle to an offscreen texture via the WebGPU backend,
// reads it back, and prints a pass/fail line for a headless-browser grade. Device
// creation + buffer mapping are async in the browser; the backend yields via
// emscripten_sleep (ASYNCIFY) so they resolve.
//
// Build: xmake f -p wasm --vri_backend_wgpu=y; xmake build vri-webgpu-triangle
// Run:   emrun (headless Chrome with --enable-unsafe-webgpu) over the .html

#include "tests/buffer_texture_copy_probe.h"
#include "tests/compressed_format_probe.h"
#include "tests/fence_destroy_probe.h"
#include "tests/map_async_probe.h"
#include "tests/map_cancel_probe.h"
#include "tests/queue_wait_probe.h"
#include "tests/wgpu_completion_probe.h"

#include <vri/vri.h>

#include <cstdint>
#include <cstdio>

#include "shaders/common/triangle_wgsl.h"    // g_triangleWgsl (solid red)
#include "shaders/tests/compute_fill_wgsl.h" // g_computeFillWgsl (out[i] = i + 100)

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

extern "C" int vriTestTimelineC(VriDevice* device, size_t coreSize);

namespace
{
    constexpr uint32_t kW = 64;
    constexpr uint32_t kH = 64;

    bool CompletionProbe()
    {
        auto creation        = vriDefaultDeviceCreationDesc();
        creation.graphicsAPI = VriGraphicsAPI_WebGPU;
        VriDevice* device    = nullptr;
        if (vriCreateDevice(&creation, &device) != VriResult_Success)
        {
            return false;
        }
        VriCoreInterface core {};
        const auto       result = vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(core), &core);
        const int        failure =
            result == VriResult_Success ? vri::test::CheckWebGPUCompletionRetirement(device, core) : __LINE__;
        if (failure)
        {
            std::printf("WebGPU completion retirement failed at line %d\n", failure);
        }
        vriDestroyDevice(device);
        return failure == 0;
    }

    // The sRGB and signed block-compressed formats the browser grants survive an upload/readback round trip.
    bool CompressedProbe()
    {
        auto creation        = vriDefaultDeviceCreationDesc();
        creation.graphicsAPI = VriGraphicsAPI_WebGPU;
        VriDevice* device    = nullptr;
        if (vriCreateDevice(&creation, &device) != VriResult_Success)
            return false;
        VriCoreInterface c {};
        bool             ok = vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success;
        for (const vri::test::CompressedFormat& f : vri::test::kAddedCompressedFormats)
        {
            if (!ok || (c.GetFormatSupport(device, f.format) & VriFormatSupport_Texture) == 0)
                continue;
            const int failure = vri::test::CheckCompressedRoundTrip(device, c, f);
            std::printf("VRI_WEBGPU_STEP: %s round trip %s\n", f.name, failure ? "FAILED" : "ok");
            ok = failure == 0;
        }
        vriDestroyDevice(device);
        return ok;
    }

    // MapBufferAsync maps a GPU buffer from the browser event loop: the callback runs after the call
    // returns, once, with the copied data.
    bool MapAsyncProbe()
    {
        auto creation        = vriDefaultDeviceCreationDesc();
        creation.graphicsAPI = VriGraphicsAPI_WebGPU;
        VriDevice* device    = nullptr;
        if (vriCreateDevice(&creation, &device) != VriResult_Success)
            return false;
        VriCoreInterface core {};
        int              failure = __LINE__;
        if (vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(core), &core) == VriResult_Success)
            failure = vri::test::CheckMapBufferAsync(device, core, true);
        if (failure)
            std::printf("WebGPU MapBufferAsync failed at line %d\n", failure);
        vriDestroyDevice(device);
        return failure == 0;
    }

    // A feature this backend does not grant: refused without bestEffort, absent from enabledFeatures with it.
    bool FeatureProbe(VriGraphicsAPI api)
    {
        VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI           = api;
        dc.enabledFeatures       = VriFeature_RayTracing;
        dc.bestEffort            = VRI_FALSE;
        VriDevice*      dev      = nullptr;
        const VriResult strict   = vriCreateDevice(&dc, &dev);
        if (dev)
            vriDestroyDevice(dev);
        dc.bestEffort = VRI_TRUE;
        dev           = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;
        VriCoreInterface c {};
        const bool       granted = vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success &&
                             c.GetDeviceDesc(dev)->enabledFeatures == 0;
        vriDestroyDevice(dev);
        return strict == VriResult_Unsupported && granted;
    }

    // Buffer <-> texture copies with padded rows and slices: an odd-width R16F texture and a 3D texture.
    bool CopyProbe()
    {
        auto creation        = vriDefaultDeviceCreationDesc();
        creation.graphicsAPI = VriGraphicsAPI_WebGPU;
        VriDevice* device    = nullptr;
        if (vriCreateDevice(&creation, &device) != VriResult_Success)
            return false;
        VriCoreInterface c {};
        bool             ok = vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success;
        for (const vri::test::CopyCase& k : vri::test::kCopyCases)
        {
            if (ok && !vri::test::CopyCaseSupported(device, c, k))
            {
                std::printf("copy of format %d cannot be read back here - skipped\n", static_cast<int>(k.format));
                continue;
            }
            const int failure = ok ? vri::test::CheckBufferTextureRoundTrip(device, c, k) : __LINE__;
            if (failure)
                std::printf("WebGPU copy of format %d failed at line %d\n", static_cast<int>(k.format), failure);
            ok = ok && failure == 0;
        }
        vriDestroyDevice(device);
        return ok;
    }

    // A submission waiting on an earlier submission's fence: both submits succeed in the same turn of the event
    // loop without a CPU wait (the fence cannot advance before the page yields, so it still reads 0), and the
    // second submission sees the first's data once the page has yielded. A never-signalled value is refused.
    bool QueueWaitProbe()
    {
        VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI           = VriGraphicsAPI_WebGPU;
        VriDevice* dev           = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;
        VriCoreInterface             c {};
        vri::test::OrderedWaitResult r;
        const bool ran = vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success &&
                         vri::test::RunOrderedWait(dev, c, r) == 0;
        const VriResult unsignaled = ran ? vri::test::SubmitUnsignaledWait(dev, c) : VriResult_Failure;
        std::printf("VRI_WEBGPU_STEP: queue wait submits %d/%d, fence after submit %llu, data %d, unsignaled wait %d\n",
                    static_cast<int>(r.submitA),
                    static_cast<int>(r.submitB),
                    static_cast<unsigned long long>(r.fenceAfterSubmit),
                    r.dataOk ? 1 : 0,
                    static_cast<int>(unsignaled));
        vriDestroyDevice(dev);
        return ran && r.submitA == VriResult_Success && r.submitB == VriResult_Success && r.fenceAfterSubmit == 0 &&
               r.dataOk && unsignaled == VriResult_InvalidArgument;
    }

    // A fence destroyed while its submission is in flight (the browser cannot wait for it first): the queue stays
    // usable, and a later fence still completes.
    bool FenceDestroyProbe()
    {
        auto creation        = vriDefaultDeviceCreationDesc();
        creation.graphicsAPI = VriGraphicsAPI_WebGPU;
        VriDevice* device    = nullptr;
        if (vriCreateDevice(&creation, &device) != VriResult_Success)
            return false;
        VriCoreInterface c {};
        int              failure = __LINE__;
        if (vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success)
            failure = vri::test::CheckFenceDestroyedInFlight(device, c);
        if (failure)
            std::printf("WebGPU fence destroyed in flight failed at line %d\n", failure);
        vriDestroyDevice(device);
        return failure == 0;
    }

    // DestroyBuffer while a MapBufferAsync is in flight cancels it: one failed callback before DestroyBuffer
    // returns, nothing more once the page yields, and the device keeps working. UnmapBuffer aborts a map inside the
    // call.
    bool MapCancelProbe()
    {
        const int failure = vri::test::CheckMapCancelledByDestroy();
        std::printf("VRI_WEBGPU_STEP: map cancelled by destroy %s (line %d)\n", failure ? "FAILED" : "ok", failure);
        return failure == 0;
    }

    bool RenderAndProbe(bool& centerRed)
    {
        centerRed                 = false;
        VriDeviceCreationDesc dc  = vriDefaultDeviceCreationDesc();
        dc.coordinates.clipSpaceY = VriClipSpaceY_Up;
        dc.graphicsAPI            = VriGraphicsAPI_WebGPU;
        dc.enableValidation       = VRI_TRUE;
        dc.bestEffort             = VRI_TRUE;
        VriDevice* dev            = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            std::printf("VRI_WEBGPU: device creation FAILED\n");
            return false;
        }
        std::printf("VRI_WEBGPU_STEP: device-created\n");
        std::fflush(stdout);

        VriCoreInterface c {};
        if (vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) != VriResult_Success)
            return false;
        VriQueue* queue = nullptr;
        c.GetQueue(dev, VriQueueType_Graphics, 0, &queue);
        const int timelineFailure = vriTestTimelineC(dev, sizeof(c));
        if (timelineFailure)
        {
            std::printf("Timeline C ABI regression failed at line %d\n", timelineFailure);
            vriDestroyDevice(dev);
            return false;
        }

        VriTextureDesc td {};
        td.type           = VriTextureType_2D;
        td.format         = VriFormat_RGBA8_UNORM;
        td.width          = kW;
        td.height         = kH;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = 1;
        td.sampleNum      = 1;
        td.usage          = VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* color = nullptr;
        c.CreateTexture(dev, &td, &color);
        VriTextureViewDesc vdsc {};
        vdsc.texture        = color;
        vdsc.viewType       = VriTextureViewType_2D;
        vdsc.format         = VriFormat_Unknown;
        vdsc.aspect         = VriImageAspect_Color;
        VriDescriptor* view = nullptr;
        c.CreateTextureView(dev, &vdsc, &view);

        VriBufferDesc bd {};
        bd.size             = static_cast<uint64_t>(kW) * kH * 4;
        bd.usage            = VriBufferUsage_TransferDst;
        bd.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        c.CreateBuffer(dev, &bd, &readback);

        VriPipelineLayoutDesc ld {};
        VriPipelineLayout*    layout = nullptr;
        c.CreatePipelineLayout(dev, &ld, &layout);
        VriShaderDesc sh[2] {};
        sh[0].stage          = VriShaderStage_Vertex;
        sh[0].bytecode       = g_triangleWgsl;
        sh[0].bytecodeSize   = sizeof(g_triangleWgsl);
        sh[0].entryPointName = "vertexMain";
        sh[1].stage          = VriShaderStage_Fragment;
        sh[1].bytecode       = g_triangleWgsl;
        sh[1].bytecodeSize   = sizeof(g_triangleWgsl);
        sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca {};
        ca.format         = VriFormat_RGBA8_UNORM;
        ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd {};
        pd.pipelineLayout          = layout;
        pd.shaders                 = sh;
        pd.shaderNum               = 2;
        pd.inputAssembly.topology  = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode  = VriCullMode_None;
        pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum   = 1;
        pd.outputMerger.colors     = &ca;
        pd.outputMerger.colorNum   = 1;
        VriPipeline* pipeline      = nullptr;
        if (c.CreateGraphicsPipeline(dev, &pd, &pipeline) != VriResult_Success)
        {
            std::printf("VRI_WEBGPU: pipeline creation FAILED\n");
            return false;
        }
        std::printf("VRI_WEBGPU_STEP: pipeline-created\n");
        std::fflush(stdout);

        VriCommandAllocator* alloc = nullptr;
        c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
        VriCommandBuffer* cmd = nullptr;
        c.CreateCommandBuffer(alloc, &cmd);
        VriFence* fence = nullptr;
        c.CreateFence(dev, 0, &fence);

        c.BeginCommandBuffer(cmd);
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
        rt.view                    = view;
        rt.loadOp                  = VriAttachmentLoadOp_Clear;
        rt.storeOp                 = VriAttachmentStoreOp_Store;
        rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att {};
        att.colors            = &rt;
        att.colorNum          = 1;
        att.renderArea.width  = kW;
        att.renderArea.height = kH;
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp {0, 0, static_cast<float>(kW), static_cast<float>(kH), 0, 1};
        c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc {0, 0, kW, kH};
        c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, pipeline);
        VriDrawDesc draw {};
        draw.vertexNum   = 3;
        draw.instanceNum = 1;
        c.CmdDraw(cmd, &draw);
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
        VriBufferTextureCopyDesc copy {};
        copy.texture.aspect   = VriImageAspect_Color;
        copy.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, color, &copy);
        c.EndCommandBuffer(cmd);

        VriFenceSubmitDesc sig {};
        sig.fence = fence;
        sig.value = 1;
        VriQueueSubmitDesc sub {};
        sub.commandBuffers   = &cmd;
        sub.commandBufferNum = 1;
        sub.signalFences     = &sig;
        sub.signalFenceNum   = 1;
        c.QueueSubmit(queue, &sub);
        c.Wait(fence, 1);
        std::printf("VRI_WEBGPU_STEP: submitted\n");
        std::fflush(stdout);

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bd.size));
        const bool     ok = px != nullptr;
        if (ok)
        {
            const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
            centerRed        = px[o + 0] == 255 && px[o + 1] == 0 && px[o + 2] == 0;
            c.UnmapBuffer(readback);
        }

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(readback);
        c.DestroyDescriptor(view);
        c.DestroyTexture(color);
        vriDestroyDevice(dev);
        return ok;
    }
    // Browser WebGPU compute: a 64-wide workgroup fills a storage buffer with
    // out[i] = i + 100; read back and verify every element. (WebGL2 cannot do this.)
    bool ComputeProbe()
    {
        VriDeviceCreationDesc dc  = vriDefaultDeviceCreationDesc();
        dc.coordinates.clipSpaceY = VriClipSpaceY_Up;
        dc.graphicsAPI            = VriGraphicsAPI_WebGPU;
        dc.enableValidation       = VRI_TRUE;
        dc.bestEffort             = VRI_TRUE;
        VriDevice* dev            = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;

        VriCoreInterface c {};
        if (vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) != VriResult_Success)
            return false;
        if (c.GetDeviceDesc(dev)->hasComputeShader == VRI_FALSE)
        {
            vriDestroyDevice(dev);
            return false;
        }
        VriQueue* queue = nullptr;
        c.GetQueue(dev, VriQueueType_Graphics, 0, &queue);

        constexpr uint32_t kCount = 64;
        VriBufferDesc      sbd {};
        sbd.size           = kCount * 4 * 2;
        sbd.usage          = VriBufferUsage_StorageBuffer | VriBufferUsage_TransferSrc;
        sbd.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* storage = nullptr;
        c.CreateBuffer(dev, &sbd, &storage);
        VriBufferDesc rbd {};
        rbd.size            = kCount * 4;
        rbd.usage           = VriBufferUsage_TransferDst;
        rbd.memoryLocation  = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        c.CreateBuffer(dev, &rbd, &readback);

        VriDescriptorRangeDesc ranges[2] {};
        for (auto& range : ranges)
        {
            range.descriptorNum  = 1;
            range.descriptorType = VriDescriptorType_StorageBuffer;
            range.shaderStages   = VriShaderStage_Compute;
        }
        ranges[0].baseRegister = 7;
        ranges[1].baseRegister = 0;
        VriDescriptorSetDesc setDesc {};
        setDesc.ranges   = ranges;
        setDesc.rangeNum = 2;
        VriPipelineLayoutDesc ld {};
        ld.descriptorSets         = &setDesc;
        ld.descriptorSetNum       = 1;
        VriPipelineLayout* layout = nullptr;
        c.CreatePipelineLayout(dev, &ld, &layout);

        VriComputePipelineDesc cpd {};
        cpd.pipelineLayout        = layout;
        cpd.shader.stage          = VriShaderStage_Compute;
        cpd.shader.bytecode       = g_computeFillWgsl;
        cpd.shader.bytecodeSize   = sizeof(g_computeFillWgsl);
        cpd.shader.entryPointName = "computeMain";
        VriPipeline* pipeline     = nullptr;
        if (c.CreateComputePipeline(dev, &cpd, &pipeline) != VriResult_Success)
        {
            vriDestroyDevice(dev);
            return false;
        }

        VriDescriptorPoolDesc pdsc {};
        pdsc.descriptorSetMaxNum = 1;
        pdsc.storageBufferMaxNum = 2;
        VriDescriptorPool* pool  = nullptr;
        c.CreateDescriptorPool(dev, &pdsc, &pool);
        VriDescriptorSet* set = nullptr;
        c.AllocateDescriptorSets(pool, layout, 0, &set, 1);
        VriBufferViewDesc bv {};
        bv.buffer             = storage;
        bv.viewType           = VriDescriptorType_StorageBuffer;
        bv.offset             = 0;
        bv.size               = kCount * 4;
        VriDescriptor* sbView = nullptr;
        c.CreateBufferView(dev, &bv, &sbView);
        bv.offset                = kCount * 4;
        VriDescriptor* spareView = nullptr;
        c.CreateBufferView(dev, &bv, &spareView);
        const VriDescriptor*         spareDescriptors[1] = {spareView};
        const VriDescriptor*         descs[1]            = {sbView};
        VriDescriptorRangeUpdateDesc upd {};
        upd.descriptors    = descs;
        upd.descriptorNum  = 1;
        upd.baseDescriptor = 0;
        // The second write must preserve the first range, regardless of binding order.
        c.UpdateDescriptorRanges(set, 1, 1, &upd);
        upd.descriptors = spareDescriptors;
        c.UpdateDescriptorRanges(set, 0, 1, &upd);
        upd.descriptors = descs;
        c.UpdateDescriptorRanges(set, 1, 1, &upd);

        VriCommandAllocator* alloc = nullptr;
        c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
        VriCommandBuffer* cmd = nullptr;
        c.CreateCommandBuffer(alloc, &cmd);
        VriFence* fence = nullptr;
        c.CreateFence(dev, 0, &fence);

        c.BeginCommandBuffer(cmd);
        c.CmdSetPipeline(cmd, pipeline);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, set);
        VriDispatchDesc disp {};
        disp.x = 1;
        disp.y = 1;
        disp.z = 1;
        c.CmdDispatch(cmd, &disp);
        VriBufferCopyDesc copy {};
        copy.size = rbd.size;
        c.CmdCopyBuffer(cmd, readback, storage, &copy);
        c.EndCommandBuffer(cmd);

        VriFenceSubmitDesc sig {};
        sig.fence = fence;
        sig.value = 1;
        VriQueueSubmitDesc sub {};
        sub.commandBuffers   = &cmd;
        sub.commandBufferNum = 1;
        sub.signalFences     = &sig;
        sub.signalFenceNum   = 1;
        c.QueueSubmit(queue, &sub);
        c.Wait(fence, 1);

        const uint32_t* px    = static_cast<const uint32_t*>(c.MapBuffer(readback, 0, rbd.size));
        bool            allOk = px != nullptr;
        if (px)
        {
            for (uint32_t i = 0; i < kCount; ++i)
                if (px[i] != i + 100u)
                {
                    allOk = false;
                    break;
                }
            c.UnmapBuffer(readback);
        }

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyDescriptor(sbView);
        c.DestroyDescriptor(spareView);
        c.DestroyDescriptorPool(pool);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(storage);
        c.DestroyBuffer(readback);
        vriDestroyDevice(dev);
        return allOk;
    }

    // A canvas swapchain reports the Replaced texture model: one texture, index 0 at every acquire, and a view made
    // after each acquire that is used only until the Present.
    bool SwapChainModelProbe()
    {
        VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI           = VriGraphicsAPI_WebGPU;
        dc.bestEffort            = VRI_TRUE;
        VriDevice* dev           = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;
        VriCoreInterface      c {};
        VriSwapChainInterface sw {};
        VriQueue*             queue = nullptr;
        if (vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) != VriResult_Success ||
            vriGetInterface(dev, VRI_INTERFACE_SWAPCHAIN, sizeof(sw), &sw) != VriResult_Success ||
            c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) != VriResult_Success)
        {
            vriDestroyDevice(dev);
            return false;
        }
        VriSwapChainDesc sd {};
        sd.window.type                = VriWindowSystem_Web;
        sd.window.handle.web.selector = "#canvas";
        sd.queue                      = queue;
        sd.width                      = 64;
        sd.height                     = 32;
        sd.textureNum                 = 2;
        VriSwapChain* swc             = nullptr;
        VriResult     created         = VriResult_Unsupported;
        for (const VriFormat format : {VriFormat_BGRA8_UNORM, VriFormat_RGBA8_UNORM})
        {
            sd.format = format;
            created   = sw.CreateSwapChain(dev, &sd, &swc);
            if (created != VriResult_Unsupported)
                break;
        }
        if (created != VriResult_Success)
        {
            std::printf("VRI_WEBGPU_STEP: swapchain texture model: CreateSwapChain %d\n", static_cast<int>(created));
            vriDestroyDevice(dev);
            return false;
        }
        const VriSwapChainTextureModel model = sw.GetSwapChainTextureModel(swc);
        uint32_t                       count = 0;
        sw.GetSwapChainTextures(swc, nullptr, &count);
        VriTexture* texture = nullptr;
        uint32_t    one     = 1;
        sw.GetSwapChainTextures(swc, &texture, &one);

        VriCommandAllocator* alloc = nullptr;
        VriCommandBuffer*    cmd   = nullptr;
        c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
        c.CreateCommandBuffer(alloc, &cmd);
        bool framesOk = alloc && cmd && texture;
        for (int frame = 0; frame < 2 && framesOk; ++frame)
        {
            uint32_t index = 1;
            framesOk       = sw.AcquireNextTexture(swc, nullptr, 0, &index) == VriResult_Success && index == 0;
            VriTextureViewDesc vd {};
            vd.texture          = texture;
            vd.viewType         = VriTextureViewType_2D;
            vd.aspect           = VriImageAspect_Color;
            VriDescriptor* view = nullptr;
            framesOk            = framesOk && c.CreateTextureView(dev, &vd, &view) == VriResult_Success;
            if (!framesOk)
                break;
            c.ResetCommandAllocator(alloc);
            c.BeginCommandBuffer(cmd);
            VriAttachmentDesc rt {};
            rt.view                    = view;
            rt.loadOp                  = VriAttachmentLoadOp_Clear;
            rt.storeOp                 = VriAttachmentStoreOp_Store;
            rt.clearValue.color.f32[3] = 1.0f;
            VriAttachmentsDesc att {};
            att.colors            = &rt;
            att.colorNum          = 1;
            att.renderArea.width  = 64;
            att.renderArea.height = 32;
            att.layerNum          = 1;
            c.CmdBeginRendering(cmd, &att);
            c.CmdEndRendering(cmd);
            c.EndCommandBuffer(cmd);
            VriQueueSubmitDesc submit {};
            submit.commandBuffers   = &cmd;
            submit.commandBufferNum = 1;
            framesOk =
                c.QueueSubmit(queue, &submit) == VriResult_Success && sw.Present(swc, nullptr, 0) == VriResult_Success;
            c.DestroyDescriptor(view); // valid only until the Present
        }
        std::printf("VRI_WEBGPU_STEP: swapchain texture model %d (Replaced = %d), textures %u, frames %s\n",
                    static_cast<int>(model),
                    static_cast<int>(VriSwapChainTextureModel_Replaced),
                    count,
                    framesOk ? "ok" : "FAILED");
        c.DestroyCommandAllocator(alloc);
        sw.DestroySwapChain(swc);
        vriDestroyDevice(dev);
        return model == VriSwapChainTextureModel_Replaced && count == 1 && framesOk;
    }
} // namespace

int main()
{
    std::printf("VRI_WEBGPU_STEP: main\n");
    std::fflush(stdout);
    bool       centerRed = false;
    const bool ran       = RenderAndProbe(centerRed);
    std::printf("VRI_WEBGPU_STEP: triangle done\n");
    std::fflush(stdout);
    const bool compute = ComputeProbe();
    std::printf("VRI_WEBGPU_STEP: compute done\n");
    std::fflush(stdout);
    const bool completion = CompletionProbe();
    const bool mapAsync   = MapAsyncProbe();
    const bool compressed = CompressedProbe();
    const bool features   = FeatureProbe(VriGraphicsAPI_WebGPU);
    const bool copies     = CopyProbe();
    const bool queueWait  = QueueWaitProbe();
    const bool fences     = FenceDestroyProbe();
    const bool mapCancel  = MapCancelProbe();
    const bool swapModel  = SwapChainModelProbe();
    const bool pass       = ran && centerRed && compute && completion && mapAsync && compressed && features && copies &&
                      queueWait && fences && mapCancel && swapModel;
    std::printf("VRI_WEBGPU_RESULT: %s (ran=%d centerRed=%d compute=%d completion=%d mapAsync=%d compressed=%d "
                "features=%d copies=%d queueWait=%d fences=%d mapCancel=%d swapModel=%d)\n",
                pass ? "PASS" : "FAIL",
                ran ? 1 : 0,
                centerRed ? 1 : 0,
                compute ? 1 : 0,
                completion ? 1 : 0,
                mapAsync ? 1 : 0,
                compressed ? 1 : 0,
                features ? 1 : 0,
                copies ? 1 : 0,
                queueWait ? 1 : 0,
                fences ? 1 : 0,
                mapCancel ? 1 : 0,
                swapModel ? 1 : 0);
    std::fflush(stdout);
#if defined(__EMSCRIPTEN__)
    emscripten_force_exit(pass ? 0 : 1);
#endif
    return pass ? 0 : 1;
}
