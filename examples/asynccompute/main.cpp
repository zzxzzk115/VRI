// Async compute: the plasma from `computeshader`, moved to the DEDICATED COMPUTE QUEUE so it
// runs concurrently with the graphics queue's frame. Demonstrates the two pieces cross-queue
// work needs and nothing else:
//
//   1. Vri*Usage_ConcurrentQueues - the storage images are written by the compute queue and
//      sampled by the graphics queue with no ownership-transfer barriers (Vulkan: the flag
//      creates them VK_SHARING_MODE_CONCURRENT across the device's distinct families).
//   2. Timeline-fence wiring across queues - each compute submit GPU-waits the previous
//      graphics frame (VriQueueSubmitDesc::waitFences on the app's frame fence) and signals
//      its own fence, which the host waits before a graphics frame samples that batch.
//
// Double-buffered: the batch submitted at frame N writes img[N&1] while the graphics frame
// samples img[(N-1)&1] (primed once before the loop), so the two queues genuinely overlap
// inside a frame. Without a dedicated compute queue VRI aliases VriQueueType_Compute to
// graphics and everything still runs, just serialized. Requires compute - no WebGL2 fallback
// on purpose; the point IS the second queue.
#include "../common/example_app.h"

#include <cstring>

#include "shaders/examples/compute_plasma_dxbc.h" // g_computePlasmaDxbcCS (D3D12)
#include "shaders/examples/compute_plasma_spv.h"  // g_computePlasmaSpv  (Vulkan/OpenGL)
#include "shaders/examples/compute_plasma_wgsl.h" // g_computePlasmaWgsl (WebGPU)
#include "shaders/examples/show_tex_dxbc.h"       // g_showTexDxbcVS / PS
#include "shaders/examples/show_tex_spv.h"        // g_showTexSpv  (display: sample the result)
#include "shaders/examples/show_tex_wgsl.h"       // g_showTexWgsl

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480;
    struct Params
    {
        float    time;
        uint32_t width, height, pad;
    };

    vriex::ExampleApp app;

    // Everything a batch needs, file-scope: an example is a singleton and the recording path
    // is shared by the pre-loop priming batch and the per-frame one.
    VriBuffer *          g_ubo = nullptr, *g_ustg = nullptr;
    VriTexture*          g_img[2]          = {};
    VriDescriptorSet*    g_computeSet[2]   = {};
    VriDescriptorSet*    g_displaySet[2]   = {};
    VriPipelineLayout*   g_computeLayout   = nullptr;
    VriPipeline*         g_computePipeline = nullptr;
    VriQueue*            g_computeQueue    = nullptr;
    VriCommandAllocator* g_computeAlloc    = nullptr;
    VriCommandBuffer*    g_computeCmd      = nullptr;
    VriFence*            g_computeFence    = nullptr;
    uint64_t             g_computeValue    = 0;
    uint64_t             g_frameValue      = 0; // mirrored from onUpdate for onRecord
    bool                 g_imgInit[2]      = {false, false};
    float                g_time            = 0.0f;

    // Record one plasma batch into g_img[slot] and submit it on the compute queue.
    // waitGraphicsValue > 0 adds the GPU-side cross-queue wait on the app's frame fence -
    // already satisfied in this synchronous host, but it is exactly the wiring a pipelined
    // host needs, so the example shows it live.
    void SubmitPlasmaBatch(const uint32_t slot, const uint64_t waitGraphicsValue)
    {
        VriCoreInterface& c = app.c;

        // One cmd buffer, reused: the caller guarantees the previous batch retired
        // (Wait(g_computeFence, g_computeValue)).
        Params p {};
        p.time   = g_time;
        p.width  = kWidth;
        p.height = kHeight;
        std::memcpy(c.MapBuffer(g_ustg, 0, sizeof(Params)), &p, sizeof(Params));
        c.UnmapBuffer(g_ustg);

        c.ResetCommandAllocator(g_computeAlloc);
        c.BeginCommandBuffer(g_computeCmd);

        VriBufferCopyDesc ucp {};
        ucp.size = sizeof(Params);
        c.CmdCopyBuffer(g_computeCmd, g_ubo, g_ustg, &ucp);
        VriBufferBarrierDesc ub {};
        ub.buffer        = g_ubo;
        ub.before.access = VriAccess_CopyDestinationWrite;
        ub.before.stages = VriPipelineStage_Transfer;
        ub.after.access  = VriAccess_ConstantBufferRead;
        ub.after.stages  = VriPipelineStage_ComputeShader;
        VriBarrierGroupDesc ubg {};
        ubg.buffers   = &ub;
        ubg.bufferNum = 1;
        c.CmdBarrier(g_computeCmd, &ubg);

        // Storage-writable. The before half names the last GRAPHICS use (sampled two frames
        // ago); execution order across queues is the fences' job, the barrier only moves the
        // layout - legal on either queue for a concurrent-shared image.
        VriTextureBarrierDesc tw {};
        tw.texture       = g_img[slot];
        tw.before.layout = g_imgInit[slot] ? VriLayout_ShaderResource : VriLayout_Undefined;
        tw.before.stages = VriPipelineStage_None;
        tw.after.access  = VriAccess_ShaderResourceStorageWrite;
        tw.after.layout  = VriLayout_ShaderResourceStorage;
        tw.after.stages  = VriPipelineStage_ComputeShader;
        tw.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc gw {};
        gw.textures   = &tw;
        gw.textureNum = 1;
        c.CmdBarrier(g_computeCmd, &gw);
        g_imgInit[slot] = true;

        c.CmdSetPipeline(g_computeCmd, g_computePipeline);
        c.CmdSetPipelineLayout(g_computeCmd, g_computeLayout);
        c.CmdSetDescriptorSet(g_computeCmd, 0, g_computeSet[slot]);
        VriDispatchDesc disp {};
        disp.x = (kWidth + 7) / 8;
        disp.y = (kHeight + 7) / 8;
        disp.z = 1;
        c.CmdDispatch(g_computeCmd, &disp);

        VriTextureBarrierDesc tr {};
        tr.texture       = g_img[slot];
        tr.before.access = VriAccess_ShaderResourceStorageWrite;
        tr.before.layout = VriLayout_ShaderResourceStorage;
        tr.before.stages = VriPipelineStage_ComputeShader;
        tr.after.access  = VriAccess_ShaderResourceRead;
        tr.after.layout  = VriLayout_ShaderResource;
        tr.after.stages  = VriPipelineStage_FragmentShader;
        tr.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc gr {};
        gr.textures   = &tr;
        gr.textureNum = 1;
        c.CmdBarrier(g_computeCmd, &gr);
        c.EndCommandBuffer(g_computeCmd);

        VriFenceSubmitDesc wait {};
        wait.fence  = app.fence;
        wait.value  = waitGraphicsValue;
        wait.stages = VriPipelineStage_ComputeShader;
        VriFenceSubmitDesc sig {};
        sig.fence  = g_computeFence;
        sig.value  = ++g_computeValue;
        sig.stages = VriPipelineStage_AllCommands;
        VriQueueSubmitDesc sub {};
        sub.waitFences       = waitGraphicsValue ? &wait : nullptr;
        sub.waitFenceNum     = waitGraphicsValue ? 1u : 0u;
        sub.commandBuffers   = &g_computeCmd;
        sub.commandBufferNum = 1;
        sub.signalFences     = &sig;
        sub.signalFenceNum   = 1;
        c.QueueSubmit(g_computeQueue, &sub);
    }
} // namespace

int main(int, char**)
{
    app.Init("asynccompute", kWidth, kHeight, /*hasDepth*/ false);
    app.SetClearColor(0.0f, 0.0f, 0.0f);
    VriCoreInterface& c = app.c;
    if (c.GetDeviceDesc(app.dev)->hasComputeShader == VRI_FALSE)
        app.Fail("asynccompute needs compute shaders (no fallback - the example IS the second queue)");

    c.GetQueue(app.dev, VriQueueType_Compute, 0, &g_computeQueue);
    std::printf("[asynccompute] compute queue %s\n",
                g_computeQueue == app.queue ? "aliases graphics (no dedicated queue) - correct, just serialized" :
                                              "is dedicated - the plasma overlaps the graphics frame");
    std::fflush(stdout);

    // ---- shared params (host writes, compute queue copies + reads) ---------------------
    VriBufferDesc ubd {};
    ubd.size           = sizeof(Params);
    ubd.usage          = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst;
    ubd.memoryLocation = VriMemoryLocation_Device;
    c.CreateBuffer(app.dev, &ubd, &g_ubo);
    VriBufferDesc usd {};
    usd.size           = sizeof(Params);
    usd.usage          = VriBufferUsage_TransferSrc;
    usd.memoryLocation = VriMemoryLocation_HostUpload;
    c.CreateBuffer(app.dev, &usd, &g_ustg);
    VriBufferViewDesc ubv {};
    ubv.buffer             = g_ubo;
    ubv.viewType           = VriDescriptorType_ConstantBuffer;
    ubv.offset             = 0;
    ubv.size               = sizeof(Params);
    VriDescriptor* uboView = nullptr;
    c.CreateBufferView(app.dev, &ubv, &uboView);

    // ---- the cross-queue surfaces: two storage images, CONCURRENT sharing --------------
    // Written on the compute queue, sampled on the graphics queue. Without ConcurrentQueues
    // this would need queue-family ownership transfers, which VRI's barrier API deliberately
    // does not expose.
    VriDescriptor* imgView[2] = {};
    for (int i = 0; i < 2; ++i)
    {
        VriTextureDesc td {};
        td.type      = VriTextureType_2D;
        td.format    = VriFormat_RGBA8_UNORM;
        td.width     = kWidth;
        td.height    = kHeight;
        td.depth     = 1;
        td.mipNum    = 1;
        td.layerNum  = 1;
        td.sampleNum = 1;
        td.usage =
            VriTextureUsage_ShaderResourceStorage | VriTextureUsage_ShaderResource | VriTextureUsage_ConcurrentQueues;
        td.memoryLocation = VriMemoryLocation_Device;
        if (c.CreateTexture(app.dev, &td, &g_img[i]) != VriResult_Success)
            app.Fail("CreateTexture (storage) failed");
        VriTextureViewDesc vd {};
        vd.texture  = g_img[i];
        vd.viewType = VriTextureViewType_2D;
        vd.format   = VriFormat_Unknown;
        vd.aspect   = VriImageAspect_Color;
        c.CreateTextureView(app.dev, &vd, &imgView[i]);
    }
    VriSamplerDesc smp {};
    smp.magFilter          = VriFilter_Linear;
    smp.minFilter          = VriFilter_Linear;
    smp.mipmapMode         = VriMipmapMode_Nearest;
    smp.addressModeU       = VriAddressMode_ClampToEdge;
    smp.addressModeV       = VriAddressMode_ClampToEdge;
    smp.addressModeW       = VriAddressMode_ClampToEdge;
    smp.maxLod             = 1.0f;
    VriDescriptor* sampler = nullptr;
    c.CreateSampler(app.dev, &smp, &sampler);

    // ---- pipelines (identical to computeshader) ----------------------------------------
    VriDescriptorRangeDesc cr[2] {};
    cr[0].baseRegister   = 0;
    cr[0].descriptorNum  = 1;
    cr[0].descriptorType = VriDescriptorType_ConstantBuffer;
    cr[0].shaderStages   = VriShaderStage_Compute;
    cr[1].baseRegister   = 1;
    cr[1].descriptorNum  = 1;
    cr[1].descriptorType = VriDescriptorType_StorageTexture;
    cr[1].shaderStages   = VriShaderStage_Compute;
    VriDescriptorSetDesc csd {};
    csd.registerSpace = 0;
    csd.ranges        = cr;
    csd.rangeNum      = 2;
    VriPipelineLayoutDesc cld {};
    cld.descriptorSets   = &csd;
    cld.descriptorSetNum = 1;
    c.CreatePipelineLayout(app.dev, &cld, &g_computeLayout);

    VriComputePipelineDesc cpd {};
    cpd.pipelineLayout        = g_computeLayout;
    cpd.shader.stage          = VriShaderStage_Compute;
    cpd.shader.entryPointName = "computeMain";
    if (app.useDxbc)
    {
        cpd.shader.bytecode     = g_computePlasmaDxbcCS;
        cpd.shader.bytecodeSize = sizeof(g_computePlasmaDxbcCS);
    }
    else
    {
        cpd.shader.bytecode =
            app.useWgsl ? static_cast<const void*>(g_computePlasmaWgsl) : static_cast<const void*>(g_computePlasmaSpv);
        cpd.shader.bytecodeSize = app.useWgsl ? sizeof(g_computePlasmaWgsl) : sizeof(g_computePlasmaSpv);
    }
    if (c.CreateComputePipeline(app.dev, &cpd, &g_computePipeline) != VriResult_Success)
        app.Fail("CreateComputePipeline failed");

    VriDescriptorRangeDesc dr[2] {};
    dr[0].baseRegister   = 0;
    dr[0].descriptorNum  = 1;
    dr[0].descriptorType = VriDescriptorType_Texture;
    dr[0].shaderStages   = VriShaderStage_Fragment;
    dr[1].baseRegister   = 1;
    dr[1].descriptorNum  = 1;
    dr[1].descriptorType = VriDescriptorType_Sampler;
    dr[1].shaderStages   = VriShaderStage_Fragment;
    VriDescriptorSetDesc dsd {};
    dsd.registerSpace = 0;
    dsd.ranges        = dr;
    dsd.rangeNum      = 2;
    VriPipelineLayoutDesc dld {};
    dld.descriptorSets               = &dsd;
    dld.descriptorSetNum             = 1;
    VriPipelineLayout* displayLayout = nullptr;
    c.CreatePipelineLayout(app.dev, &dld, &displayLayout);

    VriShaderDesc sh[2] {};
    sh[0].stage          = VriShaderStage_Vertex;
    sh[0].entryPointName = "vertexMain";
    sh[1].stage          = VriShaderStage_Fragment;
    sh[1].entryPointName = "fragmentMain";
    if (app.useDxbc)
    {
        sh[0].bytecode     = g_showTexDxbcVS;
        sh[0].bytecodeSize = sizeof(g_showTexDxbcVS);
        sh[1].bytecode     = g_showTexDxbcPS;
        sh[1].bytecodeSize = sizeof(g_showTexDxbcPS);
    }
    else
    {
        sh[0].bytecode = sh[1].bytecode =
            app.useWgsl ? static_cast<const void*>(g_showTexWgsl) : static_cast<const void*>(g_showTexSpv);
        sh[0].bytecodeSize = sh[1].bytecodeSize = app.useWgsl ? sizeof(g_showTexWgsl) : sizeof(g_showTexSpv);
    }
    VriColorAttachmentDesc ca {};
    ca.format         = app.swapFormat;
    ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd {};
    pd.pipelineLayout            = displayLayout;
    pd.shaders                   = sh;
    pd.shaderNum                 = 2;
    pd.inputAssembly.topology    = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode    = VriCullMode_None;
    pd.rasterization.lineWidth   = 1.0f;
    pd.multisample.sampleNum     = 1;
    pd.outputMerger.colors       = &ca;
    pd.outputMerger.colorNum     = 1;
    VriPipeline* displayPipeline = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &pd, &displayPipeline) != VriResult_Success)
        app.Fail("CreateGraphicsPipeline failed");

    // One compute set + one display set per buffered image.
    VriDescriptorPoolDesc pdsc {};
    pdsc.descriptorSetMaxNum  = 4;
    pdsc.constantBufferMaxNum = 2;
    pdsc.storageTextureMaxNum = 2;
    pdsc.textureMaxNum        = 2;
    pdsc.samplerMaxNum        = 2;
    VriDescriptorPool* pool   = nullptr;
    c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    for (int i = 0; i < 2; ++i)
    {
        c.AllocateDescriptorSets(pool, g_computeLayout, 0, &g_computeSet[i], 1);
        c.AllocateDescriptorSets(pool, displayLayout, 0, &g_displaySet[i], 1);
        const VriDescriptor*         a[1] = {uboView};
        const VriDescriptor*         b[1] = {imgView[i]};
        VriDescriptorRangeUpdateDesc u[2] {};
        u[0].descriptors   = a;
        u[0].descriptorNum = 1;
        u[1].descriptors   = b;
        u[1].descriptorNum = 1;
        c.UpdateDescriptorRanges(g_computeSet[i], 0, 2, u);
        const VriDescriptor* e[1] = {imgView[i]};
        const VriDescriptor* f[1] = {sampler};
        u[0].descriptors          = e;
        u[1].descriptors          = f;
        c.UpdateDescriptorRanges(g_displaySet[i], 0, 2, u);
    }

    // ---- the compute queue's own command machinery + timeline --------------------------
    c.CreateCommandAllocator(app.dev, VriQueueType_Compute, &g_computeAlloc);
    c.CreateCommandBuffer(g_computeAlloc, &g_computeCmd);
    c.CreateFence(app.dev, 0, &g_computeFence);

    // Prime img[0] before the loop: the first frame samples it while its own batch is still
    // writing img[1]. Synchronous on purpose - it runs exactly once.
    SubmitPlasmaBatch(0, 0);
    c.Wait(g_computeFence, g_computeValue);

    app.onUpdate = [](uint64_t frameValue) {
        g_frameValue = frameValue;
        // The single compute cmd is reused: the previous batch must have retired. It ran
        // concurrent with the previous graphics frame, which the synchronous example loop has
        // already waited out, so in the steady state this wait is free.
        if (g_computeValue)
            app.c.Wait(g_computeFence, g_computeValue);
        g_time += 2.4f * app.dt;
        // This batch writes img[frameValue&1]; today's graphics frame samples the OTHER slot.
        // The GPU-side wait on the previous graphics frame keeps the pattern correct even in
        // a pipelined host (it is already satisfied in this synchronous one).
        SubmitPlasmaBatch(static_cast<uint32_t>(frameValue & 1), frameValue > 1 ? frameValue - 1 : 0);
    };

    app.onRecord = [displayPipeline, displayLayout](VriCommandBuffer* cmd) {
        // Sample what the PREVIOUS batch produced - the current one is still in flight on the
        // compute queue. That concurrency is the whole example.
        const uint32_t shown = static_cast<uint32_t>((g_frameValue - 1) & 1);
        app.c.CmdSetPipeline(cmd, displayPipeline);
        app.c.CmdSetPipelineLayout(cmd, displayLayout);
        app.c.CmdSetDescriptorSet(cmd, 0, g_displaySet[shown]);
        VriDrawDesc d {};
        d.vertexNum   = 3;
        d.instanceNum = 1;
        app.c.CmdDraw(cmd, &d);
    };

    app.Run();
    return 0;
}
