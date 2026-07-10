// Subgroup / wave operations: a compute pass paints three bands driven by real wave
// intrinsics - a WaveGetLaneIndex rainbow (one full rainbow per wave, so the stripe period
// IS the hardware subgroup size), a WavePrefixSum ramp, and a WaveActiveCountBits "ballot
// meter" - into a storage image that a fullscreen pass displays. Explicit degradation
// ladder (no silent no-op): hasShaderWaveOps + compute -> wave intrinsics (Vulkan subgroups
// / SM6.0 DXIL / WGSL 'enable subgroups'); compute only (WebGPU backend) -> the same bands
// from the thread id with an assumed width of 32; no compute (WebGL2) -> a fragment-shader
// version, so the wasm pixel self-check always passes.
// Shared scaffolding: examples/common/example_app.h.
#include "../common/example_app.h"

#include <cstring>

#include "shaders/examples/show_tex_dxbc.h"       // g_showTexDxbcVS / PS (display: sample the result)
#include "shaders/examples/show_tex_spv.h"        // g_showTexSpv
#include "shaders/examples/show_tex_wgsl.h"       // g_showTexWgsl
#include "shaders/examples/wave_vis_basic_dxbc.h" // g_waveVisBasicDxbcCS (never runs on D3D12; wave path wins)
#include "shaders/examples/wave_vis_basic_spv.h"  // g_waveVisBasicSpv
#include "shaders/examples/wave_vis_basic_wgsl.h" // g_waveVisBasicWgsl (WebGPU: compute, no waves)
#include "shaders/examples/wave_vis_dxil.h"       // g_waveVisDxilCS (D3D12, SM6.0 wave intrinsics)
#include "shaders/examples/wave_vis_frag_dxbc.h"  // g_waveVisFragDxbcVS / PS
#include "shaders/examples/wave_vis_frag_spv.h"   // g_waveVisFragSpv (no-compute fallback)
#include "shaders/examples/wave_vis_frag_wgsl.h"  // g_waveVisFragWgsl (WebGL2 fallback)
#include "shaders/examples/wave_vis_spv.h"        // g_waveVisSpv (Vulkan subgroups; no WGSL - the
                                                  // WebGPU backend reports hasShaderWaveOps=false,
                                                  // so the wave path never runs there)

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480;
    struct Params
    {
        float    time;
        uint32_t width, height, pad;
    };
    bool g_imgInit = false;
} // namespace

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.Init("waveops", kWidth, kHeight, /*hasDepth*/ false);
    app.SetClearColor(0.0f, 0.0f, 0.0f);
    VriCoreInterface&    c          = app.c;
    const VriDeviceDesc* dd         = c.GetDeviceDesc(app.dev);
    const bool           hasCompute = dd->hasComputeShader != VRI_FALSE;
    const bool           hasWave    = hasCompute && dd->hasShaderWaveOps != VRI_FALSE;
    const uint32_t       subgroup   = dd->subgroupSize;

    // params constant buffer (time + dimensions), refreshed each frame
    VriBufferDesc ubd {};
    ubd.size           = sizeof(Params);
    ubd.usage          = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst;
    ubd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* ubo     = nullptr;
    c.CreateBuffer(app.dev, &ubd, &ubo);
    VriBufferDesc usd {};
    usd.size           = sizeof(Params);
    usd.usage          = VriBufferUsage_TransferSrc;
    usd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* ustg    = nullptr;
    c.CreateBuffer(app.dev, &usd, &ustg);
    VriBufferViewDesc ubv {};
    ubv.buffer             = ubo;
    ubv.viewType           = VriDescriptorType_ConstantBuffer;
    ubv.offset             = 0;
    ubv.size               = sizeof(Params);
    VriDescriptor* uboView = nullptr;
    c.CreateBufferView(app.dev, &ubv, &uboView);

    static float speed = 1.0f, t = 0.0f;
    app.onUpdate = [ustg](uint64_t) {
        t += speed * app.dt;
        Params p {};
        p.time   = t;
        p.width  = kWidth;
        p.height = kHeight;
        std::memcpy(app.c.MapBuffer(ustg, 0, sizeof(Params)), &p, sizeof(Params));
        app.c.UnmapBuffer(ustg);
    };
    app.onGui = [=] {
        if (hasWave)
            ImGui::Text("wave intrinsics active, subgroupSize = %u\n(top band: one rainbow per wave)", subgroup);
        else if (hasCompute)
            ImGui::Text("hasShaderWaveOps = false on %s\n(compute fallback, assumed width 32)", app.apiName);
        else
            ImGui::Text("no compute on %s\n(fragment fallback, assumed width 32)", app.apiName);
        ImGui::SliderFloat("speed", &speed, 0.0f, 4.0f);
        ImGui::TextUnformatted("bands: lane index | prefix sum | ballot count");
    };

    auto uploadParams = [ubo, ustg](VriCommandBuffer* cmd, VriPipelineStageFlags stage) {
        VriCoreInterface& c = app.c;
        VriBufferCopyDesc ucp {};
        ucp.size = sizeof(Params);
        c.CmdCopyBuffer(cmd, ubo, ustg, &ucp);
        VriBufferBarrierDesc ub {};
        ub.buffer        = ubo;
        ub.before.access = VriAccess_CopyDestinationWrite;
        ub.before.stages = VriPipelineStage_Transfer;
        ub.after.access  = VriAccess_ConstantBufferRead;
        ub.after.stages  = stage;
        VriBarrierGroupDesc ubg {};
        ubg.buffers   = &ub;
        ubg.bufferNum = 1;
        c.CmdBarrier(cmd, &ubg);
    };

    if (hasCompute)
    {
        std::printf(hasWave ? "[waveops] wave intrinsics active (subgroupSize %u)\n" :
                              "[waveops] compute fallback: no wave ops on this backend (subgroupSize %u)\n",
                    subgroup);
        std::fflush(stdout);

        // storage image: written by compute, sampled by the display pass
        VriTextureDesc td {};
        td.type           = VriTextureType_2D;
        td.format         = VriFormat_RGBA8_UNORM;
        td.width          = kWidth;
        td.height         = kHeight;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = 1;
        td.sampleNum      = 1;
        td.usage          = VriTextureUsage_ShaderResourceStorage | VriTextureUsage_ShaderResource;
        td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* img   = nullptr;
        if (c.CreateTexture(app.dev, &td, &img) != VriResult_Success)
            app.Fail("CreateTexture (storage) failed");
        VriTextureViewDesc vd {};
        vd.texture             = img;
        vd.viewType            = VriTextureViewType_2D;
        vd.format              = VriFormat_Unknown;
        vd.aspect              = VriImageAspect_Color;
        VriDescriptor* imgView = nullptr;
        c.CreateTextureView(app.dev, &vd, &imgView);
        VriSamplerDesc smp {};
        smp.magFilter          = VriFilter_Nearest;
        smp.minFilter          = VriFilter_Nearest;
        smp.mipmapMode         = VriMipmapMode_Nearest;
        smp.addressModeU       = VriAddressMode_ClampToEdge;
        smp.addressModeV       = VriAddressMode_ClampToEdge;
        smp.addressModeW       = VriAddressMode_ClampToEdge;
        smp.maxLod             = 1.0f;
        VriDescriptor* sampler = nullptr;
        c.CreateSampler(app.dev, &smp, &sampler);

        // compute layout: params CB @0, storage image @1
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
        cld.descriptorSets               = &csd;
        cld.descriptorSetNum             = 1;
        VriPipelineLayout* computeLayout = nullptr;
        c.CreatePipelineLayout(app.dev, &cld, &computeLayout);

        // the wave shader where the intrinsics exist, the thread-id fallback otherwise
        const vriex::ExampleApp::ShaderVariants waveCS {
            VRI_SHADER_BLOB(g_waveVisSpv), nullptr, 0, VRI_SHADER_D3D12(g_waveVisDxilCS)};
        const vriex::ExampleApp::ShaderVariants basicCS {VRI_SHADER_BLOB(g_waveVisBasicSpv),
                                                         VRI_SHADER_BLOB(g_waveVisBasicWgsl),
                                                         VRI_SHADER_D3D12(g_waveVisBasicDxbcCS)};
        VriComputePipelineDesc                  cpd {};
        cpd.pipelineLayout           = computeLayout;
        cpd.shader                   = app.Shader(VriShaderStage_Compute, "computeMain", hasWave ? waveCS : basicCS);
        VriPipeline* computePipeline = nullptr;
        if (c.CreateComputePipeline(app.dev, &cpd, &computePipeline) != VriResult_Success)
            app.Fail("CreateComputePipeline failed");

        // display layout: sampled texture @0, sampler @1
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

        const vriex::ExampleApp::ShaderVariants dispVS {
            VRI_SHADER_BLOB(g_showTexSpv), VRI_SHADER_BLOB(g_showTexWgsl), VRI_SHADER_D3D12(g_showTexDxbcVS)};
        const vriex::ExampleApp::ShaderVariants dispPS {
            VRI_SHADER_BLOB(g_showTexSpv), VRI_SHADER_BLOB(g_showTexWgsl), VRI_SHADER_D3D12(g_showTexDxbcPS)};
        VriShaderDesc sh[2] = {
            app.Shader(VriShaderStage_Vertex, "vertexMain", dispVS),
            app.Shader(VriShaderStage_Fragment, "fragmentMain", dispPS),
        };
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

        VriDescriptorPoolDesc pdsc {};
        pdsc.descriptorSetMaxNum  = 2;
        pdsc.constantBufferMaxNum = 1;
        pdsc.storageTextureMaxNum = 1;
        pdsc.textureMaxNum        = 1;
        pdsc.samplerMaxNum        = 1;
        VriDescriptorPool* pool   = nullptr;
        c.CreateDescriptorPool(app.dev, &pdsc, &pool);
        VriDescriptorSet* computeSet = nullptr;
        c.AllocateDescriptorSets(pool, computeLayout, 0, &computeSet, 1);
        VriDescriptorSet* displaySet = nullptr;
        c.AllocateDescriptorSets(pool, displayLayout, 0, &displaySet, 1);
        {
            const VriDescriptor*         a[1] = {uboView};
            const VriDescriptor*         b[1] = {imgView};
            VriDescriptorRangeUpdateDesc u[2] {};
            u[0].descriptors   = a;
            u[0].descriptorNum = 1;
            u[1].descriptors   = b;
            u[1].descriptorNum = 1;
            c.UpdateDescriptorRanges(computeSet, 0, 2, u);
        }
        {
            const VriDescriptor*         a[1] = {imgView};
            const VriDescriptor*         b[1] = {sampler};
            VriDescriptorRangeUpdateDesc u[2] {};
            u[0].descriptors   = a;
            u[0].descriptorNum = 1;
            u[1].descriptors   = b;
            u[1].descriptorNum = 1;
            c.UpdateDescriptorRanges(displaySet, 0, 2, u);
        }

        app.onPreRender = [=](VriCommandBuffer* cmd) {
            uploadParams(cmd, VriPipelineStage_ComputeShader);
            VriTextureBarrierDesc tw {};
            tw.texture       = img;
            tw.before.layout = g_imgInit ? VriLayout_ShaderResource : VriLayout_Undefined;
            tw.before.stages = g_imgInit ? VriPipelineStage_FragmentShader : VriPipelineStage_None;
            tw.after.access  = VriAccess_ShaderResourceStorageWrite;
            tw.after.layout  = VriLayout_ShaderResourceStorage;
            tw.after.stages  = VriPipelineStage_ComputeShader;
            tw.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc gw {};
            gw.textures   = &tw;
            gw.textureNum = 1;
            c.CmdBarrier(cmd, &gw);
            g_imgInit = true;

            c.CmdSetPipeline(cmd, computePipeline);
            c.CmdSetPipelineLayout(cmd, computeLayout);
            c.CmdSetDescriptorSet(cmd, 0, computeSet);
            VriDispatchDesc disp {};
            disp.x = (kWidth + 63) / 64; // numthreads(64,1,1): one workgroup per 64-px run
            disp.y = kHeight;
            disp.z = 1;
            c.CmdDispatch(cmd, &disp);

            VriTextureBarrierDesc tr {};
            tr.texture       = img;
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
            c.CmdBarrier(cmd, &gr);
        };
        app.onRecord = [=](VriCommandBuffer* cmd) {
            c.CmdSetPipeline(cmd, displayPipeline);
            c.CmdSetPipelineLayout(cmd, displayLayout);
            c.CmdSetDescriptorSet(cmd, 0, displaySet);
            VriDrawDesc d {};
            d.vertexNum   = 3;
            d.instanceNum = 1;
            c.CmdDraw(cmd, &d);
        };
    }
    else
    {
        std::printf("[waveops] no compute (WebGL2) - fragment-shader fallback\n");
        std::fflush(stdout);

        VriDescriptorRangeDesc r {};
        r.baseRegister   = 0;
        r.descriptorNum  = 1;
        r.descriptorType = VriDescriptorType_ConstantBuffer;
        r.shaderStages   = VriShaderStage_Fragment;
        VriDescriptorSetDesc sd {};
        sd.registerSpace = 0;
        sd.ranges        = &r;
        sd.rangeNum      = 1;
        VriPipelineLayoutDesc ld {};
        ld.descriptorSets         = &sd;
        ld.descriptorSetNum       = 1;
        VriPipelineLayout* layout = nullptr;
        c.CreatePipelineLayout(app.dev, &ld, &layout);

        const vriex::ExampleApp::ShaderVariants fVS {VRI_SHADER_BLOB(g_waveVisFragSpv),
                                                     VRI_SHADER_BLOB(g_waveVisFragWgsl),
                                                     VRI_SHADER_D3D12(g_waveVisFragDxbcVS)};
        const vriex::ExampleApp::ShaderVariants fPS {VRI_SHADER_BLOB(g_waveVisFragSpv),
                                                     VRI_SHADER_BLOB(g_waveVisFragWgsl),
                                                     VRI_SHADER_D3D12(g_waveVisFragDxbcPS)};
        VriShaderDesc                           sh[2] = {
            app.Shader(VriShaderStage_Vertex, "vertexMain", fVS),
            app.Shader(VriShaderStage_Fragment, "fragmentMain", fPS),
        };
        VriColorAttachmentDesc ca {};
        ca.format         = app.swapFormat;
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
        if (c.CreateGraphicsPipeline(app.dev, &pd, &pipeline) != VriResult_Success)
            app.Fail("CreateGraphicsPipeline failed");

        VriDescriptorPoolDesc pdsc {};
        pdsc.descriptorSetMaxNum  = 1;
        pdsc.constantBufferMaxNum = 1;
        VriDescriptorPool* pool   = nullptr;
        c.CreateDescriptorPool(app.dev, &pdsc, &pool);
        VriDescriptorSet* set = nullptr;
        c.AllocateDescriptorSets(pool, layout, 0, &set, 1);
        {
            const VriDescriptor*         u[1] = {uboView};
            VriDescriptorRangeUpdateDesc up {};
            up.descriptors   = u;
            up.descriptorNum = 1;
            c.UpdateDescriptorRanges(set, 0, 1, &up);
        }

        app.onPreRender = [=](VriCommandBuffer* cmd) { uploadParams(cmd, VriPipelineStage_FragmentShader); };
        app.onRecord    = [=](VriCommandBuffer* cmd) {
            c.CmdSetPipeline(cmd, pipeline);
            c.CmdSetPipelineLayout(cmd, layout);
            c.CmdSetDescriptorSet(cmd, 0, set);
            VriDrawDesc d {};
            d.vertexNum   = 3;
            d.instanceNum = 1;
            c.CmdDraw(cmd, &d);
        };
    }

    app.SetupCapture();
    app.Run();

    // Teardown intentionally minimal (like examples/computeshader): the pipelines, pool,
    // and views of whichever path ran are released with the device.
    return 0;
}
