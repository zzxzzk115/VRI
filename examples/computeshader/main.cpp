// Compute shader: a compute pass writes an animated plasma into a storage image, then a
// fullscreen pass samples it onto the screen. Ported in spirit from Sascha Willems'
// computeshader. WebGL2 (OpenGL ES 3.0) has no compute, so when hasComputeShader is false
// the example degrades *explicitly* - it renders the same plasma directly in a fragment
// shader instead (no silent no-op). No assets; shared host scaffolding in examples/common.
#include "../common/example_app.h"

#include <cstring>
#include <utility>

#include "tests/shaders/compute_plasma_spv.h"  // g_computePlasmaSpv  (compute; Vulkan/OpenGL)
#include "tests/shaders/compute_plasma_wgsl.h" // g_computePlasmaWgsl (WebGPU)
#include "tests/shaders/compute_plasma_dxbc.h" // g_computePlasmaDxbcCS (D3D12)
#include "tests/shaders/show_tex_spv.h"        // g_showTexSpv  (display: sample the result)
#include "tests/shaders/show_tex_wgsl.h"       // g_showTexWgsl
#include "tests/shaders/show_tex_dxbc.h"       // g_showTexDxbcVS / PS
#include "tests/shaders/gen_plasma_spv.h"      // g_genPlasmaSpv  (WebGL2 fragment fallback)
#include "tests/shaders/gen_plasma_wgsl.h"     // g_genPlasmaWgsl (the fallback never runs on D3D12)

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480;
    struct Params { float time; uint32_t width, height, pad; };
    bool g_imgInit = false; // storage image has been written at least once (layout tracking)
} // namespace

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.Init("computeshader", kWidth, kHeight, /*hasDepth*/ false);
    app.SetClearColor(0.0f, 0.0f, 0.0f);
    VriCoreInterface& c = app.c;
    const bool hasCompute = c.GetDeviceDesc(app.dev)->hasComputeShader != VRI_FALSE;

    // params constant buffer (time + dimensions), refreshed each frame
    VriBufferDesc ubd{}; ubd.size = sizeof(Params); ubd.usage = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst; ubd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* ubo = nullptr; c.CreateBuffer(app.dev, &ubd, &ubo);
    VriBufferDesc usd{}; usd.size = sizeof(Params); usd.usage = VriBufferUsage_TransferSrc; usd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* ustg = nullptr; c.CreateBuffer(app.dev, &usd, &ustg);
    VriBufferViewDesc ubv{}; ubv.buffer = ubo; ubv.viewType = VriDescriptorType_ConstantBuffer; ubv.offset = 0; ubv.size = sizeof(Params);
    VriDescriptor* uboView = nullptr; c.CreateBufferView(app.dev, &ubv, &uboView);

    app.onUpdate = [ustg](uint64_t frame) {
        Params p{}; p.time = static_cast<float>(frame) * 0.04f; p.width = kWidth; p.height = kHeight;
        std::memcpy(app.c.MapBuffer(ustg, 0, sizeof(Params)), &p, sizeof(Params)); app.c.UnmapBuffer(ustg);
    };

    if (hasCompute)
    {
        std::printf("[computeshader] compute available: plasma generated on the GPU\n"); std::fflush(stdout);

        // storage image: written by compute (UAV/storage), sampled by the display pass
        VriTextureDesc td{}; td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM; td.width = kWidth; td.height = kHeight; td.depth = 1;
        td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1; td.usage = VriTextureUsage_ShaderResourceStorage | VriTextureUsage_ShaderResource; td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* img = nullptr; if (c.CreateTexture(app.dev, &td, &img) != VriResult_Success) app.Fail("CreateTexture (storage) failed");
        VriTextureViewDesc vd{}; vd.texture = img; vd.viewType = VriTextureViewType_2D; vd.format = VriFormat_Unknown; vd.aspect = VriImageAspect_Color;
        VriDescriptor* imgView = nullptr; c.CreateTextureView(app.dev, &vd, &imgView);
        VriSamplerDesc smp{}; smp.magFilter = VriFilter_Linear; smp.minFilter = VriFilter_Linear; smp.mipmapMode = VriMipmapMode_Nearest;
        smp.addressModeU = VriAddressMode_ClampToEdge; smp.addressModeV = VriAddressMode_ClampToEdge; smp.addressModeW = VriAddressMode_ClampToEdge; smp.maxLod = 1.0f;
        VriDescriptor* sampler = nullptr; c.CreateSampler(app.dev, &smp, &sampler);

        // compute pipeline layout: params CB @0, storage image @1
        VriDescriptorRangeDesc cr[2]{};
        cr[0].baseRegister = 0; cr[0].descriptorNum = 1; cr[0].descriptorType = VriDescriptorType_ConstantBuffer; cr[0].shaderStages = VriShaderStage_Compute;
        cr[1].baseRegister = 1; cr[1].descriptorNum = 1; cr[1].descriptorType = VriDescriptorType_StorageTexture; cr[1].shaderStages = VriShaderStage_Compute;
        VriDescriptorSetDesc csd{}; csd.registerSpace = 0; csd.ranges = cr; csd.rangeNum = 2;
        VriPipelineLayoutDesc cld{}; cld.descriptorSets = &csd; cld.descriptorSetNum = 1;
        VriPipelineLayout* computeLayout = nullptr; c.CreatePipelineLayout(app.dev, &cld, &computeLayout);

        VriComputePipelineDesc cpd{}; cpd.pipelineLayout = computeLayout;
        cpd.shader.stage = VriShaderStage_Compute; cpd.shader.entryPointName = "computeMain";
        if (app.useDxbc) { cpd.shader.bytecode = g_computePlasmaDxbcCS; cpd.shader.bytecodeSize = sizeof(g_computePlasmaDxbcCS); }
        else { cpd.shader.bytecode = app.useWgsl ? static_cast<const void*>(g_computePlasmaWgsl) : static_cast<const void*>(g_computePlasmaSpv);
               cpd.shader.bytecodeSize = app.useWgsl ? sizeof(g_computePlasmaWgsl) : sizeof(g_computePlasmaSpv); }
        VriPipeline* computePipeline = nullptr;
        if (c.CreateComputePipeline(app.dev, &cpd, &computePipeline) != VriResult_Success) app.Fail("CreateComputePipeline failed");

        // display pipeline layout: sampled texture @0, sampler @1
        VriDescriptorRangeDesc dr[2]{};
        dr[0].baseRegister = 0; dr[0].descriptorNum = 1; dr[0].descriptorType = VriDescriptorType_Texture; dr[0].shaderStages = VriShaderStage_Fragment;
        dr[1].baseRegister = 1; dr[1].descriptorNum = 1; dr[1].descriptorType = VriDescriptorType_Sampler; dr[1].shaderStages = VriShaderStage_Fragment;
        VriDescriptorSetDesc dsd{}; dsd.registerSpace = 0; dsd.ranges = dr; dsd.rangeNum = 2;
        VriPipelineLayoutDesc dld{}; dld.descriptorSets = &dsd; dld.descriptorSetNum = 1;
        VriPipelineLayout* displayLayout = nullptr; c.CreatePipelineLayout(app.dev, &dld, &displayLayout);

        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].entryPointName = "fragmentMain";
        if (app.useDxbc) { sh[0].bytecode = g_showTexDxbcVS; sh[0].bytecodeSize = sizeof(g_showTexDxbcVS); sh[1].bytecode = g_showTexDxbcPS; sh[1].bytecodeSize = sizeof(g_showTexDxbcPS); }
        else { sh[0].bytecode = sh[1].bytecode = app.useWgsl ? static_cast<const void*>(g_showTexWgsl) : static_cast<const void*>(g_showTexSpv);
               sh[0].bytecodeSize = sh[1].bytecodeSize = app.useWgsl ? sizeof(g_showTexWgsl) : sizeof(g_showTexSpv); }
        VriColorAttachmentDesc ca{}; ca.format = app.swapFormat; ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = displayLayout; pd.shaders = sh; pd.shaderNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
        VriPipeline* displayPipeline = nullptr;
        if (c.CreateGraphicsPipeline(app.dev, &pd, &displayPipeline) != VriResult_Success) app.Fail("CreateGraphicsPipeline failed");

        VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 2; pdsc.constantBufferMaxNum = 1; pdsc.storageTextureMaxNum = 1; pdsc.textureMaxNum = 1; pdsc.samplerMaxNum = 1;
        VriDescriptorPool* pool = nullptr; c.CreateDescriptorPool(app.dev, &pdsc, &pool);
        VriDescriptorSet* computeSet = nullptr; c.AllocateDescriptorSets(pool, computeLayout, 0, &computeSet, 1);
        VriDescriptorSet* displaySet = nullptr; c.AllocateDescriptorSets(pool, displayLayout, 0, &displaySet, 1);
        { const VriDescriptor* a[1] = {uboView}; const VriDescriptor* b[1] = {imgView};
          VriDescriptorRangeUpdateDesc u[2]{}; u[0].descriptors = a; u[0].descriptorNum = 1; u[1].descriptors = b; u[1].descriptorNum = 1; c.UpdateDescriptorRanges(computeSet, 0, 2, u); }
        { const VriDescriptor* a[1] = {imgView}; const VriDescriptor* b[1] = {sampler};
          VriDescriptorRangeUpdateDesc u[2]{}; u[0].descriptors = a; u[0].descriptorNum = 1; u[1].descriptors = b; u[1].descriptorNum = 1; c.UpdateDescriptorRanges(displaySet, 0, 2, u); }

        app.onPreRender = [ubo, ustg, img, computeLayout, computePipeline, computeSet](VriCommandBuffer* cmd) {
            VriCoreInterface& c = app.c;
            // refresh params
            VriBufferCopyDesc ucp{}; ucp.size = sizeof(Params); c.CmdCopyBuffer(cmd, ubo, ustg, &ucp);
            VriBufferBarrierDesc ub{}; ub.buffer = ubo; ub.before.access = VriAccess_CopyDestinationWrite; ub.before.stages = VriPipelineStage_Transfer;
            ub.after.access = VriAccess_ConstantBufferRead; ub.after.stages = VriPipelineStage_ComputeShader;
            VriBarrierGroupDesc ubg{}; ubg.buffers = &ub; ubg.bufferNum = 1; c.CmdBarrier(cmd, &ubg);
            // storage image -> writable by compute
            VriTextureBarrierDesc tw{}; tw.texture = img; tw.before.layout = g_imgInit ? VriLayout_ShaderResource : VriLayout_Undefined; tw.before.stages = g_imgInit ? VriPipelineStage_FragmentShader : VriPipelineStage_None;
            tw.after.access = VriAccess_ShaderResourceStorageWrite; tw.after.layout = VriLayout_ShaderResourceStorage; tw.after.stages = VriPipelineStage_ComputeShader; tw.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc gw{}; gw.textures = &tw; gw.textureNum = 1; c.CmdBarrier(cmd, &gw);
            g_imgInit = true;
            // compute the plasma
            c.CmdSetPipeline(cmd, computePipeline);
            c.CmdSetPipelineLayout(cmd, computeLayout);
            c.CmdSetDescriptorSet(cmd, 0, computeSet);
            VriDispatchDesc disp{}; disp.x = (kWidth + 7) / 8; disp.y = (kHeight + 7) / 8; disp.z = 1; c.CmdDispatch(cmd, &disp);
            // storage image -> sampleable by the display pass
            VriTextureBarrierDesc tr{}; tr.texture = img; tr.before.access = VriAccess_ShaderResourceStorageWrite; tr.before.layout = VriLayout_ShaderResourceStorage; tr.before.stages = VriPipelineStage_ComputeShader;
            tr.after.access = VriAccess_ShaderResourceRead; tr.after.layout = VriLayout_ShaderResource; tr.after.stages = VriPipelineStage_FragmentShader; tr.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc gr{}; gr.textures = &tr; gr.textureNum = 1; c.CmdBarrier(cmd, &gr);
        };
        app.onRecord = [displayPipeline, displayLayout, displaySet](VriCommandBuffer* cmd) {
            app.c.CmdSetPipeline(cmd, displayPipeline);
            app.c.CmdSetPipelineLayout(cmd, displayLayout);
            app.c.CmdSetDescriptorSet(cmd, 0, displaySet);
            VriDrawDesc d{}; d.vertexNum = 3; d.instanceNum = 1; app.c.CmdDraw(cmd, &d);
        };
    }
    else
    {
        std::printf("[computeshader] compute unavailable (WebGL2) - using the fragment-shader plasma fallback\n"); std::fflush(stdout);

        VriDescriptorRangeDesc r{}; r.baseRegister = 0; r.descriptorNum = 1; r.descriptorType = VriDescriptorType_ConstantBuffer; r.shaderStages = VriShaderStage_Fragment;
        VriDescriptorSetDesc sd{}; sd.registerSpace = 0; sd.ranges = &r; sd.rangeNum = 1;
        VriPipelineLayoutDesc ld{}; ld.descriptorSets = &sd; ld.descriptorSetNum = 1;
        VriPipelineLayout* layout = nullptr; c.CreatePipelineLayout(app.dev, &ld, &layout);

        VriShaderDesc sh[2]{};
        sh[0].stage = VriShaderStage_Vertex;   sh[0].entryPointName = "vertexMain";
        sh[1].stage = VriShaderStage_Fragment; sh[1].entryPointName = "fragmentMain";
        sh[0].bytecode = sh[1].bytecode = app.useWgsl ? static_cast<const void*>(g_genPlasmaWgsl) : static_cast<const void*>(g_genPlasmaSpv);
        sh[0].bytecodeSize = sh[1].bytecodeSize = app.useWgsl ? sizeof(g_genPlasmaWgsl) : sizeof(g_genPlasmaSpv);
        VriColorAttachmentDesc ca{}; ca.format = app.swapFormat; ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd{};
        pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
        pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
        VriPipeline* pipeline = nullptr;
        if (c.CreateGraphicsPipeline(app.dev, &pd, &pipeline) != VriResult_Success) app.Fail("CreateGraphicsPipeline failed");

        VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 1; pdsc.constantBufferMaxNum = 1;
        VriDescriptorPool* pool = nullptr; c.CreateDescriptorPool(app.dev, &pdsc, &pool);
        VriDescriptorSet* set = nullptr; c.AllocateDescriptorSets(pool, layout, 0, &set, 1);
        const VriDescriptor* d0[1] = {uboView}; VriDescriptorRangeUpdateDesc u{}; u.descriptors = d0; u.descriptorNum = 1; c.UpdateDescriptorRanges(set, 0, 1, &u);

        app.onPreRender = [ubo, ustg](VriCommandBuffer* cmd) {
            VriBufferCopyDesc ucp{}; ucp.size = sizeof(Params); app.c.CmdCopyBuffer(cmd, ubo, ustg, &ucp);
            VriBufferBarrierDesc ub{}; ub.buffer = ubo; ub.before.access = VriAccess_CopyDestinationWrite; ub.before.stages = VriPipelineStage_Transfer;
            ub.after.access = VriAccess_ConstantBufferRead; ub.after.stages = VriPipelineStage_FragmentShader;
            VriBarrierGroupDesc ubg{}; ubg.buffers = &ub; ubg.bufferNum = 1; app.c.CmdBarrier(cmd, &ubg);
        };
        app.onRecord = [pipeline, layout, set](VriCommandBuffer* cmd) {
            app.c.CmdSetPipeline(cmd, pipeline);
            app.c.CmdSetPipelineLayout(cmd, layout);
            app.c.CmdSetDescriptorSet(cmd, 0, set);
            VriDrawDesc d{}; d.vertexNum = 3; d.instanceNum = 1; app.c.CmdDraw(cmd, &d);
        };
    }

    app.SetupCapture();
    app.Run();
    return 0;
}
