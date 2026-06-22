// Descriptor indexing (bindless): a 4x4 grid of quads, each quad samples a different texture
// chosen by a per-vertex index. Ported in spirit from Sascha Willems' descriptorindexing.
//
// Where the device reports hasBindless (Vulkan, D3D12) the 16 textures live in a fixed-size
// descriptor array indexed non-uniformly in the shader (descriptorindexing.slang). WebGPU,
// WebGL2 and OpenGL can't do descriptor indexing, so they degrade *explicitly* to a single
// 2D-array texture indexed by layer (arraytex.slang) - same picture, different mechanism. The
// row of 16 cells should look identical on every backend. No assets; the textures are
// procedurally generated solid colors. Host scaffolding is examples/common/example_app.h.
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "tests/shaders/descriptorindexing_spv.h"  // g_descriptorindexingSpv  (Vulkan)
#include "tests/shaders/descriptorindexing_dxbc.h" // g_descriptorindexingDxbcVS / PS (D3D12)
#include "tests/shaders/arraytex_spv.h"            // g_arraytexSpv  (OpenGL fallback)
#include "tests/shaders/arraytex_wgsl.h"           // g_arraytexWgsl (WebGPU/WebGL2 fallback)
#include "tests/shaders/arraytex_dxbc.h"           // g_arraytexDxbcVS / PS (unused; D3D12 is bindless)

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480;
    constexpr uint32_t kTexCount = 16; // must match Texture2D uTex[kTexCount] in descriptorindexing.slang
    constexpr uint32_t kTexDim = 64;   // each texture/layer is a 64x64 solid color; 64*4=256 keeps the
                                       // upload row pitch 256-aligned (D3D12 requires it for buffer->texture copies)
    constexpr uint32_t kGrid = 4;      // 4x4 grid of quads = 16 cells

    struct Vertex { float px, py, pz; float u, v; float texIndex; };

    // HSV->RGBA8 (s,v fixed) for a distinct-per-index palette.
    void PaletteColor(uint32_t i, uint8_t out[4])
    {
        const float h = (static_cast<float>(i) / kTexCount) * 6.0f;
        const float s = 0.75f, v = 0.95f;
        const float c = v * s, x = c * (1.0f - std::fabs(std::fmod(h, 2.0f) - 1.0f)), m = v - c;
        float r = 0, g = 0, b = 0;
        if (h < 1) { r = c; g = x; } else if (h < 2) { r = x; g = c; } else if (h < 3) { g = c; b = x; }
        else if (h < 4) { g = x; b = c; } else if (h < 5) { r = x; b = c; } else { r = c; b = x; }
        out[0] = static_cast<uint8_t>((r + m) * 255.0f); out[1] = static_cast<uint8_t>((g + m) * 255.0f);
        out[2] = static_cast<uint8_t>((b + m) * 255.0f); out[3] = 255;
    }

    // Fill one kTexDim*kTexDim RGBA8 image with palette color `i`.
    void FillSolid(uint32_t i, uint8_t* dst)
    {
        uint8_t c[4]; PaletteColor(i, c);
        for (uint32_t p = 0; p < kTexDim * kTexDim; ++p) std::memcpy(dst + p * 4, c, 4);
    }
} // namespace

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.requestFeatures = VriFeature_Bindless; // bestEffort: granted on VK/D3D12, not on WebGPU/WebGL2/GL
    app.Init("descriptorindexing", kWidth, kHeight, /*hasDepth*/ false);
    VriCoreInterface& c = app.c;
    const bool bindless = c.GetDeviceDesc(app.dev)->hasBindless != VRI_FALSE;
    std::printf("[descriptorindexing] %s path: %s\n", app.apiName,
                bindless ? "bindless (descriptor array)" : "2D-array-texture fallback");
    std::fflush(stdout);

    // ---- geometry: a 4x4 grid of quads, each carrying its cell index as texIndex ----
    std::vector<Vertex> verts;
    std::vector<uint16_t> indices;
    const float pad = 0.03f, cell = 2.0f / kGrid;
    for (uint32_t gy = 0; gy < kGrid; ++gy)
        for (uint32_t gx = 0; gx < kGrid; ++gx)
        {
            const float ti = static_cast<float>(gy * kGrid + gx);
            const float x0 = -1.0f + gx * cell + pad, x1 = -1.0f + (gx + 1) * cell - pad;
            const float y0 = -1.0f + gy * cell + pad, y1 = -1.0f + (gy + 1) * cell - pad;
            const uint16_t base = static_cast<uint16_t>(verts.size());
            verts.push_back({x0, y0, 0, 0, 0, ti}); verts.push_back({x1, y0, 0, 1, 0, ti});
            verts.push_back({x1, y1, 0, 1, 1, ti}); verts.push_back({x0, y1, 0, 0, 1, ti});
            indices.push_back(base); indices.push_back(uint16_t(base + 1)); indices.push_back(uint16_t(base + 2));
            indices.push_back(base); indices.push_back(uint16_t(base + 2)); indices.push_back(uint16_t(base + 3));
        }
    const uint32_t vbSize = static_cast<uint32_t>(verts.size() * sizeof(Vertex));
    const uint32_t ibSize = static_cast<uint32_t>(indices.size() * sizeof(uint16_t));
    const uint32_t indexCount = static_cast<uint32_t>(indices.size());

    auto deviceBuf = [&](uint32_t size, VriBufferUsageFlags usage, const void* src) {
        VriBufferDesc bd{}; bd.size = size; bd.usage = usage | VriBufferUsage_TransferDst; bd.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* b = nullptr; c.CreateBuffer(app.dev, &bd, &b);
        VriBufferDesc sd{}; sd.size = size; sd.usage = VriBufferUsage_TransferSrc; sd.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* s = nullptr; c.CreateBuffer(app.dev, &sd, &s);
        std::memcpy(c.MapBuffer(s, 0, size), src, size); c.UnmapBuffer(s);
        return std::pair<VriBuffer*, VriBuffer*>{b, s};
    };
    auto [vbuf, vstg] = deviceBuf(vbSize, VriBufferUsage_VertexBuffer, verts.data());
    auto [ibuf, istg] = deviceBuf(ibSize, VriBufferUsage_IndexBuffer, indices.data());

    // identity MVP (the grid is already in clip space) - the shaders still read a camera CB.
    float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    auto [ubo, ustg] = deviceBuf(sizeof(identity), VriBufferUsage_ConstantBuffer, identity);
    VriBufferViewDesc ubv{}; ubv.buffer = ubo; ubv.viewType = VriDescriptorType_ConstantBuffer; ubv.offset = 0; ubv.size = sizeof(identity);
    VriDescriptor* uboView = nullptr; c.CreateBufferView(app.dev, &ubv, &uboView);

    VriSamplerDesc smp{}; smp.magFilter = VriFilter_Nearest; smp.minFilter = VriFilter_Nearest; smp.mipmapMode = VriMipmapMode_Nearest;
    smp.addressModeU = VriAddressMode_ClampToEdge; smp.addressModeV = VriAddressMode_ClampToEdge; smp.addressModeW = VriAddressMode_ClampToEdge; smp.maxLod = 1.0f;
    VriDescriptor* sampler = nullptr; c.CreateSampler(app.dev, &smp, &sampler);

    // ---- textures: N separate (bindless) or one N-layer array (fallback) ----
    constexpr uint32_t kImgBytes = kTexDim * kTexDim * 4;
    std::vector<VriTexture*> texes;        // bindless
    std::vector<VriDescriptor*> texViews;  // bindless
    VriTexture* arrayTex = nullptr; VriDescriptor* arrayView = nullptr; // fallback
    VriBuffer* texStaging = nullptr;

    auto makeTex = [&](uint32_t layerNum) {
        VriTextureDesc td{}; td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM; td.width = kTexDim; td.height = kTexDim; td.depth = 1;
        td.mipNum = 1; td.layerNum = layerNum; td.sampleNum = 1; td.usage = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst; td.memoryLocation = VriMemoryLocation_Device;
        VriTexture* t = nullptr; c.CreateTexture(app.dev, &td, &t); return t;
    };

    if (bindless)
    {
        std::vector<uint8_t> px(kImgBytes);
        for (uint32_t i = 0; i < kTexCount; ++i)
        {
            texes.push_back(makeTex(1));
            VriTextureViewDesc vd{}; vd.texture = texes[i]; vd.viewType = VriTextureViewType_2D; vd.format = VriFormat_Unknown; vd.aspect = VriImageAspect_Color;
            VriDescriptor* v = nullptr; c.CreateTextureView(app.dev, &vd, &v); texViews.push_back(v);
        }
        // one staging buffer holding all kTexCount images back to back
        VriBufferDesc sd{}; sd.size = kImgBytes * kTexCount; sd.usage = VriBufferUsage_TransferSrc; sd.memoryLocation = VriMemoryLocation_HostUpload;
        c.CreateBuffer(app.dev, &sd, &texStaging);
        uint8_t* m = static_cast<uint8_t*>(c.MapBuffer(texStaging, 0, kImgBytes * kTexCount));
        for (uint32_t i = 0; i < kTexCount; ++i) FillSolid(i, m + i * kImgBytes);
        c.UnmapBuffer(texStaging);
    }
    else
    {
        arrayTex = makeTex(kTexCount);
        VriTextureViewDesc vd{}; vd.texture = arrayTex; vd.viewType = VriTextureViewType_2DArray; vd.format = VriFormat_Unknown; vd.aspect = VriImageAspect_Color; vd.layerNum = kTexCount;
        c.CreateTextureView(app.dev, &vd, &arrayView);
        VriBufferDesc sd{}; sd.size = kImgBytes * kTexCount; sd.usage = VriBufferUsage_TransferSrc; sd.memoryLocation = VriMemoryLocation_HostUpload;
        c.CreateBuffer(app.dev, &sd, &texStaging);
        uint8_t* m = static_cast<uint8_t*>(c.MapBuffer(texStaging, 0, kImgBytes * kTexCount));
        for (uint32_t i = 0; i < kTexCount; ++i) FillSolid(i, m + i * kImgBytes);
        c.UnmapBuffer(texStaging);
    }

    // ---- pipeline layout: CB@0 (vertex), Sampler@1 (frag), Texture(array)@2 (frag) ----
    VriDescriptorRangeDesc ranges[3]{};
    ranges[0].baseRegister = 0; ranges[0].descriptorNum = 1;         ranges[0].descriptorType = VriDescriptorType_ConstantBuffer; ranges[0].shaderStages = VriShaderStage_Vertex;
    ranges[1].baseRegister = 1; ranges[1].descriptorNum = 1;         ranges[1].descriptorType = VriDescriptorType_Sampler;        ranges[1].shaderStages = VriShaderStage_Fragment;
    ranges[2].baseRegister = 2; ranges[2].descriptorNum = bindless ? kTexCount : 1; ranges[2].descriptorType = VriDescriptorType_Texture; ranges[2].shaderStages = VriShaderStage_Fragment;
    if (!bindless) ranges[2].viewType = VriTextureViewType_2DArray; // fallback samples a 2D-array texture (WebGPU needs the dimension)
    VriDescriptorSetDesc setDesc{}; setDesc.registerSpace = 0; setDesc.ranges = ranges; setDesc.rangeNum = 3;
    VriPipelineLayoutDesc ld{}; ld.descriptorSets = &setDesc; ld.descriptorSetNum = 1;
    VriPipelineLayout* layout = nullptr; c.CreatePipelineLayout(app.dev, &ld, &layout);

    VriShaderDesc sh[2]{};
    sh[0].stage = VriShaderStage_Vertex;   sh[0].entryPointName = "vertexMain";
    sh[1].stage = VriShaderStage_Fragment; sh[1].entryPointName = "fragmentMain";
    if (app.useDxbc)
    {
        // D3D12 always has bindless here (hasBindless gated the path); use the indexing shader.
        sh[0].bytecode = g_descriptorindexingDxbcVS; sh[0].bytecodeSize = sizeof(g_descriptorindexingDxbcVS);
        sh[1].bytecode = g_descriptorindexingDxbcPS; sh[1].bytecodeSize = sizeof(g_descriptorindexingDxbcPS);
    }
    else if (bindless) // Vulkan
    {
        sh[0].bytecode = sh[1].bytecode = g_descriptorindexingSpv;
        sh[0].bytecodeSize = sh[1].bytecodeSize = sizeof(g_descriptorindexingSpv);
    }
    else // OpenGL (SPIR-V) or WebGPU/WebGL2 (WGSL) fallback
    {
        sh[0].bytecode = sh[1].bytecode = app.useWgsl ? static_cast<const void*>(g_arraytexWgsl) : static_cast<const void*>(g_arraytexSpv);
        sh[0].bytecodeSize = sh[1].bytecodeSize = app.useWgsl ? sizeof(g_arraytexWgsl) : sizeof(g_arraytexSpv);
    }

    VriVertexAttributeDesc attrs[3]{};
    attrs[0].format = VriFormat_RGB32_SFLOAT; attrs[0].offset = 0;  attrs[0].streamIndex = 0;
    attrs[1].format = VriFormat_RG32_SFLOAT;  attrs[1].offset = 12; attrs[1].streamIndex = 0;
    attrs[2].format = VriFormat_R32_SFLOAT;   attrs[2].offset = 20; attrs[2].streamIndex = 0;
    VriVertexStreamDesc stream{}; stream.stride = sizeof(Vertex); stream.bindingSlot = 0; stream.stepRate = VriVertexStepRate_PerVertex;

    VriColorAttachmentDesc ca{}; ca.format = app.swapFormat; ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd{};
    pd.pipelineLayout = layout; pd.shaders = sh; pd.shaderNum = 2;
    pd.vertexInput.attributes = attrs; pd.vertexInput.attributeNum = 3; pd.vertexInput.streams = &stream; pd.vertexInput.streamNum = 1;
    pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode = VriCullMode_None; pd.rasterization.frontFace = VriFrontFace_CounterClockwise; pd.rasterization.lineWidth = 1.0f;
    pd.multisample.sampleNum = 1; pd.outputMerger.colors = &ca; pd.outputMerger.colorNum = 1;
    VriPipeline* pipeline = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &pd, &pipeline) != VriResult_Success) app.Fail("CreateGraphicsPipeline failed");

    VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 1; pdsc.constantBufferMaxNum = 1; pdsc.samplerMaxNum = 1; pdsc.textureMaxNum = bindless ? kTexCount : 1;
    VriDescriptorPool* pool = nullptr; c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* set = nullptr; c.AllocateDescriptorSets(pool, layout, 0, &set, 1);
    {
        const VriDescriptor* d0[1] = {uboView}; const VriDescriptor* d1[1] = {sampler};
        std::vector<const VriDescriptor*> texDescs;
        if (bindless) for (auto* v : texViews) texDescs.push_back(v); else texDescs.push_back(arrayView);
        VriDescriptorRangeUpdateDesc u[3]{};
        u[0].descriptors = d0; u[0].descriptorNum = 1;
        u[1].descriptors = d1; u[1].descriptorNum = 1;
        u[2].descriptors = texDescs.data(); u[2].descriptorNum = static_cast<uint32_t>(texDescs.size());
        c.UpdateDescriptorRanges(set, 0, 3, u);
    }

    // ---- one-time upload (fence value 1) ----
    {
        VriCommandBuffer* cmd = app.cmd;
        c.BeginCommandBuffer(cmd);
        VriBufferCopyDesc vcp{}; vcp.size = vbSize; c.CmdCopyBuffer(cmd, vbuf, vstg, &vcp);
        VriBufferCopyDesc icp{}; icp.size = ibSize; c.CmdCopyBuffer(cmd, ibuf, istg, &icp);
        VriBufferCopyDesc ucp{}; ucp.size = sizeof(identity); c.CmdCopyBuffer(cmd, ubo, ustg, &ucp); // static MVP, uploaded once
        VriBufferBarrierDesc gb[3]{};
        gb[0].buffer = vbuf; gb[0].before.access = VriAccess_CopyDestinationWrite; gb[0].before.stages = VriPipelineStage_Transfer; gb[0].after.access = VriAccess_VertexBufferRead; gb[0].after.stages = VriPipelineStage_VertexInput;
        gb[1].buffer = ibuf; gb[1].before.access = VriAccess_CopyDestinationWrite; gb[1].before.stages = VriPipelineStage_Transfer; gb[1].after.access = VriAccess_IndexBufferRead; gb[1].after.stages = VriPipelineStage_VertexInput;
        gb[2].buffer = ubo;  gb[2].before.access = VriAccess_CopyDestinationWrite; gb[2].before.stages = VriPipelineStage_Transfer; gb[2].after.access = VriAccess_ConstantBufferRead; gb[2].after.stages = VriPipelineStage_VertexShader;
        VriBarrierGroupDesc gbg{}; gbg.buffers = gb; gbg.bufferNum = 3; c.CmdBarrier(cmd, &gbg);

        auto uploadTex = [&](VriTexture* t, uint32_t layerNum) {
            VriTextureBarrierDesc tb{}; tb.texture = t; tb.before.layout = VriLayout_Undefined; tb.before.stages = VriPipelineStage_None;
            tb.after.access = VriAccess_CopyDestinationWrite; tb.after.layout = VriLayout_CopyDestination; tb.after.stages = VriPipelineStage_Transfer; tb.aspect = VriImageAspect_Color; tb.layerNum = layerNum;
            VriBarrierGroupDesc g{}; g.textures = &tb; g.textureNum = 1; c.CmdBarrier(cmd, &g);
            for (uint32_t layer = 0; layer < layerNum; ++layer)
            {
                VriBufferTextureCopyDesc up{}; up.bufferOffset = uint64_t(layer) * kImgBytes;
                up.texture.aspect = VriImageAspect_Color; up.texture.baseLayer = layer; up.texture.layerNum = 1; up.texture.width = kTexDim; up.texture.height = kTexDim;
                c.CmdUploadBufferToTexture(cmd, t, texStaging, &up);
            }
            VriTextureBarrierDesc tb2{}; tb2.texture = t; tb2.before.access = VriAccess_CopyDestinationWrite; tb2.before.layout = VriLayout_CopyDestination; tb2.before.stages = VriPipelineStage_Transfer;
            tb2.after.access = VriAccess_ShaderResourceRead; tb2.after.layout = VriLayout_ShaderResource; tb2.after.stages = VriPipelineStage_FragmentShader; tb2.aspect = VriImageAspect_Color; tb2.layerNum = layerNum;
            VriBarrierGroupDesc g2{}; g2.textures = &tb2; g2.textureNum = 1; c.CmdBarrier(cmd, &g2);
        };
        if (bindless) for (uint32_t i = 0; i < kTexCount; ++i) { /* each tex: one image at offset i */
            VriTextureBarrierDesc tb{}; tb.texture = texes[i]; tb.before.layout = VriLayout_Undefined; tb.before.stages = VriPipelineStage_None;
            tb.after.access = VriAccess_CopyDestinationWrite; tb.after.layout = VriLayout_CopyDestination; tb.after.stages = VriPipelineStage_Transfer; tb.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc g{}; g.textures = &tb; g.textureNum = 1; c.CmdBarrier(cmd, &g);
            VriBufferTextureCopyDesc up{}; up.bufferOffset = uint64_t(i) * kImgBytes; up.texture.aspect = VriImageAspect_Color; up.texture.layerNum = 1; up.texture.width = kTexDim; up.texture.height = kTexDim;
            c.CmdUploadBufferToTexture(cmd, texes[i], texStaging, &up);
            VriTextureBarrierDesc tb2{}; tb2.texture = texes[i]; tb2.before.access = VriAccess_CopyDestinationWrite; tb2.before.layout = VriLayout_CopyDestination; tb2.before.stages = VriPipelineStage_Transfer;
            tb2.after.access = VriAccess_ShaderResourceRead; tb2.after.layout = VriLayout_ShaderResource; tb2.after.stages = VriPipelineStage_FragmentShader; tb2.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc g2{}; g2.textures = &tb2; g2.textureNum = 1; c.CmdBarrier(cmd, &g2);
        }
        else uploadTex(arrayTex, kTexCount);

        c.EndCommandBuffer(cmd);
        VriFenceSubmitDesc sig{}; sig.fence = app.fence; sig.value = 1;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(app.queue, &sub); c.Wait(app.fence, 1);
    }

    app.onRecord = [pipeline, layout, set, vbuf, ibuf, indexCount](VriCommandBuffer* cmd) {
        app.c.CmdSetPipeline(cmd, pipeline);
        app.c.CmdSetPipelineLayout(cmd, layout);
        app.c.CmdSetDescriptorSet(cmd, 0, set);
        VriVertexBufferBinding vb{}; vb.buffer = vbuf; vb.offset = 0; app.c.CmdSetVertexBuffers(cmd, 0, &vb, 1);
        app.c.CmdSetIndexBuffer(cmd, ibuf, 0, VriIndexType_UInt16);
        VriDrawIndexedDesc di{}; di.indexNum = indexCount; di.instanceNum = 1; app.c.CmdDrawIndexed(cmd, &di);
    };

    app.SetupCapture();
    app.Run();
    return 0;
}
