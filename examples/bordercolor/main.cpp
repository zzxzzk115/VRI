// Custom sampler border color: a quad whose UVs span [-1, 2] samples a procedural checker
// texture with ClampToBorder addressing, so a one-texture-wide ring around the image takes
// the sampler's border color. ImGui cycles the six preset VriBorderColor values and - where
// VriDeviceDesc::hasCustomBorderColor - an arbitrary RGBA picked with a color editor
// (VriSamplerDesc::useCustomBorderColor/customBorderColor). Changing the border rebuilds the
// sampler and rewrites the descriptor set; the frame loop is fully synchronous, so the old
// sampler is idle by then. Shared scaffolding: examples/common/example_app.h.
#include "../common/example_app.h"

#include <cmath>
#include <vector>

#include "shaders/examples/border_quad_dxbc.h" // g_borderQuadDxbcVS / PS (D3D12)
#include "shaders/examples/border_quad_spv.h"  // g_borderQuadSpv (Vulkan + OpenGL)
#include "shaders/examples/border_quad_wgsl.h" // g_borderQuadWgsl (WebGPU; desktop-only target, still generated)

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480;
    constexpr uint32_t kTexSize = 64;

    // A colorful checker with a thin bright frame, so the texture region and the border
    // ring around it are unmistakable.
    std::vector<uint8_t> BuildChecker()
    {
        std::vector<uint8_t> px(kTexSize * kTexSize * 4);
        for (uint32_t y = 0; y < kTexSize; ++y)
            for (uint32_t x = 0; x < kTexSize; ++x)
            {
                uint8_t*   p    = &px[(y * kTexSize + x) * 4];
                const bool a    = ((x / 8) + (y / 8)) & 1;
                const bool edge = x == 0 || y == 0 || x == kTexSize - 1 || y == kTexSize - 1;
                p[0]            = edge ? 255 : (a ? 235 : 40);
                p[1]            = edge ? 255 : (a ? 120 : 90);
                p[2]            = edge ? 255 : (a ? 60 : 200);
                p[3]            = 255;
            }
        return px;
    }

    const char* kPresetNames[] = {
        "float transparent black",
        "float opaque black",
        "float opaque white",
        "int transparent black",
        "int opaque black",
        "int opaque white",
    };
} // namespace

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.Init("bordercolor", kWidth, kHeight, /*hasDepth*/ false);
    VriCoreInterface& c = app.c;

    const bool hasCustom = c.GetDeviceDesc(app.dev)->hasCustomBorderColor == VRI_TRUE;

    // ---- checker texture ----
    VriTextureDesc td {};
    td.type           = VriTextureType_2D;
    td.format         = VriFormat_RGBA8_UNORM;
    td.width          = kTexSize;
    td.height         = kTexSize;
    td.depth          = 1;
    td.mipNum         = 1;
    td.layerNum       = 1;
    td.sampleNum      = 1;
    td.usage          = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst;
    td.memoryLocation = VriMemoryLocation_Device;
    VriTexture* tex   = nullptr;
    if (c.CreateTexture(app.dev, &td, &tex) != VriResult_Success)
        app.Fail("CreateTexture failed");
    VriTextureViewDesc tvd {};
    tvd.texture            = tex;
    tvd.viewType           = VriTextureViewType_2D;
    tvd.format             = VriFormat_Unknown;
    tvd.aspect             = VriImageAspect_Color;
    VriDescriptor* texView = nullptr;
    if (c.CreateTextureView(app.dev, &tvd, &texView) != VriResult_Success)
        app.Fail("texture view failed");

    const std::vector<uint8_t> pixels = BuildChecker();
    app.BeginUpload();
    app.UploadTexture(tex, pixels.data(), kTexSize, kTexSize, 1, kTexSize * kTexSize * 4);
    app.EndUpload();

    // ---- layout: texture@0 + sampler@1 (fragment) ----
    VriDescriptorRangeDesc r[2] {};
    r[0].baseRegister   = 0;
    r[0].descriptorNum  = 1;
    r[0].descriptorType = VriDescriptorType_Texture;
    r[0].shaderStages   = VriShaderStage_Fragment;
    r[1].baseRegister   = 1;
    r[1].descriptorNum  = 1;
    r[1].descriptorType = VriDescriptorType_Sampler;
    r[1].shaderStages   = VriShaderStage_Fragment;
    VriDescriptorSetDesc sd {};
    sd.registerSpace = 0;
    sd.ranges        = r;
    sd.rangeNum      = 2;
    VriPipelineLayoutDesc ld {};
    ld.descriptorSets         = &sd;
    ld.descriptorSetNum       = 1;
    VriPipelineLayout* layout = nullptr;
    c.CreatePipelineLayout(app.dev, &ld, &layout);

    const vriex::ExampleApp::ShaderVariants vs {
        VRI_SHADER_BLOB(g_borderQuadSpv),
        VRI_SHADER_BLOB(g_borderQuadWgsl),
        VRI_SHADER_D3D12(g_borderQuadDxbcVS),
    };
    const vriex::ExampleApp::ShaderVariants ps {
        VRI_SHADER_BLOB(g_borderQuadSpv),
        VRI_SHADER_BLOB(g_borderQuadWgsl),
        VRI_SHADER_D3D12(g_borderQuadDxbcPS),
    };
    VriShaderDesc sh[2] = {
        app.Shader(VriShaderStage_Vertex, "vertexMain", vs),
        app.Shader(VriShaderStage_Fragment, "fragmentMain", ps),
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
        app.Fail("pipeline failed");

    VriDescriptorPoolDesc pdsc {};
    pdsc.descriptorSetMaxNum = 1;
    pdsc.textureMaxNum       = 1;
    pdsc.samplerMaxNum       = 1;
    VriDescriptorPool* pool  = nullptr;
    c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* set = nullptr;
    c.AllocateDescriptorSets(pool, layout, 0, &set, 1);

    // ---- sampler, rebuilt whenever the border selection changes ----
    static int            preset       = 2; // float opaque white: obvious against the dark clear
    static bool           useCustom    = false;
    static float          custom[4]    = {0.2f, 0.9f, 0.4f, 1.0f};
    static bool           samplerDirty = true;
    static VriDescriptor* sampler      = nullptr;

    auto rebuildSampler = [&, set] {
        if (sampler)
            c.DestroyDescriptor(sampler); // frame loop is synchronous: the GPU is done with it
        VriSamplerDesc smp {};
        smp.magFilter    = VriFilter_Nearest;
        smp.minFilter    = VriFilter_Nearest;
        smp.mipmapMode   = VriMipmapMode_Nearest;
        smp.addressModeU = VriAddressMode_ClampToBorder;
        smp.addressModeV = VriAddressMode_ClampToBorder;
        smp.addressModeW = VriAddressMode_ClampToBorder;
        smp.maxLod       = 1.0f;
        smp.borderColor  = static_cast<VriBorderColor>(preset);
        if (useCustom)
        {
            smp.useCustomBorderColor = VRI_TRUE;
            smp.customBorderColor[0] = custom[0];
            smp.customBorderColor[1] = custom[1];
            smp.customBorderColor[2] = custom[2];
            smp.customBorderColor[3] = custom[3];
        }
        if (c.CreateSampler(app.dev, &smp, &sampler) != VriResult_Success)
            app.Fail("CreateSampler failed");
        const VriDescriptor*         t[1] = {texView};
        const VriDescriptor*         s[1] = {sampler};
        VriDescriptorRangeUpdateDesc u[2] {};
        u[0].descriptors   = t;
        u[0].descriptorNum = 1;
        u[1].descriptors   = s;
        u[1].descriptorNum = 1;
        c.UpdateDescriptorRanges(set, 0, 2, u);
    };

    app.onGui = [=] {
        if (ImGui::Combo("border", &preset, kPresetNames, 6))
            samplerDirty = true;
        if (hasCustom)
        {
            if (ImGui::Checkbox("custom border color", &useCustom))
                samplerDirty = true;
            if (useCustom && ImGui::ColorEdit4("color", custom))
                samplerDirty = true;
        }
        else
        {
            ImGui::Text("hasCustomBorderColor = false on %s\n(presets only)", app.apiName);
        }
        ImGui::TextUnformatted("center: texture | ring: border color");
    };

    app.onUpdate = [=](uint64_t) {
        if (samplerDirty)
        {
            rebuildSampler();
            samplerDirty = false;
        }
    };

    app.onRecord = [=](VriCommandBuffer* cmd) {
        c.CmdSetPipeline(cmd, pipeline);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, set);
        VriDrawDesc d {};
        d.vertexNum   = 6;
        d.instanceNum = 1;
        c.CmdDraw(cmd, &d);
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyDescriptorPool(pool);
    if (sampler)
        c.DestroyDescriptor(sampler);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    c.DestroyDescriptor(texView);
    c.DestroyTexture(tex);
    app.Shutdown();
#endif
    return 0;
}
