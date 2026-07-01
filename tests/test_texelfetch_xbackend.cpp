// Cross-backend texelFetch-without-sampler regression: the fragment shader reads an integer
// texture via .Load (texel fetch, NO sampler) and the pipeline layout binds ONLY a Texture,
// no Sampler - VRI's separate-image, sampler-less path (e.g. an r8ui palette-index texture).
//
// On GL/WebGL this image has no (texture, sampler) pair to fuse, so SPIRV-Cross must synthesise
// a dummy sampler before build_combined_image_samplers(); without it the transpile aborts with
// "texelFetch without sampler" and CreateGraphicsPipeline fails. This test is that path's guard:
// building the pipeline must succeed on every enabled backend.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

#include "shaders/tests/triangle_texelfetch_spv.h" // g_triangleTexelfetchSpv (Vulkan + OpenGL)

namespace
{
    bool BuildTexelFetchPipeline(VriGraphicsAPI api, bool& ran)
    {
        ran = false;
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        VriDevice* dev      = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return false;
        ran = true;

        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);

        // pipeline layout: set 0 = {binding 0: Texture} (fragment) - NO sampler.
        VriDescriptorRangeDesc range {};
        range.baseRegister   = 0;
        range.descriptorNum  = 1;
        range.descriptorType = VriDescriptorType_Texture;
        range.shaderStages   = VriShaderStage_Fragment;
        VriDescriptorSetDesc setDesc {};
        setDesc.registerSpace = 0;
        setDesc.ranges        = &range;
        setDesc.rangeNum      = 1;
        VriPipelineLayoutDesc ld {};
        ld.descriptorSets         = &setDesc;
        ld.descriptorSetNum       = 1;
        VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriShaderDesc sh[2] {};
        sh[0].stage          = VriShaderStage_Vertex;
        sh[0].bytecode       = g_triangleTexelfetchSpv;
        sh[0].bytecodeSize   = sizeof(g_triangleTexelfetchSpv);
        sh[0].entryPointName = "vertexMain";
        sh[1].stage          = VriShaderStage_Fragment;
        sh[1].bytecode       = g_triangleTexelfetchSpv;
        sh[1].bytecodeSize   = sizeof(g_triangleTexelfetchSpv);
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
        // The regression point: on GL this transpiles the sampler-less texelFetch to GLSL, which
        // needs the synthesised dummy sampler. A failure here is the bug this test guards against.
        const bool ok = (c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);

        if (pipeline)
            c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        vriDestroyDevice(dev);
        return ok;
    }
} // namespace

TEST_CASE("texelFetch-without-sampler: sampler-less integer-texture pipeline builds on each backend")
{
    bool       ran = false;
    const bool vk  = BuildTexelFetchPipeline(VriGraphicsAPI_Vulkan, ran);
    if (ran)
    {
        CHECK(vk);
    }
    else
    {
        MESSAGE("Vulkan unavailable - skipped");
    }

    const bool gl = BuildTexelFetchPipeline(VriGraphicsAPI_OpenGL, ran);
    if (ran)
    {
        CHECK(gl);
    }
    else
    {
        MESSAGE("OpenGL unavailable - skipped");
    }
}
