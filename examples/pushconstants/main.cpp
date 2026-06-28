// Push constants: a fullscreen triangle whose color is supplied entirely by a push constant
// updated every frame (no vertex buffer, no descriptor set). Exercises CmdSetConstants across
// backends - real push constants on Vulkan/D3D12, a reserved-UBO emulation on OpenGL/WebGL +
// WebGPU. Ported in spirit from Sascha Willems' pushconstants. Shared host scaffolding in
// examples/common.
#include "../common/example_app.h"

#include <cmath>

#include "examples/shaders/pushconst_dxbc.h" // g_pushconstDxbcVS / PS (D3D12)
#include "examples/shaders/pushconst_spv.h"  // g_pushconstSpv  (Vulkan + OpenGL)
#include "examples/shaders/pushconst_wgsl.h" // g_pushconstWgsl (WebGPU)

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480;
    float              g_color[4] = {1.0f, 0.5f, 0.2f, 1.0f}; // updated each frame, pushed as a constant
} // namespace

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.Init("pushconstants", kWidth, kHeight, /*hasDepth*/ false);
    VriCoreInterface& c = app.c;

    // pipeline layout: a single 16-byte push constant (a float4 color), no descriptor sets
    VriPushConstantDesc pcDesc {};
    pcDesc.baseRegister = 0;
    pcDesc.size         = sizeof(g_color);
    pcDesc.shaderStages = VriShaderStage_Vertex | VriShaderStage_Fragment;
    VriPipelineLayoutDesc ld {};
    ld.pushConstants          = &pcDesc;
    ld.pushConstantNum        = 1;
    VriPipelineLayout* layout = nullptr;
    c.CreatePipelineLayout(app.dev, &ld, &layout);

    VriShaderDesc sh[2] {};
    sh[0].stage          = VriShaderStage_Vertex;
    sh[0].entryPointName = "vertexMain";
    sh[1].stage          = VriShaderStage_Fragment;
    sh[1].entryPointName = "fragmentMain";
    if (app.useDxbc)
    {
        sh[0].bytecode     = g_pushconstDxbcVS;
        sh[0].bytecodeSize = sizeof(g_pushconstDxbcVS);
        sh[1].bytecode     = g_pushconstDxbcPS;
        sh[1].bytecodeSize = sizeof(g_pushconstDxbcPS);
    }
    else
    {
        sh[0].bytecode = sh[1].bytecode =
            app.useWgsl ? static_cast<const void*>(g_pushconstWgsl) : static_cast<const void*>(g_pushconstSpv);
        sh[0].bytecodeSize = sh[1].bytecodeSize = app.useWgsl ? sizeof(g_pushconstWgsl) : sizeof(g_pushconstSpv);
    }

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

    static bool  animate = true;
    static float t       = 0.0f;
    app.onUpdate         = [](uint64_t) {
        if (!animate)
            return;
        t += 1.8f * app.dt; // 1.8/s (= 0.03/frame at 60fps)
        g_color[0] = 0.5f + 0.5f * std::sin(t);
        g_color[1] = 0.5f + 0.5f * std::sin(t + 2.094f);
        g_color[2] = 0.5f + 0.5f * std::sin(t + 4.188f);
    };
    app.onGui = [] {
        ImGui::Checkbox("animate", &animate);
        ImGui::BeginDisabled(animate);
        ImGui::ColorEdit3("color", g_color);
        ImGui::EndDisabled();
    };
    app.onRecord = [layout, pipeline](VriCommandBuffer* cmd) {
        app.c.CmdSetPipeline(cmd, pipeline);
        app.c.CmdSetPipelineLayout(cmd, layout);                 // must precede CmdSetConstants
        app.c.CmdSetConstants(cmd, 0, g_color, sizeof(g_color)); // the push constant
        VriDrawDesc d {};
        d.vertexNum   = 3;
        d.instanceNum = 1;
        app.c.CmdDraw(cmd, &d);
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    app.Shutdown();
#endif
    return 0;
}
