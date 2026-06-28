// Windowed triangle: the minimal VRI swapchain -> present path, identical across every
// backend. The host scaffolding (backend select, windowing, present loop, headless
// capture) lives in examples/common/example_app.h; this file just builds a pipeline and
// records a 3-vertex draw. VRI_API / ?backend force a backend; VRI_MAX_FRAMES / ?frames=N
// auto-exit.
#include "../common/example_app.h"

// Slang-authored triangle, compiled offline to per-target blobs (see `xmake shaders`).
#include "common/shaders/triangle_dxbc.h" // g_triangleDxbcVS/PS (Direct3D 12, per stage)
#include "common/shaders/triangle_spv.h"  // g_triangleSpv      (Vulkan + OpenGL via SPIRV-Cross)
#include "common/shaders/triangle_wgsl.h" // g_triangleWgsl     (WebGPU)

int main(int, char**)
{
    static vriex::ExampleApp app;
    app.Init("triangle", 640, 480, /*hasDepth*/ false); // 640*4 is 256-aligned for the readback self-check
    app.SetClearColor(0.1f, 0.1f, 0.12f);
    VriCoreInterface& c = app.c;

    VriPipelineLayoutDesc layoutDesc {};
    VriPipelineLayout*    layout = nullptr;
    c.CreatePipelineLayout(app.dev, &layoutDesc, &layout);

    VriShaderDesc shaders[2] {};
    shaders[0].stage          = VriShaderStage_Vertex;
    shaders[1].stage          = VriShaderStage_Fragment;
    shaders[0].entryPointName = "vertexMain";
    shaders[1].entryPointName = "fragmentMain";
    if (app.useDxbc)
    {
        shaders[0].bytecode     = g_triangleDxbcVS;
        shaders[0].bytecodeSize = sizeof(g_triangleDxbcVS);
        shaders[1].bytecode     = g_triangleDxbcPS;
        shaders[1].bytecodeSize = sizeof(g_triangleDxbcPS);
    }
    else
    {
        shaders[0].bytecode =
            app.useWgsl ? static_cast<const void*>(g_triangleWgsl) : static_cast<const void*>(g_triangleSpv);
        shaders[0].bytecodeSize = app.useWgsl ? sizeof(g_triangleWgsl) : sizeof(g_triangleSpv);
        shaders[1].bytecode     = shaders[0].bytecode;
        shaders[1].bytecodeSize = shaders[0].bytecodeSize;
    }

    VriColorAttachmentDesc colorAttach {};
    colorAttach.format         = app.swapFormat;
    colorAttach.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd {};
    pd.pipelineLayout          = layout;
    pd.shaders                 = shaders;
    pd.shaderNum               = 2;
    pd.inputAssembly.topology  = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode  = VriCullMode_None;
    pd.rasterization.lineWidth = 1.0f;
    pd.multisample.sampleNum   = 1;
    pd.outputMerger.colors     = &colorAttach;
    pd.outputMerger.colorNum   = 1;
    VriPipeline* pipeline      = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &pd, &pipeline) != VriResult_Success)
        app.Fail("CreateGraphicsPipeline failed");

    app.onRecord = [pipeline](VriCommandBuffer* cmd) {
        app.c.CmdSetPipeline(cmd, pipeline);
        VriDrawDesc draw {};
        draw.vertexNum   = 3;
        draw.instanceNum = 1;
        app.c.CmdDraw(cmd, &draw);
    };
    app.onGui = [] { ImGui::ColorEdit3("clear color", app.clearColor); };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    app.Shutdown();
#endif
    return 0;
}
