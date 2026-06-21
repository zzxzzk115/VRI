// Windowed triangle example: SDL3 window -> VRI swapchain -> present.
//
// Runs on Vulkan or WebGPU (set VRI_API=vulkan|webgpu; default vulkan), proving
// the windowing seam (VriWindowHandle from an SDL_Window*) and swapchain work
// across backends from one code path. Set VRI_MAX_FRAMES=N to auto-exit.

#include <SDL3/SDL.h>

#include <vri/vri.h>
#include <vri/integration/vri_sdl3.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Slang-authored triangle, compiled offline to per-target blobs (see `xmake shaders`).
#include "tests/shaders/triangle_spv.h"  // g_triangleSpv      (Vulkan + OpenGL via SPIRV-Cross)
#include "tests/shaders/triangle_wgsl.h" // g_triangleWgsl     (WebGPU)
#include "tests/shaders/triangle_dxbc.h" // g_triangleDxbcVS/PS (Direct3D 12, per stage)

namespace
{
    constexpr uint32_t kWidth = 800;
    constexpr uint32_t kHeight = 600;
    constexpr VriFormat kSwapFormat = VriFormat_BGRA8_UNORM;

    void Fail(const char* msg)
    {
        std::fprintf(stderr, "[example] %s\n", msg);
        std::exit(1);
    }
}

int main(int, char**)
{
    const char* apiEnv = std::getenv("VRI_API");
    VriGraphicsAPI api = VriGraphicsAPI_Vulkan;
    if (apiEnv && std::strcmp(apiEnv, "webgpu") == 0) api = VriGraphicsAPI_WebGPU;
    else if (apiEnv && (std::strcmp(apiEnv, "opengl") == 0 || std::strcmp(apiEnv, "gl") == 0)) api = VriGraphicsAPI_OpenGL;
    else if (apiEnv && (std::strcmp(apiEnv, "d3d12") == 0 || std::strcmp(apiEnv, "dx12") == 0)) api = VriGraphicsAPI_D3D12;
    const bool useWgsl = (api == VriGraphicsAPI_WebGPU); // GL consumes SPIR-V (transpiled), like Vulkan
    const bool useDxbc = (api == VriGraphicsAPI_D3D12);  // D3D12 consumes per-stage DXBC blobs
    const char* apiName = api == VriGraphicsAPI_WebGPU ? "WebGPU" : (api == VriGraphicsAPI_OpenGL ? "OpenGL" : (api == VriGraphicsAPI_D3D12 ? "D3D12" : "Vulkan"));

    if (!SDL_Init(SDL_INIT_VIDEO))
        Fail("SDL_Init failed");
    char title[64];
    std::snprintf(title, sizeof(title), "VRI Triangle (%s)", apiName);
    SDL_Window* window = SDL_CreateWindow(title, kWidth, kHeight, 0);
    if (!window)
        Fail("SDL_CreateWindow failed");

    VriDeviceCreationDesc deviceDesc{};
    deviceDesc.graphicsAPI = api;
    deviceDesc.enableValidation = VRI_TRUE;
    deviceDesc.bestEffort = VRI_TRUE;
    VriDevice* device = nullptr;
    if (vriCreateDevice(&deviceDesc, &device) != VriResult_Success)
        Fail("vriCreateDevice failed");

    VriCoreInterface core{};
    VriSwapChainInterface swap{};
    if (vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(core), &core) != VriResult_Success ||
        vriGetInterface(device, VRI_INTERFACE_SWAPCHAIN, sizeof(swap), &swap) != VriResult_Success)
        Fail("vriGetInterface failed");

    VriQueue* queue = nullptr;
    core.GetQueue(device, VriQueueType_Graphics, 0, &queue);

    VriSwapChainDesc scDesc{};
    scDesc.window = vriWindowHandleFromSDL3(window);
    scDesc.queue = queue;
    scDesc.format = kSwapFormat;
    scDesc.width = kWidth;
    scDesc.height = kHeight;
    scDesc.textureNum = 3;
    scDesc.vsync = VRI_TRUE;
    VriSwapChain* swapchain = nullptr;
    if (swap.CreateSwapChain(device, &scDesc, &swapchain) != VriResult_Success)
        Fail("CreateSwapChain failed");

    // pipeline
    VriPipelineLayoutDesc layoutDesc{};
    VriPipelineLayout* layout = nullptr;
    core.CreatePipelineLayout(device, &layoutDesc, &layout);

    VriShaderDesc shaders[2]{};
    shaders[0].stage = VriShaderStage_Vertex;
    shaders[1].stage = VriShaderStage_Fragment;
    shaders[0].entryPointName = "vertexMain";
    shaders[1].entryPointName = "fragmentMain";
    if (useDxbc) // D3D12: separate per-stage DXBC blobs (a graphics PSO takes one VS + one PS)
    {
        shaders[0].bytecode = g_triangleDxbcVS; shaders[0].bytecodeSize = sizeof(g_triangleDxbcVS);
        shaders[1].bytecode = g_triangleDxbcPS; shaders[1].bytecodeSize = sizeof(g_triangleDxbcPS);
    }
    else // SPIR-V (Vulkan/OpenGL) or WGSL (WebGPU): one multi-entry module for both stages
    {
        shaders[0].bytecode = useWgsl ? static_cast<const void*>(g_triangleWgsl) : static_cast<const void*>(g_triangleSpv);
        shaders[0].bytecodeSize = useWgsl ? sizeof(g_triangleWgsl) : sizeof(g_triangleSpv);
        shaders[1].bytecode = shaders[0].bytecode; shaders[1].bytecodeSize = shaders[0].bytecodeSize;
    }

    VriColorAttachmentDesc colorAttach{};
    colorAttach.format = kSwapFormat;
    colorAttach.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd{};
    pd.pipelineLayout = layout;
    pd.shaders = shaders;
    pd.shaderNum = 2;
    pd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode = VriCullMode_None;
    pd.rasterization.lineWidth = 1.0f;
    pd.multisample.sampleNum = 1;
    pd.outputMerger.colors = &colorAttach;
    pd.outputMerger.colorNum = 1;
    VriPipeline* pipeline = nullptr;
    if (core.CreateGraphicsPipeline(device, &pd, &pipeline) != VriResult_Success)
        Fail("CreateGraphicsPipeline failed");

    VriCommandAllocator* allocator = nullptr;
    core.CreateCommandAllocator(device, VriQueueType_Graphics, &allocator);
    VriCommandBuffer* cmd = nullptr;
    core.CreateCommandBuffer(allocator, &cmd);
    VriFence* fence = nullptr;
    core.CreateFence(device, 0, &fence);

    const char* maxFramesEnv = std::getenv("VRI_MAX_FRAMES");
    const uint64_t maxFrames = maxFramesEnv ? std::strtoull(maxFramesEnv, nullptr, 10) : 0;

    uint64_t frameValue = 0;
    bool running = true;
    while (running)
    {
        SDL_Event e;
        while (SDL_PollEvent(&e))
            if (e.type == SDL_EVENT_QUIT || e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED)
                running = false;
        if (!running)
            break;

        uint32_t index = 0;
        if (swap.AcquireNextTexture(swapchain, nullptr, 0, &index) == VriResult_OutOfDate)
        {
            swap.Resize(swapchain, kWidth, kHeight);
            continue;
        }

        // current backbuffer texture + a per-frame view (required for WebGPU,
        // harmless for Vulkan).
        VriTexture* textures[8] = {};
        uint32_t count = 8;
        swap.GetSwapChainTextures(swapchain, textures, &count);
        VriTexture* backbuffer = textures[index];

        VriTextureViewDesc viewDesc{};
        viewDesc.texture = backbuffer;
        viewDesc.viewType = VriTextureViewType_2D;
        viewDesc.format = VriFormat_Unknown;
        viewDesc.aspect = VriImageAspect_Color;
        VriDescriptor* view = nullptr;
        core.CreateTextureView(device, &viewDesc, &view);

        core.BeginCommandBuffer(cmd);

        VriTextureBarrierDesc toColor{};
        toColor.texture = backbuffer;
        toColor.before.layout = VriLayout_Undefined;
        toColor.before.stages = VriPipelineStage_None;
        toColor.after.access = VriAccess_ColorAttachmentWrite;
        toColor.after.layout = VriLayout_ColorAttachment;
        toColor.after.stages = VriPipelineStage_ColorAttachmentOutput;
        toColor.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc g0{};
        g0.textures = &toColor;
        g0.textureNum = 1;
        core.CmdBarrier(cmd, &g0);

        VriAttachmentDesc rt{};
        rt.view = view;
        rt.loadOp = VriAttachmentLoadOp_Clear;
        rt.storeOp = VriAttachmentStoreOp_Store;
        rt.clearValue.color.f32[0] = 0.1f;
        rt.clearValue.color.f32[1] = 0.1f;
        rt.clearValue.color.f32[2] = 0.12f;
        rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att{};
        att.colors = &rt;
        att.colorNum = 1;
        att.renderArea.width = kWidth;
        att.renderArea.height = kHeight;
        att.layerNum = 1;
        core.CmdBeginRendering(cmd, &att);
        VriViewport vp{0.0f, 0.0f, static_cast<float>(kWidth), static_cast<float>(kHeight), 0.0f, 1.0f};
        core.CmdSetViewports(cmd, &vp, 1);
        VriRect scissor{0, 0, kWidth, kHeight};
        core.CmdSetScissors(cmd, &scissor, 1);
        core.CmdSetPipeline(cmd, pipeline);
        VriDrawDesc draw{};
        draw.vertexNum = 3;
        draw.instanceNum = 1;
        core.CmdDraw(cmd, &draw);
        core.CmdEndRendering(cmd);

        VriTextureBarrierDesc toPresent{};
        toPresent.texture = backbuffer;
        toPresent.before.access = VriAccess_ColorAttachmentWrite;
        toPresent.before.layout = VriLayout_ColorAttachment;
        toPresent.before.stages = VriPipelineStage_ColorAttachmentOutput;
        toPresent.after.layout = VriLayout_Present;
        toPresent.after.stages = VriPipelineStage_AllCommands;
        toPresent.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc g1{};
        g1.textures = &toPresent;
        g1.textureNum = 1;
        core.CmdBarrier(cmd, &g1);

        core.EndCommandBuffer(cmd);

        VriFenceSubmitDesc signal{};
        signal.fence = fence;
        signal.value = ++frameValue;
        VriQueueSubmitDesc submit{};
        submit.commandBuffers = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences = &signal;
        submit.signalFenceNum = 1;
        core.QueueSubmit(queue, &submit);
        core.Wait(fence, frameValue); // serialized (no frame pipelining yet)

        swap.Present(swapchain, nullptr, 0);
        core.DestroyDescriptor(view);

        if (maxFrames != 0 && frameValue >= maxFrames)
            running = false;
    }

    core.DeviceWaitIdle(device);
    core.DestroyFence(fence);
    core.DestroyCommandAllocator(allocator);
    core.DestroyPipeline(pipeline);
    core.DestroyPipelineLayout(layout);
    swap.DestroySwapChain(swapchain);
    vriDestroyDevice(device);
    SDL_DestroyWindow(window);
    SDL_Quit();
    std::printf("[example] %s triangle: %llu frames presented\n",
                apiName, static_cast<unsigned long long>(frameValue));
    return 0;
}
