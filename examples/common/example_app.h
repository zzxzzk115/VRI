// example_app.h - shared host scaffolding for the VRI examples.
//
// Backend selection (VRI_API env / ?backend= URL / Auto), windowing (SDL3 on desktop,
// the page #canvas on the web), device + swapchain + an optional depth target, the
// present loop (desktop while-loop / emscripten main loop), headless auto-exit
// (VRI_MAX_FRAMES env / ?frames=N URL) with a pixel self-check + BMP capture - all live
// here so a backend/platform fix lands once for every example instead of being copy-pasted
// (and missed) across the three mains.
//
// An example fills in only what differs: it creates its resources/pipeline after Init(),
// sets onUpdate (per-frame uploads) + onRecord (pipeline + draw inside the pass), then
// Run(). The frame skeleton (acquire -> barriers -> begin/clear -> record -> present) is
// shared. Header-only; examples only.
#pragma once

#if defined(__EMSCRIPTEN__)
#    include <emscripten/emscripten.h>
#else
#    include <SDL3/SDL.h>
#    include <vri/integration/vri_sdl3.h>
#endif

#include <vri/vri.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "capture.h"

namespace vriex
{
    // VRI_API (vulkan|webgpu|opengl|d3d12) or, on the web, ?backend=webgpu|webgl override
    // the backend; otherwise VriGraphicsAPI_Auto lets VRI pick (WebGPU first, then WebGL).
    inline VriGraphicsAPI SelectApi()
    {
        const char* env = std::getenv("VRI_API");
        VriGraphicsAPI api = VriGraphicsAPI_Auto;
        if (env && std::strcmp(env, "webgpu") == 0) api = VriGraphicsAPI_WebGPU;
        else if (env && (std::strcmp(env, "opengl") == 0 || std::strcmp(env, "gl") == 0)) api = VriGraphicsAPI_OpenGL;
        else if (env && (std::strcmp(env, "d3d12") == 0 || std::strcmp(env, "dx12") == 0)) api = VriGraphicsAPI_D3D12;
        else if (env && std::strcmp(env, "vulkan") == 0) api = VriGraphicsAPI_Vulkan;
#if defined(__EMSCRIPTEN__)
        const int b = EM_ASM_INT({
            var s = location.search;
            if (s.indexOf('backend=webgpu') >= 0) return 1;
            if (s.indexOf('backend=webgl') >= 0 || s.indexOf('backend=opengl') >= 0) return 2;
            return 0;
        });
        if (b == 1) api = VriGraphicsAPI_WebGPU;
        else if (b == 2) api = VriGraphicsAPI_OpenGL;
#endif
        return api;
    }

    // Default diagnostic sink: every backend routes its native validation/diagnostics (VK
    // debug-utils, D3D12 InfoQueue, GL KHR_debug, WebGPU uncaptured errors) plus VRI's own
    // validation layer through this one callback. The examples install it so problems print
    // instead of vanishing; a real app would plug in its own logger here.
    inline void VRI_CALL DefaultMessageCallback(void*, VriMessageSeverity severity, const char* message)
    {
        const char* s = severity == VriMessageSeverity_Error ? "ERROR"
                      : severity == VriMessageSeverity_Warning ? "WARN" : "INFO";
        std::fprintf(stderr, "[VRI][%s] %s\n", s, message);
        std::fflush(stderr);
    }

    // VRI_MAX_FRAMES env (desktop) or ?frames=N URL (web): auto-exit after N frames.
    inline uint64_t QueryMaxFrames()
    {
        const char* env = std::getenv("VRI_MAX_FRAMES");
        uint64_t n = env ? std::strtoull(env, nullptr, 10) : 0;
#if defined(__EMSCRIPTEN__)
        const int f = EM_ASM_INT({ var s = location.search; var i = s.indexOf('frames='); return i >= 0 ? (parseInt(s.substring(i + 7)) | 0) : 0; });
        if (f > 0) n = static_cast<uint64_t>(f);
#endif
        return n;
    }

    struct ExampleApp
    {
        // ---- config (override before/at Init) ----
        const char* name = "example";
        uint32_t width = 640, height = 480;
        VriFormat swapFormat = VriFormat_BGRA8_UNORM;
        VriFormat depthFormat = VriFormat_D32_SFLOAT;
        bool hasDepth = false;
        uint64_t requestFeatures = 0; // VriFeatureBits to request at device creation (bestEffort)
        float clearColor[4] = {0.08f, 0.10f, 0.14f, 1.0f};

        // ---- valid after Init() ----
        VriDevice* dev = nullptr;
        VriCoreInterface c{};
        VriSwapChainInterface swap{};
        VriQueue* queue = nullptr;
        VriSwapChain* swapchain = nullptr;
        VriTexture* depth = nullptr; VriDescriptor* depthView = nullptr;
        VriCommandAllocator* alloc = nullptr; VriCommandBuffer* cmd = nullptr; VriFence* fence = nullptr;
        VriGraphicsAPI api = VriGraphicsAPI_Auto;
        const char* apiName = "Vulkan";
        bool useWgsl = false, useDxbc = false;

        // ---- the example fills these in before Run() ----
        // onUpdate: CPU-side per-frame work (e.g. write a staging buffer). Runs BEFORE the
        //   swapchain image is acquired - important on WebGPU, where a buffer map does an
        //   ASYNCIFY yield that must not sit between acquire and present.
        // onPreRender: record copies/barriers (e.g. staging -> constant buffer) after the
        //   command buffer begins, before the render pass opens.
        // onRecord: record pipeline + draw inside the render pass.
        std::function<void(uint64_t)> onUpdate;
        std::function<void(VriCommandBuffer*)> onPreRender;
        std::function<void(VriCommandBuffer*)> onRecord;

        // ---- internal ----
        VriBuffer* captureBuf = nullptr; const char* capturePath = nullptr;
        std::vector<VriBuffer*> uploadStaging; // staging buffers held until EndUpload() frees them
        uint64_t frameValue = 1; uint64_t maxFrames = 0; bool depthInit = false; bool running = true;
#if !defined(__EMSCRIPTEN__)
        SDL_Window* window = nullptr;
#endif

        [[noreturn]] void Fail(const char* msg) { std::fprintf(stderr, "[%s] %s\n", name, msg); std::exit(1); }

        void SetClearColor(float r, float g, float b, float a = 1.0f) { clearColor[0] = r; clearColor[1] = g; clearColor[2] = b; clearColor[3] = a; }

        // Bring up backend + window + device + swapchain (+ depth). The example creates its
        // own resources afterwards using dev / c / useWgsl / useDxbc, then calls Run().
        void Init(const char* exampleName, uint32_t w, uint32_t h, bool depthWanted)
        {
            name = exampleName; width = w; height = h; hasDepth = depthWanted;
            api = SelectApi();
#if !defined(__EMSCRIPTEN__)
            if (!SDL_Init(SDL_INIT_VIDEO)) Fail("SDL_Init failed");
#endif
            VriDeviceCreationDesc dd{}; dd.graphicsAPI = api; dd.enableValidation = VRI_TRUE; dd.bestEffort = VRI_TRUE;
            dd.enabledFeatures = requestFeatures; // bestEffort: ungranted features just leave the matching hasX false
            static VriCallbackInterface cb{}; cb.MessageCallback = DefaultMessageCallback; dd.callbackInterface = &cb;
            if (vriCreateDevice(&dd, &dev) != VriResult_Success) Fail("vriCreateDevice failed");
            if (vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) != VriResult_Success ||
                vriGetInterface(dev, VRI_INTERFACE_SWAPCHAIN, sizeof(swap), &swap) != VriResult_Success)
                Fail("vriGetInterface failed");

            api = c.GetDeviceDesc(dev)->graphicsAPI; // Auto resolves to a concrete backend
            useWgsl = api == VriGraphicsAPI_WebGPU;  // GL consumes SPIR-V (transpiled), like Vulkan
            useDxbc = api == VriGraphicsAPI_D3D12;
            // The GL backend is WebGL2 in the browser, plain OpenGL on desktop - label accordingly.
#if defined(__EMSCRIPTEN__)
            const char* glName = "WebGL";
#else
            const char* glName = "OpenGL";
#endif
            apiName = api == VriGraphicsAPI_WebGPU ? "WebGPU" : ((api == VriGraphicsAPI_OpenGL || api == VriGraphicsAPI_OpenGLES) ? glName : (api == VriGraphicsAPI_D3D12 ? "D3D12" : "Vulkan"));

            char title[64]; std::snprintf(title, sizeof(title), "VRI %s (%s)", name, apiName);
            VriWindowHandle wh{};
#if defined(__EMSCRIPTEN__)
            // Target the page canvas directly: the GL backend owns the single Emscripten
            // GLFW window, and a second glfwCreateWindow aborts. WebGPU makes its surface
            // from the same selector. The swapchain sizes this canvas to width x height.
            (void)title;
            // Feed the custom shell (examples/common/shell.html) the example name + backend.
            EM_ASM({ if (Module.vriSetTitle) Module.vriSetTitle(UTF8ToString($0), UTF8ToString($1)); }, name, apiName);
            wh.type = VriWindowSystem_Web; wh.handle.web.selector = "#canvas";
#else
            window = SDL_CreateWindow(title, static_cast<int>(width), static_cast<int>(height), 0);
            if (!window) Fail("SDL_CreateWindow failed");
            wh = vriWindowHandleFromSDL3(window);
#endif
            c.GetQueue(dev, VriQueueType_Graphics, 0, &queue);
            VriSwapChainDesc scd{}; scd.window = wh; scd.queue = queue; scd.format = swapFormat;
            scd.width = width; scd.height = height; scd.textureNum = 3; scd.vsync = VRI_TRUE;
            if (swap.CreateSwapChain(dev, &scd, &swapchain) != VriResult_Success) Fail("CreateSwapChain failed");

            if (hasDepth)
            {
                VriTextureDesc dtd{}; dtd.type = VriTextureType_2D; dtd.format = depthFormat; dtd.width = width; dtd.height = height; dtd.depth = 1;
                dtd.mipNum = 1; dtd.layerNum = 1; dtd.sampleNum = 1; dtd.usage = VriTextureUsage_DepthStencilAttachment; dtd.memoryLocation = VriMemoryLocation_Device;
                if (c.CreateTexture(dev, &dtd, &depth) != VriResult_Success) Fail("depth CreateTexture failed");
                VriTextureViewDesc dvd{}; dvd.texture = depth; dvd.viewType = VriTextureViewType_2D; dvd.format = VriFormat_Unknown; dvd.aspect = VriImageAspect_Depth;
                if (c.CreateTextureView(dev, &dvd, &depthView) != VriResult_Success) Fail("depth view failed");
            }

            c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
            c.CreateCommandBuffer(alloc, &cmd);
            c.CreateFence(dev, 0, &fence);
        }

        // ---- one-shot upload helper (kills the staging boilerplate examples kept repeating) --
        // Usage: BeginUpload(); UploadBuffer(...)/UploadTexture(...) ...; EndUpload();
        // Each Upload* creates a host staging buffer, records the copy + the
        // copy->read barrier, and EndUpload() submits, waits, and frees the staging.
        void BeginUpload() { c.BeginCommandBuffer(cmd); }

        void UploadBuffer(VriBuffer* dst, const void* data, uint64_t size, VriAccessFlags afterAccess, VriPipelineStageFlags afterStage)
        {
            VriBufferDesc sd{}; sd.size = size; sd.usage = VriBufferUsage_TransferSrc; sd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* stg = nullptr; c.CreateBuffer(dev, &sd, &stg);
            std::memcpy(c.MapBuffer(stg, 0, size), data, size); c.UnmapBuffer(stg);
            uploadStaging.push_back(stg);
            VriBufferCopyDesc cp{}; cp.size = size; c.CmdCopyBuffer(cmd, dst, stg, &cp);
            VriBufferBarrierDesc bb{}; bb.buffer = dst; bb.before.access = VriAccess_CopyDestinationWrite; bb.before.stages = VriPipelineStage_Transfer;
            bb.after.access = afterAccess; bb.after.stages = afterStage;
            VriBarrierGroupDesc g{}; g.buffers = &bb; g.bufferNum = 1; c.CmdBarrier(cmd, &g);
        }

        // Upload `layers` images (each w*h, layerBytes each, contiguous in `data`) into a 2D or
        // 2D-array texture, leaving it sampleable. Backends handle row-pitch alignment internally.
        void UploadTexture(VriTexture* dst, const void* data, uint32_t w, uint32_t h, uint32_t layers, uint32_t layerBytes)
        {
            const uint64_t total = uint64_t(layerBytes) * layers;
            VriBufferDesc sd{}; sd.size = total; sd.usage = VriBufferUsage_TransferSrc; sd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* stg = nullptr; c.CreateBuffer(dev, &sd, &stg);
            std::memcpy(c.MapBuffer(stg, 0, total), data, total); c.UnmapBuffer(stg);
            uploadStaging.push_back(stg);
            VriTextureBarrierDesc tb{}; tb.texture = dst; tb.before.layout = VriLayout_Undefined; tb.before.stages = VriPipelineStage_None;
            tb.after.access = VriAccess_CopyDestinationWrite; tb.after.layout = VriLayout_CopyDestination; tb.after.stages = VriPipelineStage_Transfer; tb.aspect = VriImageAspect_Color; tb.layerNum = layers;
            VriBarrierGroupDesc g0{}; g0.textures = &tb; g0.textureNum = 1; c.CmdBarrier(cmd, &g0);
            for (uint32_t layer = 0; layer < layers; ++layer)
            {
                VriBufferTextureCopyDesc up{}; up.bufferOffset = uint64_t(layer) * layerBytes;
                up.texture.aspect = VriImageAspect_Color; up.texture.baseLayer = layer; up.texture.layerNum = 1; up.texture.width = w; up.texture.height = h;
                c.CmdUploadBufferToTexture(cmd, dst, stg, &up);
            }
            VriTextureBarrierDesc tb2{}; tb2.texture = dst; tb2.before.access = VriAccess_CopyDestinationWrite; tb2.before.layout = VriLayout_CopyDestination; tb2.before.stages = VriPipelineStage_Transfer;
            tb2.after.access = VriAccess_ShaderResourceRead; tb2.after.layout = VriLayout_ShaderResource; tb2.after.stages = VriPipelineStage_FragmentShader; tb2.aspect = VriImageAspect_Color; tb2.layerNum = layers;
            VriBarrierGroupDesc g1{}; g1.textures = &tb2; g1.textureNum = 1; c.CmdBarrier(cmd, &g1);
        }

        void EndUpload()
        {
            c.EndCommandBuffer(cmd);
            VriFenceSubmitDesc sig{}; sig.fence = fence; sig.value = 1; // before the first frame (frameValue starts at 1 -> first frame signals 2)
            VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
            c.QueueSubmit(queue, &sub); c.Wait(fence, 1);
            for (VriBuffer* stg : uploadStaging) c.DestroyBuffer(stg);
            uploadStaging.clear();
        }

        // Enable headless auto-exit + capture. Call after the example's resources exist.
        // VRI_CAPTURE writes a BMP (desktop); on the web ?frames=N triggers a pixel self-check.
        void SetupCapture()
        {
            maxFrames = QueryMaxFrames();
            capturePath = std::getenv("VRI_CAPTURE");
            if (capturePath && maxFrames == 0) maxFrames = 40;
            bool want = capturePath != nullptr;
#if defined(__EMSCRIPTEN__)
            want = want || maxFrames != 0; // read the last frame back for the self-check
#endif
            if (want) { VriBufferDesc cb{}; cb.size = uint64_t(width) * height * 4; cb.usage = VriBufferUsage_TransferDst; cb.memoryLocation = VriMemoryLocation_HostReadback; c.CreateBuffer(dev, &cb, &captureBuf); }
        }

        void Finish()
        {
            running = false;
#if defined(__EMSCRIPTEN__)
            emscripten_cancel_main_loop();
            std::printf("[%s] %s: %llu frames presented\n", name, apiName, static_cast<unsigned long long>(frameValue - 1));
            std::fflush(stdout);
            // Close the tab when done (headless runs), deferred so emrun can POST the final
            // stdout (self-check + "frames presented") before the page goes away.
            emscripten_async_call([](void*) { EM_ASM({ if (typeof window !== 'undefined' && window.close) window.close(); }); emscripten_force_exit(0); }, nullptr, 1500);
#endif
        }

        void Frame()
        {
            if (onUpdate) onUpdate(frameValue); // CPU work (may yield on WebGPU) before acquire

            uint32_t index = 0;
            if (swap.AcquireNextTexture(swapchain, nullptr, 0, &index) == VriResult_OutOfDate) { swap.Resize(swapchain, width, height); return; }
            VriTexture* backbuffers[8] = {}; uint32_t count = 8; swap.GetSwapChainTextures(swapchain, backbuffers, &count);
            VriTexture* backbuffer = backbuffers[index];
            VriTextureViewDesc bvd{}; bvd.texture = backbuffer; bvd.viewType = VriTextureViewType_2D; bvd.format = VriFormat_Unknown; bvd.aspect = VriImageAspect_Color;
            VriDescriptor* bbView = nullptr; c.CreateTextureView(dev, &bvd, &bbView);

            c.BeginCommandBuffer(cmd);
            if (onPreRender) onPreRender(cmd);

            VriTextureBarrierDesc bgr[2]{};
            bgr[0].texture = backbuffer; bgr[0].before.layout = VriLayout_Undefined; bgr[0].before.stages = VriPipelineStage_None;
            bgr[0].after.access = VriAccess_ColorAttachmentWrite; bgr[0].after.layout = VriLayout_ColorAttachment; bgr[0].after.stages = VriPipelineStage_ColorAttachmentOutput; bgr[0].aspect = VriImageAspect_Color;
            uint32_t bn = 1;
            if (hasDepth)
            {
                bgr[1].texture = depth; bgr[1].before.layout = depthInit ? VriLayout_DepthStencilAttachment : VriLayout_Undefined; bgr[1].before.stages = VriPipelineStage_None;
                bgr[1].after.access = VriAccess_DepthStencilAttachmentWrite; bgr[1].after.layout = VriLayout_DepthStencilAttachment; bgr[1].after.stages = VriPipelineStage_EarlyFragmentTests; bgr[1].aspect = VriImageAspect_Depth;
                bn = 2;
            }
            VriBarrierGroupDesc g{}; g.textures = bgr; g.textureNum = bn; c.CmdBarrier(cmd, &g);
            depthInit = true;

            VriAttachmentDesc colorRT{}; colorRT.view = bbView; colorRT.loadOp = VriAttachmentLoadOp_Clear; colorRT.storeOp = VriAttachmentStoreOp_Store;
            colorRT.clearValue.color.f32[0] = clearColor[0]; colorRT.clearValue.color.f32[1] = clearColor[1]; colorRT.clearValue.color.f32[2] = clearColor[2]; colorRT.clearValue.color.f32[3] = clearColor[3];
            VriAttachmentDesc depthRT{}; depthRT.view = depthView; depthRT.loadOp = VriAttachmentLoadOp_Clear; depthRT.storeOp = VriAttachmentStoreOp_DontCare; depthRT.clearValue.depthStencil.depth = 1.0f;
            VriAttachmentsDesc att{}; att.colors = &colorRT; att.colorNum = 1; if (hasDepth) att.depth = &depthRT;
            att.renderArea.width = width; att.renderArea.height = height; att.layerNum = 1;
            c.CmdBeginRendering(cmd, &att);
            VriViewport vp{0, 0, float(width), float(height), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
            VriRect scis{0, 0, width, height}; c.CmdSetScissors(cmd, &scis, 1);
            if (onRecord) onRecord(cmd);
            c.CmdEndRendering(cmd);

            const bool capturing = captureBuf && maxFrames != 0 && frameValue >= maxFrames;
            if (capturing)
            {
                VriTextureBarrierDesc toSrc{}; toSrc.texture = backbuffer; toSrc.before.access = VriAccess_ColorAttachmentWrite; toSrc.before.layout = VriLayout_ColorAttachment; toSrc.before.stages = VriPipelineStage_ColorAttachmentOutput;
                toSrc.after.access = VriAccess_CopySourceRead; toSrc.after.layout = VriLayout_CopySource; toSrc.after.stages = VriPipelineStage_Transfer; toSrc.aspect = VriImageAspect_Color;
                VriBarrierGroupDesc gs{}; gs.textures = &toSrc; gs.textureNum = 1; c.CmdBarrier(cmd, &gs);
                VriBufferTextureCopyDesc rc{}; rc.texture.aspect = VriImageAspect_Color; rc.texture.layerNum = 1; c.CmdReadbackTextureToBuffer(cmd, captureBuf, backbuffer, &rc);
                VriTextureBarrierDesc toPresent{}; toPresent.texture = backbuffer; toPresent.before.access = VriAccess_CopySourceRead; toPresent.before.layout = VriLayout_CopySource; toPresent.before.stages = VriPipelineStage_Transfer;
                toPresent.after.layout = VriLayout_Present; toPresent.after.stages = VriPipelineStage_AllCommands; toPresent.aspect = VriImageAspect_Color;
                VriBarrierGroupDesc gp{}; gp.textures = &toPresent; gp.textureNum = 1; c.CmdBarrier(cmd, &gp);
            }
            else
            {
                VriTextureBarrierDesc toPresent{}; toPresent.texture = backbuffer; toPresent.before.access = VriAccess_ColorAttachmentWrite; toPresent.before.layout = VriLayout_ColorAttachment; toPresent.before.stages = VriPipelineStage_ColorAttachmentOutput;
                toPresent.after.layout = VriLayout_Present; toPresent.after.stages = VriPipelineStage_AllCommands; toPresent.aspect = VriImageAspect_Color;
                VriBarrierGroupDesc gp{}; gp.textures = &toPresent; gp.textureNum = 1; c.CmdBarrier(cmd, &gp);
            }
            c.EndCommandBuffer(cmd);

            VriFenceSubmitDesc sig{}; sig.fence = fence; sig.value = ++frameValue; sig.stages = VriPipelineStage_AllCommands;
            VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
            c.QueueSubmit(queue, &sub);
            c.Wait(fence, frameValue);
            swap.Present(swapchain, nullptr, 0);
            c.DestroyDescriptor(bbView);

            if (capturing)
            {
                const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(captureBuf, 0, uint64_t(width) * height * 4));
#if defined(__EMSCRIPTEN__)
                // Headless self-check: count center pixels that differ from a corner (the
                // cleared background) -> the geometry actually drew. Channel order agnostic.
                const uint8_t* cor = px + (size_t(8) * width + 8) * 4;
                const uint8_t* ctr = px + (size_t(height / 2) * width + width / 2) * 4;
                int drawn = 0; const int x0 = width / 4, x1 = 3 * width / 4, y0 = height / 4, y1 = 3 * height / 4;
                // Also count DISTINCT colors over a coarse grid (quantized to 5 bits/channel):
                // "differs from bg" alone can't tell a correct multi-color image from one that
                // is mostly black (black differs from the bg too) - distinct-color count can.
                bool seen[32768] = {}; int distinct = 0;
                for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x)
                {
                    const uint8_t* p = px + (size_t(y) * width + x) * 4;
                    int d0 = p[0] - cor[0], d1 = p[1] - cor[1], d2 = p[2] - cor[2]; if (d0 * d0 + d1 * d1 + d2 * d2 > 900) ++drawn;
                    int key = ((p[0] >> 3) << 10) | ((p[1] >> 3) << 5) | (p[2] >> 3);
                    if (!seen[key]) { seen[key] = true; ++distinct; }
                }
                std::printf("[%s] %s: %d/%d center px differ from bg; %d distinct colors; center=%u,%u,%u corner=%u,%u,%u\n", name, apiName, drawn, (x1 - x0) * (y1 - y0), distinct, ctr[0], ctr[1], ctr[2], cor[0], cor[1], cor[2]); std::fflush(stdout);
#else
                if (capturePath && WriteBmpBGRA(capturePath, px, width, height)) std::printf("[%s] %s: wrote %s\n", name, apiName, capturePath);
#endif
                c.UnmapBuffer(captureBuf);
            }
            if (maxFrames != 0 && frameValue - 1 >= maxFrames) Finish();
        }

        void Run()
        {
            std::printf("[%s] %s: rendering\n", name, apiName); std::fflush(stdout);
#if defined(__EMSCRIPTEN__)
            EM_ASM({ if (Module.vriReady) Module.vriReady(); }); // reveal the canvas (hide the loading overlay)
            emscripten_set_main_loop_arg([](void* p) { static_cast<ExampleApp*>(p)->Frame(); }, this, 0, true); // never returns
#else
            while (running)
            {
                SDL_Event e; while (SDL_PollEvent(&e)) if (e.type == SDL_EVENT_QUIT || e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) running = false;
                if (!running) break;
                Frame();
            }
            c.DeviceWaitIdle(dev);
            std::printf("[%s] %s: %llu frames presented\n", name, apiName, static_cast<unsigned long long>(frameValue - 1));
#endif
        }

        // Tear down the base objects (desktop only; the web main loop never returns). The
        // example destroys its own resources before calling this.
        void Shutdown()
        {
            if (fence) c.DestroyFence(fence);
            if (alloc) c.DestroyCommandAllocator(alloc);
            if (captureBuf) c.DestroyBuffer(captureBuf);
            if (depthView) c.DestroyDescriptor(depthView);
            if (depth) c.DestroyTexture(depth);
            if (swapchain) swap.DestroySwapChain(swapchain);
            if (dev) vriDestroyDevice(dev);
#if !defined(__EMSCRIPTEN__)
            if (window) SDL_DestroyWindow(window);
            SDL_Quit();
#endif
        }
    };
} // namespace vriex
