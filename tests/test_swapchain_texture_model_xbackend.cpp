// GetSwapChainTextureModel answers how a swapchain's textures behave, on hidden Win32 windows (never shown). Vulkan,
// D3D12 and GL answer Fixed, WebGPU answers Replaced. Each swapchain then renders frames the way its answer allows:
// - Fixed: views made once are used in every frame, and every acquire picks one of the textures GetSwapChainTextures
//   returned at the start;
// - Replaced: one texture, index 0 at every acquire, and a view made after each acquire and destroyed after Present.
// A wrong answer fails. A Fixed answer on WebGPU makes views before the first acquire, from a handle with no texture
// behind it, and wgpu-native aborts ("invalid texture"). A Replaced answer on a backend with several images gets
// acquire indices other than 0.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace
{
    struct Messages
    {
        int errors = 0;
    };

    void VRI_CALL CountErrors(void* userArg, VriMessageSeverity severity, const char* message)
    {
        if (severity != VriMessageSeverity_Error)
            return;
        ++static_cast<Messages*>(userArg)->errors;
        MESSAGE(std::string(message));
    }

    VriDescriptor* ColorView(const VriCoreInterface& c, VriDevice* dev, VriTexture* texture)
    {
        VriTextureViewDesc vd {};
        vd.texture          = texture;
        vd.viewType         = VriTextureViewType_2D;
        vd.aspect           = VriImageAspect_Color;
        VriDescriptor* view = nullptr;
        REQUIRE(c.CreateTextureView(dev, &vd, &view) == VriResult_Success);
        return view;
    }

    // Clears `texture` through `view` and presents it, waiting for the frame on the host.
    void RenderAndPresent(const VriCoreInterface&      c,
                          const VriSwapChainInterface& sw,
                          VriSwapChain*                swapChain,
                          VriQueue*                    queue,
                          VriCommandAllocator*         alloc,
                          VriCommandBuffer*            cmd,
                          VriFence*                    fence,
                          uint64_t                     value,
                          VriTexture*                  texture,
                          VriDescriptor*               view)
    {
        c.ResetCommandAllocator(alloc);
        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        VriTextureBarrierDesc tb {};
        tb.texture       = texture;
        tb.before.layout = VriLayout_Undefined;
        tb.after.access  = VriAccess_ColorAttachmentWrite;
        tb.after.layout  = VriLayout_ColorAttachment;
        tb.after.stages  = VriPipelineStage_AllCommands;
        tb.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc g {};
        g.textures   = &tb;
        g.textureNum = 1;
        c.CmdBarrier(cmd, &g);
        VriAttachmentDesc rt {};
        rt.view                    = view;
        rt.loadOp                  = VriAttachmentLoadOp_Clear;
        rt.storeOp                 = VriAttachmentStoreOp_Store;
        rt.clearValue.color.f32[1] = 1.0f;
        rt.clearValue.color.f32[3] = 1.0f;
        VriAttachmentsDesc att {};
        att.colors            = &rt;
        att.colorNum          = 1;
        att.renderArea.width  = 64;
        att.renderArea.height = 32;
        att.layerNum          = 1;
        c.CmdBeginRendering(cmd, &att);
        c.CmdEndRendering(cmd);
        tb.before       = tb.after;
        tb.after.access = VriAccess_None;
        tb.after.layout = VriLayout_Present;
        tb.after.stages = VriPipelineStage_AllCommands;
        c.CmdBarrier(cmd, &g);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
        VriFenceSubmitDesc signal {};
        signal.fence = fence;
        signal.value = value;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        REQUIRE(c.QueueSubmit(queue, &submit) == VriResult_Success);
        CHECK(sw.Present(swapChain, fence, value) == VriResult_Success);
        c.Wait(fence, value);
    }

    void Check(VriGraphicsAPI api, const std::string& name, VriSwapChainTextureModel expected)
    {
        CAPTURE(name);
        Messages              messages;
        VriCallbackInterface  callbacks = {&messages, CountErrors};
        VriDeviceCreationDesc dc        = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI                  = api;
        dc.enableValidation             = VRI_TRUE;
        dc.callbackInterface            = &callbacks;
        VriDevice* dev                  = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE(name << " unavailable - skipping");
            return;
        }
        VriCoreInterface      c {};
        VriSwapChainInterface sw {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_SWAPCHAIN, sizeof(sw), &sw) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        HWND hwnd = CreateWindowExA(
            0, "STATIC", "vri-test", WS_POPUP, 0, 0, 64, 32, nullptr, nullptr, GetModuleHandleA(nullptr), nullptr);
        REQUIRE(hwnd != nullptr);
        VriSwapChainDesc sd {};
        sd.window.type                   = VriWindowSystem_Win32;
        sd.window.handle.win32.hwnd      = hwnd;
        sd.window.handle.win32.hinstance = GetModuleHandleA(nullptr);
        sd.queue                         = queue;
        sd.format                        = VriFormat_BGRA8_UNORM;
        sd.width                         = 64;
        sd.height                        = 32;
        sd.textureNum                    = 3;
        VriSwapChain* swapChain          = nullptr;
        REQUIRE(sw.CreateSwapChain(dev, &sd, &swapChain) == VriResult_Success);

        const VriSwapChainTextureModel model = sw.GetSwapChainTextureModel(swapChain);
        CHECK(model == expected);

        uint32_t count = 0;
        REQUIRE(sw.GetSwapChainTextures(swapChain, nullptr, &count) == VriResult_Success);
        REQUIRE(count >= 1);
        std::vector<VriTexture*> textures(count);
        REQUIRE(sw.GetSwapChainTextures(swapChain, textures.data(), &count) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        constexpr uint64_t kFrames = 6;
        if (model == VriSwapChainTextureModel_Fixed)
        {
            std::vector<VriDescriptor*> views;
            for (VriTexture* t : textures)
                views.push_back(ColorView(c, dev, t));
            for (uint64_t frame = 1; frame <= kFrames; ++frame)
            {
                CAPTURE(frame);
                uint32_t index = 0;
                REQUIRE(sw.AcquireNextTexture(swapChain, nullptr, 0, &index) == VriResult_Success);
                REQUIRE(index < count);
                std::vector<VriTexture*> now(count);
                uint32_t                 nowCount = count;
                REQUIRE(sw.GetSwapChainTextures(swapChain, now.data(), &nowCount) == VriResult_Success);
                CHECK(nowCount == count);
                CHECK(now == textures);
                RenderAndPresent(c, sw, swapChain, queue, alloc, cmd, fence, frame, textures[index], views[index]);
            }
            REQUIRE(c.DeviceWaitIdle(dev) == VriResult_Success);
            for (VriDescriptor* v : views)
                c.DestroyDescriptor(v);
        }
        else
        {
            CHECK(count == 1);
            for (uint64_t frame = 1; frame <= kFrames; ++frame)
            {
                CAPTURE(frame);
                uint32_t index = 1;
                REQUIRE(sw.AcquireNextTexture(swapChain, nullptr, 0, &index) == VriResult_Success);
                CHECK(index == 0);
                VriDescriptor* view = ColorView(c, dev, textures[0]);
                RenderAndPresent(c, sw, swapChain, queue, alloc, cmd, fence, frame, textures[0], view);
                c.DestroyDescriptor(view);
            }
            REQUIRE(c.DeviceWaitIdle(dev) == VriResult_Success);
        }

        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        sw.DestroySwapChain(swapChain);
        DestroyWindow(hwnd);
        vriDestroyDevice(dev);
        CHECK(messages.errors == 0);
    }
} // namespace

TEST_CASE("Swapchain texture model: the query matches how the textures behave")
{
    Check(VriGraphicsAPI_Vulkan, "Vulkan", VriSwapChainTextureModel_Fixed);
    Check(VriGraphicsAPI_D3D12, "D3D12", VriSwapChainTextureModel_Fixed);
    Check(VriGraphicsAPI_OpenGL, "OpenGL", VriSwapChainTextureModel_Fixed);
    Check(VriGraphicsAPI_WebGPU, "WebGPU", VriSwapChainTextureModel_Replaced);
}
