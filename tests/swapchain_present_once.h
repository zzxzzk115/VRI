// Presents one image of a swapchain that has not presented yet. A test presents before destroying a swapchain
// because NVIDIA's Vulkan driver, in a process where the D3D12 debug layer is on, has the layer end the process
// when a never-presented swapchain is destroyed (DETAIL.md, "Known swapchain limitations"); earlier D3D12 tests
// turn that layer on for the whole test process.
#pragma once

#include <doctest/doctest.h>

#include <vri/vri.h>

namespace vri::test
{
    inline void PresentOnce(VriDevice* dev, VriCoreInterface& c, VriSwapChainInterface& sw, VriSwapChain* swapChain)
    {
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);
        uint32_t index = 0;
        REQUIRE(sw.AcquireNextTexture(swapChain, nullptr, 0, &index) == VriResult_Success);
        VriTexture* textures[8] = {};
        uint32_t    count       = 8;
        REQUIRE(sw.GetSwapChainTextures(swapChain, textures, &count) == VriResult_Success);
        REQUIRE(index < count);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);
        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        VriTextureBarrierDesc tb {};
        tb.texture       = textures[index];
        tb.before.layout = VriLayout_Undefined;
        tb.after.layout  = VriLayout_Present;
        tb.aspect        = VriImageAspect_Color;
        VriBarrierGroupDesc g {};
        g.textures   = &tb;
        g.textureNum = 1;
        c.CmdBarrier(cmd, &g);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
        VriFenceSubmitDesc signal {};
        signal.fence = fence;
        signal.value = 1;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        REQUIRE(c.QueueSubmit(queue, &submit) == VriResult_Success);
        CHECK(sw.Present(swapChain, fence, 1) == VriResult_Success);
        REQUIRE(c.DeviceWaitIdle(dev) == VriResult_Success);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
    }
} // namespace vri::test
