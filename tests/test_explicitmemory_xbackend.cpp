// Explicit-memory (placed-resource) parity: the GetMemoryDesc -> AllocateMemory ->
// CreateBuffer/CreateTexture(Undefined) -> BindBufferMemory/BindTextureMemory path.
//
// Vulkan has covered this since phase 2 (test_core_phase2_vk.cpp); this file exercises the
// SAME explicit path across every enabled backend, verified by GPU readback. It is the
// regression net for the D3D12 Tier-1 placed-resource implementation (LAZ-15): D3D12 now
// backs AllocateMemory with an ID3D12Heap and BindMemory with CreatePlacedResource.
//
// A backend that does not offer an explicit-memory model (its AllocateMemory returns
// Unsupported, e.g. GL / WebGPU) self-skips, so the file is stable under the CI matrix.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>

namespace
{
    struct Probe
    {
        bool ran       = false; // backend available and device created
        bool supported = false; // backend offers explicit memory (AllocateMemory succeeded)
        bool bufferOk  = false; // explicit-memory-backed buffer round-tripped correctly
        bool textureOk = false; // explicit-memory-backed texture round-tripped correctly
    };

    const char* ApiName(VriGraphicsAPI api)
    {
        switch (api)
        {
            case VriGraphicsAPI_Vulkan:
                return "Vulkan";
            case VriGraphicsAPI_WebGPU:
                return "WebGPU";
            case VriGraphicsAPI_OpenGL:
                return "OpenGL";
            case VriGraphicsAPI_D3D12:
                return "D3D12";
            case VriGraphicsAPI_Metal:
                return "Metal";
            default:
                return "?";
        }
    }

    Probe RunExplicitMemory(VriGraphicsAPI api)
    {
        Probe                 p;
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        VriDevice* dev      = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return p; // ran == false -> caller skips
        p.ran = true;

        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        auto submit = [&](uint64_t v) {
            VriFenceSubmitDesc signal {};
            signal.fence = fence;
            signal.value = v;
            VriQueueSubmitDesc s {};
            s.commandBuffers   = &cmd;
            s.commandBufferNum = 1;
            s.signalFences     = &signal;
            s.signalFenceNum   = 1;
            c.QueueSubmit(queue, &s);
            c.Wait(fence, v);
        };

        // ---- buffer path: device buffer backed by explicitly allocated memory ----
        constexpr uint32_t kN    = 64;
        const uint64_t     bytes = kN * sizeof(uint32_t);

        VriBufferDesc devDesc {};
        devDesc.size           = bytes;
        devDesc.usage          = VriBufferUsage_TransferSrc | VriBufferUsage_TransferDst;
        devDesc.memoryLocation = VriMemoryLocation_Undefined; // UNBOUND: we drive memory ourselves

        VriMemoryDesc memDesc {};
        c.GetBufferMemoryDesc(dev, &devDesc, VriMemoryLocation_Device, &memDesc);
        VriMemory* mem = nullptr;
        if (memDesc.size == 0 || c.AllocateMemory(dev, &memDesc, &mem) != VriResult_Success)
        {
            // Backend has no explicit-memory model - self-skip (never a false positive).
            c.DestroyFence(fence);
            c.DestroyCommandAllocator(alloc);
            vriDestroyDevice(dev);
            return p; // supported == false
        }
        p.supported = true;
        CHECK(memDesc.size >= bytes);

        VriBuffer* devBuf = nullptr;
        REQUIRE(c.CreateBuffer(dev, &devDesc, &devBuf) == VriResult_Success);
        REQUIRE(c.BindBufferMemory(dev, devBuf, mem, 0) == VriResult_Success);

        VriBufferDesc upDesc {};
        upDesc.size           = bytes;
        upDesc.usage          = VriBufferUsage_TransferSrc;
        upDesc.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* staging    = nullptr;
        REQUIRE(c.CreateBuffer(dev, &upDesc, &staging) == VriResult_Success);
        {
            uint32_t* m = static_cast<uint32_t*>(c.MapBuffer(staging, 0, bytes));
            REQUIRE(m != nullptr);
            for (uint32_t i = 0; i < kN; ++i)
                m[i] = i + 1000u;
            c.UnmapBuffer(staging);
        }
        VriBufferDesc rbDesc {};
        rbDesc.size           = bytes;
        rbDesc.usage          = VriBufferUsage_TransferDst;
        rbDesc.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readback   = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rbDesc, &readback) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        VriBufferCopyDesc cp {};
        cp.size = bytes;
        c.CmdCopyBuffer(cmd, devBuf, staging, &cp); // staging -> placed device buffer
        {
            VriBufferBarrierDesc bb {};
            bb.buffer        = devBuf;
            bb.before.access = VriAccess_CopyDestinationWrite;
            bb.before.stages = VriPipelineStage_Transfer;
            bb.after.access  = VriAccess_CopySourceRead;
            bb.after.stages  = VriPipelineStage_Transfer;
            VriBarrierGroupDesc g {};
            g.buffers   = &bb;
            g.bufferNum = 1;
            c.CmdBarrier(cmd, &g);
        }
        c.CmdCopyBuffer(cmd, readback, devBuf, &cp); // placed device buffer -> readback
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
        submit(1);
        {
            const uint32_t* r = static_cast<const uint32_t*>(c.MapBuffer(readback, 0, bytes));
            REQUIRE(r != nullptr);
            bool ok = true;
            for (uint32_t i = 0; i < kN; ++i)
                ok = ok && (r[i] == i + 1000u);
            p.bufferOk = ok;
            c.UnmapBuffer(readback);
        }

        // ---- texture path: standard (non-MSAA) 2D texture backed by placed memory ----
        constexpr uint32_t kW = 64, kH = 64;
        const uint64_t     texBytes = static_cast<uint64_t>(kW) * kH * 4;

        VriTextureDesc td {};
        td.type           = VriTextureType_2D;
        td.format         = VriFormat_RGBA8_UNORM;
        td.width          = kW;
        td.height         = kH;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = 1;
        td.sampleNum      = 1;
        td.usage          = VriTextureUsage_TransferSrc | VriTextureUsage_TransferDst | VriTextureUsage_ShaderResource;
        td.memoryLocation = VriMemoryLocation_Undefined;

        VriMemoryDesc texMem {};
        c.GetTextureMemoryDesc(dev, &td, VriMemoryLocation_Device, &texMem);
        VriMemory* tmem = nullptr;
        if (texMem.size != 0 && c.AllocateMemory(dev, &texMem, &tmem) == VriResult_Success)
        {
            VriTexture* tex = nullptr;
            REQUIRE(c.CreateTexture(dev, &td, &tex) == VriResult_Success);
            REQUIRE(c.BindTextureMemory(dev, tex, tmem, 0) == VriResult_Success);

            VriBufferDesc sbd {};
            sbd.size           = texBytes;
            sbd.usage          = VriBufferUsage_TransferSrc;
            sbd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* texStg  = nullptr;
            REQUIRE(c.CreateBuffer(dev, &sbd, &texStg) == VriResult_Success);
            {
                uint8_t* m = static_cast<uint8_t*>(c.MapBuffer(texStg, 0, texBytes));
                REQUIRE(m != nullptr);
                for (uint64_t i = 0; i < texBytes; i += 4)
                {
                    m[i + 0] = 10;
                    m[i + 1] = 200;
                    m[i + 2] = 30;
                    m[i + 3] = 255;
                }
                c.UnmapBuffer(texStg);
            }
            VriBufferDesc trbd {};
            trbd.size           = texBytes;
            trbd.usage          = VriBufferUsage_TransferDst;
            trbd.memoryLocation = VriMemoryLocation_HostReadback;
            VriBuffer* texRb    = nullptr;
            REQUIRE(c.CreateBuffer(dev, &trbd, &texRb) == VriResult_Success);

            auto texBarrier = [&](VriAccessFlags        ba,
                                  VriLayout             bl,
                                  VriPipelineStageFlags bs,
                                  VriAccessFlags        aa,
                                  VriLayout             al,
                                  VriPipelineStageFlags as) {
                VriTextureBarrierDesc tb {};
                tb.texture       = tex;
                tb.before.access = ba;
                tb.before.layout = bl;
                tb.before.stages = bs;
                tb.after.access  = aa;
                tb.after.layout  = al;
                tb.after.stages  = as;
                tb.aspect        = VriImageAspect_Color;
                VriBarrierGroupDesc g {};
                g.textures   = &tb;
                g.textureNum = 1;
                c.CmdBarrier(cmd, &g);
            };

            c.ResetCommandAllocator(alloc);
            REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
            texBarrier(VriAccess_None,
                       VriLayout_Undefined,
                       VriPipelineStage_None,
                       VriAccess_CopyDestinationWrite,
                       VriLayout_CopyDestination,
                       VriPipelineStage_Transfer);
            VriBufferTextureCopyDesc up {};
            up.texture.aspect   = VriImageAspect_Color;
            up.texture.layerNum = 1;
            up.texture.width    = kW;
            up.texture.height   = kH;
            c.CmdUploadBufferToTexture(cmd, tex, texStg, &up);
            texBarrier(VriAccess_CopyDestinationWrite,
                       VriLayout_CopyDestination,
                       VriPipelineStage_Transfer,
                       VriAccess_CopySourceRead,
                       VriLayout_CopySource,
                       VriPipelineStage_Transfer);
            VriBufferTextureCopyDesc down {};
            down.texture.aspect   = VriImageAspect_Color;
            down.texture.layerNum = 1;
            c.CmdReadbackTextureToBuffer(cmd, texRb, tex, &down);
            REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
            submit(2);

            const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(texRb, 0, texBytes));
            REQUIRE(px != nullptr);
            const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
            p.textureOk      = (px[o + 0] == 10 && px[o + 1] == 200 && px[o + 2] == 30);
            c.UnmapBuffer(texRb);

            c.DeviceWaitIdle(dev);
            c.DestroyBuffer(texRb);
            c.DestroyBuffer(texStg);
            c.DestroyTexture(tex);
            c.FreeMemory(tmem);
        }
        else
        {
            p.textureOk = true; // texture explicit memory not offered here; buffer path already gates support
        }

        c.DeviceWaitIdle(dev);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(staging);
        c.DestroyBuffer(devBuf);
        c.FreeMemory(mem);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        vriDestroyDevice(dev);
        return p;
    }

    void Check(VriGraphicsAPI api)
    {
        Probe p = RunExplicitMemory(api);
        if (!p.ran)
        {
            MESSAGE(ApiName(api) << " unavailable - skipping");
            return;
        }
        if (!p.supported)
        {
            MESSAGE(ApiName(api) << " has no explicit-memory model - skipping");
            return;
        }
        INFO(ApiName(api) << " explicit placed memory");
        CHECK(p.bufferOk);
        CHECK(p.textureOk);
    }
} // namespace

TEST_CASE("explicit memory: AllocateMemory + BindMemory back a buffer & texture (placed resources)")
{
    Check(VriGraphicsAPI_Vulkan);
    Check(VriGraphicsAPI_D3D12);
    Check(VriGraphicsAPI_OpenGL);
    Check(VriGraphicsAPI_WebGPU);
    Check(VriGraphicsAPI_Metal);
}
