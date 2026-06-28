// example-cuda-interop - end-to-end demonstration of VRI <-> CUDA interop.
//
// VRI and CUDA share the SAME GPU memory and the SAME timeline, with zero copies between
// them and no CUDA dependency inside VRI:
//
//   1. VRI creates an exportable buffer + an exportable timeline fence (VRI_INTERFACE_EXTERNAL).
//   2. VRI uploads input  x[i] = i  into the shared buffer and signals the fence to 1.
//   3. CUDA imports the buffer + fence by OS handle, waits for fence==1, runs a kernel
//      ( x[i] = x[i]*2 + 1 ) on the shared memory, and signals the fence to 2.
//   4. VRI waits for fence==2, reads the buffer back, and verifies CUDA's transform.
//
// This is headless (console) -- interop is a compute/memory concern, not a rendering one.
// Build is opt-in: xmake f --vri_build_cuda_interop=y  (needs the CUDA Toolkit; xmake's
// `cuda` package can fetch a system install or download one).

#include <vri/vri.h>

#include "cuda_interop.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <windows.h>
namespace
{
    // Vulkan exports OPAQUE_WIN32 on Windows; the opaque memory + semaphore handle type.
    constexpr VriExternalHandleType kVkHandleType = VriExternalHandleType_OpaqueWin32;
    void                            CloseExported(void* h)
    {
        if (h)
            CloseHandle(static_cast<HANDLE>(h));
    }
} // namespace
#else
#include <unistd.h>
namespace
{
    constexpr VriExternalHandleType kVkHandleType = VriExternalHandleType_OpaqueFd;
    void                            CloseExported(void* h) { ::close(static_cast<int>(reinterpret_cast<intptr_t>(h))); }
} // namespace
#endif

namespace
{
    constexpr uint32_t kCount = 1024;

    // Tiny RAII for a VRI device.
    struct App
    {
        VriDevice*       device = nullptr;
        VriCoreInterface core {};
        ~App()
        {
            if (device)
                vriDestroyDevice(device);
        }
    };

    VriBuffer*
    MakeBuffer(const VriCoreInterface& c, VriDevice* d, uint64_t size, VriBufferUsageFlags usage, VriMemoryLocation loc)
    {
        VriBufferDesc bd {};
        bd.size           = size;
        bd.usage          = usage;
        bd.memoryLocation = loc;
        VriBuffer* b      = nullptr;
        if (c.CreateBuffer(d, &bd, &b) != VriResult_Success)
            return nullptr;
        return b;
    }
} // namespace

int main()
{
    if (!cudaInteropSupported())
    {
        std::printf("CUDA is not available in this build/runtime - cannot run the interop demo.\n");
        return 0; // not a failure: nothing to demonstrate without CUDA
    }

    // ---- 1. VRI device with external-memory export enabled ----
    // CUDA interop is wired on Vulkan and D3D12; pick with VRI_API=vulkan|d3d12 (default vulkan).
    const char* apiEnv   = std::getenv("VRI_API");
    const bool  useD3D12 = apiEnv && (std::strcmp(apiEnv, "d3d12") == 0 || std::strcmp(apiEnv, "D3D12") == 0);

    // The handle flavor + CUDA import kind differ per backend: Vulkan exports one OPAQUE type
    // for both memory and fences; D3D12 exports a shared resource for memory and a shared fence.
    const VriExternalHandleType memHandleType   = useD3D12 ? VriExternalHandleType_D3D12Resource : kVkHandleType;
    const VriExternalHandleType fenceHandleType = useD3D12 ? VriExternalHandleType_D3D12Fence : kVkHandleType;
    const int                   cudaHandleKind  = useD3D12 ? CUDA_INTEROP_KIND_D3D12 : CUDA_INTEROP_KIND_VULKAN;

    App                   app;
    VriDeviceCreationDesc desc {};
    desc.graphicsAPI      = useD3D12 ? VriGraphicsAPI_D3D12 : VriGraphicsAPI_Vulkan;
    desc.enableValidation = VRI_TRUE;
    desc.bestEffort       = VRI_TRUE;
    desc.enabledFeatures  = VriFeature_ExternalMemory;
    if (vriCreateDevice(&desc, &app.device) != VriResult_Success)
    {
        std::printf("Failed to create a %s device.\n", useD3D12 ? "D3D12" : "Vulkan");
        return 1;
    }
    if (vriGetInterface(app.device, VRI_INTERFACE_CORE, sizeof(app.core), &app.core) != VriResult_Success)
        return 1;
    const VriCoreInterface& c   = app.core;
    VriDevice*              dev = app.device;

    const VriDeviceDesc* dd = c.GetDeviceDesc(dev);
    std::printf("VRI device: %s\n", dd->adapter.name);
    if (dd->hasExternalMemory == VRI_FALSE)
    {
        std::printf("This adapter does not support external memory export - cannot demo CUDA interop.\n");
        return 0;
    }
    cudaInteropPrintDevice(); // for the demo it should be the same physical GPU

    VriExternalInterface ext {};
    if (vriGetInterface(dev, VRI_INTERFACE_EXTERNAL, sizeof(ext), &ext) != VriResult_Success)
        return 1;

    VriQueue* queue = nullptr;
    if (c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) != VriResult_Success)
        return 1;

    // ---- 2. shared (exportable) buffer + timeline fence ----
    const uint64_t bufSize = static_cast<uint64_t>(kCount) * sizeof(uint32_t);
    VriBuffer*     shared  = nullptr;
    VriBufferDesc  sharedDesc {};
    sharedDesc.size           = bufSize;
    sharedDesc.usage          = VriBufferUsage_StorageBuffer | VriBufferUsage_TransferSrc | VriBufferUsage_TransferDst;
    sharedDesc.memoryLocation = VriMemoryLocation_Device;
    if (ext.CreateExportableBuffer(dev, &sharedDesc, memHandleType, &shared) != VriResult_Success)
    {
        std::printf("CreateExportableBuffer failed.\n");
        return 1;
    }
    VriFence* fenceShared = nullptr; // the VRI <-> CUDA timeline
    if (ext.CreateExportableFence(dev, 0, fenceHandleType, &fenceShared) != VriResult_Success)
    {
        std::printf("CreateExportableFence failed.\n");
        return 1;
    }
    VriFence* fenceLocal = nullptr; // host waits this for the readback
    if (c.CreateFence(dev, 0, &fenceLocal) != VriResult_Success)
        return 1;

    // staging + readback (host-visible)
    VriBuffer* staging  = MakeBuffer(c, dev, bufSize, VriBufferUsage_TransferSrc, VriMemoryLocation_HostUpload);
    VriBuffer* readback = MakeBuffer(c, dev, bufSize, VriBufferUsage_TransferDst, VriMemoryLocation_HostReadback);
    if (!staging || !readback)
        return 1;

    // input: x[i] = i
    {
        auto* p = static_cast<uint32_t*>(c.MapBuffer(staging, 0, bufSize));
        for (uint32_t i = 0; i < kCount; ++i)
            p[i] = i;
        c.UnmapBuffer(staging);
    }

    // Two independent command buffers (upload, then readback). They must not share one
    // buffer: the upload stays "pending" until CUDA consumes it, so reusing/resetting it for
    // the readback would touch an in-flight command buffer.
    VriCommandAllocator* alloc       = nullptr;
    VriCommandBuffer*    cmdUpload   = nullptr;
    VriCommandBuffer*    cmdReadback = nullptr;
    if (c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) != VriResult_Success ||
        c.CreateCommandBuffer(alloc, &cmdUpload) != VriResult_Success ||
        c.CreateCommandBuffer(alloc, &cmdReadback) != VriResult_Success)
        return 1;

    // ---- 3. upload input -> shared, signal fence = 1 ----
    VriCommandBuffer* cmd = cmdUpload;
    c.BeginCommandBuffer(cmd);
    {
        VriBufferCopyDesc region {};
        region.size = bufSize;
        c.CmdCopyBuffer(cmd, shared, staging, &region);
        // make the copy write available before CUDA consumes the buffer
        VriBufferBarrierDesc bb {};
        bb.buffer        = shared;
        bb.before.access = VriAccess_CopyDestinationWrite;
        bb.before.stages = VriPipelineStage_Transfer;
        bb.after.access  = VriAccess_ShaderResourceStorageRead | VriAccess_ShaderResourceStorageWrite;
        bb.after.stages  = VriPipelineStage_AllCommands;
        VriBarrierGroupDesc g {};
        g.buffers   = &bb;
        g.bufferNum = 1;
        c.CmdBarrier(cmd, &g);
    }
    c.EndCommandBuffer(cmd);
    {
        VriFenceSubmitDesc signal {};
        signal.fence  = fenceShared;
        signal.value  = 1;
        signal.stages = VriPipelineStage_AllCommands;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        c.QueueSubmit(queue, &submit);
    }

    // ---- export the OS handles and hand them to CUDA ----
    VriExternalMemoryInfo memInfo {};
    if (ext.GetBufferMemoryHandle(dev, shared, memHandleType, &memInfo) != VriResult_Success)
    {
        std::printf("GetBufferMemoryHandle failed.\n");
        return 1;
    }
    void* semHandle = nullptr;
    if (ext.GetFenceHandle(dev, fenceShared, fenceHandleType, &semHandle) != VriResult_Success)
    {
        std::printf("GetFenceHandle failed.\n");
        return 1;
    }

    std::printf("[%s] exported buffer handle (%llu-byte allocation) + fence handle; handing to CUDA...\n",
                useD3D12 ? "D3D12" : "Vulkan",
                static_cast<unsigned long long>(memInfo.size));

    // CUDA: wait fence==1 -> kernel x[i]=x[i]*2+1 -> signal fence=2
    const int cudaRc =
        cudaInteropRun(cudaHandleKind, memInfo.handle, memInfo.size, semHandle, kCount, /*wait*/ 1, /*signal*/ 2);

    // the caller owns the exported handles (CUDA duplicated what it needed)
    CloseExported(memInfo.handle);
    CloseExported(semHandle);

    if (cudaRc != 0)
    {
        std::printf("CUDA interop step failed (rc=%d).\n", cudaRc);
        return 1;
    }

    // ---- 4. read the shared buffer back (wait fence==2) and verify ----
    cmd = cmdReadback;
    c.BeginCommandBuffer(cmd);
    {
        VriBufferCopyDesc region {};
        region.size = bufSize;
        c.CmdCopyBuffer(cmd, readback, shared, &region);
    }
    c.EndCommandBuffer(cmd);
    {
        VriFenceSubmitDesc wait {};
        wait.fence  = fenceShared; // wait for CUDA's signal
        wait.value  = 2;
        wait.stages = VriPipelineStage_AllCommands;
        VriFenceSubmitDesc signal {};
        signal.fence  = fenceLocal;
        signal.value  = 1;
        signal.stages = VriPipelineStage_AllCommands;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.waitFences       = &wait;
        submit.waitFenceNum     = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        c.QueueSubmit(queue, &submit);
    }
    c.Wait(fenceLocal, 1);

    int             bad = 0;
    const uint32_t* out = static_cast<const uint32_t*>(c.MapBuffer(readback, 0, bufSize));
    for (uint32_t i = 0; i < kCount; ++i)
        if (out[i] != i * 2u + 1u) // the transform CUDA applied
            ++bad;
    std::printf("Verify x[i] == i*2+1 after CUDA: samples [0]=%u [1]=%u [2]=%u [%u]=%u\n",
                out[0],
                out[1],
                out[2],
                kCount - 1,
                out[kCount - 1]);
    c.UnmapBuffer(readback);

    c.DeviceWaitIdle(dev);
    c.DestroyCommandAllocator(alloc);
    c.DestroyBuffer(readback);
    c.DestroyBuffer(staging);
    c.DestroyFence(fenceLocal);
    c.DestroyFence(fenceShared);
    c.DestroyBuffer(shared);

    if (bad == 0)
        std::printf("VRI <-> CUDA interop: PASS (CUDA transformed all %u shared elements)\n", kCount);
    else
        std::printf("VRI <-> CUDA interop: FAIL (%d/%u elements wrong)\n", bad, kCount);
    return bad == 0 ? 0 : 1;
}
