// example-profiler - a minimal GPU profiler built on VRI's query interface (ext/vri_ext_query.h).
//
// A tiny GpuProfiler wraps a timestamp query pool + a readback buffer: Stamp() records the GPU
// clock at a boundary during command recording, and after the submit's fence Report() turns the
// raw ticks into per-section milliseconds. The demo times a few device->device copies of growing
// size, so the printed times visibly scale with the work. Headless (console) and backend-agnostic
// (VriGraphicsAPI_Auto picks Vulkan/D3D12/...); needs no extra SDK.

#include <vri/vri.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace
{
    const char* ApiName(VriGraphicsAPI api)
    {
        switch (api)
        {
            case VriGraphicsAPI_Vulkan:
                return "Vulkan";
            case VriGraphicsAPI_D3D12:
                return "D3D12";
            case VriGraphicsAPI_WebGPU:
                return "WebGPU";
            case VriGraphicsAPI_OpenGL:
                return "OpenGL";
            default:
                return "?";
        }
    }

    // A bounded timestamp profiler: Stamp() at each boundary, Report() prints section deltas.
    struct GpuProfiler
    {
        const VriCoreInterface*  c = nullptr;
        VriQueryInterface        q {};
        VriDevice*               dev      = nullptr;
        VriQueryPool*            pool     = nullptr;
        VriBuffer*               readback = nullptr;
        uint32_t                 cap      = 0;
        uint32_t                 count    = 0;
        double                   periodNs = 0.0;
        std::vector<const char*> labels; // labels[i] names the work that ended at stamp i

        bool Init(VriDevice* d, const VriCoreInterface* core, const VriQueryInterface& query, uint32_t capacity)
        {
            dev      = d;
            c        = core;
            q        = query;
            cap      = capacity;
            periodNs = core->GetDeviceDesc(d)->timestampPeriodNanoseconds;

            VriQueryPoolDesc qpd {};
            qpd.type       = VriQueryType_Timestamp;
            qpd.queryCount = capacity;
            if (q.CreateQueryPool(d, &qpd, &pool) != VriResult_Success)
                return false;

            VriBufferDesc bd {};
            bd.size           = static_cast<uint64_t>(capacity) * q.GetQuerySize(pool);
            bd.usage          = VriBufferUsage_TransferDst;
            bd.memoryLocation = VriMemoryLocation_HostReadback;
            return c->CreateBuffer(d, &bd, &readback) == VriResult_Success;
        }

        void Begin(VriCommandBuffer* cmd)
        {
            count = 0;
            labels.clear();
            q.CmdResetQueries(cmd, pool, 0, cap);
        }
        // Record the GPU clock; `label` names the section that just finished (ignored for the first).
        void Stamp(VriCommandBuffer* cmd, const char* label)
        {
            if (count >= cap)
                return;
            q.CmdWriteTimestamp(cmd, pool, count);
            labels.push_back(label);
            ++count;
        }
        void End(VriCommandBuffer* cmd) { q.CmdCopyQueries(cmd, pool, 0, count, readback, 0); }

        void Report()
        {
            const uint64_t* ticks = static_cast<const uint64_t*>(
                c->MapBuffer(readback, 0, static_cast<uint64_t>(count) * sizeof(uint64_t)));
            std::printf("\n  GPU timings\n  -----------\n");
            for (uint32_t i = 1; i < count; ++i)
            {
                const double ms = static_cast<double>(ticks[i] - ticks[i - 1]) * periodNs / 1.0e6;
                std::printf("  %-14s %8.3f ms\n", labels[i], ms);
            }
            c->UnmapBuffer(readback);
        }

        void Destroy()
        {
            if (readback)
                c->DestroyBuffer(readback);
            if (pool)
                q.DestroyQueryPool(pool);
        }
    };
} // namespace

int main()
{
    // ---- device: Auto-select, or force a backend with VRI_API=vulkan|d3d12 ----
    const char*           apiEnv = std::getenv("VRI_API");
    VriDeviceCreationDesc desc {};
    desc.graphicsAPI = VriGraphicsAPI_Auto;
    if (apiEnv && std::strcmp(apiEnv, "vulkan") == 0)
        desc.graphicsAPI = VriGraphicsAPI_Vulkan;
    else if (apiEnv && (std::strcmp(apiEnv, "d3d12") == 0 || std::strcmp(apiEnv, "D3D12") == 0))
        desc.graphicsAPI = VriGraphicsAPI_D3D12;
    desc.enableValidation = VRI_TRUE;
    desc.bestEffort       = VRI_TRUE;
    VriDevice* dev        = nullptr;
    if (vriCreateDevice(&desc, &dev) != VriResult_Success)
    {
        std::printf("Failed to create a device.\n");
        return 1;
    }
    VriCoreInterface c {};
    if (vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) != VriResult_Success)
        return 1;

    const VriDeviceDesc* dd = c.GetDeviceDesc(dev);
    std::printf("device: %s [%s]\n", dd->adapter.name, ApiName(dd->graphicsAPI));

    VriQueryInterface q {};
    const bool        hasQuery = vriGetInterface(dev, VRI_INTERFACE_QUERY, sizeof(q), &q) == VriResult_Success;
    if (!hasQuery || dd->hasTimestampQueries == VRI_FALSE)
    {
        std::printf("This backend has no timestamp queries - profiler unavailable here.\n");
        vriDestroyDevice(dev);
        return 0;
    }

    VriQueue* queue = nullptr;
    c.GetQueue(dev, VriQueueType_Graphics, 0, &queue);

    // ---- work to profile: device->device copies of growing size ----
    constexpr uint64_t kMiB     = 1u << 20;
    const uint64_t     sizes[3] = {16 * kMiB, 64 * kMiB, 128 * kMiB};
    const char*        names[3] = {"copy 16 MiB", "copy 64 MiB", "copy 128 MiB"};
    const uint64_t     biggest  = sizes[2];

    VriBuffer*    src = nullptr;
    VriBuffer*    dst = nullptr;
    VriBufferDesc bd {};
    bd.memoryLocation = VriMemoryLocation_Device;
    bd.size           = biggest;
    bd.usage          = VriBufferUsage_TransferSrc;
    c.CreateBuffer(dev, &bd, &src);
    bd.usage = VriBufferUsage_TransferDst;
    c.CreateBuffer(dev, &bd, &dst);

    GpuProfiler prof;
    if (!prof.Init(dev, &c, q, 8))
    {
        std::printf("Failed to init the profiler.\n");
        return 1;
    }

    VriCommandAllocator* alloc = nullptr;
    VriCommandBuffer*    cmd   = nullptr;
    c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
    c.CreateCommandBuffer(alloc, &cmd);
    VriFence* fence = nullptr;
    c.CreateFence(dev, 0, &fence);

    c.BeginCommandBuffer(cmd);
    prof.Begin(cmd);
    prof.Stamp(cmd, "start");
    for (int i = 0; i < 3; ++i)
    {
        VriBufferCopyDesc region {};
        region.size = sizes[i];
        c.CmdCopyBuffer(cmd, dst, src, &region);
        prof.Stamp(cmd, names[i]);
    }
    prof.End(cmd);
    c.EndCommandBuffer(cmd);

    VriFenceSubmitDesc signal {};
    signal.fence  = fence;
    signal.value  = 1;
    signal.stages = VriPipelineStage_AllCommands;
    VriQueueSubmitDesc submit {};
    submit.commandBuffers   = &cmd;
    submit.commandBufferNum = 1;
    submit.signalFences     = &signal;
    submit.signalFenceNum   = 1;
    c.QueueSubmit(queue, &submit);
    c.Wait(fence, 1);

    prof.Report();

    c.DeviceWaitIdle(dev);
    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    prof.Destroy();
    c.DestroyBuffer(dst);
    c.DestroyBuffer(src);
    vriDestroyDevice(dev);
    std::printf("\nprofiler demo done.\n");
    return 0;
}
