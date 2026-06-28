// GPU query pools (ext/vri_ext_query.h): timestamp queries on whichever explicit backends are
// built (Vulkan, D3D12). Records two timestamps around a device->device copy, resolves them into
// a readback buffer, and checks the GPU clock advanced. Self-skips where the backend/adapter
// lacks timestamp queries (e.g. software rasterizers), keeping the suite green. Validation-clean.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>

namespace
{
    struct Dev
    {
        VriDevice*       device = nullptr;
        VriCoreInterface core {};
        ~Dev()
        {
            if (device)
                vriDestroyDevice(device);
        }
    };

    bool Init(Dev& d, VriGraphicsAPI api)
    {
        VriDeviceCreationDesc desc {};
        desc.graphicsAPI      = api;
        desc.enableValidation = VRI_TRUE;
        desc.bestEffort       = VRI_TRUE;
        if (vriCreateDevice(&desc, &d.device) != VriResult_Success)
            return false;
        return vriGetInterface(d.device, VRI_INTERFACE_CORE, sizeof(d.core), &d.core) == VriResult_Success;
    }

    VriBuffer* MakeBuffer(const VriCoreInterface& c,
                          VriDevice*              dev,
                          uint64_t                size,
                          VriBufferUsageFlags     usage,
                          VriMemoryLocation       loc)
    {
        VriBufferDesc bd {};
        bd.size           = size;
        bd.usage          = usage;
        bd.memoryLocation = loc;
        VriBuffer* b      = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &b) == VriResult_Success);
        return b;
    }

    // Records 2 timestamps around a buffer copy and verifies the GPU clock advanced.
    void RunTimestamp(VriGraphicsAPI api, const char* name)
    {
        Dev d;
        if (!Init(d, api))
        {
            MESSAGE("[" << name << "] unavailable - skipping query test");
            return;
        }
        const VriCoreInterface& c   = d.core;
        VriDevice*              dev = d.device;
        const VriDeviceDesc*    dd  = c.GetDeviceDesc(dev);

        VriQueryInterface q {};
        const bool        hasQuery = vriGetInterface(dev, VRI_INTERFACE_QUERY, sizeof(q), &q) == VriResult_Success;
        CHECK(hasQuery); // Vulkan + D3D12 always register the query interface
        if (!hasQuery || dd->hasTimestampQueries == VRI_FALSE)
        {
            MESSAGE("[" << name << "] lacks timestamp queries - skipping");
            return;
        }
        CHECK(dd->timestampPeriodNanoseconds > 0.0f);

        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        VriQueryPoolDesc qpd {};
        qpd.type           = VriQueryType_Timestamp;
        qpd.queryCount     = 2;
        VriQueryPool* pool = nullptr;
        REQUIRE(q.CreateQueryPool(dev, &qpd, &pool) == VriResult_Success);
        CHECK(q.GetQuerySize(pool) == sizeof(uint64_t));

        constexpr uint64_t kSize = 1u << 20; // 1 MiB of copy work between the timestamps
        VriBuffer*         src   = MakeBuffer(c, dev, kSize, VriBufferUsage_TransferSrc, VriMemoryLocation_Device);
        VriBuffer*         dst   = MakeBuffer(c, dev, kSize, VriBufferUsage_TransferDst, VriMemoryLocation_Device);
        VriBuffer*         readback =
            MakeBuffer(c, dev, 2 * sizeof(uint64_t), VriBufferUsage_TransferDst, VriMemoryLocation_HostReadback);

        VriCommandAllocator* alloc = nullptr;
        VriCommandBuffer*    cmd   = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        q.CmdResetQueries(cmd, pool, 0, 2);
        q.CmdWriteTimestamp(cmd, pool, 0);
        VriBufferCopyDesc region {};
        region.size = kSize;
        c.CmdCopyBuffer(cmd, dst, src, &region);
        q.CmdWriteTimestamp(cmd, pool, 1);
        q.CmdCopyQueries(cmd, pool, 0, 2, readback, 0);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);

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

        uint64_t        ticks[2] = {0, 0};
        const uint64_t* mapped   = static_cast<const uint64_t*>(c.MapBuffer(readback, 0, 2 * sizeof(uint64_t)));
        REQUIRE(mapped != nullptr);
        std::memcpy(ticks, mapped, sizeof(ticks));
        c.UnmapBuffer(readback);

        CHECK(ticks[0] > 0);         // GPU clock is running
        CHECK(ticks[1] >= ticks[0]); // monotonic across the copy
        const double elapsedMs = static_cast<double>(ticks[1] - ticks[0]) * dd->timestampPeriodNanoseconds / 1.0e6;
        CHECK(elapsedMs >= 0.0);
        MESSAGE("[" << name << "] GPU copy took " << elapsedMs << " ms (" << (ticks[1] - ticks[0]) << " ticks)");

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(dst);
        c.DestroyBuffer(src);
        q.DestroyQueryPool(pool);
    }
} // namespace

TEST_CASE("Query pool: Vulkan timestamps advance across GPU work") { RunTimestamp(VriGraphicsAPI_Vulkan, "Vulkan"); }

TEST_CASE("Query pool: D3D12 timestamps advance across GPU work") { RunTimestamp(VriGraphicsAPI_D3D12, "D3D12"); }
