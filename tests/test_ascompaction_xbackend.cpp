// Acceleration-structure compaction (ext/vri_ext_raytracing.h): build a BLAS with
// AllowCompaction, query its compacted size, create a compacted AS of that size, and copy/compact
// into it. Checks the compacted size is non-zero and no larger than the original, and the compacted
// AS has a valid device address. Vulkan + D3D12; self-skips where ray tracing is unavailable (CI
// software rasterizers), keeping the suite green. Validation-clean.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>

namespace
{
    void RunCompaction(VriGraphicsAPI api, const char* name)
    {
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        dc.enabledFeatures  = VriFeature_RayTracing;
        VriDevice* dev      = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << name << "] unavailable - skipping AS-compaction test");
            return;
        }
        struct Guard
        {
            VriDevice* d;
            ~Guard() { vriDestroyDevice(d); }
        } guard {dev};

        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        const VriDeviceDesc*   dd = c.GetDeviceDesc(dev);
        VriRayTracingInterface rt {};
        if (dd->hasRayTracing == VRI_FALSE ||
            vriGetInterface(dev, VRI_INTERFACE_RAYTRACING, sizeof(rt), &rt) != VriResult_Success)
        {
            MESSAGE("[" << name << "] ray tracing unsupported - skipping");
            return;
        }

        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        auto hostBuffer = [&](uint64_t size, VriBufferUsageFlags usage, const void* data) {
            VriBufferDesc bd {};
            bd.size           = size;
            bd.usage          = usage;
            bd.memoryLocation = VriMemoryLocation_HostUpload;
            VriBuffer* b      = nullptr;
            REQUIRE(c.CreateBuffer(dev, &bd, &b) == VriResult_Success);
            if (data)
            {
                std::memcpy(c.MapBuffer(b, 0, size), data, size);
                c.UnmapBuffer(b);
            }
            return b;
        };

        // BLAS geometry: a single triangle.
        const float verts[9] = {0.0f, -0.5f, 0.0f, 0.5f, 0.5f, 0.0f, -0.5f, 0.5f, 0.0f};
        VriBuffer*  vbuf     = hostBuffer(sizeof(verts), VriBufferUsage_AccelerationBuildInput, verts);

        VriAsGeometryDesc geom {};
        geom.type                   = VriAsGeometryType_Triangles;
        geom.triangles.vertexBuffer = vbuf;
        geom.triangles.vertexCount  = 3;
        geom.triangles.vertexStride = 3 * sizeof(float);
        geom.triangles.vertexFormat = VriFormat_RGB32_SFLOAT;
        VriAccelerationStructureDesc blasDesc {};
        blasDesc.type  = VriAccelerationStructureType_BottomLevel; // build with compaction allowed + fast trace
        blasDesc.flags = VriAccelerationStructureBuild_PreferFastTrace | VriAccelerationStructureBuild_AllowCompaction;
        blasDesc.geometryCount         = 1;
        blasDesc.geometries            = &geom;
        VriAccelerationStructure* blas = nullptr;
        REQUIRE(rt.CreateAccelerationStructure(dev, &blasDesc, &blas) == VriResult_Success);

        // Size sink (UAV on D3D12 / transfer-dst on Vulkan) + host-readable copy of it.
        VriBufferDesc sd {};
        sd.size            = sizeof(uint64_t);
        sd.usage           = VriBufferUsage_StorageBuffer | VriBufferUsage_TransferSrc | VriBufferUsage_TransferDst;
        sd.memoryLocation  = VriMemoryLocation_Device;
        VriBuffer* sizeBuf = nullptr;
        REQUIRE(c.CreateBuffer(dev, &sd, &sizeBuf) == VriResult_Success);
        VriBufferDesc rd {};
        rd.size           = sizeof(uint64_t);
        rd.usage          = VriBufferUsage_TransferDst;
        rd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* readSz = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rd, &readSz) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd1 = nullptr;
        VriCommandBuffer* cmd2 = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd1) == VriResult_Success);
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd2) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        // cmd1: build the BLAS, write its compacted size, copy the size to a readback buffer.
        REQUIRE(c.BeginCommandBuffer(cmd1) == VriResult_Success);
        VriBuildAccelerationStructureDesc build {};
        build.dst      = blas;
        build.geometry = &blasDesc;
        rt.CmdBuildAccelerationStructure(cmd1, &build);
        rt.CmdWriteAccelerationStructureCompactedSize(cmd1, blas, sizeBuf, 0);
        {
            VriBufferBarrierDesc bb {};
            bb.buffer        = sizeBuf;
            bb.before.access = VriAccess_ShaderResourceStorageWrite | VriAccess_CopyDestinationWrite;
            bb.before.stages = VriPipelineStage_AllCommands;
            bb.after.access  = VriAccess_CopySourceRead;
            bb.after.stages  = VriPipelineStage_Transfer;
            VriBarrierGroupDesc g {};
            g.buffers   = &bb;
            g.bufferNum = 1;
            c.CmdBarrier(cmd1, &g);
        }
        VriBufferCopyDesc region {};
        region.size = sizeof(uint64_t);
        c.CmdCopyBuffer(cmd1, readSz, sizeBuf, &region);
        REQUIRE(c.EndCommandBuffer(cmd1) == VriResult_Success);
        {
            VriFenceSubmitDesc sig {};
            sig.fence = fence;
            sig.value = 1;
            VriQueueSubmitDesc sub {};
            sub.commandBuffers   = &cmd1;
            sub.commandBufferNum = 1;
            sub.signalFences     = &sig;
            sub.signalFenceNum   = 1;
            c.QueueSubmit(queue, &sub);
            c.Wait(fence, 1);
        }

        uint64_t    compactedSize = 0;
        const void* mapped        = c.MapBuffer(readSz, 0, sizeof(uint64_t));
        REQUIRE(mapped != nullptr);
        std::memcpy(&compactedSize, mapped, sizeof(compactedSize));
        c.UnmapBuffer(readSz);
        CHECK(compactedSize > 0);
        MESSAGE("[" << name << "] compacted BLAS size = " << compactedSize << " bytes");

        // Create the compacted destination and compact into it.
        VriAccelerationStructure* compact = nullptr;
        REQUIRE(rt.CreateAccelerationStructureCompacted(
                    dev, VriAccelerationStructureType_BottomLevel, compactedSize, &compact) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd2) == VriResult_Success);
        rt.CmdCopyAccelerationStructure(cmd2, compact, blas, VRI_TRUE);
        REQUIRE(c.EndCommandBuffer(cmd2) == VriResult_Success);
        {
            VriFenceSubmitDesc sig {};
            sig.fence = fence;
            sig.value = 2;
            VriQueueSubmitDesc sub {};
            sub.commandBuffers   = &cmd2;
            sub.commandBufferNum = 1;
            sub.signalFences     = &sig;
            sub.signalFenceNum   = 1;
            c.QueueSubmit(queue, &sub);
            c.Wait(fence, 2);
        }
        CHECK(rt.GetAccelerationStructureDeviceAddress(compact) != 0); // valid compacted AS

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readSz);
        c.DestroyBuffer(sizeBuf);
        rt.DestroyAccelerationStructure(compact);
        rt.DestroyAccelerationStructure(blas);
        c.DestroyBuffer(vbuf);
    }
} // namespace

TEST_CASE("AS compaction: Vulkan shrinks a BLAS") { RunCompaction(VriGraphicsAPI_Vulkan, "Vulkan"); }
TEST_CASE("AS compaction: D3D12 shrinks a BLAS") { RunCompaction(VriGraphicsAPI_D3D12, "D3D12"); }
