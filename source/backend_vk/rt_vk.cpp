// rt_vk.cpp - ray tracing via VK_KHR_acceleration_structure + VK_KHR_ray_tracing_pipeline.
//
// Acceleration structures own their backing store + build scratch (allocated at
// creation, sized from vkGetAccelerationStructureBuildSizesKHR). CmdBuild emits an
// AS-build barrier afterwards so a dependent build (BLAS->TLAS) or a trace sees the
// result without the app inserting one. RT pipelines reuse PipelineVK (bind point
// RAY_TRACING). The SBT itself is built by the app from GetShaderGroupHandles.
#include "rt_vk.h"

#include "conversions_vk.h"
#include "device_vk.h"
#include "objects_vk.h"

#include <deque>
#include <vector>

namespace vri::vk
{
    namespace
    {
        DeviceVK*                Dev(VriDevice* h) { return reinterpret_cast<DeviceVK*>(h); }
        CommandBufferVK*         CB(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferVK*>(h); }
        BufferVK*                BUF(VriBuffer* h) { return reinterpret_cast<BufferVK*>(h); }
        PipelineVK*              Pipe(VriPipeline* h) { return reinterpret_cast<PipelineVK*>(h); }
        AccelerationStructureVK* AS(VriAccelerationStructure* h)
        {
            return reinterpret_cast<AccelerationStructureVK*>(h);
        }

        VkDeviceAddress BufAddr(DeviceVK* d, VkBuffer b)
        {
            VkBufferDeviceAddressInfo i = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            i.buffer                    = b;
            return vkGetBufferDeviceAddress(d->Device(), &i);
        }

        bool MakeBuffer(DeviceVK* d, VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buf, VmaAllocation& alloc)
        {
            VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size               = size ? size : 1;
            bci.usage              = usage;
            // Concurrent across the device's distinct families whenever there is more than one:
            // acceleration structures are commonly consumed by ray queries on the compute queue,
            // buffers have no compression to lose, and the barrier API deliberately has no
            // queue-family ownership transfer. Covers scratch too - harmless.
            uint32_t       families[VriQueueType_Count];
            const uint32_t familyCount = d->DistinctQueueFamilies(families);
            if (familyCount > 1)
            {
                bci.sharingMode           = VK_SHARING_MODE_CONCURRENT;
                bci.queueFamilyIndexCount = familyCount;
                bci.pQueueFamilyIndices   = families;
            }
            VmaAllocationCreateInfo aci = {};
            aci.usage                   = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            return vmaCreateBuffer(d->Allocator(), &bci, &aci, &buf, &alloc, nullptr) == VK_SUCCESS;
        }

        VkAccelerationStructureTypeKHR ToAsType(VriAccelerationStructureType t)
        {
            return t == VriAccelerationStructureType_TopLevel ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR :
                                                                VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        }

        VkBuildAccelerationStructureFlagsKHR ToBuildFlags(VriAccelerationStructureBuildFlags f)
        {
            VkBuildAccelerationStructureFlagsKHR r = 0;
            if (f & VriAccelerationStructureBuild_AllowUpdate)
                r |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            if (f & VriAccelerationStructureBuild_AllowCompaction)
                r |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
            if (f & VriAccelerationStructureBuild_PreferFastTrace)
                r |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            if (f & VriAccelerationStructureBuild_PreferFastBuild)
                r |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
            if (f & VriAccelerationStructureBuild_LowMemory)
                r |= VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR;
            return r;
        }

        // Build the VkAccelerationStructureGeometryKHR array + primitive counts. With
        // withAddr the buffer device addresses are filled (for build); without, only
        // formats/counts (for sizing).
        struct GeomBuild
        {
            std::vector<VkAccelerationStructureGeometryKHR>                geoms;
            std::vector<uint32_t>                                          primCounts;
            std::vector<VkAccelerationStructureBuildRangeInfoKHR>          ranges;
            std::deque<VkAccelerationStructureTrianglesOpacityMicromapEXT> ommChains; // stable addresses for pNext
        };

        void BuildGeoms(DeviceVK* d, const VriAccelerationStructureDesc* desc, bool withAddr, GeomBuild& gb)
        {
            for (uint32_t i = 0; i < desc->geometryCount; ++i)
            {
                const VriAsGeometryDesc&           g  = desc->geometries[i];
                VkAccelerationStructureGeometryKHR vg = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
                vg.flags       = (g.flags & VriAsGeometry_Opaque) ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
                uint32_t prims = 0;
                if (g.type == VriAsGeometryType_Triangles)
                {
                    const VriAsTrianglesDesc& t                         = g.triangles;
                    vg.geometryType                                     = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                    VkAccelerationStructureGeometryTrianglesDataKHR& vt = vg.geometry.triangles;
                    vt.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                    vt.vertexFormat = ToVkFormat(t.vertexFormat);
                    vt.vertexStride = t.vertexStride;
                    vt.maxVertex    = t.vertexCount ? t.vertexCount - 1 : 0;
                    vt.indexType    = t.indexBuffer ? (t.indexType == VriIndexType_UInt16 ? VK_INDEX_TYPE_UINT16 :
                                                                                            VK_INDEX_TYPE_UINT32) :
                                                      VK_INDEX_TYPE_NONE_KHR;
                    if (withAddr)
                    {
                        vt.vertexData.deviceAddress = BufAddr(d, BUF(t.vertexBuffer)->buffer) + t.vertexOffset;
                        if (t.indexBuffer)
                            vt.indexData.deviceAddress = BufAddr(d, BUF(t.indexBuffer)->buffer) + t.indexOffset;
                        if (t.transformBuffer)
                            vt.transformData.deviceAddress =
                                BufAddr(d, BUF(t.transformBuffer)->buffer) + t.transformOffset;
                    }
                    prims = t.indexBuffer ? t.indexCount / 3 : t.vertexCount / 3;

                    if (t.micromap) // attach an opacity micromap to this geometry
                    {
                        MicromapVK* mm = reinterpret_cast<MicromapVK*>(t.micromap);
                        gb.ommChains.emplace_back();
                        VkAccelerationStructureTrianglesOpacityMicromapEXT& omm = gb.ommChains.back();
                        omm = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT};
                        omm.indexType =
                            t.ommIndexType == VriIndexType_UInt16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
                        omm.micromap = mm->micromap;
                        if (withAddr && t.ommIndexBuffer)
                            omm.indexBuffer.deviceAddress =
                                BufAddr(d, BUF(t.ommIndexBuffer)->buffer) + t.ommIndexOffset;
                        vt.pNext = &omm;
                    }
                }
                else if (g.type == VriAsGeometryType_Aabbs)
                {
                    const VriAsAabbsDesc& a                         = g.aabbs;
                    vg.geometryType                                 = VK_GEOMETRY_TYPE_AABBS_KHR;
                    VkAccelerationStructureGeometryAabbsDataKHR& va = vg.geometry.aabbs;
                    va.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
                    va.stride = a.stride ? a.stride : 24; // 6 floats, tightly packed
                    if (withAddr)
                        va.data.deviceAddress = BufAddr(d, BUF(a.buffer)->buffer) + a.offset;
                    prims = a.count;
                }
                else // instances (TLAS)
                {
                    const VriAsInstancesDesc& in                        = g.instances;
                    vg.geometryType                                     = VK_GEOMETRY_TYPE_INSTANCES_KHR;
                    VkAccelerationStructureGeometryInstancesDataKHR& vi = vg.geometry.instances;
                    vi.sType           = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
                    vi.arrayOfPointers = VK_FALSE;
                    if (withAddr)
                        vi.data.deviceAddress = BufAddr(d, BUF(in.instanceBuffer)->buffer) + in.offset;
                    prims = in.instanceCount;
                }
                VkAccelerationStructureBuildRangeInfoKHR range = {};
                range.primitiveCount                           = prims;
                gb.geoms.push_back(vg);
                gb.primCounts.push_back(prims);
                gb.ranges.push_back(range);
            }
        }

        // ---- interface --------------------------------------------------------

        VriResult VRI_CALL CreateAccelerationStructure(VriDevice*                          device,
                                                       const VriAccelerationStructureDesc* desc,
                                                       VriAccelerationStructure**          out)
        {
            DeviceVK* d = Dev(device);
            GeomBuild gb;
            BuildGeoms(d, desc, false, gb);

            VkAccelerationStructureBuildGeometryInfoKHR bi = {
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
            bi.type          = ToAsType(desc->type);
            bi.flags         = ToBuildFlags(desc->flags);
            bi.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            bi.geometryCount = static_cast<uint32_t>(gb.geoms.size());
            bi.pGeometries   = gb.geoms.data();

            VkAccelerationStructureBuildSizesInfoKHR sizes = {
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
            d->Ext().GetAccelerationStructureBuildSizes(
                d->Device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, gb.primCounts.data(), &sizes);

            VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
            VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            p2.pNext                       = &asProps;
            vkGetPhysicalDeviceProperties2(d->PhysicalDevice(), &p2);
            const VkDeviceSize scratchAlign = asProps.minAccelerationStructureScratchOffsetAlignment;

            AccelerationStructureVK* a = new AccelerationStructureVK {};
            a->device                  = d;
            a->type                    = bi.type;
            if (!MakeBuffer(d,
                            sizes.accelerationStructureSize,
                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            a->buffer,
                            a->bufferAlloc))
            {
                delete a;
                return VriResult_OutOfMemory;
            }

            VkAccelerationStructureCreateInfoKHR ci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            ci.buffer                               = a->buffer;
            ci.size                                 = sizes.accelerationStructureSize;
            ci.type                                 = bi.type;
            if (d->Ext().CreateAccelerationStructure(d->Device(), &ci, nullptr, &a->as) != VK_SUCCESS)
            {
                vmaDestroyBuffer(d->Allocator(), a->buffer, a->bufferAlloc);
                delete a;
                return VriResult_Failure;
            }

            if (!MakeBuffer(d,
                            sizes.buildScratchSize + scratchAlign,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            a->scratch,
                            a->scratchAlloc))
            {
                d->Ext().DestroyAccelerationStructure(d->Device(), a->as, nullptr);
                vmaDestroyBuffer(d->Allocator(), a->buffer, a->bufferAlloc);
                delete a;
                return VriResult_OutOfMemory;
            }
            const VkDeviceAddress sbase = BufAddr(d, a->scratch);
            a->scratchAddress = (sbase + scratchAlign - 1) & ~(static_cast<VkDeviceAddress>(scratchAlign) - 1);

            VkAccelerationStructureDeviceAddressInfoKHR ai = {
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
            ai.accelerationStructure = a->as;
            a->deviceAddress         = d->Ext().GetAccelerationStructureDeviceAddress(d->Device(), &ai);

            *out = ToHandle(a);
            return VriResult_Success;
        }

        void VRI_CALL DestroyAccelerationStructure(VriAccelerationStructure* as)
        {
            if (!as)
                return;
            AccelerationStructureVK* a = AS(as);
            if (a->compactedSizePool)
                vkDestroyQueryPool(a->device->Device(), a->compactedSizePool, nullptr);
            a->device->Ext().DestroyAccelerationStructure(a->device->Device(), a->as, nullptr);
            vmaDestroyBuffer(a->device->Allocator(), a->buffer, a->bufferAlloc);
            vmaDestroyBuffer(a->device->Allocator(), a->scratch, a->scratchAlloc);
            delete a;
        }

        uint64_t VRI_CALL GetAccelerationStructureDeviceAddress(const VriAccelerationStructure* as)
        {
            return reinterpret_cast<const AccelerationStructureVK*>(as)->deviceAddress;
        }

        VriResult VRI_CALL CreateAccelerationStructureDescriptor(VriDevice*                device,
                                                                 VriAccelerationStructure* as,
                                                                 VriDescriptor**           out)
        {
            DescriptorVK* v = new DescriptorVK {};
            v->kind         = DescriptorVK::Kind::AccelerationStructure;
            v->device       = Dev(device);
            v->accel        = AS(as)->as;
            *out            = reinterpret_cast<VriDescriptor*>(v);
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateRayTracingPipeline(VriDevice*                       device,
                                                    const VriRayTracingPipelineDesc* desc,
                                                    VriPipeline**                    out)
        {
            DeviceVK* d   = Dev(device);
            VkDevice  dev = d->Device();

            std::vector<VkPipelineShaderStageCreateInfo> stages(desc->shaderNum,
                                                                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO});
            std::vector<VkShaderModule>                  modules(desc->shaderNum, VK_NULL_HANDLE);
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                const VriShaderDesc&     s  = desc->shaders[i];
                VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
                mi.codeSize                 = s.bytecodeSize;
                mi.pCode                    = static_cast<const uint32_t*>(s.bytecode);
                vkCreateShaderModule(dev, &mi, nullptr, &modules[i]);
                stages[i].stage  = ToVkShaderStage(s.stage);
                stages[i].module = modules[i];
                stages[i].pName  = s.entryPointName ? s.entryPointName : "main";
            }
            bool moduleFailed = false;
            for (VkShaderModule m : modules)
                if (m == VK_NULL_HANDLE)
                    moduleFailed = true;
            if (moduleFailed)
            {
                for (VkShaderModule m : modules)
                    if (m)
                        vkDestroyShaderModule(dev, m, nullptr);
                return VriResult_Failure;
            }

            std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups(
                desc->groupNum, {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR});
            for (uint32_t i = 0; i < desc->groupNum; ++i)
            {
                const VriShaderGroupDesc&             g  = desc->groups[i];
                VkRayTracingShaderGroupCreateInfoKHR& vg = groups[i];
                vg.type                                  = g.type == VriShaderGroupType_TrianglesHitGroup ?
                                                               VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR :
                                                           g.type == VriShaderGroupType_ProceduralHitGroup ?
                                                               VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR :
                                                               VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
                vg.generalShader = g.generalShader == VRI_SHADER_UNUSED ? VK_SHADER_UNUSED_KHR : g.generalShader;
                vg.closestHitShader =
                    g.closestHitShader == VRI_SHADER_UNUSED ? VK_SHADER_UNUSED_KHR : g.closestHitShader;
                vg.anyHitShader = g.anyHitShader == VRI_SHADER_UNUSED ? VK_SHADER_UNUSED_KHR : g.anyHitShader;
                vg.intersectionShader =
                    g.intersectionShader == VRI_SHADER_UNUSED ? VK_SHADER_UNUSED_KHR : g.intersectionShader;
            }

            VkRayTracingPipelineCreateInfoKHR ci = {VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
            ci.stageCount                        = static_cast<uint32_t>(stages.size());
            ci.pStages                           = stages.data();
            ci.groupCount                        = static_cast<uint32_t>(groups.size());
            ci.pGroups                           = groups.data();
            ci.maxPipelineRayRecursionDepth      = desc->maxRecursionDepth ? desc->maxRecursionDepth : 1;
            ci.layout = desc->pipelineLayout ? reinterpret_cast<PipelineLayoutVK*>(desc->pipelineLayout)->layout :
                                               VK_NULL_HANDLE;
            // Required for tracing against acceleration structures that carry opacity micromaps.
            if (d->EnabledFeatures() & VriFeature_OpacityMicromap)
                ci.flags |= VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT;

            VkPipeline     pipeline = VK_NULL_HANDLE;
            const VkResult vr =
                d->Ext().CreateRayTracingPipelines(dev, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline);
            for (VkShaderModule m : modules)
                vkDestroyShaderModule(dev, m, nullptr);
            if (vr != VK_SUCCESS)
                return VriResult_Failure;

            *out = ToHandle(new PipelineVK {d, pipeline, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR});
            return VriResult_Success;
        }

        VriResult VRI_CALL
        GetShaderGroupHandles(VriPipeline* pipeline, uint32_t firstGroup, uint32_t groupNum, size_t dstSize, void* dst)
        {
            PipelineVK* p = Pipe(pipeline);
            return p->device->Ext().GetRayTracingShaderGroupHandles(
                       p->device->Device(), p->pipeline, firstGroup, groupNum, dstSize, dst) == VK_SUCCESS ?
                       VriResult_Success :
                       VriResult_Failure;
        }

        void VRI_CALL CmdBuildAccelerationStructure(VriCommandBuffer*                        cmd,
                                                    const VriBuildAccelerationStructureDesc* desc)
        {
            CommandBufferVK*         c = CB(cmd);
            DeviceVK*                d = c->device;
            AccelerationStructureVK* a = AS(desc->dst);

            GeomBuild gb;
            BuildGeoms(d, desc->geometry, true, gb);

            VkAccelerationStructureBuildGeometryInfoKHR bi = {
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
            bi.type                      = a->type;
            bi.flags                     = ToBuildFlags(desc->geometry->flags);
            bi.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            bi.dstAccelerationStructure  = a->as;
            bi.geometryCount             = static_cast<uint32_t>(gb.geoms.size());
            bi.pGeometries               = gb.geoms.data();
            bi.scratchData.deviceAddress = a->scratchAddress;

            const VkAccelerationStructureBuildRangeInfoKHR* pRanges = gb.ranges.data();
            d->Ext().CmdBuildAccelerationStructures(c->cmd, 1, &bi, &pRanges);

            // Make the built AS visible to a dependent build (BLAS->TLAS) or to a trace.
            // dst = ALL_COMMANDS because the AS may be read by an RT-pipeline trace OR by
            // ray query in a compute/graphics shader (any stage).
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            mb.dstAccessMask =
                VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            vkCmdPipelineBarrier(c->cmd,
                                 VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 0,
                                 1,
                                 &mb,
                                 0,
                                 nullptr,
                                 0,
                                 nullptr);
        }

        void VRI_CALL CmdTraceRays(VriCommandBuffer* cmd, const VriDispatchRaysDesc* desc)
        {
            CommandBufferVK* c      = CB(cmd);
            DeviceVK*        d      = c->device;
            auto             region = [&](const VriStridedBufferRegion& r) {
                VkStridedDeviceAddressRegionKHR v = {};
                if (r.buffer)
                {
                    v.deviceAddress = BufAddr(d, BUF(r.buffer)->buffer) + r.offset;
                    v.stride        = r.stride;
                    v.size          = r.size;
                }
                return v;
            };
            const VkStridedDeviceAddressRegionKHR rg = region(desc->raygen);
            const VkStridedDeviceAddressRegionKHR ms = region(desc->miss);
            const VkStridedDeviceAddressRegionKHR ht = region(desc->hit);
            const VkStridedDeviceAddressRegionKHR cl = region(desc->callable);
            d->Ext().CmdTraceRays(c->cmd, &rg, &ms, &ht, &cl, desc->width, desc->height, desc->depth ? desc->depth : 1);
        }

        void VRI_CALL CmdWriteAccelerationStructureCompactedSize(VriCommandBuffer*         cmd,
                                                                 VriAccelerationStructure* as,
                                                                 VriBuffer*                dstBuffer,
                                                                 uint64_t                  dstOffset)
        {
            AccelerationStructureVK* a  = AS(as);
            DeviceVK*                d  = a->device;
            VkCommandBuffer          cb = CB(cmd)->cmd;
            if (a->compactedSizePool == VK_NULL_HANDLE)
            {
                VkQueryPoolCreateInfo qci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                qci.queryType             = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
                qci.queryCount            = 1;
                vkCreateQueryPool(d->Device(), &qci, nullptr, &a->compactedSizePool);
            }
            vkCmdResetQueryPool(cb, a->compactedSizePool, 0, 1);
            d->Ext().CmdWriteAccelerationStructuresProperties(
                cb, 1, &a->as, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, a->compactedSizePool, 0);
            vkCmdCopyQueryPoolResults(cb,
                                      a->compactedSizePool,
                                      0,
                                      1,
                                      BUF(dstBuffer)->buffer,
                                      dstOffset,
                                      sizeof(uint64_t),
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        }

        VriResult VRI_CALL CreateAccelerationStructureCompacted(VriDevice*                   device,
                                                                VriAccelerationStructureType type,
                                                                uint64_t                     size,
                                                                VriAccelerationStructure**   out)
        {
            DeviceVK*                d = Dev(device);
            AccelerationStructureVK* a = new AccelerationStructureVK {};
            a->device                  = d;
            a->type                    = ToAsType(type);
            if (!MakeBuffer(d,
                            size,
                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            a->buffer,
                            a->bufferAlloc))
            {
                delete a;
                return VriResult_OutOfMemory;
            }
            VkAccelerationStructureCreateInfoKHR ci = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            ci.buffer                               = a->buffer;
            ci.size                                 = size;
            ci.type                                 = a->type;
            if (d->Ext().CreateAccelerationStructure(d->Device(), &ci, nullptr, &a->as) != VK_SUCCESS)
            {
                vmaDestroyBuffer(d->Allocator(), a->buffer, a->bufferAlloc);
                delete a;
                return VriResult_Failure;
            }
            VkAccelerationStructureDeviceAddressInfoKHR ai = {
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
            ai.accelerationStructure = a->as;
            a->deviceAddress         = d->Ext().GetAccelerationStructureDeviceAddress(d->Device(), &ai);
            *out                     = ToHandle(a);
            return VriResult_Success;
        }

        void VRI_CALL CmdCopyAccelerationStructure(VriCommandBuffer*         cmd,
                                                   VriAccelerationStructure* dst,
                                                   VriAccelerationStructure* src,
                                                   VriBool                   compact)
        {
            AccelerationStructureVK*           s    = AS(src);
            VkCopyAccelerationStructureInfoKHR info = {VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
            info.src                                = s->as;
            info.dst                                = AS(dst)->as;
            info.mode = compact != VRI_FALSE ? VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR :
                                               VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR;
            s->device->Ext().CmdCopyAccelerationStructure(CB(cmd)->cmd, &info);
        }

        const VriRayTracingInterface g_rtVK = {
            CreateAccelerationStructure,
            DestroyAccelerationStructure,
            GetAccelerationStructureDeviceAddress,
            CreateAccelerationStructureDescriptor,
            CreateRayTracingPipeline,
            GetShaderGroupHandles,
            CmdBuildAccelerationStructure,
            CmdTraceRays,
            CmdWriteAccelerationStructureCompactedSize,
            CreateAccelerationStructureCompacted,
            CmdCopyAccelerationStructure,
        };
    } // namespace

    const VriRayTracingInterface* GetRayTracingInterfaceVK() { return &g_rtVK; }
} // namespace vri::vk
