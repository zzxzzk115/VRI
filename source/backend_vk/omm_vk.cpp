// omm_vk.cpp - opacity micromaps via VK_EXT_opacity_micromap.
//
// A micromap owns its backing store + build scratch (sized from
// vkGetMicromapBuildSizesEXT). CmdBuildMicromap emits a micromap-build barrier so a
// subsequent BLAS build that consumes it sees the result. The micromap is attached to
// a BLAS triangle geometry in rt_vk's BuildGeoms (VkAccelerationStructureTrianglesOpacityMicromapEXT).
#include "omm_vk.h"

#include "device_vk.h"
#include "objects_vk.h"

namespace vri::vk
{
    namespace
    {
        DeviceVK*        Dev(VriDevice* h) { return reinterpret_cast<DeviceVK*>(h); }
        CommandBufferVK* CB(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferVK*>(h); }
        BufferVK*        BUF(VriBuffer* h) { return reinterpret_cast<BufferVK*>(h); }
        MicromapVK*      MM(VriMicromap* h) { return reinterpret_cast<MicromapVK*>(h); }

        VkDeviceAddress BufAddr(DeviceVK* d, VkBuffer b)
        {
            VkBufferDeviceAddressInfo i = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            i.buffer                    = b;
            return vkGetBufferDeviceAddress(d->Device(), &i);
        }

        bool MakeBuffer(DeviceVK* d, VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buf, VmaAllocation& alloc)
        {
            VkBufferCreateInfo bci      = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size                    = size ? size : 1;
            bci.usage                   = usage;
            VmaAllocationCreateInfo aci = {};
            aci.usage                   = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            return vmaCreateBuffer(d->Allocator(), &bci, &aci, &buf, &alloc, nullptr) == VK_SUCCESS;
        }

        VkOpacityMicromapFormatEXT ToVkOmmFormat(VriOpacityMicromapFormat f)
        {
            return f == VriOpacityMicromapFormat_4State ? VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT :
                                                          VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT;
        }

        VriResult VRI_CALL CreateMicromap(VriDevice* device, const VriMicromapDesc* desc, VriMicromap** out)
        {
            DeviceVK*          d     = Dev(device);
            VkMicromapUsageEXT usage = {};
            usage.count              = desc->triangleCount;
            usage.subdivisionLevel   = desc->subdivisionLevel;
            usage.format             = static_cast<uint32_t>(ToVkOmmFormat(desc->format));

            VkMicromapBuildInfoEXT bi = {VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT};
            bi.type                   = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
            bi.flags                  = VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT;
            bi.mode                   = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
            bi.usageCountsCount       = 1;
            bi.pUsageCounts           = &usage;

            VkMicromapBuildSizesInfoEXT sizes = {VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT};
            d->Ext().GetMicromapBuildSizes(d->Device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &sizes);

            VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
            VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            p2.pNext                       = &asProps;
            vkGetPhysicalDeviceProperties2(d->PhysicalDevice(), &p2);
            const VkDeviceSize scratchAlign = asProps.minAccelerationStructureScratchOffsetAlignment;

            MicromapVK* m       = new MicromapVK {};
            m->device           = d;
            m->format           = ToVkOmmFormat(desc->format);
            m->subdivisionLevel = desc->subdivisionLevel;
            m->triangleCount    = desc->triangleCount;

            if (!MakeBuffer(d,
                            sizes.micromapSize,
                            VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            m->buffer,
                            m->bufferAlloc))
            {
                delete m;
                return VriResult_OutOfMemory;
            }

            VkMicromapCreateInfoEXT ci = {VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT};
            ci.buffer                  = m->buffer;
            ci.size                    = sizes.micromapSize;
            ci.type                    = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
            if (d->Ext().CreateMicromap(d->Device(), &ci, nullptr, &m->micromap) != VK_SUCCESS)
            {
                vmaDestroyBuffer(d->Allocator(), m->buffer, m->bufferAlloc);
                delete m;
                return VriResult_Failure;
            }

            if (!MakeBuffer(d,
                            sizes.buildScratchSize + scratchAlign,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            m->scratch,
                            m->scratchAlloc))
            {
                d->Ext().DestroyMicromap(d->Device(), m->micromap, nullptr);
                vmaDestroyBuffer(d->Allocator(), m->buffer, m->bufferAlloc);
                delete m;
                return VriResult_OutOfMemory;
            }
            const VkDeviceAddress sbase = BufAddr(d, m->scratch);
            m->scratchAddress = (sbase + scratchAlign - 1) & ~(static_cast<VkDeviceAddress>(scratchAlign) - 1);

            {
                VriObjectInfo info = MakeObjectInfo(VriObjectType_Micromap);
                info.memoryBytes   = sizes.micromapSize + sizes.buildScratchSize + scratchAlign;
                info.width         = static_cast<uint32_t>(sizes.micromapSize / 1024u);
                info.height        = static_cast<uint32_t>((sizes.buildScratchSize + scratchAlign) / 1024u);
                d->Objects().Track(ToHandle(m), info);
            }
            *out = ToHandle(m);
            return VriResult_Success;
        }

        void VRI_CALL DestroyMicromap(VriMicromap* micromap)
        {
            if (!micromap)
                return;
            MicromapVK* m = MM(micromap);
            m->device->Objects().Untrack(micromap);
            m->device->Ext().DestroyMicromap(m->device->Device(), m->micromap, nullptr);
            vmaDestroyBuffer(m->device->Allocator(), m->buffer, m->bufferAlloc);
            vmaDestroyBuffer(m->device->Allocator(), m->scratch, m->scratchAlloc);
            delete m;
        }

        void VRI_CALL CmdBuildMicromap(VriCommandBuffer* cmd, const VriBuildMicromapDesc* desc)
        {
            CommandBufferVK* c = CB(cmd);
            DeviceVK*        d = c->device;
            MicromapVK*      m = MM(desc->dst);

            VkMicromapUsageEXT usage = {};
            usage.count              = m->triangleCount;
            usage.subdivisionLevel   = m->subdivisionLevel;
            usage.format             = static_cast<uint32_t>(m->format);

            VkMicromapBuildInfoEXT bi      = {VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT};
            bi.type                        = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
            bi.flags                       = VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT;
            bi.mode                        = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
            bi.dstMicromap                 = m->micromap;
            bi.usageCountsCount            = 1;
            bi.pUsageCounts                = &usage;
            bi.data.deviceAddress          = BufAddr(d, BUF(desc->dataBuffer)->buffer) + desc->dataOffset;
            bi.triangleArray.deviceAddress = BufAddr(d, BUF(desc->triangleBuffer)->buffer) + desc->triangleOffset;
            bi.triangleArrayStride         = sizeof(VkMicromapTriangleEXT);
            bi.scratchData.deviceAddress   = m->scratchAddress;

            d->Ext().CmdBuildMicromaps(c->cmd, 1, &bi);

            // Make the built micromap visible to the BLAS build that consumes it.
            // Micromap stage/access bits exist only as synchronization2 (64-bit).
            VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            mb.srcStageMask     = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT;
            mb.srcAccessMask    = VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT;
            mb.dstStageMask =
                VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            mb.dstAccessMask     = VK_ACCESS_2_MICROMAP_READ_BIT_EXT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            VkDependencyInfo dep = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers    = &mb;
            d->Ext().CmdPipelineBarrier2(c->cmd, &dep);
        }

        const VriOpacityMicromapInterface g_ommVK = {
            CreateMicromap,
            DestroyMicromap,
            CmdBuildMicromap,
        };
    } // namespace

    const VriOpacityMicromapInterface* GetOpacityMicromapInterfaceVK() { return &g_ommVK; }
} // namespace vri::vk
