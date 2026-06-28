// device_vk.h - the Vulkan VriDevice implementation + factory.
#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <vri/vri.h>

#include "core/device_base.h"
#include "objects_vk.h"

namespace vri::vk
{
    class DeviceVK final : public core::DeviceBase
    {
    public:
        DeviceVK() = default;
        ~DeviceVK() override;

        // Two-phase init so failures return cleanly through the factory.
        VriResult Init(const VriDeviceCreationDesc& desc);

        // accessors used by the core interface implementation
        VkInstance           Instance() const { return m_instance; }
        VkPhysicalDevice     PhysicalDevice() const { return m_physicalDevice; }
        bool                 HasMemoryBudget() const { return m_hasMemoryBudget; }
        VkDevice             Device() const { return m_device; }
        VmaAllocator         Allocator() const { return m_allocator; }
        const VriDeviceDesc& Desc() const { return m_desc; }

        QueueVK*                     GetQueue(VriQueueType type) { return &m_queues[type]; }
        const VkAllocationCallbacks* VkAllocCallbacks() const { return nullptr; }

        // Loaded extension entry points (populated by LoadExtensionFunctions for
        // whatever feature set was granted). Zero-initialized; null == unavailable.
        struct ExtFunctions
        {
            // Core 1.3 (synchronization2 + dynamic rendering). Loaded by entry point so a
            // pre-1.3 driver (e.g. MoltenVK, which reports 1.2) resolves them to the KHR
            // aliases instead of crashing on a null core dispatch slot.
            PFN_vkCmdPipelineBarrier2 CmdPipelineBarrier2 = nullptr;
            PFN_vkCmdBeginRendering   CmdBeginRendering   = nullptr;
            PFN_vkCmdEndRendering     CmdEndRendering     = nullptr;
            PFN_vkQueueSubmit2        QueueSubmit2        = nullptr;

            PFN_vkCmdSetFragmentShadingRateKHR CmdSetFragmentShadingRate = nullptr;
            PFN_vkCmdDrawMeshTasksEXT          CmdDrawMeshTasks          = nullptr;
            PFN_vkCmdDrawMeshTasksIndirectEXT  CmdDrawMeshTasksIndirect  = nullptr;
            // ray tracing (KHR acceleration structure + ray tracing pipeline)
            PFN_vkCreateAccelerationStructureKHR              CreateAccelerationStructure              = nullptr;
            PFN_vkDestroyAccelerationStructureKHR             DestroyAccelerationStructure             = nullptr;
            PFN_vkGetAccelerationStructureBuildSizesKHR       GetAccelerationStructureBuildSizes       = nullptr;
            PFN_vkCmdBuildAccelerationStructuresKHR           CmdBuildAccelerationStructures           = nullptr;
            PFN_vkGetAccelerationStructureDeviceAddressKHR    GetAccelerationStructureDeviceAddress    = nullptr;
            PFN_vkCreateRayTracingPipelinesKHR                CreateRayTracingPipelines                = nullptr;
            PFN_vkGetRayTracingShaderGroupHandlesKHR          GetRayTracingShaderGroupHandles          = nullptr;
            PFN_vkCmdTraceRaysKHR                             CmdTraceRays                             = nullptr;
            PFN_vkCmdWriteAccelerationStructuresPropertiesKHR CmdWriteAccelerationStructuresProperties = nullptr;
            PFN_vkCmdCopyAccelerationStructureKHR             CmdCopyAccelerationStructure             = nullptr;
            // calibrated timestamps (VK_KHR/EXT_calibrated_timestamps)
            PFN_vkGetCalibratedTimestampsKHR GetCalibratedTimestamps = nullptr;
            // opacity micromap (VK_EXT_opacity_micromap)
            PFN_vkCreateMicromapEXT        CreateMicromap        = nullptr;
            PFN_vkDestroyMicromapEXT       DestroyMicromap       = nullptr;
            PFN_vkGetMicromapBuildSizesEXT GetMicromapBuildSizes = nullptr;
            PFN_vkCmdBuildMicromapsEXT     CmdBuildMicromaps     = nullptr;
        };
        const ExtFunctions& Ext() const { return m_ext; }
        uint64_t            EnabledFeatures() const { return m_enabledFeatures; }
        VkTimeDomainKHR     HostTimeDomain() const { return m_hostTimeDomain; }

        void ReportError(const char* message) const;
        void ReportWarning(const char* message) const;

    private:
        VriResult CreateInstance(const VriDeviceCreationDesc& desc);
        VriResult PickPhysicalDevice(uint32_t adapterIndex);
        VriResult CreateLogicalDevice(const VriDeviceCreationDesc& desc);
        VriResult CreateAllocator();
        void      LoadExtensionFunctions();
        void      FillDeviceDesc();
        void      FillRegistry();

        VkInstance               m_instance       = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
        VkPhysicalDevice         m_physicalDevice = VK_NULL_HANDLE;
        VkDevice                 m_device         = VK_NULL_HANDLE;
        VmaAllocator             m_allocator      = VK_NULL_HANDLE;

        QueueVK  m_queues[VriQueueType_Count]        = {};
        uint32_t m_queueFamilies[VriQueueType_Count] = {};
        bool     m_validation                        = false;
        uint64_t m_enabledFeatures                   = 0; // granted VriFeatureBits
        // Always-on-if-available pipeline-state extensions (capabilities, not opt-in features).
        bool            m_hasConservativeRaster   = false;
        bool            m_hasBarycentric          = false;
        bool            m_hasCustomBorderColor    = false;
        bool            m_hasPipelineStats        = false;
        bool            m_hasCalibratedTimestamps = false;
        bool            m_hasDrawIndirectCount    = false;
        bool            m_hasMemoryBudget         = false;
        VkTimeDomainKHR m_hostTimeDomain          = VK_TIME_DOMAIN_DEVICE_KHR; // the host clock to pair with DEVICE
        ExtFunctions    m_ext                     = {};

        VriDeviceDesc        m_desc     = {};
        VriCallbackInterface m_callback = {};
    };

    // Factory invoked from the C entry point (vri_entry.cpp). Reports the
    // precise failure code (e.g. Unsupported) via outResult.
    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult);

    // Enumerate Vulkan adapters (spins up a transient instance). Fills up to
    // `capacity` entries, returns the total adapter count.
    uint32_t EnumerateAdapters(VriAdapterDesc* outAdapters, uint32_t capacity);
} // namespace vri::vk
