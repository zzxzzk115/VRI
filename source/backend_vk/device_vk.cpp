#include "device_vk.h"
#include "core_vk.h"
#include "swapchain_vk.h"
#include "interop_vk.h"
#include "vrs_vk.h"
#include "meshshader_vk.h"
#include "rt_vk.h"
#include "omm_vk.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

namespace vri::vk
{
    namespace
    {
        VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT severity,
            VkDebugUtilsMessageTypeFlagsEXT /*types*/,
            const VkDebugUtilsMessengerCallbackDataEXT* data,
            void* userData)
        {
            if (severity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
                return VK_FALSE;

            const VriCallbackInterface* cb = static_cast<const VriCallbackInterface*>(userData);
            const char* msg = data && data->pMessage ? data->pMessage : "<null>";
            if (cb && cb->MessageCallback)
            {
                VriMessageSeverity s = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                                           ? VriMessageSeverity_Error
                                           : VriMessageSeverity_Warning;
                cb->MessageCallback(cb->userArg, s, msg);
            }
            else
            {
                std::fprintf(stderr, "[VRI/VK] %s\n", msg);
            }
            return VK_FALSE;
        }

        bool HasLayer(const char* name)
        {
            uint32_t count = 0;
            vkEnumerateInstanceLayerProperties(&count, nullptr);
            std::vector<VkLayerProperties> layers(count);
            vkEnumerateInstanceLayerProperties(&count, layers.data());
            for (const VkLayerProperties& l : layers)
                if (std::strcmp(l.layerName, name) == 0)
                    return true;
            return false;
        }
    } // namespace

    DeviceVK::~DeviceVK()
    {
        if (m_device)
            vkDeviceWaitIdle(m_device);
        if (m_allocator)
            vmaDestroyAllocator(m_allocator);
        if (m_device)
            vkDestroyDevice(m_device, nullptr);
        if (m_debugMessenger)
        {
            auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy)
                destroy(m_instance, m_debugMessenger, nullptr);
        }
        if (m_instance)
            vkDestroyInstance(m_instance, nullptr);
    }

    void DeviceVK::ReportError(const char* message) const
    {
        if (m_callback.MessageCallback)
            m_callback.MessageCallback(m_callback.userArg, VriMessageSeverity_Error, message);
        else
            std::fprintf(stderr, "[VRI/VK] %s\n", message);
    }

    VriResult DeviceVK::Init(const VriDeviceCreationDesc& desc)
    {
        if (desc.callbackInterface)
            m_callback = *desc.callbackInterface;
        m_validation = desc.enableValidation != VRI_FALSE;

        VriResult r = CreateInstance(desc);
        if (r != VriResult_Success)
            return r;
        r = PickPhysicalDevice(desc.adapterIndex);
        if (r != VriResult_Success)
            return r;
        r = CreateLogicalDevice(desc);
        if (r != VriResult_Success)
            return r;
        r = CreateAllocator();
        if (r != VriResult_Success)
            return r;

        LoadExtensionFunctions();
        FillDeviceDesc();
        FillRegistry();
        return VriResult_Success;
    }

    VriResult DeviceVK::CreateInstance(const VriDeviceCreationDesc& desc)
    {
        VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "VRI";
        app.apiVersion = VK_API_VERSION_1_3;

        std::vector<const char*> extensions;
        extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#if defined(_WIN32)
        extensions.push_back("VK_KHR_win32_surface");
#elif defined(__linux__)
        extensions.push_back("VK_KHR_xlib_surface");
        extensions.push_back("VK_KHR_wayland_surface");
#endif
        for (uint32_t i = 0; i < desc.requiredInstanceExtensionNum; ++i)
            extensions.push_back(desc.requiredInstanceExtensions[i]);

        std::vector<const char*> layers;
        if (m_validation && HasLayer("VK_LAYER_KHRONOS_validation"))
        {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        else
        {
            m_validation = false;
        }

        VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;
        ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        ci.ppEnabledExtensionNames = extensions.data();
        ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
        ci.ppEnabledLayerNames = layers.data();

        if (vkCreateInstance(&ci, nullptr, &m_instance) != VK_SUCCESS)
        {
            ReportError("vkCreateInstance failed");
            return VriResult_Failure;
        }

        if (m_validation)
        {
            auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));
            if (create)
            {
                VkDebugUtilsMessengerCreateInfoEXT dci = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
                dci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
                dci.pfnUserCallback = DebugCallback;
                dci.pUserData = &m_callback;
                create(m_instance, &dci, nullptr, &m_debugMessenger);
            }
        }

        return VriResult_Success;
    }

    VriResult DeviceVK::PickPhysicalDevice(uint32_t adapterIndex)
    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
        if (count == 0)
        {
            ReportError("no Vulkan physical devices found");
            return VriResult_Failure;
        }
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

        if (adapterIndex < count)
        {
            m_physicalDevice = devices[adapterIndex];
            return VriResult_Success;
        }

        // adapterIndex out of range: prefer a discrete GPU, else the first.
        m_physicalDevice = devices[0];
        for (VkPhysicalDevice d : devices)
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(d, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                m_physicalDevice = d;
                break;
            }
        }
        return VriResult_Success;
    }

    VriResult DeviceVK::CreateLogicalDevice(const VriDeviceCreationDesc& desc)
    {
        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &familyCount, families.data());

        uint32_t graphicsFamily = UINT32_MAX;
        for (uint32_t i = 0; i < familyCount; ++i)
        {
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                graphicsFamily = i;
                break;
            }
        }
        if (graphicsFamily == UINT32_MAX)
        {
            ReportError("no graphics-capable queue family");
            return VriResult_Failure;
        }

        // Phase 1: a single graphics queue serves all queue types.
        for (int t = 0; t < VriQueueType_Count; ++t)
            m_queueFamilies[t] = graphicsFamily;

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queueCi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueCi.queueFamilyIndex = graphicsFamily;
        queueCi.queueCount = 1;
        queueCi.pQueuePriorities = &priority;

        // ---- available device extensions ----
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> extProps(extCount);
        vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extCount, extProps.data());
        std::unordered_set<std::string> avail;
        for (const VkExtensionProperties& e : extProps)
            avail.insert(e.extensionName);
        auto hasExt = [&](const char* n) { return avail.find(n) != avail.end(); };

        // ---- query supported optional features ----
        VkPhysicalDeviceAccelerationStructureFeaturesKHR asSup = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtSup = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
        VkPhysicalDeviceRayQueryFeaturesKHR rqSup = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        VkPhysicalDeviceMeshShaderFeaturesEXT meshSup = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        VkPhysicalDeviceFragmentShadingRateFeaturesKHR vrsSup = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR};
        VkPhysicalDeviceOpacityMicromapFeaturesEXT ommSup = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT};
        VkPhysicalDeviceVulkan12Features v12Sup = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures baseSup = {}; // core features: geometry/tessellation/fillModeNonSolid
        {
            VkPhysicalDeviceFeatures2 sup = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            void** q = &sup.pNext;
            if (hasExt(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) { *q = &asSup; q = &asSup.pNext; }
            if (hasExt(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME))   { *q = &rtSup; q = &rtSup.pNext; }
            if (hasExt(VK_KHR_RAY_QUERY_EXTENSION_NAME))              { *q = &rqSup; q = &rqSup.pNext; }
            if (hasExt(VK_EXT_MESH_SHADER_EXTENSION_NAME))            { *q = &meshSup; q = &meshSup.pNext; }
            if (hasExt(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME))  { *q = &vrsSup; q = &vrsSup.pNext; }
            if (hasExt(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME))       { *q = &ommSup; q = &ommSup.pNext; }
            *q = &v12Sup; // Vulkan 1.2 features are core, always safe to query
            vkGetPhysicalDeviceFeatures2(m_physicalDevice, &sup);
            baseSup = sup.features;
        }

        // ---- base enabled features (always on) ----
        VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        f13.dynamicRendering = VK_TRUE;
        f13.synchronization2 = VK_TRUE;

        VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f12.timelineSemaphore = VK_TRUE;
        f12.bufferDeviceAddress = VK_TRUE;
        f12.pNext = &f13;

        // shaderDrawParameters: Slang emits the DrawParameters capability for
        // SV_VertexID/SV_InstanceID (base-vertex aware), so enable it by default.
        VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        f11.shaderDrawParameters = VK_TRUE;
        f11.pNext = &f12;

        VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext = &f11;
        // Legacy graphics stages + non-solid fill: enable when the GPU supports them.
        features2.features.geometryShader     = baseSup.geometryShader;
        features2.features.tessellationShader = baseSup.tessellationShader;
        features2.features.fillModeNonSolid   = baseSup.fillModeNonSolid; // polygonMode Line/Point

        std::vector<const char*> extensions;
        extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        for (uint32_t i = 0; i < desc.requiredDeviceExtensionNum; ++i)
            extensions.push_back(desc.requiredDeviceExtensions[i]);

        // Always-on-if-available pipeline-state capability extensions (no opt-in feature;
        // exposed via VriDeviceDesc::hasXxx, used by per-pipeline state when supported).
        if (hasExt(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME))
        {
            extensions.push_back(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
            m_hasConservativeRaster = true; // no VkPhysicalDevice feature to enable
        }

        // ---- optional features: query -> enable -> report (bestEffort aware) ----
        VkPhysicalDeviceAccelerationStructureFeaturesKHR asEn = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtEn = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
        VkPhysicalDeviceMeshShaderFeaturesEXT meshEn = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        VkPhysicalDeviceFragmentShadingRateFeaturesKHR vrsEn = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR};
        VkPhysicalDeviceOpacityMicromapFeaturesEXT ommEn = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT};
        VkPhysicalDeviceRayQueryFeaturesKHR rqEn = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};

        void** chainTail = &f13.pNext; // append optional feature structs here
        const uint64_t requested = desc.enabledFeatures;
        uint64_t granted = 0;
        auto unsupported = [&](const char* what) -> bool {
            // returns true if creation must abort (i.e. not best-effort)
            if (desc.bestEffort == VRI_FALSE)
            {
                ReportError(what);
                return true;
            }
            return false;
        };

        if (requested & VriFeature_Bindless)
        {
            if (v12Sup.descriptorIndexing && v12Sup.runtimeDescriptorArray && v12Sup.descriptorBindingPartiallyBound)
            {
                f12.descriptorIndexing = VK_TRUE;
                f12.runtimeDescriptorArray = VK_TRUE;
                f12.descriptorBindingPartiallyBound = VK_TRUE;
                // Variable-count (runtime-sized) bindless arrays + update-after-bind.
                f12.descriptorBindingVariableDescriptorCount = v12Sup.descriptorBindingVariableDescriptorCount;
                f12.descriptorBindingSampledImageUpdateAfterBind = v12Sup.descriptorBindingSampledImageUpdateAfterBind;
                f12.descriptorBindingUpdateUnusedWhilePending = v12Sup.descriptorBindingUpdateUnusedWhilePending;
                f12.shaderSampledImageArrayNonUniformIndexing = v12Sup.shaderSampledImageArrayNonUniformIndexing;
                granted |= VriFeature_Bindless;
            }
            else if (unsupported("bindless (descriptor indexing) not supported"))
                return VriResult_Unsupported;
        }

        // Acceleration structures back BOTH ray-tracing pipelines and ray query, so the
        // AS extension/feature is enabled once and shared. enableAS() is idempotent.
        bool asEnabled = false;
        auto enableAS = [&]() -> bool {
            if (asEnabled) return true;
            if (!hasExt(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) ||
                !hasExt(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) || !asSup.accelerationStructure)
                return false;
            extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
            extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
            asEn.accelerationStructure = VK_TRUE;
            *chainTail = &asEn; chainTail = &asEn.pNext;
            asEnabled = true;
            return true;
        };

        if (requested & VriFeature_RayTracing)
        {
            const bool ok = enableAS() && hasExt(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) && rtSup.rayTracingPipeline;
            if (ok)
            {
                extensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
                rtEn.rayTracingPipeline = VK_TRUE;
                *chainTail = &rtEn; chainTail = &rtEn.pNext;
                granted |= VriFeature_RayTracing;
            }
            else if (unsupported("ray tracing not supported"))
                return VriResult_Unsupported;
        }

        if (requested & VriFeature_RayQuery)
        {
            const bool ok = enableAS() && hasExt(VK_KHR_RAY_QUERY_EXTENSION_NAME) && rqSup.rayQuery;
            if (ok)
            {
                extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
                rqEn.rayQuery = VK_TRUE;
                *chainTail = &rqEn; chainTail = &rqEn.pNext;
                granted |= VriFeature_RayQuery;
            }
            else if (unsupported("ray query not supported"))
                return VriResult_Unsupported;
        }

        if (requested & VriFeature_MeshShader)
        {
            const bool ok = hasExt(VK_EXT_MESH_SHADER_EXTENSION_NAME) && meshSup.meshShader && meshSup.taskShader;
            if (ok)
            {
                extensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
                meshEn.meshShader = VK_TRUE;
                meshEn.taskShader = VK_TRUE;
                *chainTail = &meshEn; chainTail = &meshEn.pNext;
                granted |= VriFeature_MeshShader;
            }
            else if (unsupported("mesh shader not supported"))
                return VriResult_Unsupported;
        }

        if (requested & VriFeature_VariableShadingRate)
        {
            const bool ok = hasExt(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME) && vrsSup.pipelineFragmentShadingRate;
            if (ok)
            {
                extensions.push_back(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
                vrsEn.pipelineFragmentShadingRate = VK_TRUE;
                vrsEn.primitiveFragmentShadingRate = vrsSup.primitiveFragmentShadingRate;
                vrsEn.attachmentFragmentShadingRate = vrsSup.attachmentFragmentShadingRate;
                *chainTail = &vrsEn; chainTail = &vrsEn.pNext;
                granted |= VriFeature_VariableShadingRate;
            }
            else if (unsupported("variable rate shading not supported"))
                return VriResult_Unsupported;
        }

        if (requested & VriFeature_OpacityMicromap)
        {
            // OMM extends acceleration-structure geometry, so it needs ray tracing too.
            const bool ok = (granted & VriFeature_RayTracing) &&
                            hasExt(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME) && ommSup.micromap;
            if (ok)
            {
                extensions.push_back(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME);
                ommEn.micromap = VK_TRUE;
                *chainTail = &ommEn; chainTail = &ommEn.pNext;
                granted |= VriFeature_OpacityMicromap;
            }
            else if (unsupported("opacity micromap not supported (needs ray tracing + VK_EXT_opacity_micromap)"))
                return VriResult_Unsupported;
        }

        if ((requested & VriFeature_LowLatency) && unsupported("low-latency not implemented"))
            return VriResult_Unsupported;

        m_enabledFeatures = granted;

        VkDeviceCreateInfo ci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        ci.pNext = &features2;
        ci.queueCreateInfoCount = 1;
        ci.pQueueCreateInfos = &queueCi;
        ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        ci.ppEnabledExtensionNames = extensions.data();

        if (vkCreateDevice(m_physicalDevice, &ci, nullptr, &m_device) != VK_SUCCESS)
        {
            ReportError("vkCreateDevice failed");
            return VriResult_Failure;
        }

        for (int t = 0; t < VriQueueType_Count; ++t)
        {
            VkQueue q = VK_NULL_HANDLE;
            vkGetDeviceQueue(m_device, graphicsFamily, 0, &q);
            m_queues[t].device = this;
            m_queues[t].queue = q;
            m_queues[t].familyIndex = graphicsFamily;
            m_queues[t].indexInFamily = 0;
        }

        return VriResult_Success;
    }

    VriResult DeviceVK::CreateAllocator()
    {
        VmaAllocatorCreateInfo ci = {};
        ci.instance = m_instance;
        ci.physicalDevice = m_physicalDevice;
        ci.device = m_device;
        ci.vulkanApiVersion = VK_API_VERSION_1_3;
        ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

        if (vmaCreateAllocator(&ci, &m_allocator) != VK_SUCCESS)
        {
            ReportError("vmaCreateAllocator failed");
            return VriResult_Failure;
        }
        return VriResult_Success;
    }

    void DeviceVK::FillDeviceDesc()
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(m_physicalDevice, &props);

        m_desc = {};
        std::strncpy(m_desc.adapter.name, props.deviceName, sizeof(m_desc.adapter.name) - 1);
        m_desc.adapter.deviceId = props.deviceID;
        m_desc.adapter.vendorId = props.vendorID;
        switch (props.deviceType)
        {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   m_desc.adapter.type = VriAdapterType_Discrete; break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: m_desc.adapter.type = VriAdapterType_Integrated; break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:            m_desc.adapter.type = VriAdapterType_Software; break;
            default:                                     m_desc.adapter.type = VriAdapterType_Unknown; break;
        }

        VkPhysicalDeviceMemoryProperties mem;
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &mem);
        for (uint32_t i = 0; i < mem.memoryHeapCount; ++i)
        {
            if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                m_desc.adapter.videoMemorySize += mem.memoryHeaps[i].size;
            else
                m_desc.adapter.sharedMemorySize += mem.memoryHeaps[i].size;
        }

        m_desc.graphicsAPI = VriGraphicsAPI_Vulkan;
        m_desc.apiVersionMajor = VK_API_VERSION_MAJOR(props.apiVersion);
        m_desc.apiVersionMinor = VK_API_VERSION_MINOR(props.apiVersion);
        m_desc.viewportMaxNum = props.limits.maxViewports;
        m_desc.attachmentColorMaxNum = props.limits.maxColorAttachments;
        m_desc.attachmentMaxDim = props.limits.maxFramebufferWidth;
        m_desc.texture1DMaxDim = props.limits.maxImageDimension1D;
        m_desc.texture2DMaxDim = props.limits.maxImageDimension2D;
        m_desc.texture3DMaxDim = props.limits.maxImageDimension3D;
        m_desc.textureArrayLayerMaxNum = props.limits.maxImageArrayLayers;
        m_desc.bufferMaxSize = props.limits.maxStorageBufferRange;
        m_desc.uploadBufferTextureRowAlignment = static_cast<uint32_t>(props.limits.optimalBufferCopyRowPitchAlignment);
        m_desc.constantBufferOffsetAlignment = static_cast<uint32_t>(props.limits.minUniformBufferOffsetAlignment);
        m_desc.storageBufferOffsetAlignment = static_cast<uint32_t>(props.limits.minStorageBufferOffsetAlignment);
        for (int t = 0; t < VriQueueType_Count; ++t)
            m_desc.queueCount[t] = 1;
        m_desc.hasComputeShader = VRI_TRUE;
        VkPhysicalDeviceFeatures baseFeatures;
        vkGetPhysicalDeviceFeatures(m_physicalDevice, &baseFeatures);
        m_desc.hasGeometryShader = baseFeatures.geometryShader ? VRI_TRUE : VRI_FALSE;
        m_desc.hasTessellation = baseFeatures.tessellationShader ? VRI_TRUE : VRI_FALSE;

        m_desc.enabledFeatures = m_enabledFeatures;
        m_desc.hasRayTracing = (m_enabledFeatures & VriFeature_RayTracing) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasMeshShader = (m_enabledFeatures & VriFeature_MeshShader) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasBindless = (m_enabledFeatures & VriFeature_Bindless) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasVariableShadingRate = (m_enabledFeatures & VriFeature_VariableShadingRate) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasOpacityMicromap = (m_enabledFeatures & VriFeature_OpacityMicromap) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasRayQuery = (m_enabledFeatures & VriFeature_RayQuery) ? VRI_TRUE : VRI_FALSE;
        m_desc.hasConservativeRaster = m_hasConservativeRaster ? VRI_TRUE : VRI_FALSE;
        m_desc.hasFragmentShaderBarycentric = m_hasBarycentric ? VRI_TRUE : VRI_FALSE;

        if (m_enabledFeatures & VriFeature_RayTracing)
        {
            VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
            VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            props2.pNext = &rtProps;
            vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);
            m_desc.rtShaderGroupHandleSize = rtProps.shaderGroupHandleSize;
            m_desc.rtShaderGroupBaseAlignment = rtProps.shaderGroupBaseAlignment;
            m_desc.rtShaderGroupHandleAlignment = rtProps.shaderGroupHandleAlignment;
        }
    }

    // Resolve extension entry points for the feature set granted at creation.
    void DeviceVK::LoadExtensionFunctions()
    {
        if (m_enabledFeatures & VriFeature_VariableShadingRate)
            m_ext.CmdSetFragmentShadingRate = reinterpret_cast<PFN_vkCmdSetFragmentShadingRateKHR>(
                vkGetDeviceProcAddr(m_device, "vkCmdSetFragmentShadingRateKHR"));
        if (m_enabledFeatures & VriFeature_MeshShader)
        {
            m_ext.CmdDrawMeshTasks = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(
                vkGetDeviceProcAddr(m_device, "vkCmdDrawMeshTasksEXT"));
            m_ext.CmdDrawMeshTasksIndirect = reinterpret_cast<PFN_vkCmdDrawMeshTasksIndirectEXT>(
                vkGetDeviceProcAddr(m_device, "vkCmdDrawMeshTasksIndirectEXT"));
        }
        if (m_enabledFeatures & VriFeature_RayTracing)
        {
            auto L = [&](const char* n) { return vkGetDeviceProcAddr(m_device, n); };
            m_ext.CreateAccelerationStructure = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(L("vkCreateAccelerationStructureKHR"));
            m_ext.DestroyAccelerationStructure = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(L("vkDestroyAccelerationStructureKHR"));
            m_ext.GetAccelerationStructureBuildSizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(L("vkGetAccelerationStructureBuildSizesKHR"));
            m_ext.CmdBuildAccelerationStructures = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(L("vkCmdBuildAccelerationStructuresKHR"));
            m_ext.GetAccelerationStructureDeviceAddress = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(L("vkGetAccelerationStructureDeviceAddressKHR"));
            m_ext.CreateRayTracingPipelines = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(L("vkCreateRayTracingPipelinesKHR"));
            m_ext.GetRayTracingShaderGroupHandles = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(L("vkGetRayTracingShaderGroupHandlesKHR"));
            m_ext.CmdTraceRays = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(L("vkCmdTraceRaysKHR"));
        }
        if (m_enabledFeatures & VriFeature_OpacityMicromap)
        {
            auto L = [&](const char* n) { return vkGetDeviceProcAddr(m_device, n); };
            m_ext.CreateMicromap = reinterpret_cast<PFN_vkCreateMicromapEXT>(L("vkCreateMicromapEXT"));
            m_ext.DestroyMicromap = reinterpret_cast<PFN_vkDestroyMicromapEXT>(L("vkDestroyMicromapEXT"));
            m_ext.GetMicromapBuildSizes = reinterpret_cast<PFN_vkGetMicromapBuildSizesEXT>(L("vkGetMicromapBuildSizesEXT"));
            m_ext.CmdBuildMicromaps = reinterpret_cast<PFN_vkCmdBuildMicromapsEXT>(L("vkCmdBuildMicromapsEXT"));
        }
    }

    void DeviceVK::FillRegistry()
    {
        m_registry.Register(VRI_INTERFACE_CORE, GetCoreInterfaceVK(), sizeof(VriCoreInterface));
        m_registry.Register(VRI_INTERFACE_SWAPCHAIN, GetSwapChainInterfaceVK(), sizeof(VriSwapChainInterface));
        m_registry.Register(VRI_INTERFACE_INTEROP, GetInteropInterfaceVK(), sizeof(VriInteropInterface));
        // Optional interfaces: registered only when the backing feature was granted,
        // so vriGetInterface returns Unsupported on adapters/runs that lack it.
        if (m_enabledFeatures & VriFeature_VariableShadingRate)
            m_registry.Register(VRI_INTERFACE_VRS, GetShadingRateInterfaceVK(), sizeof(VriShadingRateInterface));
        if (m_enabledFeatures & VriFeature_MeshShader)
            m_registry.Register(VRI_INTERFACE_MESHSHADER, GetMeshShaderInterfaceVK(), sizeof(VriMeshShaderInterface));
        // The ray-tracing interface also provides acceleration-structure creation, which
        // ray query needs (inline tracing has no RT pipeline of its own).
        if (m_enabledFeatures & (VriFeature_RayTracing | VriFeature_RayQuery))
            m_registry.Register(VRI_INTERFACE_RAYTRACING, GetRayTracingInterfaceVK(), sizeof(VriRayTracingInterface));
        if (m_enabledFeatures & VriFeature_OpacityMicromap)
            m_registry.Register(VRI_INTERFACE_OMM, GetOpacityMicromapInterfaceVK(), sizeof(VriOpacityMicromapInterface));
    }

    core::DeviceBase* CreateDevice(const VriDeviceCreationDesc& desc, VriResult& outResult)
    {
        DeviceVK* device = new DeviceVK();
        outResult = device->Init(desc);
        if (outResult != VriResult_Success)
        {
            delete device;
            return nullptr;
        }
        return device;
    }

    uint32_t EnumerateAdapters(VriAdapterDesc* outAdapters, uint32_t capacity)
    {
        VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;

        VkInstance instance = VK_NULL_HANDLE;
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS)
            return 0;

        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());

        const uint32_t fill = outAdapters ? (count < capacity ? count : capacity) : 0;
        for (uint32_t i = 0; i < fill; ++i)
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devices[i], &props);
            VriAdapterDesc& a = outAdapters[i];
            a = VriAdapterDesc{};
            std::strncpy(a.name, props.deviceName, sizeof(a.name) - 1);
            a.deviceId = props.deviceID;
            a.vendorId = props.vendorID;
            switch (props.deviceType)
            {
                case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   a.type = VriAdapterType_Discrete; break;
                case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: a.type = VriAdapterType_Integrated; break;
                case VK_PHYSICAL_DEVICE_TYPE_CPU:            a.type = VriAdapterType_Software; break;
                default:                                     a.type = VriAdapterType_Unknown; break;
            }

            VkPhysicalDeviceMemoryProperties mem;
            vkGetPhysicalDeviceMemoryProperties(devices[i], &mem);
            for (uint32_t h = 0; h < mem.memoryHeapCount; ++h)
            {
                if (mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    a.videoMemorySize += mem.memoryHeaps[h].size;
                else
                    a.sharedMemorySize += mem.memoryHeaps[h].size;
            }
        }

        vkDestroyInstance(instance, nullptr);
        return count;
    }
} // namespace vri::vk
