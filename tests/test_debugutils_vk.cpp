#include <doctest/doctest.h>

#include "backend_vk/device_vk.h"
#include "backend_vk/objects_vk.h"

#include <string>
#include <vector>

namespace
{
    std::vector<VkObjectType> g_types;
    std::vector<uint64_t>     g_handles;
    std::vector<std::string>  g_names;
    std::vector<std::string>  g_labels;
    uint32_t                  g_ends = 0;

    VKAPI_ATTR VkResult VKAPI_CALL CaptureName(VkDevice, const VkDebugUtilsObjectNameInfoEXT* info)
    {
        g_types.push_back(info->objectType);
        g_handles.push_back(info->objectHandle);
        g_names.emplace_back(info->pObjectName ? info->pObjectName : "");
        return VK_SUCCESS;
    }
    VKAPI_ATTR void VKAPI_CALL CaptureBegin(VkCommandBuffer, const VkDebugUtilsLabelEXT* label)
    {
        g_labels.emplace_back(label->pLabelName);
    }
    VKAPI_ATTR void VKAPI_CALL CaptureEnd(VkCommandBuffer) { ++g_ends; }

    template<typename H>
    H Native(uintptr_t value)
    {
        if constexpr (std::is_pointer_v<H>)
            return reinterpret_cast<H>(value);
        else
            return static_cast<H>(value);
    }
} // namespace

TEST_CASE("Vulkan debug-utils Release annotations dispatch native object types without validation")
{
    using namespace vri::vk;
    VriDeviceCreationDesc desc {};
    desc.graphicsAPI      = VriGraphicsAPI_Vulkan;
    desc.enableValidation = VRI_FALSE;
    desc.bestEffort       = VRI_TRUE;
    VriDevice* device     = nullptr;
    if (vriCreateDevice(&desc, &device) != VriResult_Success)
    {
        MESSAGE("Vulkan unavailable - skipped");
        return;
    }
    struct DeviceGuard
    {
        VriDevice* device;
        ~DeviceGuard() { vriDestroyDevice(device); }
    } guard {device};
    auto* backend = reinterpret_cast<DeviceVK*>(device);
    if (!backend->Ext().SetDebugUtilsObjectName)
    {
        MESSAGE("VK_EXT_debug_utils unavailable - annotations degrade to no-ops");
        return;
    }
    CHECK(backend->Ext().CmdBeginDebugUtilsLabel != nullptr);
    CHECK(backend->Ext().CmdEndDebugUtilsLabel != nullptr);

    struct DispatchGuard
    {
        DeviceVK::ExtFunctions& functions;
        DeviceVK::ExtFunctions  saved;
        ~DispatchGuard() { functions = saved; }
    } dispatch {const_cast<DeviceVK::ExtFunctions&>(backend->Ext()), backend->Ext()};
    dispatch.functions.SetDebugUtilsObjectName = CaptureName;
    dispatch.functions.CmdBeginDebugUtilsLabel = CaptureBegin;
    dispatch.functions.CmdEndDebugUtilsLabel   = CaptureEnd;
    VriCoreInterface core {};
    REQUIRE(vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(core), &core) == VriResult_Success);
    g_types.clear();
    g_handles.clear();
    g_names.clear();
    g_labels.clear();
    g_ends = 0;

    // Intercept native dispatch to verify exact handle/type pairs. A descriptor's
    // first member is its kind, so treating it as a device pointer would crash here.
    BufferVK     buffer {backend, Native<VkBuffer>(101), nullptr, 64};
    PipelineVK   pipeline {backend, Native<VkPipeline>(102), VK_PIPELINE_BIND_POINT_COMPUTE};
    DescriptorVK sampler {};
    sampler.device  = backend;
    sampler.kind    = DescriptorVK::Kind::Sampler;
    sampler.sampler = Native<VkSampler>(103);
    DescriptorVK view {};
    view.device    = backend;
    view.kind      = DescriptorVK::Kind::TextureView;
    view.imageView = Native<VkImageView>(104);
    CommandBufferVK command {backend, Native<VkCommandBuffer>(105), nullptr, VK_PIPELINE_BIND_POINT_GRAPHICS};
    auto*           bufferHandle = ToHandle(&buffer);
    core.SetDebugName(bufferHandle, "buffer");
    core.SetDebugName(ToHandle(&pipeline), "pipeline");
    core.SetDebugName(ToHandle(&sampler), "sampler");
    core.SetDebugName(ToHandle(&view), "view");
    core.SetDebugName(device, "device");
    core.SetDebugName(nullptr, "ignored");
    core.SetDebugName(reinterpret_cast<void*>(uintptr_t {1}), "unknown handle");
    REQUIRE(g_types.size() == 5);
    CHECK(g_types[0] == VK_OBJECT_TYPE_BUFFER);
    CHECK(g_types[1] == VK_OBJECT_TYPE_PIPELINE);
    CHECK(g_types[2] == VK_OBJECT_TYPE_SAMPLER);
    CHECK(g_types[3] == VK_OBJECT_TYPE_IMAGE_VIEW);
    CHECK(g_types[4] == VK_OBJECT_TYPE_DEVICE);
    CHECK(g_handles[0] == 101);
    CHECK(g_handles[1] == 102);
    CHECK(g_handles[2] == 103);
    CHECK(g_handles[3] == 104);

    DescriptorVK acceleration {};
    acceleration.device = backend;
    acceleration.kind   = DescriptorVK::Kind::AccelerationStructure;
    acceleration.accel  = Native<VkAccelerationStructureKHR>(107);
    core.SetDebugName(ToHandle(&acceleration), "AS view");
    CHECK(g_types.back() == VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR);
    CHECK(g_handles.back() == 107);

    // This is the same logical handle with a recreated native object (e.g. swapchain resize).
    buffer.buffer = Native<VkBuffer>(108);
    ToHandle(&buffer);
    CHECK(g_handles.back() == 108);
    CHECK(g_names.back() == "buffer");
    core.SetDebugName(bufferHandle, nullptr);
    const auto namedCount = g_names.size();
    buffer.buffer         = Native<VkBuffer>(109);
    ToHandle(&buffer);
    CHECK(g_names.size() == namedCount);

    auto* commandHandle = ToHandle(&command);
    core.CmdBeginDebugGroup(commandHandle, "outer");
    core.CmdBeginDebugGroup(commandHandle, "inner");
    core.CmdEndDebugGroup(commandHandle);
    core.CmdEndDebugGroup(commandHandle);
    REQUIRE(g_labels.size() == 2);
    CHECK(g_labels[0] == "outer");
    CHECK(g_labels[1] == "inner");
    CHECK(g_ends == 2);
    {
        auto* temporary       = new BufferVK {backend, Native<VkBuffer>(106), nullptr, 32};
        auto* temporaryHandle = ToHandle(temporary);
        CHECK(DebugObjectsVK::Find(temporaryHandle).handle == 106);
        delete temporary;
        CHECK(DebugObjectsVK::Find(temporaryHandle).device == nullptr);
    }
}
