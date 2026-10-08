#include "debug_vk.h"
#include "device_vk.h"

namespace vri::vk
{
    void DebugObjectsVK::ApplyName(const Object& object)
    {
        const auto setName = object.device->Ext().SetDebugUtilsObjectName;
        if (!setName || !object.handle || object.type == VK_OBJECT_TYPE_UNKNOWN)
            return;
        VkDebugUtilsObjectNameInfoEXT info {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        info.objectType   = object.type;
        info.objectHandle = object.handle;
        info.pObjectName  = object.name.c_str();
        setName(object.device->Device(), &info);
    }
} // namespace vri::vk
