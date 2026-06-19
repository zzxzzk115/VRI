// vri_entry.cpp - implementation of the global C entry points.
//
// These dispatch by VriGraphicsAPI to the selected backend. Phase 0 ships the
// dispatch seam with no backends wired in yet (every API returns Unsupported);
// Phase 1 plugs the Vulkan backend factory into CreateDevice().

#include <vri/vri_core.h>

#include "device_base.h"

#if defined(VRI_BACKEND_VULKAN)
#    include "device_vk.h"
#endif
#if defined(VRI_BACKEND_WGPU)
#    include "device_wgpu.h"
#endif
#if defined(VRI_BACKEND_GL)
#    include "device_gl.h"
#endif

using vri::core::DeviceBase;

namespace
{
    DeviceBase* ToBase(VriDevice* d) { return reinterpret_cast<DeviceBase*>(d); }
    const DeviceBase* ToBase(const VriDevice* d) { return reinterpret_cast<const DeviceBase*>(d); }
    VriDevice* ToHandle(DeviceBase* d) { return reinterpret_cast<VriDevice*>(d); }
} // namespace

extern "C" {

uint32_t VRI_CALL vriEnumerateAdapters(VriGraphicsAPI api,
                                       VriAdapterDesc* outAdapters,
                                       uint32_t adapterCapacity)
{
    switch (api)
    {
#if defined(VRI_BACKEND_VULKAN)
        case VriGraphicsAPI_Vulkan: return vri::vk::EnumerateAdapters(outAdapters, adapterCapacity);
#endif
        default:
            return 0;
    }
}

VriResult VRI_CALL vriCreateDevice(const VriDeviceCreationDesc* desc, VriDevice** outDevice)
{
    if (desc == nullptr || outDevice == nullptr)
        return VriResult_InvalidArgument;

    *outDevice = nullptr;

    DeviceBase* device = nullptr;
    VriResult result = VriResult_Unsupported;
    switch (desc->graphicsAPI)
    {
#if defined(VRI_BACKEND_VULKAN)
        case VriGraphicsAPI_Vulkan: device = vri::vk::CreateDevice(*desc, result); break;
#endif
#if defined(VRI_BACKEND_WGPU)
        case VriGraphicsAPI_WebGPU: device = vri::wgpu::CreateDevice(*desc, result); break;
#endif
#if defined(VRI_BACKEND_GL)
        // The GL backend serves both desktop GL and the GLES3/WebGL2 profile; the
        // device picks the context + shader target from graphicsAPI.
        case VriGraphicsAPI_OpenGL:
        case VriGraphicsAPI_OpenGLES: device = vri::gl::CreateDevice(*desc, result); break;
#endif
        default:
            return VriResult_Unsupported;
    }

    if (device == nullptr)
        return result; // backend reported the precise reason (e.g. Unsupported)

    *outDevice = ToHandle(device);
    return VriResult_Success;
}

void VRI_CALL vriDestroyDevice(VriDevice* device)
{
    delete ToBase(device);
}

VriResult VRI_CALL vriGetInterface(const VriDevice* device, const char* name, size_t size, void* out)
{
    if (device == nullptr || name == nullptr)
        return VriResult_InvalidArgument;

    return ToBase(device)->GetInterface(name, size, out);
}

} // extern "C"
