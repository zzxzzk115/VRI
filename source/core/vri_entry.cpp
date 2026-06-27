// vri_entry.cpp - implementation of the global C entry points.
//
// These dispatch by VriGraphicsAPI to the selected backend. Phase 0 ships the
// dispatch seam with no backends wired in yet (every API returns Unsupported);
// Phase 1 plugs the Vulkan backend factory into CreateDevice().

#include <vri/vri_core.h>

#include "device_base.h"

#if defined(VRI_BACKEND_VULKAN)
#include "device_vk.h"
#endif
#if defined(VRI_BACKEND_WGPU)
#include "device_wgpu.h"
#endif
#if defined(VRI_BACKEND_GL)
#include "device_gl.h"
#endif
#if defined(VRI_BACKEND_D3D12)
#include "device_d3d12.h"
#endif
#if defined(VRI_BACKEND_METAL)
// ObjC-free factory decl: device_mtl.h imports <Metal/Metal.h>, which a plain
// .cpp TU can't compile.
#include "factory_mtl.h"
#endif

using vri::core::DeviceBase;

namespace vri::core
{
    // VRI Validation layer (validation_layer.cpp). Active when the device was
    // created with enableValidation; wraps the device so it can't be confused with
    // a raw backend handle.
    VriDevice* WrapValidationDevice(VriDevice* realDevice, const VriDeviceCreationDesc& desc);
    bool       IsValidationDevice(const VriDevice* device);
    VriResult  ValidationGetInterface(const VriDevice* device, const char* name, size_t size, void* out);
    void       DestroyValidationDevice(VriDevice* device);
} // namespace vri::core

namespace
{
    DeviceBase*       ToBase(VriDevice* d) { return reinterpret_cast<DeviceBase*>(d); }
    const DeviceBase* ToBase(const VriDevice* d) { return reinterpret_cast<const DeviceBase*>(d); }
    VriDevice*        ToHandle(DeviceBase* d) { return reinterpret_cast<VriDevice*>(d); }

    // Create a specific backend. Backends not compiled in fall through to Unsupported.
    DeviceBase* CreateBackend(VriGraphicsAPI api, const VriDeviceCreationDesc& desc, VriResult& result)
    {
        result = VriResult_Unsupported;
        switch (api)
        {
#if defined(VRI_BACKEND_VULKAN)
            case VriGraphicsAPI_Vulkan:
                return vri::vk::CreateDevice(desc, result);
#endif
#if defined(VRI_BACKEND_WGPU)
            case VriGraphicsAPI_WebGPU:
                return vri::wgpu::CreateDevice(desc, result);
#endif
#if defined(VRI_BACKEND_GL)
            // The GL backend serves both desktop GL and the GLES3/WebGL2 profile; the
            // device picks the context + shader target from graphicsAPI.
            case VriGraphicsAPI_OpenGL:
            case VriGraphicsAPI_OpenGLES:
                return vri::gl::CreateDevice(desc, result);
#endif
#if defined(VRI_BACKEND_D3D12)
            case VriGraphicsAPI_D3D12:
                return vri::d3d12::CreateDevice(desc, result);
#endif
#if defined(VRI_BACKEND_METAL)
            case VriGraphicsAPI_Metal:
                return vri::mtl::CreateDevice(desc, result);
#endif
            default:
                return nullptr;
        }
    }

    // Platform-ordered candidates for VriGraphicsAPI_Auto. Only the compiled-in ones
    // actually create a device (the rest return Unsupported and are skipped); the order
    // prefers the most capable / best-tested backend per platform.
    const VriGraphicsAPI* AutoOrder(uint32_t& count)
    {
#if defined(__EMSCRIPTEN__)
        static const VriGraphicsAPI o[] = {VriGraphicsAPI_WebGPU, VriGraphicsAPI_OpenGLES};
#elif defined(_WIN32)
        static const VriGraphicsAPI o[] = {
            VriGraphicsAPI_Vulkan, VriGraphicsAPI_D3D12, VriGraphicsAPI_WebGPU, VriGraphicsAPI_OpenGL};
#elif defined(__APPLE__)
        static const VriGraphicsAPI o[] = {
            VriGraphicsAPI_Metal, VriGraphicsAPI_Vulkan, VriGraphicsAPI_WebGPU, VriGraphicsAPI_OpenGL};
#else // Linux / Android
        static const VriGraphicsAPI o[] = {VriGraphicsAPI_Vulkan, VriGraphicsAPI_WebGPU, VriGraphicsAPI_OpenGL};
#endif
        count = static_cast<uint32_t>(sizeof(o) / sizeof(o[0]));
        return o;
    }
} // namespace

extern "C"
{

    uint32_t VRI_CALL vriEnumerateAdapters(VriGraphicsAPI api, VriAdapterDesc* outAdapters, uint32_t adapterCapacity)
    {
        switch (api)
        {
#if defined(VRI_BACKEND_VULKAN)
            case VriGraphicsAPI_Vulkan:
                return vri::vk::EnumerateAdapters(outAdapters, adapterCapacity);
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
        VriResult   result = VriResult_Unsupported;
        if (desc->graphicsAPI == VriGraphicsAPI_Auto)
        {
            // Try the platform's preferred backends in order until one creates a device.
            uint32_t              n     = 0;
            const VriGraphicsAPI* order = AutoOrder(n);
            for (uint32_t i = 0; i < n && device == nullptr; ++i)
            {
                VriDeviceCreationDesc copy = *desc;
                copy.graphicsAPI           = order[i];
                device                     = CreateBackend(order[i], copy, result);
            }
        }
        else
        {
            device = CreateBackend(desc->graphicsAPI, *desc, result);
        }

        if (device == nullptr)
            return result; // backend reported the precise reason (e.g. Unsupported)

        VriDevice* handle = ToHandle(device);
        // Insert the VRI Validation layer if requested (debug builds default it on).
        *outDevice = desc->enableValidation ? vri::core::WrapValidationDevice(handle, *desc) : handle;
        return VriResult_Success;
    }

    void VRI_CALL vriDestroyDevice(VriDevice* device)
    {
        if (device == nullptr)
            return;
        if (vri::core::IsValidationDevice(device))
            vri::core::DestroyValidationDevice(device);
        else
            delete ToBase(device);
    }

    VriResult VRI_CALL vriGetInterface(const VriDevice* device, const char* name, size_t size, void* out)
    {
        if (device == nullptr || name == nullptr)
            return VriResult_InvalidArgument;

        if (vri::core::IsValidationDevice(device))
            return vri::core::ValidationGetInterface(device, name, size, out);
        return ToBase(device)->GetInterface(name, size, out);
    }

} // extern "C"
