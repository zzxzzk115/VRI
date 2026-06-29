-- example-openxr: VRI single-pass-stereo rendering driven by OpenXR.
--
-- Opt-in (off by default) via --vri_build_openxr=y: it links the OpenXR loader and needs an
-- OpenXR runtime (a headset or a runtime simulator) to actually run, so it is never built in CI.
--
-- This demonstrates the SDK-independent seam: VRI itself does NOT depend on OpenXR. The example
-- links the OpenXR loader, asks OpenXR which Vulkan instance/device extensions it requires, hands
-- those to vriCreateDevice, reads VRI's native Vulkan handles back through VRI_INTERFACE_INTEROP
-- to build the XrGraphicsBindingVulkanKHR, wraps OpenXR's swapchain VkImages as VriTextures, and
-- renders both eyes in one pass with VRI multiview (viewMask). vulkan-headers is needed only for
-- the Vulkan types in the OpenXR <-> Vulkan binding structs (no Vulkan functions are called).

if has_config("vri_build_openxr") then

    add_requires("openxr", {optional = true})
    add_requires("vulkan-headers 1.4.335", {optional = true})

    target("example-openxr")
        set_kind("binary")
        set_languages("cxx23")
        set_default(false)

        add_deps("vri")
        add_includedirs("$(projectdir)") -- shaders/<bucket>/X.h are included from the repo root

        add_files("main.cpp")
        add_packages("openxr", "vulkan-headers")
    target_end()

end
