-- External dependencies for VRI.
-- Versions/choices are aligned with the libvultra / vshadersystem ecosystem
-- (same my-xmake-repo) so VRI can be adopted by libvultra later.

-- Tests
if has_config("vri_build_tests") then
    add_requires("doctest")
end

-- Vulkan backend (MVP / reference backend)
if has_config("vri_backend_vulkan") then
    add_requires("vulkan-headers 1.4.335")
    add_requires("vulkan-memory-allocator-hpp")
end

-- WebGPU backend (prebuilt wgpu-native C API)
if has_config("vri_backend_wgpu") then
    add_requires("webgpu-sdk v0.1.2")
end

-- Examples windowing (matches libvultra)
if has_config("vri_build_examples") then
    add_requires("libsdl3")
end

-- Host tools: vri-shaderc links the Slang compiler bundled with the Vulkan SDK
-- (prebuilt) via the `slangsdk` rule, driving it in-process rather than building
-- Slang from source or shelling out to slangc.
