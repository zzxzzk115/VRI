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
    -- The backend uses the C VMA (vk_mem_alloc.h), not the C++ wrapper. The -hpp
    -- package (v3.2.1) fails to compile against vulkan-headers 1.4.335
    -- (vk::ValidationFailedEXTError was removed in newer vulkan-hpp).
    add_requires("vulkan-memory-allocator")
end

-- WebGPU backend (prebuilt wgpu-native C API)
if has_config("vri_backend_wgpu") then
    add_requires("webgpu-sdk v0.1.2")
end

-- OpenGL / OpenGL ES / WebGL backend: spirv-cross (SPIR-V -> GLSL/ESSL) always;
-- on wasm the GL loader (WebGL2) and GLFW come from Emscripten ports, so glad/glfw
-- packages are native-only.
if has_config("vri_backend_gl") then
    -- Pin a recent spirv-cross: 1.3.268 emits invalid tessellation-control GLSL
    -- (non-array output, not indexed by gl_InvocationID); vulkan-sdk-1.4.335 fixes it.
    add_requires("spirv-cross vulkan-sdk-1.4.335")
    if not is_plat("wasm") then
        add_requires("glad", {configs = {profile = "core", api = "gl=4.6"}})
        add_requires("glfw")
    end
end

-- Examples windowing (matches libvultra)
if has_config("vri_build_examples") then
    add_requires("libsdl3")
end

-- Host tools: vri-shaderc links the Slang compiler bundled with the Vulkan SDK
-- (prebuilt) via the `slangsdk` rule, driving it in-process rather than building
-- Slang from source or shelling out to slangc.
