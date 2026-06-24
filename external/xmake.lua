-- External dependencies for VRI.
-- Versions/choices are aligned with the libvultra / vshadersystem ecosystem
-- (same my-xmake-repo) so VRI can be adopted by libvultra later.

-- Tests
if has_config("vri_build_tests") then
    add_requires("doctest")
end

-- Vulkan backend (MVP / reference backend). Not available on wasm (VMA is unsupported
-- there), so guard on the platform too: that way a sticky `--vri_backend_vulkan=y` left
-- over from a desktop config can't pull these packages when you `xmake f -p wasm`.
if has_config("vri_backend_vulkan") and not is_plat("wasm") then
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
    -- glad (GL loader) is needed for every desktop-GL build (Windows/macOS/Linux-desktop);
    -- the native OpenGL ES build uses system GLES headers instead. glfw provides the context
    -- on Windows/macOS only - Linux (both desktop GL and GLES) uses EGL directly, no glfw.
    if not is_plat("wasm") then
        if not has_config("vri_backend_gl_es") then
            add_requires("glad", {configs = {profile = "core", api = "gl=4.6"}})
        end
        if not is_plat("linux") then
            add_requires("glfw")
        end
    end
end

-- Direct3D 12 backend: the DirectX Agility SDK headers expose modern D3D12 the
-- stock Windows SDK lacks (DXR 1.2 / opacity micromaps, mesh shaders, VRS). Only
-- needed when the D3D12 backend is built (Windows). dxgi/d3d12/d3dcompiler libs
-- themselves come from the system (add_syslinks in source/xmake.lua).
if has_config("vri_backend_d3d12") and is_plat("windows") then
    add_requires("directx12-agility")
end

-- Examples windowing (matches libvultra)
if has_config("vri_build_examples") then
    add_requires("libsdl3")
    -- Dear ImGui for the examples' UI. The examples draw ImDrawData through VRI
    -- (examples/common/imgui_vri.h), so we don't need an imgui render backend - only the
    -- core, plus the SDL3 *input* backend on desktop. On wasm input is fed via Emscripten.
    if is_plat("wasm") then
        add_requires("imgui v1.92.5")
    else
        add_requires("imgui v1.92.5", {configs = {sdl3 = true}})
    end
    -- glTF model loading for example-raytracing (examples/common/gltf_model.h). Header-only,
    -- the same tinygltf library Sascha Willems' Vulkan-Examples use. Desktop only (RT pipeline).
    if not is_plat("wasm") then
        add_requires("tinygltf")
    end
end

-- Host tools: vri-shaderc links the Slang compiler bundled with the Vulkan SDK
-- (prebuilt) via the `slangsdk` rule, driving it in-process rather than building
-- Slang from source or shelling out to slangc.
