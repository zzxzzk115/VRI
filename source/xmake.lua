-- The VRI library target.

target("vri")
    set_kind("static")
    set_languages("cxx23")

    -- public C/C++ headers
    add_includedirs("$(projectdir)/include", {public = true})

    -- internal headers (core, backends)
    add_includedirs(os.scriptdir(), {public = false})

    -- backend-agnostic core
    add_files("core/*.cpp")
    add_headerfiles("$(projectdir)/include/(vri/**.h)")
    add_headerfiles("$(projectdir)/include/(vri/**.hpp)")

    -- defined while building the library (controls dllexport on Windows)
    add_defines("VRI_BUILD")

    -- backend selection (implementations land Phase 1+)
    if has_config("vri_backend_vulkan") then
        add_defines("VRI_BACKEND_VULKAN")
        add_rules("vulkansdk")
        add_packages("vulkan-headers", "vulkan-memory-allocator", {public = false})
        add_files("backend_vk/**.cpp")
        add_includedirs("backend_vk", {public = false})
    end
    if has_config("vri_backend_wgpu") then
        add_defines("VRI_BACKEND_WGPU")
        -- public: wgpu_native must link into the final exe + its DLL on runtime PATH
        add_packages("webgpu-sdk", {public = true})
        add_files("backend_wgpu/**.cpp")
        add_includedirs("backend_wgpu", {public = false})
    end
    if has_config("vri_backend_gl") then
        add_defines("VRI_BACKEND_GL")
        add_packages("spirv-cross", {public = true})
        add_files("backend_gl/**.cpp")
        add_includedirs("backend_gl", {public = false})
        if is_plat("wasm") then
            -- WebGL2 + GLFW3 from Emscripten ports. The port flags are needed at
            -- compile (for the GLFW header) and link (for the final exe), and must
            -- propagate to dependents, hence {public = true}.
            add_cxflags("-sUSE_GLFW=3", {public = true, force = true})
            add_ldflags("-sUSE_GLFW=3", "-sFULL_ES3",
                        "-sMIN_WEBGL_VERSION=2", "-sMAX_WEBGL_VERSION=2",
                        {public = true, force = true})
        else
            -- public: glad/glfw must link into the final exe (+ their runtime needs)
            add_packages("glad", "glfw", {public = true})
            -- Windowed GL present (swapchain_gl) retargets the context to the window via
            -- WGL/GLX: GetDC/SetPixelFormat/SwapBuffers live in gdi32 (Windows), and the
            -- glX* entry points live in libGL (Linux).
            if is_plat("windows") then
                add_syslinks("gdi32", {public = true})
            elseif is_plat("linux") then
                add_syslinks("GL", {public = true})
            end
        end
    end
    if has_config("vri_backend_d3d11") and is_plat("windows") then
        add_defines("VRI_BACKEND_D3D11")
    end
    if has_config("vri_backend_d3d12") and is_plat("windows") then
        add_defines("VRI_BACKEND_D3D12")
        add_files("backend_d3d12/**.cpp")
        add_includedirs("backend_d3d12", {public = false})
        -- D3D12 core + DXGI; d3dcompiler (FXC) for runtime HLSL->DXBC in early bring-up
        add_syslinks("d3d12", "dxgi", "d3dcompiler", {public = true})
    end
target_end()
