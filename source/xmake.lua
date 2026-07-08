-- The VRI library target.

target("vri")
    set_kind("static")
    set_languages("cxx23")

    -- public C/C++ headers
    add_includedirs("$(projectdir)/include", {public = true})

    -- vri/vri_version.h is generated from set_version() in the root xmake.lua, so the
    -- version macros live in exactly one place (no hand-edited header to drift). Output
    -- lands in the build dir, which is put on the public include path and installed with
    -- the package headers.
    add_configfiles("$(projectdir)/include/vri/vri_version.h.in", {filename = "vri/vri_version.h"})
    add_includedirs("$(builddir)", {public = true})
    add_headerfiles("$(builddir)/(vri/vri_version.h)")

    -- internal headers (core, backends)
    add_includedirs(os.scriptdir(), {public = false})

    -- repo root, so the built-in ImGui renderer can embed "shaders/lib/imgui_*.h"
    add_includedirs("$(projectdir)", {public = false})

    -- backend-agnostic core
    add_files("core/*.cpp")
    add_headerfiles("$(projectdir)/include/(vri/**.h)")
    add_headerfiles("$(projectdir)/include/(vri/**.hpp)")

    -- defined while building the library (controls dllexport on Windows)
    add_defines("VRI_BUILD")

    -- backend selection (implementations land Phase 1+).
    -- Each backend is gated on its option AND on platform support, so a sticky option
    -- value from a previous config (xmake remembers explicit --flags across `xmake f`)
    -- can never try to build/require a backend the target platform can't use:
    --   Vulkan -> not wasm   |   D3D11/D3D12 -> windows   |   GL/WebGPU -> all (incl. wasm)
    if has_config("vri_backend_vulkan") and not is_plat("wasm") then
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
        elseif is_plat("linux") then
            -- Linux: the context + present go through EGL (VRI_GL_EGL), GLFW-free, so one
            -- path serves Wayland, X11, and headless - for BOTH desktop GL and native GLES.
            -- EGL drives the context; wayland-egl provides the Wayland window surface
            -- (wl_egl_window); X11 present needs no extra lib (uses the window XID).
            add_defines("VRI_GL_EGL")
            add_syslinks("EGL", "wayland-egl", {public = true})
            if has_config("vri_backend_gl_es") then
                -- Native OpenGL ES: system libGLESv2 exports the ES entry points directly
                -- (no loader). The GLES-subset path (shared with WebGL2) is VRI_GL_NATIVE_ES.
                add_defines("VRI_GL_NATIVE_ES")
                add_syslinks("GLESv2", {public = true})
            else
                -- Desktop GL: glad loads the GL entry points through eglGetProcAddress; link
                -- libGL too so the core symbols resolve on every driver.
                add_packages("glad", {public = true})
                add_syslinks("GL", {public = true})
            end
        else
            -- Windows / macOS: GLFW owns the context; present retargets it via WGL/NSGL.
            add_packages("glad", "glfw", {public = true})
            if is_plat("windows") then
                -- GetDC/SetPixelFormat/SwapBuffers live in gdi32.
                add_syslinks("gdi32", {public = true})
            elseif is_plat("macosx") then
                -- macOS GL present retargets the device's NSOpenGL context to the window view and
                -- paces on a CVDisplayLink (nsgl_present_gl.mm). Needs Cocoa + the (deprecated)
                -- OpenGL framework + CoreVideo (CVDisplayLink). Public so libvri consumers that don't
                -- otherwise pull CoreVideo in (e.g. vri-tests, which doesn't link SDL3) still link it.
                add_files("backend_gl/**.mm")
                add_frameworks("Cocoa", "OpenGL", "CoreVideo", {public = true})
            end
        end
    end
    if has_config("vri_backend_metal") and is_plat("macosx") then
        add_defines("VRI_BACKEND_METAL")
        -- SPIR-V is VRI's portable shader IR; SPIRV-Cross transpiles it to MSL at pipeline
        -- creation (same package the GL backend uses for SPIRV->GLSL).
        add_packages("spirv-cross", {public = true})
        -- The backend manages MTL object lifetimes by hand (retain/release): VRI hands
        -- back opaque handles via reinterpret_cast over new/delete, which ARC can't track.
        -- mxflags is the Objective-C++ flag category (applies to every .mm in the target).
        add_mxflags("-fno-objc-arc", {force = true})
        add_files("backend_metal/**.mm")
        add_includedirs("backend_metal", {public = false})
        -- Metal + CAMetalLayer (QuartzCore) + the SDL3 metal view's NSWindow/AppKit chain.
        add_frameworks("Metal", "QuartzCore", "Foundation", "Cocoa", {public = true})
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
        -- Agility SDK: its d3d12.h (DXR 1.2 / opacity micromaps etc.) takes precedence
        -- over the stock SDK header. {public=true} so a consuming exe can deploy the
        -- Agility runtime (D3D12Core.dll) + set D3D12SDKVersion when it needs DXR 1.2.
        add_packages("directx12-agility", {public = true})
    end
target_end()
