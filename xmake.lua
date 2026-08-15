-- set project name
set_project("VRI")

-- set project version
set_version("0.1.10")

-- set language version: C++ 23
set_languages("cxx23")

if is_plat("android") then
    set_toolchains("@ndk", {sdkver = "26"})
    -- Vulkan handles are 64-bit; keep arm64 only to avoid handle truncation on 32-bit.
    set_allowedarchs("arm64-v8a")
end

-- root ?
-- False when another xmake project includes this file (VRI as a subproject/submodule). In that
-- case xmake's project dir belongs to the CONSUMER, so every includes()/add_repositories() path
-- below is anchored to os.scriptdir() instead of being project-relative, and the examples, tests
-- and host tools - which only make sense in a standalone checkout - default to off.
local is_root = (os.projectdir() == os.scriptdir())
set_config("root", is_root)
set_config("project_dir", os.scriptdir())

-- global options
option("vri_build_examples") -- build examples?
    set_default(is_root and not is_plat("android") and not is_plat("wasm"))
    set_showmenu(true)
    set_description("Enable VRI examples")
option_end()

option("vri_build_tests") -- build tests?
    set_default(is_root and not is_plat("android") and not is_plat("wasm"))
    set_showmenu(true)
    set_description("Enable VRI tests")
option_end()

option("vri_build_tools") -- build host tools (vri-shaderc)?
    set_default(is_root and not is_plat("android") and not is_plat("wasm"))
    set_showmenu(true)
    set_description("Enable VRI host tools (vri-shaderc)")
option_end()

option("vri_build_cuda_interop") -- build the CUDA interop example? (needs the CUDA Toolkit)
    set_default(false)
    set_showmenu(true)
    set_description("Build example-cuda-interop (requires the CUDA Toolkit; off by default)")
option_end()

option("vri_build_openxr") -- build the OpenXR stereo example? (needs the OpenXR loader + an XR runtime to run)
    set_default(false)
    set_showmenu(true)
    set_description("Build example-openxr (links openxr-loader; needs an OpenXR runtime to run; off by default)")
option_end()

-- backend options (MVP: Vulkan on by default; others enabled as they come online).
--
-- Platform-dependent defaults are set in on_check, NOT set_default: an option's set_default is
-- baked in an early pass where the -p target isn't resolved yet (get_config("plat") == nil, so
-- is_plat("wasm") is false there). on_check runs after the platform is known, so is_plat() is
-- correct - and option:enable() there does not clobber a value the user passed explicitly, so
-- --vri_backend_*=y/n is still honored. (set_default would also suppress on_check, hence neither.)
option("vri_backend_vulkan")
    -- on by default on desktop; unavailable on the web (-> off on wasm).
    set_showmenu(true)
    set_description("Enable the Vulkan backend")
    on_check(function (option) if not is_plat("wasm") then option:enable(true) end end)
option_end()

option("vri_backend_wgpu")
    -- opt-in on desktop; the primary web backend on wasm (Auto picks it first, GL/WebGL2 is the
    -- fallback) -> on by default there.
    set_showmenu(true)
    set_description("Enable the WebGPU backend")
    on_check(function (option) if is_plat("wasm") then option:enable(true) end end)
option_end()

option("vri_backend_gl")
    -- default on everywhere: GL (desktop GL / GLES / WebGL2) is near-universally available,
    -- so like Vulkan it ships by default (on wasm it is the WebGL2 fallback behind WebGPU).
    set_default(true)
    set_showmenu(true)
    set_description("Enable the OpenGL / OpenGL ES / WebGL backend")
option_end()

option("vri_backend_gl_es")
    -- Build the GL backend as native OpenGL ES (EGL context, system libGLESv2) instead
    -- of desktop GL (glad + GLX/WGL). For embedded targets / Raspberry Pi. Off by default
    -- (desktop GL); turn on with `xmake f --vri_backend_gl_es=y`. Ignored on wasm (the
    -- web build is always WebGL2) and when the GL backend itself is disabled.
    set_default(false)
    set_showmenu(true)
    set_description("Build the OpenGL backend as native OpenGL ES (EGL; embedded / Raspberry Pi)")
option_end()

-- No vri_backend_d3d11 option: there is no D3D11 backend implementation (no source/backend_d3d11,
-- no dispatch case). The VriGraphicsAPI_D3D11 enum is kept as the deliberate "never-compiled"
-- probe (vriCreateDevice -> VriResult_Unsupported; see tests/test_main.cpp), but a build option
-- that only defined VRI_BACKEND_D3D11 without adding any sources was a dead, misleading switch.

option("vri_backend_d3d12")
    set_default(false)
    set_showmenu(true)
    set_description("Enable the Direct3D 12 backend (Windows only)")
option_end()

option("vri_backend_metal")
    -- default on for macOS: the native, first-class Apple backend (Auto picks it first),
    -- ahead of the MoltenVK/Vulkan and OpenGL fallbacks. Ignored off-Apple.
    set_default(is_plat("macosx"))
    set_showmenu(true)
    set_description("Enable the native Metal backend (macOS only)")
option_end()

option("vri_validation")
    set_default(is_mode("debug"))
    set_showmenu(true)
    set_description("Enable graphics API validation layers by default")
option_end()

-- align package runtimes with the consuming targets (matches libvultra/vshadersystem ecosystem)
if is_plat("windows") then
    add_requireconfs("**", {configs = {runtimes = is_mode("debug") and "MTd" or "MT"}})
end

-- if build on windows
if is_plat("windows") then
    add_cxxflags("/Zc:__cplusplus", {tools = {"msvc", "cl"}}) -- fix __cplusplus == 199711L error
    add_cxxflags("/bigobj") -- avoid big obj
    add_cxxflags("-D_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING")
    add_cxxflags("/EHsc")
    if is_mode("debug") then
        set_runtimes("MTd")
    else
        set_runtimes("MT")
    end
else
    add_cxxflags("-fexceptions")
end

-- add rules
rule("clangd.config")
    on_config(function (target)
        -- Only meaningful in a standalone checkout: it drops a .clangd next to VRI's own
        -- sources. When VRI is a subproject the consumer owns its editor config, and the
        -- template files are not in the consumer's project dir at all - so skip, and use
        -- absolute paths rather than relying on the working directory.
        if not get_config("root") then
            return
        end
        local dir = get_config("project_dir") or os.projectdir()
        if is_host("windows") then
            os.cp(path.join(dir, ".clangd.win"), path.join(dir, ".clangd"))
        else
            os.cp(path.join(dir, ".clangd.nowin"), path.join(dir, ".clangd"))
        end
    end)
rule_end()

-- Link only the Vulkan loader (not glslang/spirv-cross/etc. that ship in the SDK).
-- Ported from libvultra. Headers come from the vulkan-headers package; this rule
-- just locates and links the loader (vulkan-1 / libvulkan).
-- https://github.com/xmake-io/xmake-repo/issues/3962#issuecomment-2096205856
rule("vulkansdk")
    on_config(function (target)
        if target:is_plat("android") then
            target:add("syslinks", "vulkan", { public = true })
            return
        end

        import("lib.detect.find_library")
        import("detect.sdks.find_vulkansdk")

        -- Distro install (libvulkan-dev): there is no VULKAN_SDK tree, so find_vulkansdk()
        -- either fails outright or reports link dirs that don't hold libvulkan.so. Without
        -- this fallback the rule links nothing and every vk* symbol stays undefined unless
        -- the user passes --ldflags="-lvulkan" by hand.
        local function fallback_syslink()
            if target:is_plat("linux") or target:is_plat("bsd") then
                target:add("syslinks", "vulkan", { public = true })
                return true
            end
            return false
        end

        local vulkansdk = find_vulkansdk()
        if vulkansdk then
            target:add("runenvs", "PATH", vulkansdk.bindir)

            local suffix
            if target:is_plat("windows") then
                suffix = ".lib"
            elseif target:is_plat("macosx") then
                suffix = ".dylib"
            else
                suffix = ".so"
            end

            local util = target:is_plat("windows") and "vulkan-1" or "vulkan"

            if target:is_plat("macosx") then
                target:add("rpathdirs", vulkansdk.linkdirs[1], { public = true })
                target:add("ldflags", "-Wl,-rpath," .. vulkansdk.linkdirs[1], { force = true, public = true })
            end

            if not find_library(util, vulkansdk.linkdirs) then
                if not fallback_syslink() then
                    wprint(format("The Vulkan loader %s for %s is not found!", util, target:arch()))
                end
                return
            end

            local lib_name = target:is_plat("windows") and util or "lib" .. util
            local lib_path = path.join(vulkansdk.linkdirs[1], lib_name .. suffix)
            target:add("links", lib_path, { public = true })
        else
            fallback_syslink()
        end
    end)
rule_end()

-- Link the Slang compiler library bundled with the Vulkan SDK (prebuilt), so
-- vri-shaderc drives Slang in-process without building Slang from source.
rule("slangsdk")
    on_config(function (target)
        import("detect.sdks.find_vulkansdk")
        local sdk = find_vulkansdk()
        if not sdk then
            wprint("Vulkan SDK not found; vri-shaderc needs its bundled Slang")
            return
        end
        if sdk.includedirs and sdk.includedirs[1] then
            target:add("includedirs", path.join(sdk.includedirs[1], "slang"))
        end
        if sdk.linkdirs and sdk.linkdirs[1] then
            target:add("linkdirs", sdk.linkdirs[1])
        end
        target:add("links", "slang")
        if sdk.bindir then
            target:add("runenvs", "PATH", sdk.bindir) -- find slang.dll at runtime
        end
    end)
    -- copy the Slang DLLs next to the tool so it runs without the SDK on PATH
    -- (slang.dll + its on-demand helpers: slang-glslang for spirv-opt, etc.)
    after_build(function (target)
        if not target:is_plat("windows") then
            return
        end
        import("detect.sdks.find_vulkansdk")
        local sdk = find_vulkansdk()
        if sdk and sdk.bindir then
            for _, dll in ipairs(os.files(path.join(sdk.bindir, "slang*.dll"))) do
                os.cp(dll, target:targetdir())
            end
        end
    end)
rule_end()

add_rules("mode.debug", "mode.release")
add_rules("plugin.vsxmake.autoupdate")
add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode", lsp = "clangd"})
add_rules("clangd.config")

-- add repositories
add_repositories("my-xmake-repo https://github.com/zzxzzk115/xmake-repo.git backup")
-- local package overrides (prebuilt Slang 2026.11 for vri-shaderc; see the package).
-- Anchored to os.scriptdir() so it still resolves when VRI is included as a subproject.
add_repositories("vri-local-repo " .. path.join(os.scriptdir(), "xmake", "xmake-repo"))

-- vri-shaderc links a prebuilt Slang that can emit tessellation SPIR-V (the Vulkan
-- SDK's 2025.11 crashes on hull shaders). Local-developer tool only (tools are off
-- in CI), so requiring it here is gated on the tools option.
if has_config("vri_build_tools") then
    add_requires("slang-prebuilt 2026.11")
end

-- shared registries/helpers
includes(path.join(os.scriptdir(), "xmake", "examples.lua"))

-- tasks (e.g. `xmake shaders` to compile test .slang -> SPIR-V headers)
includes(path.join(os.scriptdir(), "xmake", "tasks", "shaders.lua"))
includes(path.join(os.scriptdir(), "xmake", "tasks", "examples.lua"))

-- include external libraries
includes(path.join(os.scriptdir(), "external"))

-- include source
includes(path.join(os.scriptdir(), "source"))

-- host tools (vri-shaderc, ...)
if has_config("vri_build_tools") then
    includes(path.join(os.scriptdir(), "tools"))
end

-- include tests
if has_config("vri_build_tests") then
    includes(path.join(os.scriptdir(), "tests"))
end

-- if build examples, then include examples
if has_config("vri_build_examples") then
    includes(path.join(os.scriptdir(), "examples"))
end

-- Emscripten end-to-end tests (tests/wasm/): headless triangles that render via the
-- GL (WebGL2) and/or WebGPU backends and report a pass/fail pixel check. Browser-run
-- via emrun. Wasm-only, so they live outside the desktop `tests` target but with it.
if is_plat("wasm") then
    includes(path.join(os.scriptdir(), "tests", "wasm"))
end