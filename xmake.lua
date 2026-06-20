-- set project name
set_project("VRI")

-- set project version
set_version("0.1.0")

-- set language version: C++ 23
set_languages("cxx23")

if is_plat("android") then
    set_toolchains("@ndk", {sdkver = "26"})
    -- Vulkan handles are 64-bit; keep arm64 only to avoid handle truncation on 32-bit.
    set_allowedarchs("arm64-v8a")
end

-- root ?
local is_root = (os.projectdir() == os.scriptdir())
set_config("root", is_root)
set_config("project_dir", os.scriptdir())

-- global options
option("vri_build_examples") -- build examples?
    set_default(not is_plat("android") and not is_plat("wasm"))
    set_showmenu(true)
    set_description("Enable VRI examples")
option_end()

option("vri_build_tests") -- build tests?
    set_default(not is_plat("android") and not is_plat("wasm"))
    set_showmenu(true)
    set_description("Enable VRI tests")
option_end()

option("vri_build_tools") -- build host tools (vri-shaderc)?
    set_default(not is_plat("android") and not is_plat("wasm"))
    set_showmenu(true)
    set_description("Enable VRI host tools (vri-shaderc)")
option_end()

-- backend options (MVP: Vulkan on by default; others enabled as they come online)
-- On wasm the only viable backend is GL (-> WebGL2); Vulkan is unavailable there.
option("vri_backend_vulkan")
    set_default(not is_plat("wasm"))
    set_showmenu(true)
    set_description("Enable the Vulkan backend")
option_end()

option("vri_backend_wgpu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable the WebGPU backend")
option_end()

option("vri_backend_gl")
    -- default on for wasm so `xmake f -p wasm` gives the WebGL backend out of the box
    set_default(is_plat("wasm"))
    set_showmenu(true)
    set_description("Enable the OpenGL / OpenGL ES / WebGL backend")
option_end()

option("vri_backend_d3d11")
    set_default(false)
    set_showmenu(true)
    set_description("Enable the Direct3D 11 backend (Windows only)")
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
        if is_host("windows") then
            os.cp(".clangd.win", ".clangd")
        else
            os.cp(".clangd.nowin", ".clangd")
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

        local vulkansdk = find_vulkansdk()
        if vulkansdk then
            target:add("runevs", "PATH", vulkansdk.bindir)

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
                wprint(format("The Vulkan loader %s for %s is not found!", util, target:arch()))
                return
            end

            local lib_name = target:is_plat("windows") and util or "lib" .. util
            local lib_path = path.join(vulkansdk.linkdirs[1], lib_name .. suffix)
            target:add("links", lib_path, { public = true })
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
-- local package overrides (prebuilt Slang 2026.11 for vri-shaderc; see the package)
add_repositories("vri-local-repo xmake/xmake-repo")

-- vri-shaderc links a prebuilt Slang that can emit tessellation SPIR-V (the Vulkan
-- SDK's 2025.11 crashes on hull shaders). Local-developer tool only (tools are off
-- in CI), so requiring it here is gated on the tools option.
if has_config("vri_build_tools") then
    add_requires("slang-prebuilt 2026.11")
end

-- tasks (e.g. `xmake shaders` to compile test .slang -> SPIR-V headers)
includes("xmake/tasks/shaders.lua")

-- include external libraries
includes("external")

-- include source
includes("source")

-- host tools (vri-shaderc, ...)
if has_config("vri_build_tools") then
    includes("tools")
end

-- include tests
if has_config("vri_build_tests") then
    includes("tests")
end

-- if build examples, then include examples
if has_config("vri_build_examples") then
    includes("examples")
end

-- Emscripten end-to-end targets: headless triangles that render via the GL (WebGL2)
-- and/or WebGPU backends and report a pass/fail pixel check. Browser-run via emrun.
if is_plat("wasm") then
    includes("wasm")
end