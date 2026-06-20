-- Prebuilt Slang compiler release (shader-slang/slang).
--
-- vri-shaderc links the Slang compiler in-process. The Vulkan SDK's bundled Slang
-- (2025.11) crashes emitting tessellation hull shaders to SPIR-V
-- ("unimplemented: BuiltinCast in spirv-emit"); 2026.11 fixes it. This package
-- pins a known-good prebuilt so shader regeneration is reproducible. Tools are not
-- built in CI (vri_build_tools=n), so this is a local-developer dependency only.
package("slang-prebuilt")
    set_kind("library")
    set_homepage("https://github.com/shader-slang/slang")
    set_description("Slang shading language compiler (prebuilt release).")
    set_license("Apache-2.0")

    if is_plat("windows") and is_arch("x64", "x86_64") then
        add_urls("https://github.com/shader-slang/slang/releases/download/v$(version)/slang-$(version)-windows-x86_64.zip")
        add_versions("2026.11", "e070e239290108dd7a83094cf72fb94ae753b2d5331adba76874bf17dd5c8b6a")
    elseif is_plat("linux") and is_arch("x86_64") then
        add_urls("https://github.com/shader-slang/slang/releases/download/v$(version)/slang-$(version)-linux-x86_64.zip")
        add_versions("2026.11", "d83ef21fc08d2c035ed15b5ce7541f27de8ec8d1db42b27c72bdc1567e5c4b37")
    end

    on_install(function (package)
        os.cp("include/*.h", package:installdir("include"))
        os.cp("lib/*", package:installdir("lib"))
        -- Ship the whole runtime (DLLs/.so + slangc + the standard module dir) so
        -- the linked slang resolves its dependencies and shader regen just works.
        os.cp("bin/*", package:installdir("bin"))
        package:add("links", "slang")
        package:addenv("PATH", "bin")
    end)
