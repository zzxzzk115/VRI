-- vri-shaderc: offline Slang -> SPIR-V/WGSL compiler that links the Slang compiler
-- library (drives it in-process), rather than shelling out to slangc. Uses a pinned
-- prebuilt Slang (2026.11) package: the Vulkan SDK's bundled Slang crashes emitting
-- tessellation hull shaders to SPIR-V. Local-developer tool (not built in CI).
target("vri-shaderc")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_packages("slang-prebuilt")

    add_files("main.cpp")

    -- The shaders task runs this exe directly (not via `xmake run`), so the Slang
    -- runtime must sit next to it. Copy the package's runtime (DLLs/.so + the
    -- standard module dir) into the target dir after build.
    after_build(function (target)
        local pkg = target:pkg("slang-prebuilt")
        if not pkg then
            return
        end
        local installdir = pkg:installdir()
        if not installdir then
            return
        end
        local bindir = path.join(installdir, "bin")
        if os.isdir(bindir) then
            for _, f in ipairs(os.files(path.join(bindir, "*"))) do
                os.trycp(f, target:targetdir())
            end
            for _, d in ipairs(os.dirs(path.join(bindir, "*"))) do
                os.trycp(d, target:targetdir())
            end
        end
    end)
target_end()
