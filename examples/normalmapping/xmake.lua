target("example-normalmapping")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_deps("vri")

    -- for the shared Slang-compiled shader headers under tests/shaders
    add_includedirs("$(projectdir)")

    add_files("main.cpp")

    if is_plat("wasm") then
        set_extension(".html")
        add_ldflags("-sASYNCIFY", "-sALLOW_MEMORY_GROWTH=1", "-sEXIT_RUNTIME=1", "-fexceptions", "--emrun", {force = true})
        add_ldflags("--shell-file=" .. path.join(os.scriptdir(), "..", "common", "shell.html"), {force = true}) -- Bevy-style page
        add_ldflags("--preload-file=" .. path.join(os.scriptdir(), "assets", "stonefloor01_color_rgba.ktx") .. "@/stonefloor01_color_rgba.ktx", {force = true})
        add_ldflags("--preload-file=" .. path.join(os.scriptdir(), "assets", "stonefloor01_normal_rgba.ktx") .. "@/stonefloor01_normal_rgba.ktx", {force = true})
    else
        add_packages("libsdl3")
        after_build(function (target)
            os.cp(path.join(os.scriptdir(), "assets", "stonefloor01_color_rgba.ktx"), target:targetdir())
            os.cp(path.join(os.scriptdir(), "assets", "stonefloor01_normal_rgba.ktx"), target:targetdir())
        end)
    end
target_end()
