target("example-cube")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_deps("vri")
    add_packages("libsdl3")

    -- for the shared Slang-compiled shader headers under tests/shaders
    add_includedirs("$(projectdir)")

    add_files("main.cpp")

    -- copy the .ktx asset next to the executable (loaded via SDL_GetBasePath)
    after_build(function (target)
        os.cp(path.join(os.scriptdir(), "assets", "metalplate01_rgba.ktx"), target:targetdir())
    end)
target_end()
