target("example-instancing")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_deps("vri")
    add_packages("libsdl3")

    -- for the shared Slang-compiled shader headers under tests/shaders
    add_includedirs("$(projectdir)")

    add_files("main.cpp")

    -- reuse the cube example's texture (no asset duplication); copy next to the exe
    after_build(function (target)
        os.cp(path.join(os.scriptdir(), "..", "cube", "assets", "metalplate01_rgba.ktx"), target:targetdir())
    end)
target_end()
