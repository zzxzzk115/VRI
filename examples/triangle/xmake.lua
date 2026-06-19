target("example-triangle")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_deps("vri")
    add_packages("libsdl3")

    -- for the shared Slang-compiled SPIR-V header under tests/shaders
    add_includedirs("$(projectdir)")

    add_files("main.cpp")
target_end()
