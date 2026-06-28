-- example-profiler: a minimal headless GPU profiler built on VRI's query interface.
-- Console-only (no windowing/UI), so it needs none of the SDL3/ImGui example scaffolding.

target("example-profiler")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_deps("vri")
    add_files("main.cpp")
target_end()
