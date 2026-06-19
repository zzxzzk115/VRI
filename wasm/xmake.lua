-- WebGL (Emscripten) end-to-end target. Only meaningful on -p wasm.

target("vri-webgl-triangle")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_deps("vri")
    add_includedirs("$(projectdir)/tests") -- for "shaders/triangle_spv.h"
    add_files("webgl_triangle.cpp")

    -- Emit a .html (+ .js + .wasm) so emrun can host it; --emrun pipes the page's
    -- stdout/exit code back to the terminal for a headless grade. WebGL2 + GLFW3
    -- port flags arrive transitively from the vri target (public ldflags).
    set_extension(".html")
    add_ldflags("-sEXIT_RUNTIME=1", "-sALLOW_MEMORY_GROWTH=1", "--emrun", {force = true})
    -- spirv-cross uses C++ exceptions; enable them at link, plus assertions so a
    -- runtime abort prints a real message instead of "Aborted(undefined)".
    add_ldflags("-fexceptions", "-sASSERTIONS=2", {force = true})
target_end()
