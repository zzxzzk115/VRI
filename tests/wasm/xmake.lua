-- Emscripten end-to-end tests (the wasm counterpart of the desktop `tests`). Only
-- meaningful on -p wasm; built explicitly (set_default(false)) and graded by emrun.

-- WebGL2 (GL backend) headless triangle/feature suite.
if has_config("vri_backend_gl") then
target("vri-webgl-triangle")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_deps("vri")
    add_includedirs("$(projectdir)") -- shaders are included as "shaders/<owner>/X.h" from the repo root
    add_files("webgl_triangle.cpp")

    -- Emit a .html (+ .js + .wasm) so emrun can host it; --emrun pipes the page's
    -- stdout/exit code back to the terminal for a headless grade. WebGL2 + GLFW3
    -- port flags arrive transitively from the vri target (public ldflags).
    set_extension(".html")
    add_ldflags("-sEXIT_RUNTIME=1", "-sALLOW_MEMORY_GROWTH=1", "--emrun", {force = true})
    -- Emscripten 6.0.2 flipped GROWABLE_ARRAYBUFFERS to default-on under
    -- ALLOW_MEMORY_GROWTH; on Chrome 149/150 that path breaks this page (hang
    -- after resources-created). Opt out until the resizable-ArrayBuffer path is
    -- stable.
    add_ldflags("-sGROWABLE_ARRAYBUFFERS=0", {force = true})
    -- spirv-cross uses C++ exceptions; enable them at link, plus assertions so a
    -- runtime abort prints a real message instead of "Aborted(undefined)".
    add_ldflags("-fexceptions", "-sASSERTIONS=2", {force = true})
target_end()
end

-- Browser WebGPU (WebGPU backend) headless triangle.
if has_config("vri_backend_wgpu") then
target("vri-webgpu-triangle")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_deps("vri")
    add_includedirs("$(projectdir)") -- shaders are included as "shaders/<owner>/X.h" from the repo root
    add_files("webgpu_triangle.cpp")

    set_extension(".html")
    -- ASYNCIFY: the backend yields with emscripten_sleep while the browser resolves
    -- the async WebGPU adapter/device/buffer-map promises.
    add_ldflags("-sEXIT_RUNTIME=1", "-sALLOW_MEMORY_GROWTH=1", "-sASYNCIFY", "--emrun", {force = true})
    -- Same GROWABLE_ARRAYBUFFERS opt-out as vri-webgl-triangle (emscripten 6.0.2
    -- default flip); here the page died right after device-created.
    add_ldflags("-sGROWABLE_ARRAYBUFFERS=0", {force = true})
    add_ldflags("-fexceptions", "-sASSERTIONS=2", {force = true})
target_end()
end
