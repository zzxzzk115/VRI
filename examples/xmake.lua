-- VRI examples. Web-capable unless noted: SDL3 windowing on desktop, the page canvas on wasm; the
-- shared host scaffolding lives in examples/common/example_app.h.
includes("../xmake/examples.lua")

vri_include_examples()

-- Aggregate compile gate: every example target is set_default(false) (so a bare `xmake build`
-- skips them), which leaves CI with no single handle to compile them all. `xmake build vri-examples`
-- pulls in each enabled example via add_deps, giving CI one command to enforce that shader headers
-- and the public API still compile against every example. Phony -> nothing links for the target
-- itself; it only forces its dependencies to build.
target("vri-examples")
    set_kind("phony")
    set_default(false)
    for _, _example_target in ipairs(vri_get_examples()) do
        add_deps(_example_target)
    end
target_end()
