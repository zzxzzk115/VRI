target("example-msaa")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_deps("vri")

    -- for the shared Slang-compiled shader headers under tests/shaders
    add_includedirs("$(projectdir)")

    add_files("main.cpp")
    add_packages("imgui") -- ImGui UI (drawn through VRI by examples/common/imgui_vri.h)

    if is_plat("wasm") then
        -- Web build: GLFW (Emscripten port) for windowing, ASYNCIFY for WebGPU's async
        -- device/adapter/map. Emitted as a .html so emrun can host it. No assets to preload.
        set_extension(".html")
        add_ldflags("-sASYNCIFY", "-sALLOW_MEMORY_GROWTH=1", "-sEXIT_RUNTIME=1", "-fexceptions", "--emrun", {force = true})
        add_ldflags("--shell-file=" .. path.join(os.scriptdir(), "..", "common", "shell.html"), {force = true}) -- Bevy-style page
    else
        add_packages("libsdl3")
    end
target_end()
