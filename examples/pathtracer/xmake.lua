target("example-pathtracer")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_deps("vri")
    add_includedirs("$(projectdir)") -- shared Slang-compiled shader headers under tests/shaders

    add_files("main.cpp")
    add_packages("imgui")
    add_packages("tinygltf") -- glTF model loading (examples/common/gltf_model.h)

    -- Default model: the vendored FlightHelmet glTF (assets/, CC0). Pass argv[1] to override.
    local modelDir = path.join(os.projectdir(), "assets", "models", "FlightHelmet")
    if is_plat("wasm") then
        -- Web build: WebGPU has no ray query, so it runs the compute-BVH software megakernel. Bake the
        -- glTF tree into MEMFS and point the loader at the virtual path; ASYNCIFY for WebGPU's async ops.
        set_extension(".html")
        add_defines("VRI_GLTF_MODEL_PATH=\"/FlightHelmet/glTF/FlightHelmet.gltf\"")
        add_ldflags("-sASYNCIFY", "-sALLOW_MEMORY_GROWTH=1", "-sEXIT_RUNTIME=1", "-fexceptions", "--emrun", {force = true})
        add_ldflags("--shell-file=" .. path.join(os.scriptdir(), "..", "common", "shell.html"), {force = true}) -- Bevy-style page
        add_ldflags("--preload-file=" .. modelDir .. "@/FlightHelmet", {force = true})
    else
        add_packages("libsdl3")
        local model = path.join(modelDir, "glTF", "FlightHelmet.gltf")
        add_defines("VRI_GLTF_MODEL_PATH=\"" .. model:gsub("\\", "/") .. "\"")
    end
target_end()
