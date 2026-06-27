target("example-raytracing")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_deps("vri")

    -- for the shared Slang-compiled shader headers under tests/shaders
    add_includedirs("$(projectdir)")

    add_files("main.cpp")
    add_packages("imgui")    -- ImGui UI (drawn through VRI by examples/common/imgui_vri.h)
    add_packages("tinygltf") -- glTF model loading (examples/common/gltf_model.h), same lib as ../Vulkan

    -- Default model: the FlightHelmet glTF, vendored under assets/ (CC0, from the Khronos sample
    -- models) so the example runs out of the box. Pass a different model as argv[1] to override.
    local modelDir = path.join(os.projectdir(), "assets", "models", "FlightHelmet")
    if is_plat("wasm") then
        -- Web build: no hardware ray tracing on WebGPU, so it runs the compute-BVH software fallback.
        -- Bake the whole glTF tree into MEMFS via --preload-file and point the loader at that virtual
        -- path; ASYNCIFY for WebGPU's async device/adapter/map; emitted as .html for emrun.
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
