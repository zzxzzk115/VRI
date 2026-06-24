target("example-pathtracer")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_deps("vri")
    add_includedirs("$(projectdir)") -- shared Slang-compiled shader headers under tests/shaders

    add_files("main.cpp")
    add_packages("imgui")
    add_packages("tinygltf") -- glTF model loading (examples/common/gltf_model.h)
    add_packages("libsdl3")

    -- Default model: the vendored FlightHelmet glTF (assets/, CC0). Pass argv[1] to override.
    local model = path.join(os.projectdir(), "assets", "models", "FlightHelmet", "glTF", "FlightHelmet.gltf")
    add_defines("VRI_GLTF_MODEL_PATH=\"" .. model:gsub("\\", "/") .. "\"")

    -- Inline ray query (Metal/Vulkan/D3D12) - desktop only, excluded from the wasm build.
target_end()
