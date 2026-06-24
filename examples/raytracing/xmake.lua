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
    add_packages("libsdl3")

    -- Default model: the FlightHelmet glTF, vendored under assets/ (CC0, from the Khronos sample
    -- models) so the example runs out of the box. Pass a different model as argv[1] to override.
    local model = path.join(os.projectdir(), "assets", "models", "FlightHelmet", "glTF", "FlightHelmet.gltf")
    add_defines("VRI_GLTF_MODEL_PATH=\"" .. model:gsub("\\", "/") .. "\"")

    -- RT pipeline (raygen/SBT/TraceRays) exists only on Vulkan + D3D12, so this example is desktop
    -- only - it is not part of the wasm build (handled by the guard in examples/xmake.lua).
target_end()
