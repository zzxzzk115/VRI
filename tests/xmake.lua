-- VRI tests. Includes a pure C translation unit so the public headers are
-- guaranteed to stay C-clean.

target("vri-tests")
    set_kind("binary")
    set_languages("cxx23")
    set_default(false)

    add_deps("vri")
    add_packages("doctest")

    add_files("test_main.cpp")
    add_files("test_api_usage.cpp")
    add_files("test_descriptor_xbackend.cpp") -- runs on whichever backends are enabled
    add_files("test_texture_xbackend.cpp")    -- sampled-texture (separate texture+sampler) parity
    add_files("test_vertex_xbackend.cpp")     -- vertex-buffer + indexed-draw parity
    add_files("test_depth_xbackend.cpp")      -- depth attachment + depth-test parity
    add_files("test_blend_xbackend.cpp")      -- alpha-blending parity
    add_files("test_coordsys_xbackend.cpp")   -- Y-up coordinate parity across backends
    add_files("c_clean_check.c")
    if has_config("vri_backend_vulkan") then
        add_files("test_triangle_vk.cpp")
        add_files("test_core_phase2_vk.cpp")
        add_files("test_features_vk.cpp")
    end
    if has_config("vri_backend_wgpu") then
        add_files("test_device_wgpu.cpp")
        add_files("test_triangle_wgpu.cpp")
    end
    if has_config("vri_backend_gl") then
        add_files("test_device_gl.cpp")
    end

    add_tests("default")
target_end()
