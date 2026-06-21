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
    add_files("test_validation.cpp")          -- VRI Validation layer catches misuse
    add_files("test_descriptor_xbackend.cpp") -- runs on whichever backends are enabled
    add_files("test_texture_xbackend.cpp")    -- sampled-texture (separate texture+sampler) parity
    add_files("test_vertex_xbackend.cpp")     -- vertex-buffer + indexed-draw parity
    add_files("test_depth_xbackend.cpp")      -- depth attachment + depth-test parity
    add_files("test_blend_xbackend.cpp")      -- alpha-blending parity
    add_files("test_cull_xbackend.cpp")       -- back-face culling + winding parity
    add_files("test_mrt_xbackend.cpp")        -- multiple render targets parity
    add_files("test_instance_xbackend.cpp")   -- instanced draw + SV_InstanceID parity
    add_files("test_baseoffset_xbackend.cpp")  -- base-vertex parity (vertexOffset honored, not silently dropped)
    add_files("test_indirect_xbackend.cpp")    -- indirect draw (CmdDrawIndirect reads args from a buffer)
    add_files("test_scissor_xbackend.cpp")    -- scissor-rect clipping parity
    add_files("test_stencil_xbackend.cpp")    -- stencil write/test (front/back ops, masks, ref)
    add_files("test_msaa_xbackend.cpp")       -- 4x MSAA + resolve (antialiased edges)
    add_files("test_mipmap_xbackend.cpp")     -- per-mip upload + mip-range view sampling
    add_files("test_compute_xbackend.cpp")    -- compute pipeline + dispatch + storage buffer
    add_files("test_geometry_xbackend.cpp")   -- legacy geometry shader (gated where absent)
    add_files("test_tessellation_xbackend.cpp") -- legacy tessellation hull/domain (gated where absent)
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
    if has_config("vri_backend_d3d12") then
        add_files("test_device_d3d12.cpp")
        add_files("test_clear_d3d12.cpp")
        add_files("test_triangle_d3d12.cpp")
        add_files("test_vertex_d3d12.cpp")
        add_files("test_ubo_d3d12.cpp")
        add_files("test_texture_d3d12.cpp")
        add_files("test_depth_d3d12.cpp")
    end

    add_tests("default")
target_end()
