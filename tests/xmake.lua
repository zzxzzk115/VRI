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
    add_files("test_rayquery_xbackend.cpp")   -- inline ray tracing (VK ray_query + DXR 1.1) in compute
    add_files("test_consrast_xbackend.cpp")   -- conservative rasterization (VK EXT + D3D12 tier)
    add_files("test_barycentric_xbackend.cpp")-- fragment-shader barycentrics (VK KHR + D3D12 SM6.1)
    add_files("test_bordercolor_xbackend.cpp")-- custom sampler border color (VK EXT + D3D12 native)
    add_files("test_wave_xbackend.cpp")       -- subgroup/wave ops (VK subgroups + D3D12 SM6.0)
    add_files("test_cube_xbackend.cpp")       -- examples/cube regression: MVP + sampled texture
    add_files("test_instancing_xbackend.cpp") -- per-instance vertex stream (instanced matrices)
    add_files("c_clean_check.c")
    if has_config("vri_backend_vulkan") then
        add_files("test_triangle_vk.cpp")
        add_files("test_core_phase2_vk.cpp")
        add_files("test_features_vk.cpp")
        add_files("test_vrs_vk.cpp")          -- variable rate shading (VK_KHR_fragment_shading_rate)
        add_files("test_meshshader_vk.cpp")   -- mesh shaders (VK_EXT_mesh_shader)
        add_files("test_raytracing_vk.cpp")   -- ray tracing (KHR accel struct + RT pipeline)
        add_files("test_bindless_vk.cpp")     -- bindless textures (descriptor indexing)
        add_files("test_omm_vk.cpp")          -- opacity micromaps (VK_EXT_opacity_micromap)
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
        add_files("test_msaa_d3d12.cpp")
        add_files("test_vrs_d3d12.cpp")       -- variable rate shading (RSSetShadingRate)
        add_files("test_meshshader_d3d12.cpp")-- mesh shaders (DispatchMesh, DXIL)
        add_files("test_raytracing_d3d12.cpp")-- DXR ray tracing (state object + AS + DispatchRays)
        add_files("test_omm_d3d12.cpp")       -- opacity micromap: explicit "unsupported" contract
    end

    add_tests("default")
target_end()
