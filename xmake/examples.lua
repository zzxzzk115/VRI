-- Shared example registry. `examples/xmake.lua` includes targets from this list,
-- and `xmake examples` reads the same list to decide what to run.

local _vri_examples = {
    {name = "triangle"},
    {name = "cube"},
    {name = "instancing"},
    {name = "pbrbasic"},
    {name = "pushconstants"},
    {name = "normalmapping"},
    {name = "computeshader"},
    {name = "offscreen"},
    {name = "descriptorindexing"},
    {name = "shadowmapping"},
    {name = "msaa"},
    {name = "cubemap"},
    {name = "stenciloutline"},
    {name = "rayquery"},
    {name = "deferred"},
    -- raytracing/pathtracer run everywhere: hardware RT pipeline / ray query on Vulkan/D3D12/Metal,
    -- and a compute-BVH software fallback on OpenGL and WebGPU (wasm).
    {name = "raytracing"},
    {name = "pathtracer"},
    -- Desktop-only feature demos: these exercise capabilities the web backends
    -- (WebGPU / WebGL 2) don't expose, so they are excluded from the wasm build.
    {name = "conservativeraster", desktop_only = true},
    {name = "bordercolor", desktop_only = true},
    {name = "geometryshader", desktop_only = true},
    {name = "tessellation", desktop_only = true},
    {name = "barycentrics", desktop_only = true},
    {name = "vrs", desktop_only = true},
    {name = "queries", desktop_only = true},
    -- Wave/subgroup visualization: wave intrinsics on Vulkan/D3D12, explicit compute or
    -- fragment fallbacks elsewhere - so it runs on the web too.
    {name = "waveops"},
    -- Headless GPU profiler built on the timestamp query interface (console-only, no UI).
    {name = "profiler", desktop_only = true},
    -- CUDA <-> VRI external-memory interop. Desktop-only, and opt-in via
    -- --vri_build_cuda_interop=y because it needs the CUDA Toolkit (nvcc + cudart).
    {name = "cuda_interop", desktop_only = true, requires_cuda = true},
    -- OpenXR single-pass-stereo demo. Desktop-only, opt-in via --vri_build_openxr=y: it links
    -- the OpenXR loader and needs an OpenXR runtime (HMD / simulator) to actually run.
    {name = "openxr", desktop_only = true, requires_openxr = true}
}

local function _vri_example_enabled(example)
    if example.desktop_only and is_plat("wasm") then
        return false
    end
    if example.requires_cuda and not has_config("vri_build_cuda_interop") then
        return false
    end
    if example.requires_openxr and not has_config("vri_build_openxr") then
        return false
    end
    return true
end

local function _vri_example_target(example)
    return example.target or ("example-" .. example.name)
end

function vri_get_examples()
    local out = {}
    for _, example in ipairs(_vri_examples) do
        if _vri_example_enabled(example) then
            table.insert(out, _vri_example_target(example))
        end
    end
    return out
end

function vri_include_examples()
    for _, example in ipairs(_vri_examples) do
        if _vri_example_enabled(example) then
            includes(path.join(os.projectdir(), "examples", example.name))
        end
    end
end
