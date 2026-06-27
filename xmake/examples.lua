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
    {name = "pathtracer"}
}

local function _vri_example_enabled(example)
    return not (example.desktop_only and is_plat("wasm"))
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
