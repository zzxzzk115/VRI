-- VRI examples. Web-capable unless noted: SDL3 windowing on desktop, the page canvas on wasm; the
-- shared host scaffolding lives in examples/common/example_app.h.
includes("triangle")
includes("cube")
includes("instancing")
includes("pbrbasic")
includes("pushconstants")
includes("normalmapping")
includes("computeshader")
includes("offscreen")
includes("descriptorindexing")
includes("shadowmapping")
includes("msaa")
includes("cubemap")
includes("stenciloutline")
includes("rayquery")
includes("deferred")
-- RT pipeline (raygen/SBT/TraceRays) is Vulkan + D3D12 only -> desktop, not part of the wasm build.
-- pathtracer uses inline ray query (Metal/Vulkan/D3D12). Both desktop only.
if not is_plat("wasm") then
    includes("raytracing")
    includes("pathtracer")
end
