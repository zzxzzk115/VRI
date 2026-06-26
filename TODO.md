# TODO

## Ray tracing via compute on WebGPU / OpenGL — `example-raytracing` & `example-pathtracer`

**Goal:** make the `raytracing` and `pathtracer` examples render on backends that have *compute but
no ray-tracing hardware* (WebGPU, desktop OpenGL) by tracing rays in a compute shader — exactly the
way `example-rayquery` already does with `tests/shaders/rt_software.slang`. Today these two examples
degrade explicitly (an "unsupported" message) on those backends.

This is what the **🟡 Simulated** entries for *Ray tracing* on WebGPU/OpenGL in the README feature
table refer to. `example-rayquery` already delivers it for a small triangle scene; the work below
extends the same idea to the glTF-model examples.

### Phase 1 — native WebGPU + desktop OpenGL (no web-asset work needed)

Both examples load the FlightHelmet glTF from disk on desktop regardless of backend, so this phase
needs **no** new asset infrastructure — only a CPU BVH + software-traversal shaders + descriptor
wiring. Mirror `example-rayquery`'s `useHw ? hardware : software` branch.

1. **CPU BVH builder** — add a small header (e.g. `examples/common/bvh.h`). A median/SAH-split
   BVH over the model's triangles; emit two storage buffers:
   - `BvhNode[]` — `{ float3 aabbMin; uint leftOrFirst; float3 aabbMax; uint countOrSecond; }`
     (16-byte-aligned; pack as two `float4` like `GltfVertex`/`Vertex` already do to avoid
     std430 stride surprises).
   - `triRef[]` — per leaf-triangle: `{ uint indexBase; uint geomNodeIndex; }` so the shader can
     fetch the 3 vertices via `vertices[indices[indexBase + k]]` and the material via
     `geometryNodes[geomNodeIndex]` — identical data to the hardware hit path.
   - Retain CPU copies of positions/indices: `gltf_model.h` currently drops its local `verts`/
     `indices` after upload. Add optional `std::vector<GltfVertex> cpuVerts; std::vector<uint32_t>
     cpuIndices;` (kept only when a flag asks) to feed the builder.

2. **Software shaders** — new Slang sources next to the existing ones:
   - `tests/shaders/rt_gltf_software.slang` — the software twin of `rt_gltf_rayquery.slang`:
     replace `RaytracingAccelerationStructure` + `RayQuery` with a stack-based BVH traversal
     (Möller–Trumbore per leaf triangle). Keep the shading body identical so the image matches.
   - `tests/shaders/rt_pathtrace_software.slang` — the software twin of `rt_pathtrace.slang`
     (same BVH traversal inside the bounce loop; keep accumulation/tonemap unchanged).
   - Add BVH-node + triRef SSBO bindings after the existing ranges. Regenerate headers with
     `xmake shaders` → produces `_spv.h` (desktop GL), `_wgsl.h` (WebGPU), `_dxbc.h`/`_dxil.h`.
     Add the new stems to the DXIL `dxilProfiles` map in `xmake/tasks/shaders.lua` if needed
     (`cs_6_5`).

3. **Wire up the examples** — in `examples/raytracing/main.cpp` and `examples/pathtracer/main.cpp`,
   add a `hasCompute && !hasRayQuery && !hasRtPipeline` software branch alongside the existing
   hardware branch (build BVH, create the extra storage buffers/views, select the `_wgsl`/`_spv`
   shader via `app.useWgsl`, extend the descriptor pool/set). The display pass + accumulation are
   unchanged.

4. **Verify** — run `VRI_API=webgpu xmake run example-raytracing` and `VRI_API=opengl ...` and
   confirm the image matches the hardware (Vulkan/D3D12) path; same for `example-pathtracer`.

### Phase 2 — browser (wasm)

Once Phase 1 works on native WebGPU, enable both examples in the **browser** build:

1. Remove the `if not is_plat("wasm")` guard in `examples/xmake.lua` for these two targets and the
   `#if defined(__EMSCRIPTEN__)` early-out stubs in both `main.cpp`.
2. **Preload the model + textures** into the Emscripten virtual FS (`--preload-file
   assets/models/FlightHelmet@/...` or a packaged `.data`), and make `gltf_model.h`'s tinygltf
   image loading work under wasm (it should, via stb_image; verify external `.bin`/image paths
   resolve inside the FS).
3. Build with `xmake f -p wasm`, confirm both run in headless Chrome (WebGPU) the same way the
   other web examples are validated.

Note: WebGL 2 / OpenGL ES have **no compute**, so ray tracing stays genuinely unsupported there
(explicit message) — that's correct, not a gap.
