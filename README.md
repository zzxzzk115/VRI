# VRI

[![CI](https://github.com/zzxzzk115/VRI/actions/workflows/ci.yml/badge.svg)](https://github.com/zzxzzk115/VRI/actions/workflows/ci.yml)

VRI is a cross-API **Render Hardware Interface** — "[NRI](https://github.com/NVIDIA-RTX/NRI) but with more graphics APIs and more platforms." It exposes one explicit, modern rendering API (command buffers, descriptor sets, explicit barriers/synchronization) over many backends, with full feature/extension query + enable.

## Design

- **C ABI core + header-only C++23 wrapper.** Public headers under `include/vri/` are C-clean (guarded by a pure-C translation unit in the tests); `include/vri/vri.hpp` adds a `std::expected`-based RAII layer.
- **Explicit modern surface, NRI-style.** `vriCreateDevice` → opaque `VriDevice*`; `vriGetInterface(name, size, out)` copies a backend-filled function table (the core interface, or optional/queryable extension interfaces like swapchain, interop, ray tracing).
- **Legacy backends emulate the explicit semantics** (descriptor flattening, barrier translation). The OpenGL backend detects each context's version/feature tier and uses the highest path it supports — never a blanket downgrade: desktop GL 4.5+ takes the AZDO/modern route (direct state access, `glClipControl`, FBO cache, multisample textures), while GLES3/WebGL2 (and pre-4.5 desktop) fall back explicitly to the non-DSA subset. The same source drives both (see [docs/gl-backend-audit.md](docs/gl-backend-audit.md)).
- **Standard coordinate system: Y-up clip space, depth [0,1], top-left origin** (D3D12/WebGPU/Metal convention). Vulkan is aligned with a negative-height viewport; desktop GL 4.5+ matches it natively via `glClipControl(UPPER_LEFT, ZERO_TO_ONE)` (exact depth, no per-vertex work), while GLES/WebGL2 and pre-4.5 desktop flip clip-space Y in-shader (SPIRV-Cross `flip_vert_y`) since they have no `glClipControl`.
- **VRI Validation layer** (NRI/Vulkan-style), enabled by `enableValidation`: a per-device wrapper that checks preconditions (capability gating, command-buffer lifecycle, render-pass scope) and reports via the message callback, then forwards to the backend. Off = the raw backend table (zero cost).
- **One diagnostic sink for every backend.** A single `VriCallbackInterface` (set in `VriDeviceCreationDesc`) receives the VRI validation layer *and* each backend's native diagnostics — Vulkan debug-utils, D3D12 `ID3D12InfoQueue1`, OpenGL `KHR_debug`, WebGPU uncaptured errors — tagged with a `VriMessageSeverity`. The app sets the callback once and logs however it likes (the examples just print); nothing is swallowed into a platform-specific console.
- **Explicit capability degradation, never silent.** Unsupported features are reported through `VriDeviceDesc` (e.g. `hasComputeShader`) and the relevant `Create*` returns `VriResult_Unsupported` — so e.g. compute on WebGL2 fails loudly, it doesn't no-op.
- **Shaders authored in [Slang](https://github.com/shader-slang/slang)**, compiled offline (`tools/vri-shaderc`) to per-target output — SPIR-V (Vulkan, and OpenGL/WebGL via SPIRV-Cross→GLSL/ESSL) and WGSL (WebGPU); MSL/DXIL later. The runtime consumes precompiled bytecode; it never links the Slang frontend.

## Backends

| Backend | Platform | Status |
| --- | --- | --- |
| Vulkan | desktop (Windows/Linux) | ✅ resources, explicit memory, descriptor sets, graphics + compute pipelines, command recording, swapchain (Win32), interop, feature query+enable |
| WebGPU (`webgpu-sdk`) | desktop **+ Web** (Emscripten) | ✅ resources, bind groups, graphics + compute pipelines, command recording, swapchain (Win32). Same backend runs native (wgpu-native) and **in-browser** (emdawnwebgpu) |
| OpenGL (desktop, 4.x) | desktop (Windows/Linux) | ✅ resources, descriptor sets (flattened), graphics + compute pipelines (SPIR-V→GLSL via SPIRV-Cross), storage-image compute (image load/store + memory barriers), modern-GL fast paths (DSA, separate-format VAOs, base-vertex/instance, indirect draw/dispatch, immutable+persistent buffers — version/feature-gated), headless rendering + windowed swapchain present (Win32) |
| WebGL 2 (OpenGL ES 3.0) | Web (Emscripten) | ✅ the same GL backend on the GLES3/WebGL2 subset; runs in-browser, incl. canvas swapchain present. No compute (ES 3.0) — reported via `hasComputeShader` |
| Direct3D 12 | desktop (Windows) | ✅ resources, descriptor sets (root CBV/UAV + SRV/sampler tables), graphics + compute pipelines (Slang→DXBC, reflection-built input layout), depth-stencil, 4x MSAA + resolve, command recording, swapchain (Win32). Tests run on the WARP software adapter in CI |
| OpenGL ES (native) / Metal | — | ⏳ planned |

> **The Web has two backends:** OpenGL→**WebGL2** and **WebGPU** (browser WebGPU via `webgpu-sdk`/emdawnwebgpu), both via Emscripten. The single WebGPU backend compiles for native and the browser — `wgpu-native` specifics are isolated behind `wgpu_native.h`, and the browser's async adapter/device/buffer-map promises are awaited by yielding with `emscripten_sleep` (`-sASYNCIFY`). Verified end-to-end in headless Chrome.

## Tested feature coverage

The same Slang source + the same VRI calls are verified to produce identical results across backends. Desktop tests (`vri-tests`, doctest) run on Vulkan + WebGPU + OpenGL + Direct3D 12 (the latter on the WARP software adapter in CI); both Web backends are verified end-to-end in headless Chrome — WebGL2 (via SwiftShader) as a 13-probe program (incl. a canvas-swapchain present), and browser WebGPU as a render-to-texture readback **plus a compute (storage-buffer fill) check** — i.e. compute on the Web, which WebGL2 cannot do.

Covered: solid + Y-up coordinate orientation, UBO descriptors, sampled textures (separate texture + sampler), vertex buffers + indexed draw, depth test, stencil (write-then-test with front/back ops, masks, reference), alpha blending, back-face culling/winding, multiple render targets, instancing (`SV_InstanceID`), scissor, compute (storage-buffer fill), 4× MSAA + resolve (antialiased edges; **all backends** — multisample textures on VK/WebGPU/desktop-GL, multisample renderbuffers on WebGL2, blit/auto resolve), and mipmaps (per-mip upload + mip-range view selection) — verified identical on Vulkan, WebGPU, and desktop OpenGL, with WebGL2 MSAA verified in the headless E2E. **Legacy graphics stages** — geometry shaders and tessellation (hull/domain), both on **Vulkan + desktop OpenGL** — are **capability-gated**: `hasGeometryShader`/`hasTessellation` are reported honestly and a pipeline using a stage the backend lacks (WebGPU, WebGL2, …) fails with `VriResult_Unsupported` rather than silently dropping it. The validation layer and the explicit "compute unavailable" contract on WebGL2 are tested too.

**Modern-feature extensions** are exposed as separate queryable interfaces (`vriGetInterface`), registered by a backend only when the backing feature was granted at device creation — so absence is explicit (`Unsupported`), never a silent no-op. On Vulkan: **Variable Rate Shading** (`VRI_INTERFACE_VRS`, per-draw coarse shading rate, `VK_KHR_fragment_shading_rate`) and **mesh shaders** (`VRI_INTERFACE_MESHSHADER`, mesh/task pipelines + `CmdDrawMeshTasks`, `VK_EXT_mesh_shader`) are implemented and tested (each renders a triangle and reads it back; tests self-skip on adapters that lack the feature, e.g. CI software rasterizers). **ray tracing** (`VRI_INTERFACE_RAYTRACING`, `VK_KHR_acceleration_structure` + `VK_KHR_ray_tracing_pipeline`: BLAS/TLAS build, RT pipelines with raygen/miss/hit groups, shader-binding-table handles, `CmdTraceRays`, acceleration-structure descriptors). The RT test builds a BLAS+TLAS from a triangle and traces it into a storage image (center pixel hits → red, corners miss → black). And **opacity micromaps** (`VRI_INTERFACE_OMM`, `VK_EXT_opacity_micromap`: build a micromap, attach it to BLAS triangle geometry) — the OMM test marks the triangle fully transparent and confirms the ray passes through (the center pixel goes black instead of red). The validation layer wraps all these interfaces too, so handles are unwrapped correctly under `enableValidation` (backend-agnostically — the same wrappers serve every backend's table). The same queryable-interface model also covers **D3D12**: Variable Rate Shading (`RSSetShadingRate`), mesh shaders (`DispatchMesh`, pipeline-state-stream PSO), and **DXR ray tracing** (state objects, acceleration structures, shader tables, `DispatchRays`, TLAS-SRV descriptors) are implemented and tested there — the D3D12 RT test traces the same triangle into a storage image (center hits → red, corners miss → black), exactly like the Vulkan RT test. **opacity micromaps** (`VRI_INTERFACE_OMM`, DXR 1.2 — built through the AS-build path, attached to BLAS triangle geometry, `ALLOW_OPACITY_MICROMAPS` pipeline flag) are implemented and tested too (the same transparent-OMM-makes-the-triangle-vanish test as Vulkan). Mesh/RT shaders are Slang→DXIL (`sm_6_5` / `lib_6_3`); the single VRI RT/mesh/VRS/OMM interfaces drive both Vulkan and D3D12 unchanged. D3D12 OMM needs the DirectX Agility SDK headers (pulled via xmake when the D3D12 backend is built) and a DXR-1.2 runtime — on adapters/runtimes below Raytracing Tier 1.2 (e.g. WARP) VRI reports `hasOpacityMicromap=false` and the test self-skips, so the degradation is always explicit. **All four modern features (VRS, mesh shaders, ray tracing, opacity micromaps) now work on both Vulkan and D3D12.**

**Bindless textures** (descriptor indexing) are a core-API feature, not a separate interface: a descriptor range marked `VriDescriptorRange_PartiallyBound | VriDescriptorRange_VariableSized` becomes a runtime-sized array that the shader indexes dynamically. On Vulkan this maps to `descriptorIndexing` (partially-bound + variable-count + update-after-bind, enabled via `VriFeature_Bindless`); the test fills a few slots of a 64-entry `Texture2D[]` array and samples a UBO-selected index (→ the matching color). It self-skips where the adapter lacks descriptor indexing. The same `VriFeature_Bindless` will map to D3D12 `ResourceDescriptorHeap` / Metal argument buffers as those land — the portable concept stays in the core API (Vulkan-only escape hatches will instead use a `VRI_INTERFACE_VK_*` naming convention).

**Ray query** (inline ray tracing, `VriFeature_RayQuery`) works on **both Vulkan (`VK_KHR_ray_query`) and D3D12 (DXR 1.1 inline)** from one VRI code path: it reuses the acceleration-structure interface, compute pipelines, and the AS/storage-image descriptors — no ray-tracing pipeline, no shader binding table. The shared test builds a BLAS+TLAS and a **compute** shader traces rays inline against the TLAS (`RayQuery`/`TraceRayInline`), writing hit→red / miss→black; it self-skips where ray query is unavailable. AS creation is enabled by either `VriFeature_RayTracing` or `VriFeature_RayQuery`.

**Conservative rasterization** (`VriRasterizationDesc::conservativeRaster`, `VriDeviceDesc::hasConservativeRaster`) is a portable pipeline-state capability on Vulkan (`VK_EXT_conservative_rasterization`, always enabled when available) and D3D12 (`ConservativeRasterizationTier`); the cross-backend test draws a sub-pixel-thin triangle and confirms the conservative draw lights up strictly more pixels than the normal one. It's a capability (no opt-in feature), reported honestly and self-skipped where absent.

**Fragment-shader barycentrics** (`VriDeviceDesc::hasFragmentShaderBarycentric`) are likewise portable: Vulkan `VK_KHR_fragment_shader_barycentric` (enabled when available) and D3D12 SM6.1 `SV_Barycentrics`. The cross-backend test outputs the rasterizer barycentrics as color and checks the center pixel is `{0.5, 0.25, 0.25}` (they sum to 1).

**Custom sampler border color** (`VriSamplerDesc::useCustomBorderColor` + `customBorderColor`, `VriDeviceDesc::hasCustomBorderColor`): an arbitrary RGBA border for `ClampToBorder` addressing — Vulkan `VK_EXT_custom_border_color`, D3D12 native (`D3D12_SAMPLER_DESC::BorderColor`). The cross-backend test samples a blue texture far outside `[0,1]` with a custom green border and reads back green.

**Subgroup / wave operations** (`VriDeviceDesc::hasShaderWaveOps` + `subgroupSize`): Vulkan subgroups and D3D12 SM6.0 wave intrinsics, queried uniformly. The cross-backend test runs a compute shader where each lane adds 1 to `WaveActiveSum` and checks the result equals the reported `subgroupSize` (proving wave ops work and the size is consistent) — runs on WARP too.

## Platforms

Windows / Linux (Vulkan, WebGPU, OpenGL) and Web (Emscripten → WebGL2 **and** WebGPU), all built + tested in CI. macOS is **built + run in CI** (compile-verified; GPU backends pending real-device validation — CI has no MoltenVK ICD, and desktop GL caps at 4.1). Android / iOS are planned per backend.

## Building

VRI uses [xmake](https://xmake.io).

```sh
# configure (enable the backends you want)
xmake f --vri_backend_vulkan=y --vri_backend_wgpu=y

# build everything
xmake

# run the tests
xmake run vri-tests

# windowed examples (backend auto-picked; override with VRI_API=vulkan|webgpu|opengl|d3d12)
xmake run example-triangle
xmake run example-cube        # textured, depth-tested, animated-MVP rotating cube
xmake run example-instancing  # a field of textured cubes via per-instance matrices
xmake run example-pbrbasic    # PBR metallic-roughness sphere grid lit by 4 point lights
xmake run example-pushconstants # push constants across backends (VK/D3D12 native, GL/WebGPU emulated)
xmake run example-normalmapping # tangent-space normal mapping (color + normal map, orbiting light)
xmake run example-computeshader # compute writes a plasma to a storage image, then samples it
                                # (WebGL2 has no compute -> explicit fragment-shader fallback)
xmake run example-offscreen     # render-to-texture + MRT: a cube drawn into two offscreen color
                                # targets (albedo + face normals), then composited side by side

# (re)compile the Slang test shaders to embedded SPIR-V/WGSL headers
xmake shaders
```

The examples request `VriGraphicsAPI_Auto`, so `vriCreateDevice` picks the best backend for
the platform (Vulkan on desktop, WebGPU then WebGL in the browser) and reports the choice in
`VriDeviceDesc::graphicsAPI`. They share one host layer ([examples/common/example_app.h](examples/common/example_app.h))
across **desktop (SDL3) and the web (Emscripten)** — backend selection, windowing, the
present loop (while-loop on desktop, `emscripten_set_main_loop` in the browser), and headless
capture all live there, so each example only builds its resources + records a draw.

Building for the web (Emscripten):

```bash
xmake f -p wasm                  # WebGPU + WebGL2 are both default-on (no backend flags needed)
                                 #   add --emsdk=<path> only if xmake can't locate emscripten
xmake build example-cube         # or example-pbrbasic / example-pushconstants / ...
xmake run example-cube           # emrun hosts the .html in your browser
```

The page uses a small themed shell ([examples/common/shell.html](examples/common/shell.html))
with a **WebGPU/WebGL switch** and an in-page console; you can also force a backend with the
`?backend=webgpu` / `?backend=webgl` URL query (or `VRI_API=vulkan|webgpu|opengl|d3d12` on
desktop). Default (no query) is Auto — WebGPU first, WebGL2 fallback.

Two caveats:

- **After switching platform or backend flags** (e.g. `windows` → `wasm`, or toggling
  `--vri_backend_*`), wipe the wasm build once before rebuilding — otherwise a stale
  `vri_entry.cpp` (gated on `#if defined(VRI_BACKEND_*)`) can leave the lib with no backends
  and `vriCreateDevice` fails: `rm -rf build/wasm` (PowerShell: `Remove-Item -Recurse -Force
  build\wasm`), then reconfigure + build. Repeated builds without reconfiguring are fine.
- **After editing `examples/common/shell.html`**, force a relink (the shell file isn't a
  tracked dependency): `xmake build -r example-cube`.

Dependencies are pulled via xmake from the `my-xmake-repo` repository (Vulkan headers + VMA, `webgpu-sdk`, libsdl3, doctest); the Vulkan SDK provides the Vulkan loader and the Slang compiler that `vri-shaderc` links.
