# VRI

[![CI](https://github.com/zzxzzk115/VRI/actions/workflows/ci.yml/badge.svg)](https://github.com/zzxzzk115/VRI/actions/workflows/ci.yml)

VRI is a cross-API **Render Hardware Interface** — "[NRI](https://github.com/NVIDIA-RTX/NRI) but with more graphics APIs and more platforms." It exposes one explicit, modern rendering API (command buffers, descriptor sets, explicit barriers/synchronization) over many backends, with full feature/extension query + enable.

## Design

- **C ABI core + header-only C++23 wrapper.** Public headers under `include/vri/` are C-clean (guarded by a pure-C translation unit in the tests); `include/vri/vri.hpp` adds a `std::expected`-based RAII layer.
- **Explicit modern surface, NRI-style.** `vriCreateDevice` → opaque `VriDevice*`; `vriGetInterface(name, size, out)` copies a backend-filled function table (the core interface, or optional/queryable extension interfaces like swapchain, interop, ray tracing).
- **Legacy backends emulate the explicit semantics** (descriptor flattening, barrier translation). The OpenGL backend uses the GLES3/WebGL2-compatible (non-DSA) subset so the same code drives desktop GL and WebGL2.
- **Standard coordinate system: Y-up clip space, depth [0,1], top-left origin** (D3D12/WebGPU/Metal convention). Vulkan is aligned with a negative-height viewport; OpenGL/WebGL flips clip-space Y in-shader (SPIRV-Cross `flip_vert_y`) so the same convention holds without `glClipControl` (absent on GLES/WebGL2).
- **VRI Validation layer** (NRI/Vulkan-style), enabled by `enableValidation`: a per-device wrapper that checks preconditions (capability gating, command-buffer lifecycle, render-pass scope) and reports via the message callback, then forwards to the backend. Off = the raw backend table (zero cost).
- **Explicit capability degradation, never silent.** Unsupported features are reported through `VriDeviceDesc` (e.g. `hasComputeShader`) and the relevant `Create*` returns `VriResult_Unsupported` — so e.g. compute on WebGL2 fails loudly, it doesn't no-op.
- **Shaders authored in [Slang](https://github.com/shader-slang/slang)**, compiled offline (`tools/vri-shaderc`) to per-target output — SPIR-V (Vulkan, and OpenGL/WebGL via SPIRV-Cross→GLSL/ESSL) and WGSL (WebGPU); MSL/DXIL later. The runtime consumes precompiled bytecode; it never links the Slang frontend.

## Backends

| Backend | Platform | Status |
| --- | --- | --- |
| Vulkan | desktop (Windows/Linux) | ✅ resources, explicit memory, descriptor sets, graphics + compute pipelines, command recording, swapchain (Win32), interop, feature query+enable |
| WebGPU (`webgpu-sdk`) | desktop **+ Web** (Emscripten) | ✅ resources, bind groups, graphics + compute pipelines, command recording, swapchain (Win32). Same backend runs native (wgpu-native) and **in-browser** (emdawnwebgpu) |
| OpenGL (desktop, 4.x) | desktop (Windows/Linux) | ✅ resources, descriptor sets (flattened), graphics + compute pipelines (SPIR-V→GLSL 430 via SPIRV-Cross), headless rendering |
| WebGL 2 (OpenGL ES 3.0) | Web (Emscripten) | ✅ the same GL backend on the GLES3/WebGL2 subset; runs in-browser. No compute (ES 3.0) — reported via `hasComputeShader` |
| OpenGL ES (native) / D3D11 / D3D12 / Metal | — | ⏳ planned |

> **The Web has two backends:** OpenGL→**WebGL2** and **WebGPU** (browser WebGPU via `webgpu-sdk`/emdawnwebgpu), both via Emscripten. The single WebGPU backend compiles for native and the browser — `wgpu-native` specifics are isolated behind `wgpu_native.h`, and the browser's async adapter/device/buffer-map promises are awaited by yielding with `emscripten_sleep` (`-sASYNCIFY`). Verified end-to-end in headless Chrome.

## Tested feature coverage

The same Slang source + the same VRI calls are verified to produce identical results across backends. Desktop tests (`vri-tests`, doctest) run on Vulkan + WebGPU + OpenGL; both Web backends are verified end-to-end in headless Chrome — WebGL2 (via SwiftShader) as an 11-probe program, and browser WebGPU as a render-to-texture readback **plus a compute (storage-buffer fill) check** — i.e. compute on the Web, which WebGL2 cannot do.

Covered: solid + Y-up coordinate orientation, UBO descriptors, sampled textures (separate texture + sampler), vertex buffers + indexed draw, depth test, stencil (write-then-test with front/back ops, masks, reference), alpha blending, back-face culling/winding, multiple render targets, instancing (`SV_InstanceID`), scissor, compute (storage-buffer fill), 4× MSAA + resolve (antialiased edges; Vulkan + WebGPU + desktop OpenGL via multisample-texture blit), and mipmaps (per-mip upload + mip-range view selection) — verified identical on Vulkan, WebGPU, and desktop OpenGL. **Legacy graphics stages** — geometry shaders (Vulkan + desktop OpenGL) and tessellation (hull/domain; Vulkan today, the OpenGL transpile is being hardened) — are **capability-gated**: `hasGeometryShader`/`hasTessellation` are reported honestly and a pipeline using a stage the backend lacks (WebGPU, WebGL2, …) fails with `VriResult_Unsupported` rather than silently dropping it. The validation layer and the explicit "compute unavailable" contract on WebGL2 are tested too.

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

# windowed triangle example (VRI_API=vulkan|webgpu)
xmake run example-triangle

# (re)compile the Slang test shaders to embedded SPIR-V/WGSL headers
xmake shaders
```

Dependencies are pulled via xmake from the `my-xmake-repo` repository (Vulkan headers + VMA, `webgpu-sdk`, libsdl3, doctest); the Vulkan SDK provides the Vulkan loader and the Slang compiler that `vri-shaderc` links.
