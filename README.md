# VRI

[![CI](https://github.com/zzxzzk115/VRI/actions/workflows/ci.yml/badge.svg)](https://github.com/zzxzzk115/VRI/actions/workflows/ci.yml)

VRI is a cross-API **Render Hardware Interface** — "[NRI](https://github.com/NVIDIA-RTX/NRI) but with more graphics APIs and more platforms." It exposes one explicit, modern rendering API (command buffers, descriptor sets, explicit barriers/synchronization) over many backends, with full feature/extension query + enable.

## Design

- **C ABI core + header-only C++23 wrapper.** Public headers under `include/vri/` are C-clean (guarded by a pure-C translation unit in the tests); `include/vri/vri.hpp` adds a `std::expected`-based RAII layer.
- **Explicit modern surface, NRI-style.** `vriCreateDevice` → opaque `VriDevice*`; `vriGetInterface(name, size, out)` copies a backend-filled function table (the core interface, or optional/queryable extension interfaces like swapchain, interop, ray tracing).
- **Legacy backends emulate the explicit semantics** (deferred command recording, descriptor flattening, barrier translation) — those land later.
- **Standard coordinate system: Y-up clip space, depth [0,1], top-left origin** (D3D12/WebGPU/Metal convention). Vulkan is aligned with a negative-height viewport; OpenGL/WebGL flips clip-space Y in-shader (SPIRV-Cross `flip_vert_y`) so the same convention holds without `glClipControl` (absent on GLES/WebGL2).
- **Shaders authored in [Slang](https://github.com/shader-slang/slang)**, compiled offline (`tools/vri-shaderc`) to per-target output — SPIR-V (Vulkan), WGSL (WebGPU), and later GLSL/MSL/DXIL. The runtime consumes precompiled bytecode; it never links the Slang frontend.

## Backends

| Backend | Status |
| --- | --- |
| Vulkan | ✅ device, resources, explicit memory, descriptor sets, pipelines, command recording, swapchain (Win32), interop, feature/extension query+enable |
| WebGPU (wgpu-native) | ✅ device, resources, bind groups, pipelines, command recording, swapchain (Win32) |
| OpenGL (desktop, 4.x) | ✅ device, resources, descriptor sets (flattened), pipelines (SPIR-V→GLSL via SPIRV-Cross), headless command recording |
| WebGL 2 (OpenGL ES 3.0, Emscripten) | ✅ same GL backend on the GLES3/WebGL2 subset; triangle + UBO render in-browser |
| OpenGL ES (native) / D3D11 / D3D12 / Metal | ⏳ planned |

Cross-backend parity is verified by tests across **four backends** (Vulkan, WebGPU, OpenGL, WebGL): the same Slang source and the same VRI calls produce identical results — rendered output, descriptor-driven shading, and coordinate-system orientation. The WebGL backend is verified end-to-end in headless Chrome (WebGL2 via SwiftShader) in CI.

## Platforms

Windows / Linux / macOS / Android / Web (Emscripten) / iOS (rolling out per backend).

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
