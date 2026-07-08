# VRI

[![CI](https://github.com/zzxzzk115/VRI/actions/workflows/ci.yml/badge.svg)](https://github.com/zzxzzk115/VRI/actions/workflows/ci.yml)

**VRI** is a cross-platform **Render Hardware Interface**: one explicit, modern rendering API
(command buffers, descriptor sets, explicit synchronization) that runs on many graphics backends
and platforms. Write your renderer once — VRI runs it on Vulkan, Direct3D 12, Metal, WebGPU, and
OpenGL, on desktop, mobile/embedded, and the web.

Think "[NRI](https://github.com/NVIDIA-RTX/NRI), but with more graphics APIs and more platforms."

## Highlights

- **One API, every backend.** A single explicit, low-level surface (à la Vulkan/D3D12) that maps
  onto each platform's native graphics API.
- **C ABI core + header-only C++23 wrapper.** Use it from C, or from modern C++ with a
  `std::expected`-based RAII layer (`vri/vri.hpp`).
- **Honest capabilities, never silent.** Every optional feature is queryable. If a backend can't do
  something, it tells you — it never silently no-ops.
- **Portable by default.** Quirks that aren't real capability differences (row-pitch alignment,
  integer vertex attributes, coordinate conventions, …) are absorbed by VRI, so the same code
  produces the same image everywhere.
- **One diagnostics sink.** A single message callback receives both VRI's validation messages and
  each backend's native debug output.
- **Built-in validation layer.** Opt-in precondition/lifecycle checks; zero cost when off.
- **Shaders authored in [Slang](https://github.com/shader-slang/slang)**, compiled offline to each
  target's native bytecode.

## Supported platforms & backends

| Backend | Platforms |
| --- | --- |
| **Vulkan** | Windows, Linux (and macOS via MoltenVK) |
| **Direct3D 12** | Windows |
| **Metal** | macOS (Apple Silicon) |
| **WebGPU** | Windows, Linux, macOS, **Web** |
| **OpenGL** | Windows, Linux (and macOS, capped at 4.1) |
| **WebGL 2** | **Web** |
| **OpenGL ES** | Linux / embedded (e.g. Raspberry Pi) |
| **Software (CPU)** | Windows, Linux, macOS — anywhere with a software Vulkan ICD |

On the web, VRI runs through **two** backends — WebGPU and WebGL 2 — both via Emscripten. The
desktop application requests `Auto` and VRI picks the best backend for the platform.

**Software (CPU) rendering** (`VriGraphicsAPI_Software`) runs the Vulkan backend on a software
Vulkan implementation — no GPU required — so rendering works on headless CI, minimal VMs, and
machines with no usable GPU. It has the full Vulkan feature set at CPU speed, and is the last
`Auto` fallback (only reached when no GPU backend can create a device). It needs a software Vulkan
ICD present: **[SwiftShader](https://github.com/google/swiftshader)** (cross-platform) or **Mesa
lavapipe** (`apt install mesa-vulkan-drivers` — the usual Linux-CI choice). See
[DETAIL.md](DETAIL.md#software-cpu-rendering) for how VRI selects it.

## Feature support

How each feature is supported on each backend:

**✅ Native** &nbsp;·&nbsp; **🟡 Emulated / Simulated** &nbsp;·&nbsp; **⬜ Planned** (the API supports it; not yet in VRI) &nbsp;·&nbsp; **❌ Unsupported by the API**

| Feature | Vulkan | Direct3D 12 | Metal | WebGPU | OpenGL | WebGL 2 | OpenGL ES |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Core rendering & swapchain | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Depth / stencil, blending, MRT | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MSAA + resolve | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Mipmaps, instancing, cubemaps | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Compute shaders | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Push constants | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 |
| Bindless / descriptor indexing | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
| Geometry shaders | ✅ | ⬜ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Tessellation | ✅ | ⬜ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Mesh shaders | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Ray tracing | ✅ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ❌ |
| Opacity micromaps | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Variable rate shading | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Conservative rasterization | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Fragment-shader barycentrics | ✅ | ✅ | ⬜ | ❌ | ❌ | ❌ | ❌ |
| Custom sampler border color | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Subgroup / wave operations | ✅ | ✅ | ⬜ | ❌ | ❌ | ❌ | ❌ |
| Multiview (single-pass stereo) | ✅ | ⬜ | ✅ | ❌ | ✅ | ❌ | ✅ |
| External memory / interop (CUDA) | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GPU timestamp queries | ✅ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ❌ |
| Occlusion queries | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Pipeline-statistics queries | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Calibrated GPU+CPU timestamps | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Pipeline cache | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Indirect draw count (GPU-driven) | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Video memory budget | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Clear storage buffer / texture | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Built-in Dear ImGui renderer | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

Per-backend implementation notes — the fallbacks, caveats, and the native extension each feature
uses — are in **[DETAIL.md](DETAIL.md)**.

## Building

VRI uses [xmake](https://xmake.io).

```sh
# configure (enable the backends you want)
xmake f --vri_backend_vulkan=y --vri_backend_wgpu=y

# build everything
xmake

# run the tests
xmake run vri-tests
```

### Web (Emscripten)

```sh
xmake f -p wasm          # WebGPU + WebGL 2 are both enabled by default
xmake build example-cube
xmake run example-cube   # hosts the page in your browser
```

## Examples

Every example runs across the backends and the web, with a Dear ImGui control panel (rendered
through VRI's built-in ImGui renderer, `ext/vri_ext_imgui.h`). Backend is auto-selected; override on desktop with
`VRI_API=vulkan|webgpu|opengl|d3d12|metal|software`, or in the browser with `?backend=webgpu|webgl`.

```sh
xmake run example-triangle            # hello triangle
xmake run example-cube                # textured, depth-tested, animated cube
xmake run example-instancing          # a field of textured cubes
xmake run example-pbrbasic            # PBR metallic-roughness sphere grid
xmake run example-pushconstants       # push constants across backends
xmake run example-normalmapping       # tangent-space normal mapping
xmake run example-computeshader       # compute writes a plasma to a storage image
xmake run example-offscreen           # render-to-texture + multiple render targets
xmake run example-descriptorindexing  # bindless texturing (with fallback)
xmake run example-shadowmapping       # directional-light PCF shadow mapping
xmake run example-msaa                # 4x MSAA, toggle to see the edges go jagged
xmake run example-cubemap             # skybox + reflective chrome sphere
xmake run example-stenciloutline      # two-pass stencil silhouette outline
xmake run example-deferred            # deferred shading with a G-buffer + many lights
xmake run example-rayquery            # ray-traced hard shadows (HW ray query / compute fallback)
xmake run example-raytracing          # ray-traced glTF model
xmake run example-pathtracer          # progressive path tracer with global illumination
```

A console-only example (no window/UI) demonstrates GPU timing:

```sh
xmake run example-profiler            # GPU timestamp profiler (VRI_API=vulkan|d3d12)
```

One example is opt-in because it needs an extra SDK:

```sh
# VRI <-> CUDA external-memory interop: VRI and a CUDA kernel share the same GPU buffer and
# timeline. Needs the CUDA Toolkit (xmake's `cuda` package fetches/installs it if absent).
xmake f --vri_build_cuda_interop=y && xmake run example-cuda-interop
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the one-time hook setup, the clang-format convention
(pinned to 20.1.0), and commit style. TL;DR: run `scripts/setup-hooks.sh` (or
`scripts\setup-hooks.ps1`) once — it enables a pre-commit hook and provisions the pinned
clang-format, so your formatting stays CI-clean automatically.

## License

VRI is released under the [MIT License](LICENSE). The bundled FlightHelmet model is CC0, from the
Khronos glTF sample models.
