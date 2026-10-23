# VRI — Feature & backend details

Detailed notes on how VRI implements each feature across backends — the fallbacks, caveats, and
the native extension each path uses. For the at-a-glance support matrix, see the
[feature support table](README.md#feature-support) in the README. For the thread-safety contract
(which calls may run concurrently and where the caller must synchronize), see
[docs/threading.md](docs/threading.md).

## For NRI users

VRI shares NRI's explicit programming model, not its complete API. This comparison uses
[NRI revision `79d76a9`](https://github.com/NVIDIA-RTX/NRI/tree/79d76a9a78689ef58ee7962b1d8ed7e7a3efe18d)
(header version 190). NRI evolves independently; matching names do not establish matching signatures or capabilities.

| Area | Familiar model | VRI difference |
| --- | --- | --- |
| Dispatch and objects | Create a device, query a sized interface table, use opaque handles. | C declarations use `Vri*` types and pointer parameters; the C++ wrapper is optional. Function-table layout is not NRI-compatible. |
| Command allocation | CommandAllocator owns command-recording storage. | `CreateCommandAllocator` takes a device and queue type, rather than NRI's queue. |
| Resources and memory | Texture/Buffer descriptors, memory locations and explicit binding. | `CreateBuffer`/`CreateTexture` normally allocate and bind automatically. The explicit memory path queries a creation descriptor, closer to NRI's `Get*MemoryDesc2`; do not assume NRI's resource-query or allocation signatures. |
| Descriptor sets | PipelineLayout contains set/range declarations; pools allocate sets. | `AllocateDescriptorSets` has no variable-descriptor-count argument. `UpdateDescriptorRanges` takes one set and consecutive range indices, rather than NRI's array of update records. |
| Constants and binding | Bind the pipeline layout before descriptor sets and constants. | `CmdSetConstants` uses a push-constant declaration index, data and size; it is not NRI's `CmdSetRootConstants` descriptor with offset/bind point. |
| Barriers | Explicit before/after access, stages and texture layout. | Inspect `VriBarrierGroupDesc` and its supported fields; NRI flags and numeric enum values are not interchangeable. |
| Synchronization | Fence is a monotonic timeline used by queues and the host. | VRI submit/idle functions currently return `void`; NRI returns a result. A VRI Fence is not Vulkan's binary host fence. |
| Optional features | Query device capabilities and extension tables. | The supported surface and table boundaries differ. Gate each path on the returned VRI capabilities, not the NRI feature list. |

The normal sequence remains device -> core interface -> queue/allocator -> resources/layout/pipeline ->
record barriers and commands -> submit with a timeline signal -> wait before reuse or destruction.
For descriptor updates, `baseRange` indexes the declared `ranges` array. It is **not** a shader binding number:
range index 0 can declare `baseRegister = 7`; its descriptors are updated through range index 0.

## Coordinate conventions

`vriDefaultDeviceCreationDesc()` sets the required `structSize` and Vulkan defaults. An incorrectly sized
creation descriptor or invalid convention enum fails before backend initialization. Select a preset with
`vriGetConventionPreset()`, then override any field before creation. `vriGetCoordinateConventions()` returns a
copy of the immutable device configuration. Vulkan/Metal presets choose right-handed, column-major,
column-vector math; Direct3D chooses left-handed math. These are documented helper policies, not API mandates.

The authoring-to-rasterizer transform returned by `vriGetCoordinateTransform()` is explicit:

- Vulkan uses positive viewport height for Y-down, negative height for Y-up. Other backends retain their portable
  Y-up raster contract; the queried factor converts Y-down shader outputs when needed.
- A [-1, 1] authored clip depth is converted to [0, 1] by `z = 0.5 * z + 0.5 * w`. Authored [0, 1] is unchanged.
- Bottom-left authored UV becomes top-left through `v = 1 - v`. Upload/readback row zero always remains the top
  row; changing UV policy never changes raw texture bytes or pixel coordinates.
- `vriMakePerspectiveProjection()` produces an authored projection; `vriTransformProjection()` makes it ready
  for the queried rasterizer, preserving off-center and jitter terms for row/column storage and multiplication.
  Apply the conversion once, either in the projection or in the shader. World handedness is not converted by
  the clip correction; it is selected when constructing the projection/view data.

Raw precompiled shaders must obey their authored contract and explicitly use correction factors if needed;
VRI does not rewrite arbitrary SPIR-V, DXIL or WGSL. The Slang helpers and shader include path support the same
conversion without target-specific shader copies. ImGui always declares its own Y-up source convention.
OpenGL retains its current-native-context requirement; immutable convention queries do not activate a context.

The OpenGL/WebGPU timeline paths currently publish requested signal values at submission, and WebGPU descriptor
updates do not yet preserve all prior writes. These are implementation gaps to fix, not intended NRI semantics
or evidence of completion. Existing successful image tests do not establish correctness of these untested cases.

## Backend details

**✅ Native** &nbsp;·&nbsp; **🟡 Emulated / Simulated** &nbsp;·&nbsp; **⬜ Planned** (the API supports it; not yet in VRI) &nbsp;·&nbsp; **❌ Unsupported by the API**

- **Planned (⬜).** A few cells are things the backend's API *can* do but VRI hasn't wired yet —
  distinct from ❌ (the API genuinely can't). Currently: **Metal** subgroup / wave operations
  (SIMD-group functions) and fragment-shader barycentrics (`[[barycentric_coord]]`) — these await
  Apple hardware to implement and verify.
- **Push constants / bindless** are exposed uniformly: where the backend lacks the hardware feature,
  VRI provides a transparent fallback (an emulated uniform path, or a texture-array path) that
  produces the same result.
- **Ray tracing** is hardware-accelerated on Vulkan & Direct3D 12 (ray query *and* a full
  raygen/miss/hit pipeline, plus **acceleration-structure compaction** to shrink a built BLAS/TLAS
  to its exact size) and on Metal (inline ray query). On **WebGPU and OpenGL** — which have
  compute but no ray-tracing hardware — VRI traces rays in a **compute shader** instead (🟡). Where
  there's no compute at all (WebGL 2 / OpenGL ES), ray tracing is unavailable.
- **Bindless on Metal** is implemented but currently unverified (no Apple hardware in CI).
- **External memory / interop** exports OS handles (Win32 / fd) for a buffer/texture's backing
  memory and for timeline fences (`ext/vri_ext_external.h`), so a **CUDA** kernel — or OptiX,
  NVENC, another process, or another graphics API — can import the same memory and timeline.
  VRI itself does **not** depend on CUDA; it only produces the handles the consumer imports.
  Implemented on **Vulkan** (`VK_KHR_external_memory/semaphore`) and **Direct3D 12** (shared
  committed resources + shared fences); the opt-in `example-cuda-interop` runs on both
  (`VRI_API=vulkan|d3d12`).
- **Native-handle interop** (`ext/vri_ext_interop.h`, `VRI_INTERFACE_INTEROP`) is the **OpenXR**
  seam: it hands out VRI's native objects (`GetDeviceNativeHandles` → `VkInstance`/`VkPhysicalDevice`/
  `VkDevice`/queue) to build an `XrGraphicsBindingVulkanKHR`, and wraps externally-created textures
  (`WrapTexture`, e.g. OpenXR swapchain `VkImage`s) back into `VriTexture`. For `XR_KHR_vulkan_enable2`,
  `VriDeviceCreationDesc::nativeCreateInfo` (a `VriVulkanCreateHooks`) lets OpenXR interpose
  `vkCreate{Instance,Device}` while VRI still builds the create-infos. VRI does **not** depend on
  OpenXR — the opt-in `example-openxr` (`--vri_build_openxr=y`) links the loader and renders both eyes
  in one multiview pass; verified rendering on the Meta XR Simulator. Needs an OpenXR runtime to run,
  so it is never built in CI.
- **GPU timestamp queries** (`ext/vri_ext_query.h`) record the GPU clock at points in a command
  buffer and resolve the ticks into a buffer; scale by `VriDeviceDesc::timestampPeriodNanoseconds`
  for nanoseconds. The basis of a GPU profiler — see `example-profiler`. On **Vulkan**, **Direct3D
  12**, **Metal** (an `MTLCounterSampleBuffer`; Apple GPUs sample only at encoder stage boundaries,
  so each timestamp opens a one-shot blit encoder, and the counter buffer is resolved CPU-side in the
  submit's completion handler), **WebGPU** (desktop wgpu-native, via the `timestamp-query` +
  `TimestampQueryInsideEncoders` features; browser WebGPU timestamps are pass-scoped and not yet
  wired), and **desktop OpenGL 4.4+** (`glQueryCounter` + a query buffer object; GLES/WebGL have no
  core timer query). **Occlusion queries** (samples-passed) are also supported on Vulkan + D3D12 +
  Metal (a native visibility-result buffer + `MTLVisibilityResultModeCounting`) + desktop OpenGL
  (WebGPU's occlusion binds the query set at pass-begin, which this API doesn't express),
  **pipeline-statistics queries** (per-stage invocation counts, a portable `VriPipelineStatistics`
  struct) on Vulkan + D3D12 + desktop OpenGL 4.6 (via `ARB_pipeline_statistics_query`, one GL query
  object per stat; Apple GPUs expose no statistic counter set), and **calibrated timestamps** (a
  correlated GPU+CPU clock pair, for aligning GPU spans with a CPU trace) on Vulkan + D3D12 + Metal
  (`[MTLDevice sampleTimestamps:gpuTimestamp:]`).
- **Multiview** (single-pass layered rendering) renders to several layers of an array target in one
  pass, selected by a `viewMask` on both the graphics pipeline (`VriOutputMergerDesc::viewMask`) and
  the render pass (`VriAttachmentsDesc::viewMask`); the shader reads the per-view index via
  `SV_ViewID` (→ SPIR-V `gl_ViewIndex` / `gl_ViewID_OVR`). This is the basis of single-pass VR stereo.
  Implemented on **Vulkan** (core 1.1 multiview), **Direct3D 12** (`ViewInstancing` — one draw fans
  out to N view instances, each routed to its own render-target array slice via the pipeline's
  per-view `RenderTargetArrayIndex`; the shader reads `SV_ViewID`), **Metal** (SPIRV-Cross emits the
  instanced form — the draw's instance count is multiplied by the view count and each view writes its
  array slice via `[[render_target_array_index]]`), and **OpenGL / OpenGL ES** (`GL_OVR_multiview2`,
  the standalone-headset path; SPIRV-Cross emits the `gl_ViewID_OVR` form). Exposed via
  `VriDeviceDesc::hasMultiview` + `maxViewCount`.
- **Built-in Dear ImGui renderer** (`ext/vri_ext_imgui.h`, `VRI_INTERFACE_IMGUI`) draws ImGui
  through VRI's own core interface, so a single renderer covers every backend with no per-backend
  `imgui_impl_*` and it flows through the validation layer for free. VRI does **not** depend on or
  link Dear ImGui: the application owns the ImGui environment (context, input, `NewFrame`) and each
  frame hands VRI a backend-neutral, flattened `VriImguiDrawData` (so no ImGui types cross into
  VRI). Each draw command carries the texture it samples (`VriImguiDrawCommand::textureView`), so
  user images and **ImGui 1.92's dynamic, host-owned textures** work, not just the font atlas — the
  host creates textures with the core interface, routes them through ImGui's texture id, and tells
  the renderer to drop a cached binding (`FreeImguiTexture`) when a texture is destroyed.
  **Docking + multi-viewport** are supported: each detached OS window ImGui spawns gets its own
  `VriImguiViewport` (independent geometry buffers, sharing the pipeline/font/texture cache) driven
  by the `*To` calls, so several windows render per frame without clobbering each other. The examples
  enable docking and multi-viewport (detached OS windows) by default — drag a panel out of the main
  window to pop it into its own OS window; opt out with `VRI_IMGUI_VIEWPORTS=0` (auto-off in headless
  capture). The examples' control panel is drawn with it.
- **Pipeline cache** (`ext/vri_ext_pipeline_cache.h`, `VRI_INTERFACE_PIPELINE_CACHE`) lets the
  driver reuse work (shader compilation, PSO assembly) across pipeline creations and across runs:
  pass a cache in `VriGraphicsPipelineDesc::pipelineCache`, serialize it on exit
  (`GetPipelineCacheData`), and seed the next launch (`CreatePipelineCache`) for fast warm startup.
  A stale or foreign blob is detected and ignored, so seeding is always safe. On **Vulkan**
  (`VkPipelineCache`), **Direct3D 12** (`ID3D12PipelineLibrary`, keyed by a stable hash of each
  pipeline's definition), **Metal** (an `MTLBinaryArchive` populated at pipeline creation; the blob
  is bridged through a temp file since Metal only serializes to a file URL), and **desktop OpenGL**
  (emulated with program binaries — a cache hit restores the linked program via `glProgramBinary`,
  skipping the GLSL compile + link). WebGPU and WebGL 2 expose no cache blob (their runtime caches
  shaders internally), so they report `Unsupported`.
- **GPU-driven draw count.** `CmdDrawIndirectCount` / `CmdDrawIndexedIndirectCount` take the *number*
  of draws from a GPU buffer (a `uint32` at an offset, clamped to `maxDrawNum`) instead of the CPU —
  so a compute pass can decide how many draws to issue. Gated by `VriDeviceDesc::hasDrawIndirectCount`.
  On **Vulkan** (`vkCmdDraw*IndirectCount`), **Direct3D 12** (`ExecuteIndirect` with a count buffer,
  command signature cached by stride), and **desktop OpenGL 4.6** (`glMultiDraw*IndirectCount`).
  (Plain `CmdDrawIndirect` / `CmdDrawIndexedIndirect`, with a CPU-known count, are separate and more
  widely supported.)
- **Live VRAM budget.** `GetVideoMemoryInfo(device, location, &info)` reports the OS-current memory
  budget and current usage (bytes) for a memory location, for streaming / eviction decisions. On
  **Vulkan** (`VK_EXT_memory_budget`), **Direct3D 12** (`IDXGIAdapter3::QueryVideoMemoryInfo`), and
  **Metal** (working-set size); **OpenGL** and **WebGPU** have no portable query, so they return
  `Unsupported`.
- **Clear storage buffer.** `CmdClearStorageBuffer(cmd, buffer, offset, size, value)` fills a storage
  buffer region with a repeated `uint32` outside a render pass — e.g. resetting an atomic counter or
  accumulation buffer before a compute pass. Gated by `VriDeviceDesc::hasClearStorageBuffer`. On
  **Vulkan** (`vkCmdFillBuffer`), **Direct3D 12** (`ClearUnorderedAccessView` via a transient UAV),
  and **desktop OpenGL 4.3** (`glClearBufferSubData`); WebGPU/Metal (no arbitrary-value buffer fill)
  report `Unsupported`. **`CmdClearStorageTexture`** clears a storage texture to a color the same way
  (`vkCmdClearColorImage` / `ClearUnorderedAccessView{Float,Uint}` / `glClearTexImage`, GL 4.4),
  gated by `hasClearStorageTexture`.

## API migration scope

The comprehensive Vulkan-style API redesign, configurable coordinate conventions and **0.2.0** version bump
belong to the separate PR #24 task. The cooperative-vector work in PR #23 adds an optional interface to the
current API; it does not change existing coordinate conventions or complete the broader migration.

## Cooperative-vector shader inference

`VriFeature_CooperativeVector` enables the optional `VRI_INTERFACE_COOPERATIVE_VECTOR` table and
`VriDeviceDesc::hasCooperativeVector`. First implementation: Vulkan; other backends do not provide the table.
A required request (`bestEffort = false`) fails with `Unsupported` if unavailable. A best-effort request may
create a device with the flag unset. This is not a general ML runtime and does not silently replace inference
with scalar arithmetic.

- `GetCooperativeVectorDeviceProperties()` reports supported shader stages and maximum vector length.
- `GetCooperativeVectorProperties()` enumerates complete input/interpretation/matrix/bias/result type tuples.
  Query the count, allocate the array, then read it. Independently supported scalar types cannot be freely combined.
- `ConvertCooperativeVectorMatrix()` takes `VriConvertCooperativeVectorMatrixInfo`: borrowed host addresses,
  byte capacities, component types, matrix dimensions, layouts and strides. A null destination queries its size.
  Short source/destination buffers and overlapping ranges fail before conversion writes. Same-type layout conversion,
  Float32/Float16 conversion and Float32/Float16 to/from supported Float8 are implemented.

The inference-optimal representation is device/driver-specific. Store portable weights and convert on load;
use the normal VRI buffer upload path afterward. Matrix offsets passed to Slang must satisfy its alignment rules;
the test packs each matrix/bias at a 64-byte boundary. Shaders own network shape and activation functions.
Training, GPU matrix conversion and inference on D3D12/Metal/WebGPU/GL are not implemented. Availability of a
ray/mesh stage here does not enable that separate stage feature.

The Vulkan shader baseline enables queried cooperative vectors, replicated composites and Vulkan memory model,
which the pinned Slang CoopVec lowering requires. Available Float16/Int8 arithmetic is enabled too; component-type
extensions are enabled when advertised. All feature checks and native conversions stay inside VRI.
See [Khronos cooperative vectors](https://docs.vulkan.org/features/latest/features/proposals/VK_NV_cooperative_vector.html),
[host matrix conversion](https://docs.vulkan.org/refpages/latest/refpages/source/vkConvertCooperativeVectorMatrixNV.html)
and [Slang CoopVec](https://docs.shader-slang.org/en/stable/external/core-module-reference/types/coopvec-04/index.html).

```sh
xmake f -m release --vri_build_examples=n --vri_build_tests=y --vri_build_tools=y
xmake build vri-tests vri-shaderc
xmake run vri-tests --test-case='cooperative vector:*'

# Regenerate only this test shader, with the required capability explicit.
build/linux/x86_64/release/vri-shaderc shaders/tests/neural_inference.slang \
    -o shaders/tests/neural_inference_spv.h --var g_neuralInferenceSpv \
    --target spirv --capability spvCooperativeVectorNV
```

The offline test evaluates 65 inputs through an 8 → 16 → 4 FP16 MLP with ReLU, compares every output against
independent CPU arithmetic, checks opt-in/unsupported behavior, count queries, conversion bounds and exact
Float16/Float8 weight round trips. Both raw and validation-wrapped devices run the same test. Linux RTX 4060
Laptop maximum MLP error: 0.000122070 (limit 0.002). This is not acceptance on other devices or backends.

The `example-mlp` image demo uses a trained 2 -> 32 -> 32 -> 3 ReLU network. Its FP32 portable weights
are converted once to FP16/device-optimal matrices and FP16 biases, then uploaded through ordinary VRI buffers.
The default SDL3/ImGui window compares reference/prediction/error, with live error amplification and preview zoom.
`--offline` runs the same compute/verification path without initializing SDL video or ImGui; automated example runs
select it explicitly. No external model file, Python environment or inference SDK is needed for either mode.
GPU resources are reused; command recording stays before the display pass, and the GUI cache is released only
after GPU completion and before destroying its texture view. Both modes verify and export the three panels.

On the tested RTX 4060 Laptop, 256x256 inference has maximum CPU-reference error 0.0023849 (limit 0.01),
analytic-fit RMSE 0.0130625 (limit 0.03), and at most one UNORM byte of image error. The 137x137 case also passes.
These are correctness checks, not inference performance measurements. OpenGL reports unsupported (exit 77);
`xmake examples` counts such skips separately and still fails on diagnostics or other nonzero exit codes.

The original procedural target, deterministic training seed, Adam implementation and generated weights are in
`examples/mlp/`. Retraining is optional and requires NumPy; use the normal formatter after generation:

```sh
python examples/mlp/train.py
clang-format -i examples/mlp/model_weights.h
build/linux/x86_64/release/vri-shaderc shaders/examples/mlp_image.slang \
    -o shaders/examples/mlp_image_spv.h --var g_mlpImageSpv \
    --target spirv --capability spvCooperativeVectorNV
```

`VriCooperativeVectorProperties::transpose` advertises *additional* support for transposition of optimal-layout
matrices; a true value does not prevent ordinary non-transposed multiplication.
[Khronos property contract](https://docs.vulkan.org/refpages/latest/refpages/source/VkCooperativeVectorPropertiesNV.html).

## Software (CPU) rendering

`VriGraphicsAPI_Software` renders **without a GPU** by running VRI's Vulkan backend on top of a
*software Vulkan implementation* — a CPU ICD that JITs SPIR-V shaders. This is the same approach
Chrome and the Android emulator use for their GPU-less fallback. It exposes the **full Vulkan
feature set** (the whole Vulkan column above) at CPU speed, and the device reports
`VriAdapterType_Software`.

**Selecting it.** Pass `VriGraphicsAPI_Software` to `vriCreateDevice`, or run the examples with
`VRI_API=software`. It is also the **last `Auto` fallback** on every non-web platform, so a machine
with no usable GPU still renders instead of failing outright. If no software Vulkan device is
present, a `Software` request returns `VriResult_Unsupported` — it never silently falls back to a
GPU, because "software" means the CPU path specifically.

**Providing the ICD (bring-your-own).** VRI does not bundle a software renderer; it uses whichever
software Vulkan ICD is on the system, resolved in this order: (1) a caller-set `VK_ICD_FILENAMES` /
`VK_DRIVER_FILES` (the standard Vulkan loader env vars) is respected as-is; (2) otherwise, a
**[SwiftShader](https://github.com/google/swiftshader)** manifest (`vk_swiftshader_icd.json`)
sitting next to the executable is selected automatically. If neither is found, the request fails
with a message naming both options.

Per platform:

- **Linux** — `sudo apt install mesa-vulkan-drivers` installs **lavapipe**; the loader discovers it
  automatically, so `VRI_API=software` just works with no env var. This is the standard headless-CI
  setup (equivalents: `dnf install mesa-vulkan-drivers`, `pacman -S vulkan-swrast`, …).

  ```sh
  sudo apt install -y mesa-vulkan-drivers
  VRI_API=software xmake run example-triangle
  ```

- **Windows** — download a Mesa build that includes lavapipe (e.g.
  [pal1000/mesa-dist-win](https://github.com/pal1000/mesa-dist-win/releases), the
  `mesa3d-<ver>-release-msvc.7z`), extract it, and point the loader at the ICD manifest:

  ```powershell
  $env:VK_ICD_FILENAMES = "C:\path\to\mesa\x64\lvp_icd.x86_64.json"
  $env:VRI_API = "software"
  xmake run example-triangle
  ```

  Or build **SwiftShader** and drop `vk_swiftshader.dll` + `vk_swiftshader_icd.json` next to the
  app for zero-config auto-discovery (no env var needed).

- **macOS** — build **SwiftShader** (`vk_swiftshader.dylib` + manifest) and either set
  `VK_ICD_FILENAMES` or place it beside the app. Note **MoltenVK is *not* a software renderer** — it
  maps Vulkan onto the Metal GPU; for a genuine no-GPU path use SwiftShader.

  ```sh
  cmake -S swiftshader -B swiftshader/build -DSWIFTSHADER_BUILD_TESTS=OFF
  cmake --build swiftshader/build --config Release
  export VK_ICD_FILENAMES=.../swiftshader/build/Darwin/vk_swiftshader_icd.json
  VRI_API=software xmake run example-triangle
  ```

The selection logic lives in `source/backend_vk/software_icd_vk.cpp`.
