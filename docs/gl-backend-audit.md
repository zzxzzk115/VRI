# OpenGL backend audit — version/feature tiers & "no arbitrary downgrade"

**Status:** audit only (no code changes yet). This documents where the GL backend
currently sacrifices desktop capability/performance for GLES3/WebGL2 compatibility,
and the design to fix it.

## Principle

The GL backend must **detect the runtime GL version + extensions and use the highest /
most performant path each context actually supports** — not blanket-target the
GLES3/WebGL2 lowest common denominator. Downgrades are allowed **only** when the
running context genuinely lacks a feature, and then they must be **explicit**
(capability-gated, reported via `VriDeviceDesc`), never silent.

Today the backend creates a 4.6 core context on desktop but runs the GLES3/WebGL2
non-DSA LCD path *uniformly* — i.e. it downgrades a 4.6 desktop GPU to ES3-level
behavior. That is the "arbitrary downgrade" to remove.

## Tiers

| Tier | Context | Notes |
|---|---|---|
| **ES / WebGL2** | GLES 3.0 | true LCD; many features genuinely absent (legit gating) |
| **Legacy desktop** | GL 3.3 core | broad-compat desktop baseline |
| **Apple desktop** | GL 4.1 (Apple max) | geometry/tess yes; **no** compute(4.3)/DSA(4.5)/clipControl(4.5)/gl_spirv(4.6) |
| **Modern AZDO** | GL 4.6 core | DSA, glClipControl, ARB_gl_spirv, separate vtx format, MDI, base-instance, persistent map |

`m_es` (ES vs desktop) is too coarse. Replace with a detected feature set, e.g.:

```cpp
struct GlFeatures {
    int  major, minor;     // detected GL/ES version
    bool clipControl;      // GL 4.5 / ARB_clip_control      -> native top-left, depth [0,1]
    bool spirvIngest;      // GL 4.6 / ARB_gl_spirv           -> glShaderBinary + glSpecializeShader
    bool dsa;              // GL 4.5 / ARB_direct_state_access
    bool separateAttrib;   // GL 4.3 / ARB_vertex_attrib_binding
    bool baseInstance;     // GL 4.2 / ARB_base_instance
    bool drawIndirect;     // GL 4.3 multi-draw-indirect
    bool bufferStorage;    // GL 4.4 / ARB_buffer_storage     -> persistent-mapped buffers
    bool compute;          // GL 4.3 (already gated)
    bool geometry, tess;   // already gated
};
```

Each hot path branches on the relevant flag; ES/old-desktop fall back (explicitly).

## Compatibility taxes (current downgrades on capable desktop)

| # | Tax (current LCD path) | Best path (gate) | Impact | File |
|---|---|---|---|---|
| 1 | In-shader `flip_vert_y` for Y/depth | `glClipControl(UPPER_LEFT, ZERO_TO_ONE)` (4.5) | per-vertex negate; forces winding swap + readback row-flip; caused the TCS C7544 bug | `core_gl.cpp:151,644,1099` |
| 2 | SPIR-V→GLSL via SPIRV-Cross + driver GLSL recompile | `glShaderBinary` + `glSpecializeShader` ingesting SPIR-V directly (4.6 ARB_gl_spirv) | **biggest**: transpile cost + driver recompile on every pipeline create | `core_gl.cpp:128,254` |
| 3 | Transient FBO per render pass (gen+delete) | cache FBOs keyed by attachment set | per-pass alloc/free | `core_gl.cpp:788,865` |
| 4 | Per-draw `glVertexAttribPointer` + enable/disable on one VAO | separate attrib format (`glVertexAttribFormat`/`glBindVertexBuffer`, 4.3) + per-layout VAO | per-draw attrib churn | `core_gl.cpp:988` |
| 5 | `glDrawArraysInstanced` only; baseVertex/baseInstance unsupported | `*BaseVertexBaseInstance` (3.2/4.2) | correctness gap on desktop (silent) | `core_gl.cpp:1016` |
| 6 | No state-redundancy cache (re-apply all FF state per pipeline bind) | `StateCache` dedup | per-bind GL calls | `CmdSetPipeline` |
| 7 | bind-to-edit resource creation/update (non-DSA) | DSA (`glTextureStorage2D`/`glNamedBuffer*`, 4.5) | extra binds; clobbers binding state | throughout |

Notes:
- #5 is also a **silent downgrade** (base offsets quietly ignored) — should at minimum
  be wired on desktop and validated/reported where unsupported.
- #1 is the highest-value cleanup: removing `flip_vert_y` on desktop also deletes the
  winding-swap + readback-flip special cases and the class of bugs they cause.
- #2 is the highest-perf win.

## Legitimate (explicit) downgrades — keep as-is

These are genuine ES/WebGL2 capability gaps, already reported via `VriDeviceDesc` and
rejected with `VriResult_Unsupported` — correct per the philosophy:
- compute (ES needs 3.1; WebGL2 none), geometry/tessellation (desktop only),
  multisample **textures** (WebGL2 uses renderbuffers instead — already handled).

## Roadmap (priority)

1. ✅ **GlFeatures detection** (`device_gl.h`/`.cpp`) — per-device tier flags derived
   from the GL version; capability flags stay derived from it. (version-gated; extension
   scan can refine later)
2. ✅ **glClipControl on 4.5+** — desktop now uses `glClipControl(UPPER_LEFT, ZERO_TO_ONE)`:
   no `flip_vert_y`, depth is native `[0,1]` (was `[-1,1]`→compressed). Winding branches
   on clipControl (VRI CCW = GL CCW with clipControl; GL CW on the flip path). Readback
   is unchanged (both conventions read back top-left). ES/WebGL2/pre-4.5 keep the in-shader
   flip. Coordsys/cull/depth/tessellation all green; WebGL2 E2E unchanged.
3. **ARB_gl_spirv on 4.6** — `glShaderBinary` SPIR-V ingestion; SPIRV-Cross only for
   ES/legacy/Apple. **Prereq:** a SPIR-V binding-remap patcher — GL ingests the raw
   `(set,binding)`, but VRI flattens those to per-type GL units across sets and fuses
   texture+sampler, so the Vulkan SPIR-V must be patched first. Driver-gate (Mesa
   llvmpipe `gl_spirv` support is uncertain). Self-contained sub-project.
4. **AZDO draw path:**
   - ✅ **FBO cache** — render-pass FBOs are cached on the device keyed by their
     attachment set (configured once, reused as-is); evicted when a referenced texture
     is destroyed. No per-pass `glGen/DeleteFramebuffers`. All-GL (desktop + WebGL2).
   - ✅ **separate attribute format + per-pipeline VAO (4.3)** — each graphics
     pipeline bakes its vertex *format* into its own VAO once (`glVertexArrayAttrib*`
     under DSA, else `glVertexAttribFormat`/`glVertexAttribBinding`); at draw we only
     point each stream binding at a buffer (`glVertexArrayVertexBuffer` /
     `glBindVertexBuffer`), replacing per-draw `glVertexAttribPointer` + enable/disable
     churn on the shared default VAO. Gated on `GlFeatures::separateAttrib`; pre-4.3 /
     GLES / WebGL2 keep the classic per-draw path (the 4.3+ entry points aren't even
     declared in the Emscripten GLES3 headers, so the modern block is `#if`-compiled out).
   - ✅ **base-vertex/base-instance** — `CmdDraw` now honors `baseInstance`
     (`glDrawArraysInstancedBaseInstance`, GL 4.2) and `CmdDrawIndexed` honors
     `vertexOffset` (`glDrawElements(Instanced)BaseVertex`, GL 3.2 — all desktop incl.
     macOS 4.1) + `baseInstance` (`glDrawElementsInstancedBaseVertexBaseInstance`, 4.2),
     replacing the previous **silent** drop. Parity for the `VertexIndex`/`InstanceIndex`
     builtins is handled by SPIRV-Cross (`gl_BaseVertexARB`/`gl_BaseInstanceARB` auto-set
     by the `*Base*` draws, or the `SPIRV_Cross_Base*` uniform fallback). Where a base
     offset can't be honored (baseInstance pre-4.2; either on WebGL2), it's reported via
     the message callback — explicit, never silent. Note: Slang's `SV_VertexID`/
     `SV_InstanceID` are *local* (D3D semantics, base excluded), so base offsets are
     observable through attribute fetching — covered by `test_baseoffset_xbackend`
     (a vertexOffset draw selecting a green vertex set over a red decoy).
   - ✅ **indirect draw/dispatch + MDI** — `CmdDrawIndirect` (was a **silent no-op**) now
     issues `glDrawArraysIndirect` (GL 4.0; loops for >1 draw, or `glMultiDrawArraysIndirect`
     when `GlFeatures::drawIndirect`/4.3 is present) and `CmdDispatchIndirect` issues
     `glDispatchComputeIndirect` (4.3). The indirect command layout matches VK/WebGPU.
     Unavailable on WebGL2 → reported, not silent. Covered by `test_indirect_xbackend`.
   - persistent-mapped buffers (later).
5. ✅ **DSA on 4.5** — desktop GL 4.5+ now creates/updates resources by name, no
   bind-to-edit: `glCreateBuffers`/`glNamedBuffer*` (create/map/unmap/copy),
   `glCreateTextures`/`glTextureStorage2D`/`glTextureSubImage2D` (incl. multisample),
   `glCreateSamplers`, `glBindTextureUnit` + `glTextureParameteri` at descriptor bind,
   and named render-pass FBO config + clears (`glCreateFramebuffers`/
   `glNamedFramebuffer*`/`glClearNamedFramebuffer*`). No active-texture / COPY_WRITE /
   PIXEL_UNPACK binding-state clobber. ES/WebGL2 + pre-4.5 desktop keep the bind path
   (so CI's macOS 4.1 exercises the fallback; Linux llvmpipe + Windows exercise DSA).

Each step is independent, desktop-gated, and must keep ES/WebGL2 + the full CI matrix
green.
