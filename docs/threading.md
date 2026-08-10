# Threading & synchronization model

**Status:** contract documentation (behavior already in place; adds one opt-in validation check).

VRI follows the same **externally synchronized** model as Vulkan and Direct3D 12: the library
adds **no internal locking** around per-object operations, and the application owns all
synchronization. This document states, per object and per backend, which calls may run
concurrently and where the caller must serialize. It also records the places where the current
implementation is stricter or looser than this contract.

The short version: **one thread per queue timeline, one thread per command buffer.** You get
multi-threaded command recording by giving each recording thread *its own* command allocator and
command buffers; you serialize submission per queue. OpenGL is the exception — it is single-threaded
for the whole device.

## Core contract

Nothing in VRI is implicitly locked. Two calls that touch the **same** object at the same time from
two threads are a data race unless this document says otherwise. Two calls that touch **different**
objects are fine. Concretely:

| Object | Concurrency rule |
| --- | --- |
| `VriDevice` | Object *creation* calls (`CreateBuffer`, `CreateTexture`, `CreateSampler`, `Create*Pipeline`, `CreatePipelineLayout`, `AllocateMemory`, `CreateFence`, …) are **not** serialized by VRI. Treat device-level creation/destruction as externally synchronized: serialize it, or confine it to one thread. (On Vulkan it happens to be safe — see the backend notes — but that is not the portable contract.) |
| `VriCommandAllocator` | **Externally synchronized.** An allocator and every command buffer created from it belong to one thread at a time. `CreateCommandBuffer`, recording into its buffers, and `ResetCommandAllocator` must not run concurrently with each other. Reset must also not race pending GPU execution — fence first. Reuse is lock-free by design, which is *why* the single-owner rule exists. Use one allocator per recording thread. |
| `VriCommandBuffer` | **Externally synchronized, single-threaded.** A command buffer must be recorded start-to-finish (`BeginCommandBuffer` → `Cmd*` → `EndCommandBuffer`) by one thread. Different command buffers may be recorded on different threads at the same time — that is the parallel-recording path. |
| `VriQueue` | **Externally synchronized.** `QueueSubmit` / `QueueWaitIdle` on a given queue must be serialized; never submit to the same `VriQueue` from two threads at once. Distinct queues (Graphics / Compute / Transfer) are independent and may be driven from different threads; cross-queue ordering is the caller's job via timeline fences. |
| `VriDescriptorPool` | **Externally synchronized.** `AllocateDescriptorSets`, `ResetDescriptorPool`, and `UpdateDescriptorRanges` on the same pool/its sets must be serialized. Reuse is lock-free. Use one pool per thread for parallel descriptor allocation. |
| `VriFence` (timeline) | `GetFenceValue` and `Wait` are safe to call from any thread concurrently (a timeline value is monotonic and only observed here). A fence is *signaled* through `QueueSubmit`, which follows the queue rule above. |
| `VriBuffer` mapping | `MapBuffer` / `UnmapBuffer` on the same buffer must be serialized; do not map one buffer from two threads. Reading/writing the mapped pointer is ordinary CPU memory the caller must synchronize. |
| `VriDevice` teardown | `DeviceWaitIdle` before destroying objects. Destruction (`Destroy*`, `FreeMemory`) must not race any in-flight use of the object on another thread or the GPU. |

There is exactly one lock inside VRI: a process-global `std::mutex` in the validation layer
(`source/core/validation_layer.cpp`) that guards the set of live validation devices so validation
devices can be created/destroyed from multiple threads. It protects **bookkeeping only** — it does
**not** make any rendering, recording, or submission call thread-safe.

## Recommended usage pattern

```
per frame:
  for each recording thread T:
     T owns allocator A_T and command buffers from A_T   # no sharing between threads
     T: BeginCommandBuffer(cmd) ... Cmd*(cmd) ... EndCommandBuffer(cmd)   # all on T
  join threads
  submitter thread only: QueueSubmit(graphicsQueue, {all cmds})           # one thread per queue
```

To parallelize: partition work across threads, give each thread its own allocator + descriptor pool
+ command buffers, then funnel the finished command buffers to a single submitting thread per queue.

## Per-backend differences

### OpenGL / OpenGL ES / WebGL — single-threaded, device is context-thread-bound

The strongest constraint. The whole device is bound to **one** GL context, which is current on **one**
thread. Every VRI call — create, record, submit, present — must happen on the thread that owns that
context.

- The context is made current at device creation (`eglMakeCurrent` / `glfwMakeContextCurrent` /
  WGL / NSGL in `source/backend_gl/device_gl.cpp`) and VRI never migrates it to another thread.
- Commands execute **immediately** when recorded — the command buffer is a thin recorder and
  `QueueSubmit` is essentially a `glFlush` (`source/backend_gl/core_gl.cpp`). There is no deferred
  command stream, so multi-threaded recording is not available on this backend.
- If you must drive GL from a different thread, you have to move the context there yourself (make it
  current on the new thread, clear it on the old). That is outside VRI's scope.

This is the one backend where the model is *stricter* than the general contract above.

### Vulkan — full external-sync model

- `vkQueueSubmit`, command pools (`vkResetCommandPool`, `vkAllocateCommandBuffers`), and descriptor
  pools follow Vulkan's own external-synchronization rules; VRI adds no locking on top, so the
  per-object rules in the table apply verbatim.
- The VMA memory allocator is created **without** `VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT`
  (`source/backend_vk/device_vk.cpp`), so VMA uses its **internal** lock: `AllocateMemory` /
  `FreeMemory` (and the memory-desc queries) are internally thread-safe on Vulkan. This is a bonus,
  not the portable contract — code that must run on every backend should still serialize device-level
  allocation.

### Direct3D 12 — external-sync, per-engine queues

- One `ID3D12CommandAllocator` per recording thread; `Reset()` must not race recording.
- `ExecuteCommandLists` on a given queue must be serialized (no VRI lock).
- The shader-visible descriptor heaps are bump-allocated per pool with no lock, so descriptor
  allocation/reset on one pool is single-owner (the table's descriptor-pool rule).
- Each `VriQueueType` maps to its own engine — Graphics→DIRECT, Compute→COMPUTE, Transfer→COPY
  (`source/backend_d3d12/device_d3d12.h`) — so the three queues run independently; cross-queue
  ordering is the caller's via timeline fences.

### Metal — underlying runtime is thread-safe, but keep the discipline

- `MTLCommandQueue` submission and `MTLDevice` resource creation are thread-safe per Apple, and Metal
  has no command-pool / descriptor-pool object to serialize. So Metal is *looser* than the general
  contract.
- Even so, record a single command buffer on a single thread and treat the per-object rules as the
  portable baseline, so the same application code stays correct on the other backends.

### WebGPU — serialize per the WebGPU model

- Native `wgpu` queues and command encoders are not reentrant; serialize submission per queue and
  record one command buffer on one thread. There is no command-pool / descriptor-pool object.

## Validation aid (opt-in)

When `VriDeviceCreationDesc::enableValidation` is set, the validation layer flags the most common
cross-thread mistake: **recording a single command buffer from more than one thread.** It records the
thread that called `BeginCommandBuffer` and emits a `Warning` (via the device message callback) if a
later `Cmd*` or `EndCommandBuffer` call on that same command buffer arrives from a different thread.

- It is a **warning**, not an error: a caller may legitimately hand a command buffer to another thread
  under its own happens-before barrier. The check catches the far more common accidental race.
- It is **opt-in and zero-cost when off** — with validation disabled the app gets the raw backend
  table, so this never affects a release/no-validation build or any rendering behavior. It only emits
  a diagnostic.
- It does **not** try to detect cross-thread misuse of allocators, pools, or queues; those remain the
  caller's responsibility per the contract above.

## Known gaps / TODO

- **Device-level creation thread-safety is not uniform.** Only Vulkan is actually safe today (via
  VMA's internal lock); the portable contract is "serialize device object creation." Making resource
  creation uniformly safe (or documenting each backend cell explicitly) is future work — not changed
  here to keep this task documentation-first.
- **No cross-thread assertions beyond the single-command-buffer check.** Detecting concurrent
  `QueueSubmit` on one queue, or concurrent allocator reset/record, would need per-object owner
  tracking; left as a follow-up.
