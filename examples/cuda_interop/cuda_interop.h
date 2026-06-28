/*
 * cuda_interop.h - C boundary between the VRI host side (main.cpp) and the CUDA side
 * (cuda_interop.cu). Keeps the CUDA runtime headers out of the host translation unit so
 * main.cpp compiles with the normal C++ toolchain and only the .cu is built by nvcc.
 *
 * The host exports OS handles for a VRI buffer's backing memory and a VRI timeline fence
 * (via VRI_INTERFACE_EXTERNAL), then hands them here. CUDA imports the same memory + timeline,
 * waits for the host's write, transforms the data in place, and signals back.
 */
#ifndef VRI_EXAMPLE_CUDA_INTEROP_H
#define VRI_EXAMPLE_CUDA_INTEROP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /* Whether the build links CUDA on this platform (Win32 OPAQUE_WIN32 path is implemented). */
    int cudaInteropSupported(void);

    /* Print the selected CUDA device's name + compute capability. Returns 0 on success. */
    int cudaInteropPrintDevice(void);

/* Which kind of handle VRI exported, so CUDA picks the matching import type. */
#define CUDA_INTEROP_KIND_VULKAN 0 /* Vulkan OPAQUE_WIN32/FD memory + timeline semaphore */
#define CUDA_INTEROP_KIND_D3D12 1  /* D3D12 shared resource + shared ID3D12Fence */

    /*
     * Import a VRI-exported buffer + timeline fence, then on a CUDA stream:
     *   wait(fence == waitValue)  ->  kernel: x[i] = x[i] * 2 + 1  ->  signal(fence = signalValue)
     * and block until the stream finishes. The OS handles stay owned by the caller (it closes
     * them after this returns). Returns 0 on success, non-zero (a cudaError_t) on failure.
     *
     *   handleKind  : CUDA_INTEROP_KIND_* (Vulkan opaque vs D3D12 shared)
     *   memHandle   : VriExternalMemoryInfo::handle (Win32 HANDLE / fd)
     *   allocSize   : VriExternalMemoryInfo::size (total backing allocation)
     *   semHandle   : exported timeline-fence handle
     *   elementCount: number of uint32_t elements CUDA should transform
     */
    int cudaInteropRun(int      handleKind,
                       void*    memHandle,
                       uint64_t allocSize,
                       void*    semHandle,
                       uint32_t elementCount,
                       uint64_t waitValue,
                       uint64_t signalValue);

#ifdef __cplusplus
}
#endif

#endif /* VRI_EXAMPLE_CUDA_INTEROP_H */
