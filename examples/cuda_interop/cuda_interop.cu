// cuda_interop.cu - the CUDA consumer side of the VRI external-memory demo.
//
// Imports a VRI-exported buffer (external memory) and timeline fence (external semaphore)
// via the CUDA Runtime API, then on a stream: waits for the host's write, runs a kernel that
// transforms the shared buffer in place, and signals the fence back. VRI produced the handles
// through VRI_INTERFACE_EXTERNAL (ext/vri_ext_external.h) -- CUDA and the graphics queue share
// the *same* device memory and the *same* timeline, which is the whole point of interop.

#include "cuda_interop.h"

#include <cuda_runtime.h>
#include <cstdio>

namespace
{
    // Pick the CUDA import types matching what VRI exported (Vulkan opaque vs D3D12 shared).
    cudaExternalMemoryHandleType MemType(int kind)
    {
        if (kind == CUDA_INTEROP_KIND_D3D12)
            return cudaExternalMemoryHandleTypeD3D12Resource;
#if defined(_WIN32)
        return cudaExternalMemoryHandleTypeOpaqueWin32;
#else
        return cudaExternalMemoryHandleTypeOpaqueFd;
#endif
    }

    cudaExternalSemaphoreHandleType SemType(int kind)
    {
        if (kind == CUDA_INTEROP_KIND_D3D12)
            return cudaExternalSemaphoreHandleTypeD3D12Fence;
#if defined(_WIN32)
        return cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
#else
        return cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
#endif
    }

    bool Check(cudaError_t e, const char* what)
    {
        if (e != cudaSuccess)
        {
            std::fprintf(stderr, "[cuda] %s failed: %s\n", what, cudaGetErrorString(e));
            return false;
        }
        return true;
    }

    // The transform the host verifies after CUDA runs: x[i] = x[i] * 2 + 1.
    __global__ void TransformKernel(uint32_t* data, uint32_t count)
    {
        const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count)
            data[i] = data[i] * 2u + 1u;
    }

    void SetMemHandle(cudaExternalMemoryHandleDesc& d, void* handle)
    {
#if defined(_WIN32)
        d.handle.win32.handle = handle;
#else
        d.handle.fd = static_cast<int>(reinterpret_cast<intptr_t>(handle));
#endif
    }

    void SetSemHandle(cudaExternalSemaphoreHandleDesc& d, void* handle)
    {
#if defined(_WIN32)
        d.handle.win32.handle = handle;
#else
        d.handle.fd = static_cast<int>(reinterpret_cast<intptr_t>(handle));
#endif
    }
} // namespace

extern "C" int cudaInteropSupported(void)
{
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) ? 1 : 0;
}

extern "C" int cudaInteropPrintDevice(void)
{
    int dev = 0;
    if (!Check(cudaGetDevice(&dev), "cudaGetDevice"))
        return 1;
    cudaDeviceProp prop {};
    if (!Check(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties"))
        return 1;
    std::printf("[cuda] device %d: %s (compute %d.%d)\n", dev, prop.name, prop.major, prop.minor);
    return 0;
}

extern "C" int cudaInteropRun(int      handleKind,
                              void*    memHandle,
                              uint64_t allocSize,
                              void*    semHandle,
                              uint32_t elementCount,
                              uint64_t waitValue,
                              uint64_t signalValue)
{
    // ---- import the VRI buffer's backing memory as a CUDA device pointer ----
    cudaExternalMemory_t         extMem = nullptr;
    cudaExternalMemoryHandleDesc memDesc {};
    memDesc.type  = MemType(handleKind);
    memDesc.size  = allocSize;
    memDesc.flags = cudaExternalMemoryDedicated; // VRI exports dedicated allocations
    SetMemHandle(memDesc, memHandle);
    if (!Check(cudaImportExternalMemory(&extMem, &memDesc), "cudaImportExternalMemory"))
        return 1;

    void*                        devPtr = nullptr;
    cudaExternalMemoryBufferDesc bufDesc {};
    bufDesc.offset = 0;
    bufDesc.size   = static_cast<uint64_t>(elementCount) * sizeof(uint32_t);
    bufDesc.flags  = 0;
    if (!Check(cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufDesc), "cudaExternalMemoryGetMappedBuffer"))
    {
        cudaDestroyExternalMemory(extMem);
        return 1;
    }

    // ---- import the VRI timeline fence as a CUDA external semaphore ----
    cudaExternalSemaphore_t         extSem = nullptr;
    cudaExternalSemaphoreHandleDesc semDesc {};
    semDesc.type = SemType(handleKind);
    SetSemHandle(semDesc, semHandle);
    if (!Check(cudaImportExternalSemaphore(&extSem, &semDesc), "cudaImportExternalSemaphore"))
    {
        cudaFree(devPtr);
        cudaDestroyExternalMemory(extMem);
        return 1;
    }

    cudaStream_t stream = nullptr;
    int          rc     = 0;
    if (Check(cudaStreamCreate(&stream), "cudaStreamCreate"))
    {
        // wait for the host's upload (fence == waitValue) ...
        cudaExternalSemaphoreWaitParams waitParams {};
        waitParams.params.fence.value = waitValue;
        rc |= !Check(cudaWaitExternalSemaphoresAsync(&extSem, &waitParams, 1, stream),
                     "cudaWaitExternalSemaphoresAsync");

        // ... transform the shared buffer in place ...
        const uint32_t threads = 256;
        const uint32_t blocks  = (elementCount + threads - 1) / threads;
        TransformKernel<<<blocks, threads, 0, stream>>>(static_cast<uint32_t*>(devPtr), elementCount);
        rc |= !Check(cudaGetLastError(), "TransformKernel launch");

        // ... and signal the host that CUDA is done (fence = signalValue).
        cudaExternalSemaphoreSignalParams sigParams {};
        sigParams.params.fence.value = signalValue;
        rc |= !Check(cudaSignalExternalSemaphoresAsync(&extSem, &sigParams, 1, stream),
                     "cudaSignalExternalSemaphoresAsync");

        rc |= !Check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        cudaStreamDestroy(stream);
    }
    else
    {
        rc = 1;
    }

    cudaDestroyExternalSemaphore(extSem);
    cudaFree(devPtr);
    cudaDestroyExternalMemory(extMem);
    return rc;
}
