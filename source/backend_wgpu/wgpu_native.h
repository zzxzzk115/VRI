// wgpu_native.h - isolates wgpu-native-only bits from browser WebGPU.
//
// <webgpu/webgpu.h> (the standard API) is available on both native (wgpu-native
// via webgpu-sdk) and Emscripten (browser WebGPU via webgpu-sdk). Only
// <webgpu/wgpu.h> (wgpu-native extensions: wgpuDevicePoll, wgpuInstanceProcessEvents)
// is native-only. On Emscripten a wait instead yields to the browser event loop with
// emscripten_sleep so the async WebGPU callbacks fire, which only a program linked with
// -sASYNCIFY can do.
#pragma once

#include <webgpu/webgpu.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#else
#include <webgpu/wgpu.h>
#endif

namespace vri::wgpu
{
    // Async callbacks: native pumps them via wgpuInstanceProcessEvents/wgpuDevicePoll
    // (AllowProcessEvents); the browser resolves promises via its event loop, so they
    // must be allowed to fire spontaneously (during emscripten_sleep).
#if defined(__EMSCRIPTEN__)
    constexpr WGPUCallbackMode kCallbackMode = WGPUCallbackMode_AllowSpontaneous;
#else
    constexpr WGPUCallbackMode kCallbackMode = WGPUCallbackMode_AllowProcessEvents;
#endif

    // Whether this program can block until WebGPU work completes. The browser runs WebGPU's
    // callbacks from its event loop, so a wait there must yield, and only Asyncify can yield.
    inline bool CanWait()
    {
#if defined(__EMSCRIPTEN__)
        return emscripten_has_asyncify() != 0;
#else
        return true;
#endif
    }

    // Pump pending events while waiting on an async request (adapter/device/map).
    inline void PumpInstance(WGPUInstance instance)
    {
#if defined(__EMSCRIPTEN__)
        (void)instance;
        if (CanWait())
            emscripten_sleep(1); // yield to the browser; pending promises resolve
#else
        wgpuInstanceProcessEvents(instance);
#endif
    }

    // Block (native) / yield (browser) until the device's submitted work settles. Without
    // Asyncify the browser keeps the released objects alive until their work is done itself.
    inline void PollDevice(WGPUDevice device)
    {
#if defined(__EMSCRIPTEN__)
        (void)device;
        if (CanWait())
            emscripten_sleep(1); // real readback sync happens at buffer mapAsync
#else
        wgpuDevicePoll(device, /*wait*/ true, nullptr);
#endif
    }
} // namespace vri::wgpu
