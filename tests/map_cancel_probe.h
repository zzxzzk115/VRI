// Browser WebGPU: what becomes of a MapBufferAsync that is still in flight. DestroyBuffer cancels it: the callback
// runs once, with VriResult_Failure, before DestroyBuffer returns, and the browser's late answer neither calls back
// again nor reports an error. UnmapBuffer while in flight is recorded too. Only meaningful in the browser, where
// the map completes from the event loop.
#pragma once

#include <vri/vri.h>

#include <cstdio>
#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

namespace vri::test
{
    struct MapCancelState
    {
        int       calls  = 0;
        VriResult result = VriResult_Success;
        void*     data   = nullptr;
    };

    inline void VRI_CALL RecordCancelledMap(void* userData, VriResult result, void* data)
    {
        auto* state = static_cast<MapCancelState*>(userData);
        ++state->calls;
        state->result = result;
        state->data   = data;
    }

    inline void VRI_CALL CountMapErrors(void* userArg, VriMessageSeverity severity, const char*)
    {
        if (severity == VriMessageSeverity_Error)
            ++*static_cast<int*>(userArg);
    }

    inline void YieldToBrowser()
    {
#if defined(__EMSCRIPTEN__)
        for (int i = 0; i < 50; ++i)
            emscripten_sleep(1);
#endif
    }

    inline VriBuffer* CreateReadback(VriDevice* device, const VriCoreInterface& c)
    {
        VriBufferDesc bd {};
        bd.size           = 256;
        bd.usage          = VriBufferUsage_TransferDst;
        bd.memoryLocation = VriMemoryLocation_HostReadback;
        VriBuffer* buffer = nullptr;
        return c.CreateBuffer(device, &bd, &buffer) == VriResult_Success ? buffer : nullptr;
    }

    // Returns 0 when DestroyBuffer cancelled the map as specified, else the failing line. `unmapCalls`,
    // `unmapResult` and `unmapErrors` report what UnmapBuffer during the map led to, once the page had yielded.
    inline int CheckMapCancelledByDestroy(int& unmapCalls, VriResult& unmapResult, int& unmapErrors)
    {
        int                   errors    = 0;
        VriCallbackInterface  callbacks = {&errors, CountMapErrors};
        VriDeviceCreationDesc dc        = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI                  = VriGraphicsAPI_WebGPU;
        dc.callbackInterface            = &callbacks;
        VriDevice* device               = nullptr;
        if (vriCreateDevice(&dc, &device) != VriResult_Success)
            return __LINE__;
        VriCoreInterface c {};
        if (vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(c), &c) != VriResult_Success)
            return __LINE__;

        int failure = 0;
        // DestroyBuffer with the map in flight: one failed callback before it returns.
        MapCancelState destroyed;
        VriBuffer*     buffer = CreateReadback(device, c);
        if (!buffer || c.MapBufferAsync(buffer, 0, 256, RecordCancelledMap, &destroyed) != VriResult_Success)
            failure = __LINE__;
        if (!failure && destroyed.calls != 0)
            failure = __LINE__; // the browser map cannot have completed yet
        c.DestroyBuffer(buffer);
        if (!failure && (destroyed.calls != 1 || destroyed.result == VriResult_Success || destroyed.data))
            failure = __LINE__;
        YieldToBrowser(); // the browser's answer to the cancelled map arrives here
        if (!failure && (destroyed.calls != 1 || errors != 0))
            failure = __LINE__;

        // UnmapBuffer with the map in flight: recorded, not required to succeed.
        MapCancelState unmapped;
        const int      errorsBeforeUnmap = errors;
        buffer                           = CreateReadback(device, c);
        if (!failure && buffer && c.MapBufferAsync(buffer, 0, 256, RecordCancelledMap, &unmapped) == VriResult_Success)
        {
            c.UnmapBuffer(buffer);
            YieldToBrowser();
        }
        unmapCalls  = unmapped.calls;
        unmapResult = unmapped.result;
        unmapErrors = errors - errorsBeforeUnmap;
        if (buffer)
            c.DestroyBuffer(buffer);

        // The device is still usable: a fresh map completes.
        MapCancelState fresh;
        buffer = CreateReadback(device, c);
        if (!failure && (!buffer || c.MapBufferAsync(buffer, 0, 256, RecordCancelledMap, &fresh) != VriResult_Success))
            failure = __LINE__;
        for (int i = 0; i < 10000 && !failure && fresh.calls == 0; ++i)
            YieldToBrowser();
        if (!failure && (fresh.calls != 1 || fresh.result != VriResult_Success || !fresh.data))
            failure = __LINE__;
        if (buffer)
        {
            if (fresh.data)
                c.UnmapBuffer(buffer);
            c.DestroyBuffer(buffer);
        }
        vriDestroyDevice(device);
        return failure;
    }
} // namespace vri::test
