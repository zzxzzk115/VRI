// Browser WebGPU: what becomes of a MapBufferAsync that has not called back yet. DestroyBuffer cancels it: the
// callback runs once, with VriResult_Failure, before DestroyBuffer returns, and nothing more arrives, neither a
// second callback nor an error. UnmapBuffer aborts it inside the call, so a map issued after it is the only one
// left for DestroyBuffer to cancel. Only meaningful in the browser, where a map completes from the event loop.
#pragma once

#include <vri/vri.h>

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

    inline bool Failed(const MapCancelState& state) { return state.calls == 1 && state.result != VriResult_Success; }

    // Returns 0 when in-flight maps ended as specified, else the failing line.
    inline int CheckMapCancelledByDestroy()
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
        // DestroyBuffer with the map in flight: one failed callback before it returns, nothing after.
        MapCancelState destroyed;
        VriBuffer*     buffer = CreateReadback(device, c);
        if (!buffer || c.MapBufferAsync(buffer, 0, 256, RecordCancelledMap, &destroyed) != VriResult_Success)
            failure = __LINE__;
        if (!failure && destroyed.calls != 0)
            failure = __LINE__; // the browser map cannot have completed yet
        if (buffer)
            c.DestroyBuffer(buffer);
        if (!failure && (!Failed(destroyed) || destroyed.data))
            failure = __LINE__;
        YieldToBrowser();
        if (!failure && (destroyed.calls != 1 || errors != 0))
            failure = __LINE__;

        // UnmapBuffer aborts the map inside the call and reports an error; DestroyBuffer then cancels only the map
        // issued after it.
        MapCancelState aborted;
        MapCancelState remapped;
        buffer = CreateReadback(device, c);
        if (!failure &&
            (!buffer || c.MapBufferAsync(buffer, 0, 256, RecordCancelledMap, &aborted) != VriResult_Success))
            failure = __LINE__;
        if (!failure)
        {
            c.UnmapBuffer(buffer);
            if (!Failed(aborted) || errors != 1)
                failure = __LINE__;
        }
        if (!failure && c.MapBufferAsync(buffer, 0, 256, RecordCancelledMap, &remapped) != VriResult_Success)
            failure = __LINE__;
        if (buffer)
            c.DestroyBuffer(buffer);
        if (!failure && (aborted.calls != 1 || !Failed(remapped)))
            failure = __LINE__;
        YieldToBrowser();
        if (!failure && (aborted.calls != 1 || remapped.calls != 1 || errors != 1))
            failure = __LINE__;

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
