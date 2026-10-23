#include <vri/vri.h>

_Static_assert(sizeof(VriResult) == sizeof(int32_t), "Result width must be stable for C callers");

int vriTestTimelineC(VriDevice* device, size_t coreSize)
{
    VriCoreInterface   core   = {0};
    VriQueue*          queue  = NULL;
    VriFence*          fence  = NULL;
    VriQueueSubmitDesc submit = {0};
    VriFenceSubmitDesc signal = {0};
    VriResult          result;
    int                failure = 0;

    if (coreSize != sizeof(core) ||
        vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(core), &core) != VriResult_Success || !core.WaitForFence ||
        core.GetQueue(device, VriQueueType_Graphics, 0, &queue) != VriResult_Success ||
        core.CreateFence(device, 7, &fence) != VriResult_Success)
    {
        return __LINE__;
    }
#define VERIFY(expression) \
    do \
    { \
        if (!(expression)) \
        { \
            failure = __LINE__; \
            goto cleanup; \
        } \
    } while (0)

    VERIFY(core.GetFenceValue(fence) == 7);
    VERIFY(core.WaitForFence(fence, 7, 0) == VriResult_Success);
    VERIFY(core.WaitForFence(fence, 8, 0) == VriResult_Timeout);
    VERIFY(core.WaitForFence(fence, 8, 1000000) == VriResult_Timeout);
    VERIFY(core.GetFenceValue(fence) == 7);
    {
        unsigned int attempt;
        for (attempt = 0; attempt < 16; ++attempt)
        {
            VERIFY(core.WaitForFence(fence, 8, 10000) == VriResult_Timeout);
            VERIFY(core.GetFenceValue(fence) == 7);
        }
    }
    VERIFY(core.WaitForFence(NULL, 8, 0) == VriResult_InvalidArgument);
    VERIFY(core.QueueSubmit(queue, NULL) == VriResult_InvalidArgument);
    submit.commandBufferNum = 1;
    VERIFY(core.QueueSubmit(queue, &submit) == VriResult_InvalidArgument);
    submit.commandBufferNum = 0;
    signal.fence            = fence;
    signal.value            = 8;
    submit.signalFences     = &signal;
    submit.signalFenceNum   = 1;
    VERIFY(core.QueueSubmit(queue, &submit) == VriResult_Success);
    VERIFY(core.WaitForFence(fence, 8, UINT64_C(5000000000)) == VriResult_Success);
    VERIFY(core.GetFenceValue(fence) == 8);
    core.Wait(fence, 8);
    {
        VriFenceSubmitDesc wait = signal;
        signal.value            = 9;
        submit.waitFences       = &wait;
        submit.waitFenceNum     = 1;
        VERIFY(core.QueueSubmit(queue, &submit) == VriResult_Success);
    }
    VERIFY(core.WaitForFence(fence, 9, UINT64_C(5000000000)) == VriResult_Success);
    VERIFY(core.GetFenceValue(fence) == 9);
    submit.waitFences   = NULL;
    submit.waitFenceNum = 0;
    {
        uint64_t value;
        for (value = 10; value <= 41; ++value)
        {
            signal.value = value;
            VERIFY(core.QueueSubmit(queue, &submit) == VriResult_Success);
        }
    }
    VERIFY(core.WaitForFence(fence, 41, UINT64_C(5000000000)) == VriResult_Success);
    VERIFY(core.GetFenceValue(fence) == 41);

cleanup:
    result = core.QueueWaitIdle(queue);
    if (result != VriResult_Success && !failure)
    {
        failure = __LINE__;
    }
    result = core.DeviceWaitIdle(device);
    if (result != VriResult_Success && !failure)
    {
        failure = __LINE__;
    }
    core.DestroyFence(fence);
#undef VERIFY
    return failure;
}
