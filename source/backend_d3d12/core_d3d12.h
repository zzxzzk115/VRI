// core_d3d12.h - accessor for the D3D12 core function table.
#pragma once

#include <vri/vri_core.h>

namespace vri::d3d12
{
    struct QueueD3D12;

    const VriCoreInterface* GetCoreInterfaceD3D12();

    // Signals the queue's idle fence and waits on the host until the queue has finished everything submitted to it,
    // presents included (QueueWaitIdle; the swapchain also uses it before releasing its buffers).
    VriResult WaitQueueIdle(QueueD3D12* queue);
} // namespace vri::d3d12
