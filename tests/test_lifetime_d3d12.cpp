// D3D12 object lifetime: destroying a command allocator frees the command buffers created from it
// (VRI has no per-buffer destroy), so nothing the backend created keeps the ID3D12Device alive
// after vriDestroyDevice. D3D12 hands out one device per adapter and process: a device kept alive
// by a leaked command list is what the next device creation on that adapter gets back, with its old
// debug settings, and enabling GPU-based validation then removes it.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>

using Microsoft::WRL::ComPtr;

// The device of the adapter VRI's D3D12 backend picks: the first hardware adapter with feature
// level 11_0, otherwise WARP. D3D12CreateDevice returns the existing device of that adapter if
// anything still holds one.
static ComPtr<ID3D12Device> AdapterDevice()
{
    ComPtr<IDXGIFactory4> factory;
    REQUIRE(SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))));
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device>  device;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i)
    {
        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))))
            return device;
        adapter.Reset();
    }
    ComPtr<IDXGIAdapter> warp;
    REQUIRE(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))));
    REQUIRE(SUCCEEDED(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))));
    return device;
}

TEST_CASE("D3D12: destroying the device releases every command list its allocators created")
{
    VriDeviceCreationDesc dc = vriDefaultDeviceCreationDesc();
    dc.graphicsAPI           = VriGraphicsAPI_D3D12;
    dc.enableValidation      = VRI_TRUE;
    dc.bestEffort            = VRI_TRUE;
    VriDevice* dev           = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);
    VriCommandAllocator* alloc = nullptr;
    REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
    VriFence* fence = nullptr;
    REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);
    for (uint64_t value = 1; value <= 2; ++value)
    {
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
        VriFenceSubmitDesc signal {};
        signal.fence = fence;
        signal.value = value;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        REQUIRE(c.QueueSubmit(queue, &submit) == VriResult_Success);
        c.Wait(fence, value);
    }
    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    vriDestroyDevice(dev);

    const ComPtr<ID3D12Device> device = AdapterDevice();
    device->AddRef();
    const ULONG others = device->Release() - 1; // references besides this test's own
    CHECK(others == 0);
}
