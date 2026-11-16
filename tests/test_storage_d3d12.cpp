// D3D12 storage textures written by compute shaders. These tests turn on GPU-based validation
// before the device is created (it is a process-wide debug-layer setting, switched off again once
// the device is destroyed) and fail on any error the debug layer reports through the message callback:
// resource-state mistakes that drivers tolerate are only visible there. Changing the setting while
// another D3D12 device is alive in the process removes that device on WARP, so the tests then run
// without it and say so.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "shaders/tests/mip_reduce_dxbc.h"          // g_mipReduceDxbcCS
#include "shaders/tests/sample_filter_dxbc.h"       // g_sampleFilterDxbcCS
#include "shaders/tests/storage_constant_dxbc.h"    // g_storageConstantDxbcCS
#include "shaders/tests/storage_write_3d_dxbc.h"    // g_storageWrite3dDxbcCS
#include "shaders/tests/storage_write_array_dxbc.h" // g_storageWriteArrayDxbcCS
#include "shaders/tests/storage_write_dxbc.h"       // g_storageWriteDxbcCS
#include "shaders/tests/structured_read_dxbc.h"     // g_structuredReadDxbcCS
#include "shaders/tests/two_sets_dxbc.h"            // g_twoSetsDxbcCS

namespace
{
    struct Texel
    {
        float x, y, z, w;
    };

    void VRI_CALL CountErrors(void* user, VriMessageSeverity severity, const char* message)
    {
        if (severity == VriMessageSeverity_Error)
        {
            ++*static_cast<int*>(user);
            MESSAGE(std::string(message));
        }
    }

    void SetGpuBasedValidation(BOOL enable)
    {
        Microsoft::WRL::ComPtr<ID3D12Debug1> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
            debug->SetEnableGPUBasedValidation(enable);
    }

    // Whether any adapter already has a D3D12 device in this process: another test's that is still
    // referenced, or the one wgpu creates on WARP. D3D12CreateDevice returns an adapter's existing
    // device, so a reference count above ours shows one.
    bool AnyDeviceAlive()
    {
        Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory))))
            return true;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i)
        {
            Microsoft::WRL::ComPtr<ID3D12Device> device;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))))
            {
                device->AddRef();
                if (device->Release() > 1)
                    return true;
            }
            adapter.Reset();
        }
        return false;
    }

    // A D3D12 device, with GPU-based validation unless another device is alive, and one compute pipeline over one
    // descriptor set of `ranges`. Commands are recorded into `cmd` between Begin() and Submit().
    struct ComputeFixture
    {
        int                  errors = 0;
        VriCallbackInterface callback {&errors, CountErrors};
        VriDevice*           dev = nullptr;
        VriCoreInterface     c {};
        VriQueue*            queue      = nullptr; // runs the dispatches
        VriQueue*            graphics   = nullptr; // runs the readbacks
        VriPipelineLayout*   layout     = nullptr;
        VriPipeline*         pipeline   = nullptr;
        VriDescriptorPool*   pool       = nullptr;
        VriDescriptorSet*    set        = nullptr;
        VriCommandAllocator* alloc      = nullptr;
        VriCommandBuffer*    cmd        = nullptr;
        VriCommandAllocator* readAlloc  = nullptr;
        VriCommandBuffer*    readCmd    = nullptr;
        VriFence*            fence      = nullptr;
        uint64_t             fenceValue = 0;
        bool                 gbv        = false; // GPU-based validation, switched on by this fixture

        bool Create(const VriDescriptorRangeDesc* ranges,
                    uint32_t                      rangeNum,
                    const void*                   cs,
                    size_t                        csSize,
                    VriQueueType                  queueType = VriQueueType_Graphics,
                    uint32_t                      pushSize  = 0,
                    const VriDescriptorSetDesc*   secondSet = nullptr)
        {
            VriDeviceCreationDesc dc  = vriDefaultDeviceCreationDesc();
            dc.coordinates.clipSpaceY = VriClipSpaceY_Up;
            dc.graphicsAPI            = VriGraphicsAPI_D3D12;
            dc.enableValidation       = VRI_TRUE;
            dc.bestEffort             = VRI_TRUE;
            dc.callbackInterface      = &callback;
            gbv                       = !AnyDeviceAlive();
            if (gbv)
                SetGpuBasedValidation(TRUE);
            else
                MESSAGE("another D3D12 device is alive in this process; GPU-based validation stays off");
            const VriResult created = vriCreateDevice(&dc, &dev);
            if (created != VriResult_Success)
            {
                if (gbv)
                    SetGpuBasedValidation(FALSE);
                dev = nullptr;
                return false;
            }
            REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
            REQUIRE(c.GetQueue(dev, queueType, 0, &queue) == VriResult_Success);
            REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &graphics) == VriResult_Success);
            VriDescriptorSetDesc sets[2] {};
            sets[0].ranges   = ranges;
            sets[0].rangeNum = rangeNum;
            if (secondSet)
                sets[1] = *secondSet;
            VriPipelineLayoutDesc ld {};
            ld.descriptorSets   = sets;
            ld.descriptorSetNum = secondSet ? 2u : 1u;
            ld.shaderStages     = VriShaderStage_Compute;
            const VriPushConstantDesc push {0, pushSize, VriShaderStage_Compute};
            ld.pushConstants   = pushSize ? &push : nullptr;
            ld.pushConstantNum = pushSize ? 1u : 0u;
            REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);
            VriComputePipelineDesc pd {};
            pd.pipelineLayout        = layout;
            pd.shader.stage          = VriShaderStage_Compute;
            pd.shader.bytecode       = cs;
            pd.shader.bytecodeSize   = csSize;
            pd.shader.entryPointName = "computeMain";
            REQUIRE(c.CreateComputePipeline(dev, &pd, &pipeline) == VriResult_Success);
            VriDescriptorPoolDesc poolDesc {};
            poolDesc.descriptorSetMaxNum    = 1;
            poolDesc.textureMaxNum          = 4;
            poolDesc.storageTextureMaxNum   = 4;
            poolDesc.structuredBufferMaxNum = 4;
            poolDesc.samplerMaxNum          = 4;
            REQUIRE(c.CreateDescriptorPool(dev, &poolDesc, &pool) == VriResult_Success);
            REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);
            REQUIRE(c.CreateCommandAllocator(dev, queueType, &alloc) == VriResult_Success);
            REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
            REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &readAlloc) == VriResult_Success);
            REQUIRE(c.CreateCommandBuffer(readAlloc, &readCmd) == VriResult_Success);
            REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);
            return true;
        }

        ~ComputeFixture()
        {
            if (!dev)
                return;
            c.DestroyFence(fence);
            c.DestroyCommandAllocator(alloc);
            c.DestroyCommandAllocator(readAlloc);
            c.DestroyDescriptorPool(pool);
            c.DestroyPipeline(pipeline);
            c.DestroyPipelineLayout(layout);
            vriDestroyDevice(dev);
            if (gbv)
                SetGpuBasedValidation(FALSE);
        }

        VriTexture* Texture(VriTextureType type, uint32_t size, uint32_t depth, uint32_t mips, uint32_t layers)
        {
            VriTextureDesc td {};
            td.type           = type;
            td.format         = VriFormat_RGBA32_SFLOAT;
            td.width          = size;
            td.height         = size;
            td.depth          = depth;
            td.mipNum         = mips;
            td.layerNum       = layers;
            td.sampleNum      = 1;
            td.usage          = VriTextureUsage_ShaderResource | VriTextureUsage_ShaderResourceStorage;
            td.memoryLocation = VriMemoryLocation_Device;
            VriTexture* t     = nullptr;
            REQUIRE(c.CreateTexture(dev, &td, &t) == VriResult_Success);
            return t;
        }

        VriDescriptor*
        View(VriTexture* t, VriTextureViewType type, uint32_t baseMip, uint32_t baseLayer, uint32_t layerNum)
        {
            VriTextureViewDesc vd {};
            vd.texture          = t;
            vd.viewType         = type;
            vd.aspect           = VriImageAspect_Color;
            vd.baseMip          = baseMip;
            vd.mipNum           = 1;
            vd.baseLayer        = baseLayer;
            vd.layerNum         = layerNum;
            VriDescriptor* view = nullptr;
            REQUIRE(c.CreateTextureView(dev, &vd, &view) == VriResult_Success);
            return view;
        }

        void Bind(const VriDescriptor* const* views, uint32_t num)
        {
            std::vector<VriDescriptorRangeUpdateDesc> updates(num);
            for (uint32_t i = 0; i < num; ++i)
            {
                updates[i].descriptors   = &views[i];
                updates[i].descriptorNum = 1;
            }
            c.UpdateDescriptorRanges(set, 0, num, updates.data());
        }

        void Barrier(VriTexture* t, VriLayout before, VriLayout after, uint32_t baseMip, uint32_t mipNum)
        {
            VriTextureBarrierDesc tb {};
            tb.texture       = t;
            tb.before.layout = before;
            tb.after.layout  = after;
            tb.aspect        = VriImageAspect_Color;
            tb.baseMip       = baseMip;
            tb.mipNum        = mipNum;
            switch (after)
            {
                case VriLayout_ShaderResourceStorage:
                    tb.after.access = VriAccess_ShaderResourceStorageWrite;
                    tb.after.stages = VriPipelineStage_ComputeShader;
                    break;
                case VriLayout_ShaderResource:
                    tb.after.access = VriAccess_ShaderResourceRead;
                    tb.after.stages = VriPipelineStage_ComputeShader;
                    break;
                default:
                    tb.after.access = VriAccess_CopySourceRead;
                    tb.after.stages = VriPipelineStage_Transfer;
                    break;
            }
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        }

        void Begin() { REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success); }

        void Dispatch(uint32_t x, uint32_t y, uint32_t z)
        {
            c.CmdSetPipeline(cmd, pipeline);
            c.CmdSetPipelineLayout(cmd, layout);
            c.CmdSetDescriptorSet(cmd, 0, set);
            VriDispatchDesc groups {x, y, z};
            c.CmdDispatch(cmd, &groups);
        }

        void Submit() { Submit(cmd, queue); }

        void Submit(VriCommandBuffer* buffer, VriQueue* target)
        {
            REQUIRE(c.EndCommandBuffer(buffer) == VriResult_Success);
            VriFenceSubmitDesc signal {};
            signal.fence = fence;
            signal.value = ++fenceValue;
            VriQueueSubmitDesc submit {};
            submit.commandBuffers   = &buffer;
            submit.commandBufferNum = 1;
            submit.signalFences     = &signal;
            submit.signalFenceNum   = 1;
            REQUIRE(c.QueueSubmit(target, &submit) == VriResult_Success);
            c.Wait(fence, fenceValue);
        }

        // Reads one texel of a subresource (mip and layer) on the graphics queue. `offset` is the
        // texel's byte offset in the readback footprint (rows are 256 bytes apart).
        Texel Read(VriTexture* t, uint32_t mip, uint32_t layer, size_t offset = 0)
        {
            VriBufferDesc bd {};
            bd.size             = 256 * 64; // 256-byte rows; enough for the 16-texel tests
            bd.usage            = VriBufferUsage_TransferDst;
            bd.memoryLocation   = VriMemoryLocation_HostReadback;
            VriBuffer* readback = nullptr;
            REQUIRE(c.CreateBuffer(dev, &bd, &readback) == VriResult_Success);
            REQUIRE(c.BeginCommandBuffer(readCmd) == VriResult_Success);
            VriBufferTextureCopyDesc cp {};
            cp.texture.mip       = mip;
            cp.texture.baseLayer = layer;
            cp.texture.layerNum  = 1;
            cp.texture.aspect    = VriImageAspect_Color;
            c.CmdReadbackTextureToBuffer(readCmd, readback, t, &cp);
            Submit(readCmd, graphics);
            Texel texel {};
            std::memcpy(&texel, static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bd.size)) + offset, sizeof(texel));
            c.UnmapBuffer(readback);
            c.DestroyBuffer(readback);
            return texel;
        }
    };
} // namespace

TEST_CASE("D3D12: a storage texture is in the unordered-access state when a dispatch writes it")
{
    const VriDescriptorRangeDesc ranges[1] {{0, 1, VriDescriptorType_StorageTexture, VriShaderStage_Compute}};
    ComputeFixture               f;
    if (!f.Create(ranges, 1, g_storageWriteDxbcCS, sizeof(g_storageWriteDxbcCS)))
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    VriTexture*          t    = f.Texture(VriTextureType_2D, 16, 1, 1, 1);
    const VriDescriptor* view = f.View(t, VriTextureViewType_2D, 0, 0, 1);
    f.Bind(&view, 1);
    f.Begin();
    f.Barrier(t, VriLayout_Undefined, VriLayout_ShaderResourceStorage, 0, 0);
    f.Dispatch(2, 2, 1);
    // A second dependent write needs a UAV barrier between the dispatches.
    f.Barrier(t, VriLayout_ShaderResourceStorage, VriLayout_ShaderResourceStorage, 0, 0);
    f.Dispatch(2, 2, 1);
    f.Barrier(t, VriLayout_ShaderResourceStorage, VriLayout_CopySource, 0, 0);
    f.Submit();
    const Texel texel = f.Read(t, 0, 0);
    CHECK(texel.x == 16.0f);
    CHECK(texel.z == 7.0f);
    CHECK(f.errors == 0);
    f.c.DestroyDescriptor(const_cast<VriDescriptor*>(view));
    f.c.DestroyTexture(t);
}

TEST_CASE("D3D12: a compute queue hands a storage texture to the graphics queue as a shader resource")
{
    const VriDescriptorRangeDesc ranges[1] {{0, 1, VriDescriptorType_StorageTexture, VriShaderStage_Compute}};
    ComputeFixture               f;
    if (!f.Create(ranges, 1, g_storageWriteDxbcCS, sizeof(g_storageWriteDxbcCS), VriQueueType_Compute))
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    VriTextureDesc td {};
    td.type      = VriTextureType_2D;
    td.format    = VriFormat_RGBA32_SFLOAT;
    td.width     = 16;
    td.height    = 16;
    td.depth     = 1;
    td.mipNum    = 1;
    td.layerNum  = 1;
    td.sampleNum = 1;
    td.usage =
        VriTextureUsage_ShaderResource | VriTextureUsage_ShaderResourceStorage | VriTextureUsage_ConcurrentQueues;
    td.memoryLocation = VriMemoryLocation_Device;
    VriTexture* t     = nullptr;
    REQUIRE(f.c.CreateTexture(f.dev, &td, &t) == VriResult_Success);
    const VriDescriptor* view = f.View(t, VriTextureViewType_2D, 0, 0, 1);
    f.Bind(&view, 1);
    // Written on the compute queue and left readable by the graphics queue, as an async-compute
    // frame does; the readback then runs on the graphics queue.
    for (int frame = 0; frame < 2; ++frame)
    {
        f.Begin();
        f.Barrier(t, frame ? VriLayout_ShaderResource : VriLayout_Undefined, VriLayout_ShaderResourceStorage, 0, 0);
        f.Dispatch(2, 2, 1);
        f.Barrier(t, VriLayout_ShaderResourceStorage, VriLayout_ShaderResource, 0, 0);
        f.Submit();
        const Texel texel = f.Read(t, 0, 0);
        CHECK(texel.x == 16.0f);
        CHECK(texel.z == 7.0f);
    }
    CHECK(f.errors == 0);
    f.c.DestroyDescriptor(const_cast<VriDescriptor*>(view));
    f.c.DestroyTexture(t);
}

TEST_CASE("D3D12: a storage view writes the mip it selects")
{
    const VriDescriptorRangeDesc ranges[1] {{0, 1, VriDescriptorType_StorageTexture, VriShaderStage_Compute}};
    ComputeFixture               f;
    if (!f.Create(ranges, 1, g_storageWriteDxbcCS, sizeof(g_storageWriteDxbcCS)))
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    VriTexture* t = f.Texture(VriTextureType_2D, 16, 1, 3, 1);
    for (uint32_t mip = 1; mip < 3; ++mip)
    {
        CAPTURE(mip);
        const VriDescriptor* view = f.View(t, VriTextureViewType_2D, mip, 0, 1);
        f.Bind(&view, 1);
        f.Begin();
        f.Barrier(t, VriLayout_Undefined, VriLayout_ShaderResourceStorage, 0, 0);
        f.Dispatch(1, 1, 1);
        f.Barrier(t, VriLayout_ShaderResourceStorage, VriLayout_CopySource, 0, 0);
        f.Submit();
        // The shader sees the mip's own size and writes it there.
        const Texel texel = f.Read(t, mip, 0);
        CHECK(texel.x == float(16 >> mip));
        CHECK(texel.z == 7.0f);
        f.c.DestroyDescriptor(const_cast<VriDescriptor*>(view));
    }
    CHECK(f.errors == 0);
    f.c.DestroyTexture(t);
}

TEST_CASE("D3D12: an array storage view writes the layers it selects")
{
    const VriDescriptorRangeDesc ranges[1] {{0, 1, VriDescriptorType_StorageTexture, VriShaderStage_Compute}};
    ComputeFixture               f;
    if (!f.Create(ranges, 1, g_storageWriteArrayDxbcCS, sizeof(g_storageWriteArrayDxbcCS)))
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    VriTexture*          t    = f.Texture(VriTextureType_2DArray, 16, 1, 1, 3);
    const VriDescriptor* view = f.View(t, VriTextureViewType_2DArray, 0, 1, 2);
    f.Bind(&view, 1);
    f.Begin();
    f.Barrier(t, VriLayout_Undefined, VriLayout_ShaderResourceStorage, 0, 0);
    f.Dispatch(2, 2, 2);
    f.Barrier(t, VriLayout_ShaderResourceStorage, VriLayout_CopySource, 0, 0);
    f.Submit();
    for (uint32_t layer = 1; layer < 3; ++layer)
    {
        CAPTURE(layer);
        const Texel texel = f.Read(t, 0, layer);
        CHECK(texel.x == 16.0f);
        CHECK(texel.z == 2.0f); // the view's layer count
    }
    CHECK(f.errors == 0);
    f.c.DestroyDescriptor(const_cast<VriDescriptor*>(view));
    f.c.DestroyTexture(t);
}

TEST_CASE("D3D12: a 3D texture is written and read back per depth slice")
{
    const VriDescriptorRangeDesc ranges[1] {{0, 1, VriDescriptorType_StorageTexture, VriShaderStage_Compute}};
    ComputeFixture               f;
    if (!f.Create(ranges, 1, g_storageWrite3dDxbcCS, sizeof(g_storageWrite3dDxbcCS)))
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    VriTexture*          t    = f.Texture(VriTextureType_3D, 4, 4, 1, 1);
    const VriDescriptor* view = f.View(t, VriTextureViewType_3D, 0, 0, 1);
    f.Bind(&view, 1);
    f.Begin();
    f.Barrier(t, VriLayout_Undefined, VriLayout_ShaderResourceStorage, 0, 0);
    f.Dispatch(1, 1, 1);
    f.Barrier(t, VriLayout_ShaderResourceStorage, VriLayout_CopySource, 0, 0);
    f.Submit();
    // Texel (1, 2, 3): rows are 256 bytes apart, slices 4 rows apart, texels 16 bytes wide.
    const Texel texel = f.Read(t, 0, 0, 3 * 4 * 256 + 2 * 256 + 1 * 16);
    CHECK(texel.x == 1.0f);
    CHECK(texel.y == 2.0f);
    CHECK(texel.z == 3.0f);
    CHECK(f.errors == 0);
    f.c.DestroyDescriptor(const_cast<VriDescriptor*>(view));
    f.c.DestroyTexture(t);
}

TEST_CASE("D3D12: a mip reduction reads one mip while it writes the next")
{
    const VriDescriptorRangeDesc ranges[2] {{0, 1, VriDescriptorType_Texture, VriShaderStage_Compute},
                                            {1, 1, VriDescriptorType_StorageTexture, VriShaderStage_Compute}};
    ComputeFixture               f;
    if (!f.Create(ranges, 2, g_mipReduceDxbcCS, sizeof(g_mipReduceDxbcCS)))
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    VriTexture* t = f.Texture(VriTextureType_2D, 16, 1, 3, 1);
    f.Begin();
    f.Barrier(t, VriLayout_Undefined, VriLayout_ShaderResourceStorage, 0, 0);
    VriClearColor clear {};
    clear.f32[0] = 1.0f;
    clear.f32[1] = 2.0f;
    clear.f32[2] = 3.0f;
    clear.f32[3] = 4.0f;
    f.c.CmdClearStorageTexture(f.cmd, t, &clear); // mip 0
    f.Submit();
    // Each step samples mip - 1 and writes mip: only those two mips change state, so mip - 1
    // stays a shader resource while mip is an unordered-access target.
    for (uint32_t mip = 1; mip < 3; ++mip)
    {
        const VriDescriptor* views[2] {f.View(t, VriTextureViewType_2D, mip - 1, 0, 1),
                                       f.View(t, VriTextureViewType_2D, mip, 0, 1)};
        f.Bind(views, 2);
        f.Begin();
        f.Barrier(t, VriLayout_ShaderResourceStorage, VriLayout_ShaderResource, mip - 1, 1);
        f.Barrier(t, VriLayout_Undefined, VriLayout_ShaderResourceStorage, mip, 1);
        f.Dispatch(1, 1, 1);
        f.Submit();
        for (const VriDescriptor* view : views)
            f.c.DestroyDescriptor(const_cast<VriDescriptor*>(view));
    }
    f.Begin();
    f.Barrier(t, VriLayout_ShaderResourceStorage, VriLayout_CopySource, 0, 0);
    f.Submit();
    const Texel texel = f.Read(t, 2, 0);
    CHECK(texel.x == 1.0f);
    CHECK(texel.z == 5.0f); // 3 + 1 per reduction step
    CHECK(f.errors == 0);
    f.c.DestroyTexture(t);
}

TEST_CASE("D3D12: compute layouts, constants and descriptors bind before the pipeline")
{
    // On the compute queue, and on the graphics queue where compute and graphics share a list.
    for (const VriQueueType queueType : {VriQueueType_Compute, VriQueueType_Graphics})
    {
        CAPTURE(queueType);
        const VriDescriptorRangeDesc ranges[1] {{0, 1, VriDescriptorType_StorageTexture, VriShaderStage_Compute}};
        ComputeFixture               f;
        if (!f.Create(ranges, 1, g_storageConstantDxbcCS, sizeof(g_storageConstantDxbcCS), queueType, 16))
        {
            MESSAGE("D3D12 unavailable - skipping");
            return;
        }
        VriTexture*          t    = f.Texture(VriTextureType_2D, 16, 1, 1, 1);
        const VriDescriptor* view = f.View(t, VriTextureViewType_2D, 0, 0, 1);
        f.Bind(&view, 1);
        f.Begin();
        f.Barrier(t, VriLayout_Undefined, VriLayout_ShaderResourceStorage, 0, 0);
        // The layout and its bindings come before the pipeline, as a render graph records them.
        const float value[4] {0.25f, 0.5f, 0.75f, 1.0f};
        f.c.CmdSetPipelineLayout(f.cmd, f.layout);
        f.c.CmdSetDescriptorSet(f.cmd, 0, f.set);
        f.c.CmdSetConstants(f.cmd, 0, value, sizeof(value));
        f.c.CmdSetPipeline(f.cmd, f.pipeline);
        VriDispatchDesc groups {2, 2, 1};
        f.c.CmdDispatch(f.cmd, &groups);
        f.Barrier(t, VriLayout_ShaderResourceStorage, VriLayout_ShaderResource, 0, 0);
        f.Submit();
        const Texel texel = f.Read(t, 0, 0);
        CHECK(texel.x == 0.25f);
        CHECK(texel.z == 0.75f);
        CHECK(f.errors == 0);
        f.c.DestroyDescriptor(const_cast<VriDescriptor*>(view));
        f.c.DestroyTexture(t);
    }
}

TEST_CASE("D3D12: a structured buffer is readable through a descriptor table")
{
    const VriDescriptorRangeDesc ranges[2] {{0, 1, VriDescriptorType_StructuredBuffer, VriShaderStage_Compute},
                                            {1, 1, VriDescriptorType_StorageTexture, VriShaderStage_Compute}};
    ComputeFixture               f;
    if (!f.Create(ranges, 2, g_structuredReadDxbcCS, sizeof(g_structuredReadDxbcCS)))
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    const Texel   elements[2] {{9.0f, 9.0f, 9.0f, 9.0f}, {0.25f, 0.5f, 0.75f, 1.0f}};
    VriBufferDesc bd {};
    bd.size            = sizeof(elements);
    bd.structureStride = sizeof(Texel);
    bd.usage           = VriBufferUsage_StorageBuffer;
    bd.memoryLocation  = VriMemoryLocation_HostUpload;
    VriBuffer* input   = nullptr;
    REQUIRE(f.c.CreateBuffer(f.dev, &bd, &input) == VriResult_Success);
    std::memcpy(f.c.MapBuffer(input, 0, sizeof(elements)), elements, sizeof(elements));
    f.c.UnmapBuffer(input);
    VriBufferViewDesc vd {};
    vd.buffer                = input;
    vd.viewType              = VriDescriptorType_StructuredBuffer;
    VriDescriptor* inputView = nullptr;
    REQUIRE(f.c.CreateBufferView(f.dev, &vd, &inputView) == VriResult_Success);
    VriTexture*          t = f.Texture(VriTextureType_2D, 16, 1, 1, 1);
    const VriDescriptor* views[2] {inputView, f.View(t, VriTextureViewType_2D, 0, 0, 1)};
    f.Bind(views, 2);
    f.Begin();
    f.Barrier(t, VriLayout_Undefined, VriLayout_ShaderResourceStorage, 0, 0);
    f.Dispatch(2, 2, 1);
    f.Barrier(t, VriLayout_ShaderResourceStorage, VriLayout_CopySource, 0, 0);
    f.Submit();
    const Texel texel = f.Read(t, 0, 0);
    CHECK(texel.x == 0.25f);
    CHECK(texel.z == 0.75f);
    CHECK(f.errors == 0);
    for (const VriDescriptor* view : views)
        f.c.DestroyDescriptor(const_cast<VriDescriptor*>(view));
    f.c.DestroyTexture(t);
    f.c.DestroyBuffer(input);
}

TEST_CASE("D3D12: a readback copies only the region it names")
{
    const VriDescriptorRangeDesc ranges[1] {{0, 1, VriDescriptorType_StorageTexture, VriShaderStage_Compute}};
    ComputeFixture               f;
    if (!f.Create(ranges, 1, g_storageWrite3dDxbcCS, sizeof(g_storageWrite3dDxbcCS)))
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    VriTexture*          t    = f.Texture(VriTextureType_3D, 4, 4, 1, 1);
    const VriDescriptor* view = f.View(t, VriTextureViewType_3D, 0, 0, 1);
    f.Bind(&view, 1);
    f.Begin();
    f.Barrier(t, VriLayout_Undefined, VriLayout_ShaderResourceStorage, 0, 0);
    f.Dispatch(1, 1, 1);
    f.Barrier(t, VriLayout_ShaderResourceStorage, VriLayout_CopySource, 0, 0);
    f.Submit();
    // Depth slice 3, texels (1, 2) to (2, 3): a 2x2x1 box into a buffer sized for it alone (two
    // 256-byte rows), as a per-slice readback of a volume does.
    VriBufferDesc bd {};
    bd.size             = 2 * 256;
    bd.usage            = VriBufferUsage_TransferDst;
    bd.memoryLocation   = VriMemoryLocation_HostReadback;
    VriBuffer* readback = nullptr;
    REQUIRE(f.c.CreateBuffer(f.dev, &bd, &readback) == VriResult_Success);
    REQUIRE(f.c.BeginCommandBuffer(f.readCmd) == VriResult_Success);
    VriBufferTextureCopyDesc cp {};
    cp.texture.layerNum = 1;
    cp.texture.aspect   = VriImageAspect_Color;
    cp.texture.x        = 1;
    cp.texture.y        = 2;
    cp.texture.z        = 3;
    cp.texture.width    = 2;
    cp.texture.height   = 2;
    cp.texture.depth    = 1;
    f.c.CmdReadbackTextureToBuffer(f.readCmd, readback, t, &cp);
    f.Submit(f.readCmd, f.graphics);
    const auto* bytes = static_cast<const uint8_t*>(f.c.MapBuffer(readback, 0, bd.size));
    Texel       first {};
    Texel       last {};
    std::memcpy(&first, bytes, sizeof(Texel));
    std::memcpy(&last, bytes + 256 + sizeof(Texel), sizeof(Texel));
    f.c.UnmapBuffer(readback);
    CHECK(first.x == 1.0f);
    CHECK(first.y == 2.0f);
    CHECK(first.z == 3.0f);
    CHECK(last.x == 2.0f);
    CHECK(last.y == 3.0f);
    CHECK(f.errors == 0);
    f.c.DestroyBuffer(readback);
    f.c.DestroyDescriptor(const_cast<VriDescriptor*>(view));
    f.c.DestroyTexture(t);
}

TEST_CASE("D3D12: a storage-texture clear leaves the other mips in the caller's state")
{
    const VriDescriptorRangeDesc ranges[1] {{0, 1, VriDescriptorType_StorageTexture, VriShaderStage_Compute}};
    ComputeFixture               f;
    if (!f.Create(ranges, 1, g_storageWriteDxbcCS, sizeof(g_storageWriteDxbcCS)))
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    VriTexture*   target = f.Texture(VriTextureType_2D, 16, 1, 2, 1);
    VriTexture*   source = f.Texture(VriTextureType_2D, 8, 1, 1, 1);
    VriClearColor red {};
    red.f32[0] = 1.0f;
    red.f32[3] = 1.0f;
    VriClearColor blue {};
    blue.f32[2] = 1.0f;
    blue.f32[3] = 1.0f;
    // The clear is a transfer-domain operation: the caller barriers to CopyDestination, and a copy
    // into mip 1, which the clear does not touch, needs no further barrier.
    f.Begin();
    f.Barrier(source, VriLayout_Undefined, VriLayout_CopyDestination, 0, 0);
    f.Barrier(target, VriLayout_Undefined, VriLayout_CopyDestination, 0, 0);
    f.c.CmdClearStorageTexture(f.cmd, source, &red);
    f.c.CmdClearStorageTexture(f.cmd, target, &blue);
    f.Barrier(source, VriLayout_CopyDestination, VriLayout_CopySource, 0, 0);
    VriTextureCopyDesc copy {};
    copy.src.layerNum = 1;
    copy.src.aspect   = VriImageAspect_Color;
    copy.dst          = copy.src;
    copy.dst.mip      = 1;
    f.c.CmdCopyTexture(f.cmd, target, source, &copy);
    f.Barrier(target, VriLayout_CopyDestination, VriLayout_CopySource, 0, 0);
    f.Submit();
    const Texel mip0 = f.Read(target, 0, 0);
    const Texel mip1 = f.Read(target, 1, 0);
    CHECK(mip0.z == 1.0f);
    CHECK(mip1.x == 1.0f);
    CHECK(mip1.z == 0.0f);
    CHECK(f.errors == 0);
    f.c.DestroyTexture(source);
    f.c.DestroyTexture(target);
}

TEST_CASE("D3D12: descriptor sets from different pools bind together")
{
    const VriDescriptorRangeDesc first[2] {{0, 1, VriDescriptorType_StorageTexture, VriShaderStage_Compute},
                                           {1, 1, VriDescriptorType_Texture, VriShaderStage_Compute}};
    const VriDescriptorRangeDesc second[1] {{0, 1, VriDescriptorType_Texture, VriShaderStage_Compute}};
    VriDescriptorSetDesc         secondSet {};
    secondSet.registerSpace = 1;
    secondSet.ranges        = second;
    secondSet.rangeNum      = 1;
    ComputeFixture f;
    if (!f.Create(first, 2, g_twoSetsDxbcCS, sizeof(g_twoSetsDxbcCS), VriQueueType_Graphics, 0, &secondSet))
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    // Set 1 comes from a pool of its own, as a material set does next to a frame set.
    VriDescriptorPoolDesc poolDesc {};
    poolDesc.descriptorSetMaxNum = 1;
    poolDesc.textureMaxNum       = 1;
    VriDescriptorPool* pool      = nullptr;
    REQUIRE(f.c.CreateDescriptorPool(f.dev, &poolDesc, &pool) == VriResult_Success);
    VriDescriptorSet* set1 = nullptr;
    REQUIRE(f.c.AllocateDescriptorSets(pool, f.layout, 1, &set1, 1) == VriResult_Success);

    VriTexture*   target = f.Texture(VriTextureType_2D, 16, 1, 1, 1);
    VriTexture*   a      = f.Texture(VriTextureType_2D, 16, 1, 1, 1);
    VriTexture*   b      = f.Texture(VriTextureType_2D, 16, 1, 1, 1);
    VriClearColor one {};
    one.f32[0] = 1.0f;
    VriClearColor two {};
    two.f32[2] = 2.0f;
    f.Begin();
    f.Barrier(a, VriLayout_Undefined, VriLayout_CopyDestination, 0, 0);
    f.Barrier(b, VriLayout_Undefined, VriLayout_CopyDestination, 0, 0);
    f.c.CmdClearStorageTexture(f.cmd, a, &one);
    f.c.CmdClearStorageTexture(f.cmd, b, &two);
    f.Barrier(a, VriLayout_CopyDestination, VriLayout_ShaderResource, 0, 0);
    f.Barrier(b, VriLayout_CopyDestination, VriLayout_ShaderResource, 0, 0);
    f.Barrier(target, VriLayout_Undefined, VriLayout_ShaderResourceStorage, 0, 0);
    const VriDescriptor* views0[2] {f.View(target, VriTextureViewType_2D, 0, 0, 1),
                                    f.View(a, VriTextureViewType_2D, 0, 0, 1)};
    const VriDescriptor* view1 = f.View(b, VriTextureViewType_2D, 0, 0, 1);
    f.Bind(views0, 2);
    VriDescriptorRangeUpdateDesc update {};
    update.descriptors   = &view1;
    update.descriptorNum = 1;
    f.c.UpdateDescriptorRanges(set1, 0, 1, &update);
    f.c.CmdSetPipeline(f.cmd, f.pipeline);
    f.c.CmdSetPipelineLayout(f.cmd, f.layout);
    f.c.CmdSetDescriptorSet(f.cmd, 0, f.set);
    f.c.CmdSetDescriptorSet(f.cmd, 1, set1);
    VriDispatchDesc groups {2, 2, 1};
    f.c.CmdDispatch(f.cmd, &groups);
    f.Barrier(target, VriLayout_ShaderResourceStorage, VriLayout_CopySource, 0, 0);
    f.Submit();
    const Texel texel = f.Read(target, 0, 0);
    CHECK(texel.x == 1.0f);
    CHECK(texel.z == 2.0f);
    CHECK(f.errors == 0);
    for (const VriDescriptor* view : views0)
        f.c.DestroyDescriptor(const_cast<VriDescriptor*>(view));
    f.c.DestroyDescriptor(const_cast<VriDescriptor*>(view1));
    f.c.DestroyDescriptorPool(pool);
    for (VriTexture* t : {target, a, b})
        f.c.DestroyTexture(t);
}

TEST_CASE("D3D12: views may be destroyed once written into a descriptor set")
{
    const VriDescriptorRangeDesc ranges[2] {{0, 1, VriDescriptorType_StructuredBuffer, VriShaderStage_Compute},
                                            {1, 1, VriDescriptorType_StorageTexture, VriShaderStage_Compute}};
    ComputeFixture               f;
    if (!f.Create(ranges, 2, g_structuredReadDxbcCS, sizeof(g_structuredReadDxbcCS)))
    {
        MESSAGE("D3D12 unavailable - skipping");
        return;
    }
    const Texel   elements[2] {{9.0f, 9.0f, 9.0f, 9.0f}, {0.25f, 0.5f, 0.75f, 1.0f}};
    VriBufferDesc bd {};
    bd.size            = sizeof(elements);
    bd.structureStride = sizeof(Texel);
    bd.usage           = VriBufferUsage_StorageBuffer;
    bd.memoryLocation  = VriMemoryLocation_HostUpload;
    VriBuffer* input   = nullptr;
    REQUIRE(f.c.CreateBuffer(f.dev, &bd, &input) == VriResult_Success);
    std::memcpy(f.c.MapBuffer(input, 0, sizeof(elements)), elements, sizeof(elements));
    f.c.UnmapBuffer(input);
    VriBufferViewDesc vd {};
    vd.buffer                = input;
    vd.viewType              = VriDescriptorType_StructuredBuffer;
    VriDescriptor* inputView = nullptr;
    REQUIRE(f.c.CreateBufferView(f.dev, &vd, &inputView) == VriResult_Success);
    VriTexture*          t = f.Texture(VriTextureType_2D, 16, 1, 1, 1);
    const VriDescriptor* views[2] {inputView, f.View(t, VriTextureViewType_2D, 0, 0, 1)};
    f.Bind(views, 2);
    // The set holds copies of the descriptors: the views go before the set is bound.
    for (const VriDescriptor* view : views)
        f.c.DestroyDescriptor(const_cast<VriDescriptor*>(view));
    f.Begin();
    f.Barrier(t, VriLayout_Undefined, VriLayout_ShaderResourceStorage, 0, 0);
    f.Dispatch(2, 2, 1);
    f.Barrier(t, VriLayout_ShaderResourceStorage, VriLayout_CopySource, 0, 0);
    f.Submit();
    const Texel texel = f.Read(t, 0, 0);
    CHECK(texel.x == 0.25f);
    CHECK(texel.z == 0.75f);
    CHECK(f.errors == 0);
    f.c.DestroyTexture(t);
    f.c.DestroyBuffer(input);
}

TEST_CASE("D3D12: samplers filter magnification and minification separately")
{
    const VriDescriptorRangeDesc ranges[3] {{0, 1, VriDescriptorType_StorageTexture, VriShaderStage_Compute},
                                            {1, 1, VriDescriptorType_Texture, VriShaderStage_Compute},
                                            {2, 1, VriDescriptorType_Sampler, VriShaderStage_Compute}};
    for (const bool magnifyLinear : {false, true})
    {
        CAPTURE(magnifyLinear);
        ComputeFixture f;
        if (!f.Create(ranges, 3, g_sampleFilterDxbcCS, sizeof(g_sampleFilterDxbcCS)))
        {
            MESSAGE("D3D12 unavailable - skipping");
            return;
        }
        // A 2x2 source: red at (0, 0), black elsewhere.
        VriTexture*   source = f.Texture(VriTextureType_2D, 2, 1, 1, 1);
        VriTexture*   target = f.Texture(VriTextureType_2D, 16, 1, 1, 1);
        const Texel   texels[4] {{1, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}};
        VriBufferDesc bd {};
        bd.size           = sizeof(texels);
        bd.usage          = VriBufferUsage_TransferSrc;
        bd.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* upload = nullptr;
        REQUIRE(f.c.CreateBuffer(f.dev, &bd, &upload) == VriResult_Success);
        std::memcpy(f.c.MapBuffer(upload, 0, sizeof(texels)), texels, sizeof(texels));
        f.c.UnmapBuffer(upload);
        // The minification filter is the opposite of the magnification filter, which LOD 0 selects.
        VriSamplerDesc sd {};
        sd.magFilter       = magnifyLinear ? VriFilter_Linear : VriFilter_Nearest;
        sd.minFilter       = magnifyLinear ? VriFilter_Nearest : VriFilter_Linear;
        sd.addressModeU    = VriAddressMode_ClampToEdge;
        sd.addressModeV    = VriAddressMode_ClampToEdge;
        sd.addressModeW    = VriAddressMode_ClampToEdge;
        VriDescriptor* smp = nullptr;
        REQUIRE(f.c.CreateSampler(f.dev, &sd, &smp) == VriResult_Success);
        const VriDescriptor* views[3] {
            f.View(target, VriTextureViewType_2D, 0, 0, 1), f.View(source, VriTextureViewType_2D, 0, 0, 1), smp};
        f.Bind(views, 3);
        f.Begin();
        f.Barrier(source, VriLayout_Undefined, VriLayout_CopyDestination, 0, 0);
        VriBufferTextureCopyDesc cp {};
        cp.texture.layerNum = 1;
        cp.texture.aspect   = VriImageAspect_Color;
        f.c.CmdUploadBufferToTexture(f.cmd, source, upload, &cp);
        f.Barrier(source, VriLayout_CopyDestination, VriLayout_ShaderResource, 0, 0);
        f.Barrier(target, VriLayout_Undefined, VriLayout_ShaderResourceStorage, 0, 0);
        f.Dispatch(2, 2, 1);
        f.Barrier(target, VriLayout_ShaderResourceStorage, VriLayout_CopySource, 0, 0);
        f.Submit();
        // uv (0.4, 0.4) lies in texel (0, 0), between the four texel centers.
        const Texel texel = f.Read(target, 0, 0);
        if (magnifyLinear)
        {
            CHECK(texel.x > 0.0f);
            CHECK(texel.x < 1.0f);
        }
        else
            CHECK(texel.x == 1.0f);
        CHECK(f.errors == 0);
        for (const VriDescriptor* view : views)
            f.c.DestroyDescriptor(const_cast<VriDescriptor*>(view));
        f.c.DestroyBuffer(upload);
        f.c.DestroyTexture(source);
        f.c.DestroyTexture(target);
    }
}
