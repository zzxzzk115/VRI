// D3D12 storage textures written by compute shaders. These tests turn on GPU-based validation
// before the device is created (it is a process-wide debug-layer setting, switched off again once
// the device is destroyed) and fail on any error the debug layer reports through the message callback:
// resource-state mistakes that drivers tolerate are only visible there.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#include <d3d12.h>
#include <wrl/client.h>

#include "shaders/tests/storage_write_dxbc.h" // g_storageWriteDxbcCS

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

    // A D3D12 device with GPU-based validation and one compute pipeline over one descriptor set
    // of `ranges`. Commands are recorded into `cmd` between Begin() and Submit().
    struct ComputeFixture
    {
        int                  errors = 0;
        VriCallbackInterface callback {&errors, CountErrors};
        VriDevice*           dev = nullptr;
        VriCoreInterface     c {};
        VriQueue*            queue      = nullptr;
        VriPipelineLayout*   layout     = nullptr;
        VriPipeline*         pipeline   = nullptr;
        VriDescriptorPool*   pool       = nullptr;
        VriDescriptorSet*    set        = nullptr;
        VriCommandAllocator* alloc      = nullptr;
        VriCommandBuffer*    cmd        = nullptr;
        VriFence*            fence      = nullptr;
        uint64_t             fenceValue = 0;

        bool Create(const VriDescriptorRangeDesc* ranges, uint32_t rangeNum, const void* cs, size_t csSize)
        {
            VriDeviceCreationDesc dc  = vriDefaultDeviceCreationDesc();
            dc.coordinates.clipSpaceY = VriClipSpaceY_Up;
            dc.graphicsAPI            = VriGraphicsAPI_D3D12;
            dc.enableValidation       = VRI_TRUE;
            dc.bestEffort             = VRI_TRUE;
            dc.callbackInterface      = &callback;
            SetGpuBasedValidation(TRUE);
            const VriResult created = vriCreateDevice(&dc, &dev);
            if (created != VriResult_Success)
            {
                SetGpuBasedValidation(FALSE);
                dev = nullptr;
                return false;
            }
            REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
            REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);
            VriDescriptorSetDesc sd {};
            sd.ranges   = ranges;
            sd.rangeNum = rangeNum;
            VriPipelineLayoutDesc ld {};
            ld.descriptorSets   = &sd;
            ld.descriptorSetNum = 1;
            ld.shaderStages     = VriShaderStage_Compute;
            REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);
            VriComputePipelineDesc pd {};
            pd.pipelineLayout        = layout;
            pd.shader.stage          = VriShaderStage_Compute;
            pd.shader.bytecode       = cs;
            pd.shader.bytecodeSize   = csSize;
            pd.shader.entryPointName = "computeMain";
            REQUIRE(c.CreateComputePipeline(dev, &pd, &pipeline) == VriResult_Success);
            VriDescriptorPoolDesc pool_desc {};
            pool_desc.descriptorSetMaxNum  = 1;
            pool_desc.textureMaxNum        = 4;
            pool_desc.storageTextureMaxNum = 4;
            REQUIRE(c.CreateDescriptorPool(dev, &pool_desc, &pool) == VriResult_Success);
            REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);
            REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
            REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
            REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);
            return true;
        }

        ~ComputeFixture()
        {
            if (!dev)
                return;
            c.DestroyFence(fence);
            c.DestroyCommandAllocator(alloc);
            c.DestroyDescriptorPool(pool);
            c.DestroyPipeline(pipeline);
            c.DestroyPipelineLayout(layout);
            vriDestroyDevice(dev);
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

        void Submit()
        {
            REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
            VriFenceSubmitDesc signal {};
            signal.fence = fence;
            signal.value = ++fenceValue;
            VriQueueSubmitDesc submit {};
            submit.commandBuffers   = &cmd;
            submit.commandBufferNum = 1;
            submit.signalFences     = &signal;
            submit.signalFenceNum   = 1;
            REQUIRE(c.QueueSubmit(queue, &submit) == VriResult_Success);
            c.Wait(fence, fenceValue);
        }

        // Reads texel (0, 0) of one subresource (2D: mip/layer, 3D: mip and depth slice z).
        Texel Read(VriTexture* t, uint32_t mip, uint32_t layer, int32_t z = 0)
        {
            VriBufferDesc bd {};
            bd.size             = 256 * 64; // 256-byte rows; enough for the 16-texel tests
            bd.usage            = VriBufferUsage_TransferDst;
            bd.memoryLocation   = VriMemoryLocation_HostReadback;
            VriBuffer* readback = nullptr;
            REQUIRE(c.CreateBuffer(dev, &bd, &readback) == VriResult_Success);
            Begin();
            VriBufferTextureCopyDesc cp {};
            cp.texture.mip       = mip;
            cp.texture.baseLayer = layer;
            cp.texture.layerNum  = 1;
            cp.texture.aspect    = VriImageAspect_Color;
            cp.texture.z         = z;
            c.CmdReadbackTextureToBuffer(cmd, readback, t, &cp);
            Submit();
            Texel texel {};
            std::memcpy(&texel, c.MapBuffer(readback, 0, sizeof(texel)), sizeof(texel));
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
