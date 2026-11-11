#include <doctest/doctest.h>
#include <vri/vri.h>

#include <array>
#include <cstring>

#include "shaders/tests/buffer_dimensions_spv.h"

TEST_CASE("Metal buffer dimensions use descriptor ranges across encoder changes")
{
    VriDeviceCreationDesc dc {};
    dc.graphicsAPI      = VriGraphicsAPI_Metal;
    dc.enableValidation = VRI_TRUE;
    dc.enabledFeatures  = VriFeature_Bindless;
    VriDevice* dev      = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("Metal with bindless unavailable");
        return;
    }
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    VriQueue* queue = nullptr;
    REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

    for (bool argumentBuffers : {false, true})
    {
        CAPTURE(argumentBuffers);
        VriBufferDesc bd {};
        bd.size           = 1024;
        bd.usage          = VriBufferUsage_StorageBuffer | VriBufferUsage_TransferSrc;
        bd.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* data   = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &data) == VriResult_Success);
        auto* values = static_cast<float*>(c.MapBuffer(data, 0, bd.size));
        REQUIRE(values);
        std::memset(values, 0, bd.size);
        values[256 / sizeof(float)] = 17.f;
        values[512 / sizeof(float)] = 29.f;
        c.UnmapBuffer(data);
        VriBuffer* output = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &output) == VriResult_Success);
        bd.size             = 96;
        bd.usage            = VriBufferUsage_TransferDst;
        bd.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &readback) == VriResult_Success);

        VriDescriptorRangeDesc ranges[2] {};
        VriDescriptorSetDesc   sets[2] {};
        for (unsigned i = 0; i < 2; ++i)
        {
            ranges[i].descriptorNum  = 1;
            ranges[i].descriptorType = i ? VriDescriptorType_StorageBuffer : VriDescriptorType_StructuredBuffer;
            ranges[i].shaderStages   = VriShaderStage_Compute;
            ranges[i].flags          = argumentBuffers ? VriDescriptorRange_PartiallyBound : 0;
            sets[i].registerSpace    = i;
            sets[i].ranges           = &ranges[i];
            sets[i].rangeNum         = 1;
        }
        VriPipelineLayoutDesc ld {};
        ld.descriptorSets         = sets;
        ld.descriptorSetNum       = 2;
        VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);
        VriComputePipelineDesc pd {};
        pd.pipelineLayout        = layout;
        pd.shader.stage          = VriShaderStage_Compute;
        pd.shader.bytecode       = g_bufferDimensionsSpv;
        pd.shader.bytecodeSize   = sizeof(g_bufferDimensionsSpv);
        pd.shader.entryPointName = "computeMain";
        VriPipeline* pipeline    = nullptr;
        REQUIRE(c.CreateComputePipeline(dev, &pd, &pipeline) == VriResult_Success);
        VriDescriptorPoolDesc poolDesc {};
        poolDesc.descriptorSetMaxNum    = 3;
        poolDesc.structuredBufferMaxNum = 2;
        poolDesc.storageBufferMaxNum    = 1;
        VriDescriptorPool* pool         = nullptr;
        REQUIRE(c.CreateDescriptorPool(dev, &poolDesc, &pool) == VriResult_Success);
        std::array<VriDescriptorSet*, 3> descriptors {};
        std::array<VriDescriptor*, 3>    views {};
        for (unsigned i = 0; i < 3; ++i)
        {
            const unsigned setIndex = i == 2 ? 1 : 0;
            REQUIRE(c.AllocateDescriptorSets(pool, layout, setIndex, &descriptors[i], 1) == VriResult_Success);
            VriBufferViewDesc view {};
            view.buffer   = i == 2 ? output : data;
            view.viewType = ranges[setIndex].descriptorType;
            view.offset   = i == 1 ? 512 : 256;
            view.size     = i == 0 ? 80 : i == 2 ? 32 : 0;
            REQUIRE(c.CreateBufferView(dev, &view, &views[i]) == VriResult_Success);
            const VriDescriptor*         descriptor = views[i];
            VriDescriptorRangeUpdateDesc update {};
            update.descriptors   = &descriptor;
            update.descriptorNum = 1;
            c.UpdateDescriptorRanges(descriptors[i], 0, 1, &update);
        }
        VriCommandAllocator* allocator = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &allocator) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(allocator, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);
        VriTextureDesc scratchDesc {};
        scratchDesc.type   = VriTextureType_2D;
        scratchDesc.format = VriFormat_R32_UINT;
        scratchDesc.width  = 8;
        scratchDesc.height = 8;
        scratchDesc.depth = scratchDesc.layerNum = scratchDesc.mipNum = scratchDesc.sampleNum = 1;
        scratchDesc.usage          = VriTextureUsage_ShaderResourceStorage | VriTextureUsage_TransferDst;
        scratchDesc.memoryLocation = VriMemoryLocation_Device;
        VriTexture* scratch        = nullptr;
        REQUIRE(c.CreateTexture(dev, &scratchDesc, &scratch) == VriResult_Success);
        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetPipeline(cmd, pipeline);
        c.CmdSetDescriptorSet(cmd, 1, descriptors[2]);
        VriDispatchDesc dispatch {1, 1, 1};
        for (unsigned i = 0; i < 3; ++i)
        {
            // The third dispatch must restore both sets and lengths after a blit,
            // without another pipeline or descriptor command.
            if (i < 2)
                c.CmdSetDescriptorSet(cmd, 0, descriptors[i]);
            VriClearColor color {};
            color.u32[0] = 7;
            c.CmdClearStorageTexture(cmd, scratch, &color);
            c.CmdDispatch(cmd, &dispatch);
            VriBufferCopyDesc copy {};
            copy.srcOffset = 256;
            copy.dstOffset = i * 32;
            copy.size      = 32;
            c.CmdCopyBuffer(cmd, readback, output, &copy);
        }
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
        VriFenceSubmitDesc signal {};
        signal.fence = fence;
        signal.value = 1;
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        c.QueueSubmit(queue, &submit);
        c.Wait(fence, 1);
        const auto* result = static_cast<const uint32_t*>(c.MapBuffer(readback, 0, 96));
        REQUIRE(result);
        for (unsigned i = 0; i < 3; ++i)
        {
            CHECK(result[i * 8] == (i == 0 ? 5 : 32));
            CHECK(result[i * 8 + 1] == 8);
            CHECK(result[i * 8 + 2] == (i == 0 ? 17 : 29));
        }
        c.UnmapBuffer(readback);
        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(allocator);
        c.DestroyDescriptorPool(pool);
        for (auto* view : views)
            c.DestroyDescriptor(view);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(output);
        c.DestroyBuffer(data);
        c.DestroyTexture(scratch);
    }
    vriDestroyDevice(dev);
}
