#include "shaders/tests/neural_inference_spv.h"

#include <vri/vri.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" int vriTestCooperativeVectorC(VriDevice* device, size_t tableSize, size_t infoSize, uint32_t* propertyCount);

namespace
{
    constexpr uint32_t kSampleNum = 65;

    struct Context
    {
        VriDevice*                    device = nullptr;
        VriCoreInterface              core {};
        VriCooperativeVectorInterface neural {};
        int                           errors = 0;

        ~Context()
        {
            if (device)
                vriDestroyDevice(device);
        }

        VriResult Init(VriBool enabled, VriBool validation, VriBool bestEffort = VRI_TRUE)
        {
            VriCallbackInterface callback {};
            callback.userArg         = this;
            callback.MessageCallback = [](void* user, VriMessageSeverity severity, const char* message) {
                if (severity == VriMessageSeverity_Error)
                    ++static_cast<Context*>(user)->errors;
                std::fprintf(stderr, "[cooperative vector] %s\n", message);
            };
            VriDeviceCreationDesc desc  = vriDefaultDeviceCreationDesc();
            desc.coordinates.clipSpaceY = VriClipSpaceY_Up;
            desc.graphicsAPI            = VriGraphicsAPI_Vulkan;
            desc.enabledFeatures        = enabled ? VriFeature_CooperativeVector : VriFeature_None;
            desc.enableValidation       = validation;
            desc.bestEffort             = bestEffort;
            desc.callbackInterface      = &callback;
            const auto result           = vriCreateDevice(&desc, &device);
            if (result == VriResult_Success)
            {
                REQUIRE(vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(core), &core) == VriResult_Success);
                vriGetInterface(device, VRI_INTERFACE_COOPERATIVE_VECTOR, sizeof(neural), &neural);
            }
            return result;
        }
    };

    bool HasVulkan() { return vriEnumerateAdapters(VriGraphicsAPI_Vulkan, nullptr, 0) != 0; }

    std::vector<VriCooperativeVectorProperties> Combinations(Context& ctx)
    {
        uint32_t count = 0;
        REQUIRE(ctx.neural.GetCooperativeVectorProperties(ctx.device, &count, nullptr) == VriResult_Success);
        REQUIRE(count > 0);
        std::vector<VriCooperativeVectorProperties> result(count);
        REQUIRE(ctx.neural.GetCooperativeVectorProperties(ctx.device, &count, result.data()) == VriResult_Success);
        REQUIRE(count == result.size());
        return result;
    }

    uint16_t HalfSixteenth(int numerator)
    {
        // Binary-exact fixtures avoid depending on another half-conversion implementation.
        constexpr uint16_t magnitudes[] = {0, 0x2c00, 0x3000, 0x3200, 0x3400, 0x3500, 0x3600, 0x3700, 0x3800};
        return static_cast<uint16_t>(magnitudes[std::abs(numerator)] | (numerator < 0 ? 0x8000 : 0));
    }

    VriConvertCooperativeVectorMatrixInfo MatrixDesc(uint32_t rows, uint32_t columns)
    {
        VriConvertCooperativeVectorMatrixInfo desc {};
        desc.srcComponentType = VriComponentType_Float16;
        desc.dstComponentType = VriComponentType_Float16;
        desc.numRows          = rows;
        desc.numColumns       = columns;
        desc.srcLayout        = VriCooperativeVectorMatrixLayout_RowMajor;
        desc.srcStride        = columns * sizeof(uint16_t);
        desc.dstLayout        = VriCooperativeVectorMatrixLayout_InferencingOptimal;
        return desc;
    }

    void CheckWeightConversion(Context& ctx, VriComponentType type)
    {
        std::array<float, 128> source {};
        for (size_t i = 0; i < source.size(); ++i)
            source[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 16.0f;
        auto forward             = MatrixDesc(16, 8);
        forward.srcComponentType = VriComponentType_Float32;
        forward.dstComponentType = type;
        forward.srcStride        = 8 * sizeof(float);
        size_t packedSize        = 0;
        forward.pDstSize         = &packedSize;
        REQUIRE(ctx.neural.ConvertCooperativeVectorMatrix(ctx.device, &forward) == VriResult_Success);
        std::vector<std::byte> packed(packedSize);
        forward.pSrcData = source.data();
        forward.srcSize  = sizeof(source);
        forward.pDstData = packed.data();
        REQUIRE(ctx.neural.ConvertCooperativeVectorMatrix(ctx.device, &forward) == VriResult_Success);
        VriConvertCooperativeVectorMatrixInfo backward {};
        backward.numRows          = 16;
        backward.numColumns       = 8;
        backward.srcComponentType = type;
        backward.dstComponentType = VriComponentType_Float32;
        backward.srcLayout        = VriCooperativeVectorMatrixLayout_InferencingOptimal;
        backward.dstLayout        = VriCooperativeVectorMatrixLayout_ColumnMajor;
        backward.dstStride        = 16 * sizeof(float);
        size_t restoredSize       = 0;
        backward.pDstSize         = &restoredSize;
        REQUIRE(ctx.neural.ConvertCooperativeVectorMatrix(ctx.device, &backward) == VriResult_Success);
        REQUIRE(restoredSize == sizeof(source));
        std::array<float, 128> restored {};
        backward.pSrcData = packed.data();
        backward.srcSize  = packed.size();
        backward.pDstData = restored.data();
        REQUIRE(ctx.neural.ConvertCooperativeVectorMatrix(ctx.device, &backward) == VriResult_Success);
        for (size_t row = 0; row < 16; ++row)
        {
            for (size_t col = 0; col < 8; ++col)
                CHECK(restored[col * 16 + row] == source[row * 8 + col]);
        }
    }

    struct Inference
    {
        Context&             ctx;
        VriBuffer*           weights     = nullptr;
        VriBuffer*           output      = nullptr;
        VriBuffer*           readback    = nullptr;
        VriDescriptor*       weightsView = nullptr;
        VriDescriptor*       outputView  = nullptr;
        VriDescriptorPool*   pool        = nullptr;
        VriPipelineLayout*   layout      = nullptr;
        VriPipeline*         pipeline    = nullptr;
        VriCommandAllocator* allocator   = nullptr;
        VriFence*            fence       = nullptr;

        ~Inference()
        {
            const auto& c = ctx.core;
            // A failed assertion may leave submitted work alive too.
            c.DeviceWaitIdle(ctx.device);
            if (fence)
                c.DestroyFence(fence);
            if (allocator)
                c.DestroyCommandAllocator(allocator);
            if (pipeline)
                c.DestroyPipeline(pipeline);
            if (pool)
                c.DestroyDescriptorPool(pool);
            if (layout)
                c.DestroyPipelineLayout(layout);
            if (weightsView)
                c.DestroyDescriptor(weightsView);
            if (outputView)
                c.DestroyDescriptor(outputView);
            if (readback)
                c.DestroyBuffer(readback);
            if (output)
                c.DestroyBuffer(output);
            if (weights)
                c.DestroyBuffer(weights);
        }
    };

    void RunInference(Context& ctx)
    {
        Inference               gpu {ctx};
        const auto&             c = ctx.core;
        const auto&             n = ctx.neural;
        std::array<uint32_t, 4> offsets {};
        std::vector<std::byte>  packed;
        const auto              append = [&](const void* data, size_t size) {
            packed.resize((packed.size() + 63) & ~size_t {63});
            const auto offset = static_cast<uint32_t>(packed.size());
            packed.resize(packed.size() + size);
            std::memcpy(packed.data() + offset, data, size);
            return offset;
        };
        for (uint32_t layer = 0; layer < 2; ++layer)
        {
            const uint32_t        rows    = layer == 0 ? 16 : 4;
            const uint32_t        columns = layer == 0 ? 8 : 16;
            std::vector<uint16_t> matrix(rows * columns);
            std::vector<uint16_t> bias(rows);
            for (uint32_t r = 0; r < rows; ++r)
            {
                bias[r] = HalfSixteenth(layer == 0 ? static_cast<int>(r) - 8 : (static_cast<int>(r) - 2) * 2);
                for (uint32_t col = 0; col < columns; ++col)
                    matrix[r * columns + col] = HalfSixteenth(layer == 0 ? static_cast<int>((r * 3 + col * 5) % 9) - 4 :
                                                                           static_cast<int>((r * 5 + col * 3) % 7) - 3);
            }
            auto   info   = MatrixDesc(rows, columns);
            size_t bytes  = 0;
            info.pDstSize = &bytes;
            REQUIRE(n.ConvertCooperativeVectorMatrix(ctx.device, &info) == VriResult_Success);
            REQUIRE(bytes > 0);
            std::vector<std::byte> converted(bytes);
            info.pSrcData = matrix.data();
            info.srcSize  = matrix.size() * sizeof(uint16_t);
            info.pDstData = converted.data();
            REQUIRE(n.ConvertCooperativeVectorMatrix(ctx.device, &info) == VriResult_Success);
            offsets[layer * 2]     = append(converted.data(), bytes);
            offsets[layer * 2 + 1] = append(bias.data(), bias.size() * sizeof(uint16_t));
        }
        VriBufferDesc buffer {};
        buffer.size           = packed.size();
        buffer.usage          = VriBufferUsage_StorageBuffer;
        buffer.memoryLocation = VriMemoryLocation_HostUpload;
        REQUIRE(c.CreateBuffer(ctx.device, &buffer, &gpu.weights) == VriResult_Success);
        void* mapped = c.MapBuffer(gpu.weights, 0, buffer.size);
        REQUIRE(mapped != nullptr);
        std::memcpy(mapped, packed.data(), packed.size());
        c.UnmapBuffer(gpu.weights);
        constexpr size_t kOutputBytes = kSampleNum * 4 * sizeof(float);
        buffer.size                   = kOutputBytes;
        buffer.usage                  = VriBufferUsage_StorageBuffer | VriBufferUsage_TransferSrc;
        buffer.memoryLocation         = VriMemoryLocation_Device;
        REQUIRE(c.CreateBuffer(ctx.device, &buffer, &gpu.output) == VriResult_Success);
        buffer.usage          = VriBufferUsage_TransferDst;
        buffer.memoryLocation = VriMemoryLocation_HostReadback;
        REQUIRE(c.CreateBuffer(ctx.device, &buffer, &gpu.readback) == VriResult_Success);

        std::array<VriDescriptorRangeDesc, 2> ranges {};
        for (uint32_t i = 0; i < ranges.size(); ++i)
        {
            ranges[i].baseRegister   = i;
            ranges[i].descriptorType = VriDescriptorType_StorageBuffer;
            ranges[i].descriptorNum  = 1;
            ranges[i].shaderStages   = VriShaderStage_Compute;
        }
        VriDescriptorSetDesc setDesc {};
        setDesc.ranges   = ranges.data();
        setDesc.rangeNum = static_cast<uint32_t>(ranges.size());
        VriPushConstantDesc push {};
        push.size         = sizeof(offsets);
        push.shaderStages = VriShaderStage_Compute;
        VriPipelineLayoutDesc layout {};
        layout.descriptorSets   = &setDesc;
        layout.descriptorSetNum = 1;
        layout.pushConstants    = &push;
        layout.pushConstantNum  = 1;
        REQUIRE(c.CreatePipelineLayout(ctx.device, &layout, &gpu.layout) == VriResult_Success);
        VriComputePipelineDesc pipeline {};
        pipeline.pipelineLayout        = gpu.layout;
        pipeline.shader.stage          = VriShaderStage_Compute;
        pipeline.shader.bytecode       = g_neuralInferenceSpv;
        pipeline.shader.bytecodeSize   = sizeof(g_neuralInferenceSpv);
        pipeline.shader.entryPointName = "computeMain";
        REQUIRE(c.CreateComputePipeline(ctx.device, &pipeline, &gpu.pipeline) == VriResult_Success);
        VriDescriptorPoolDesc pool {};
        pool.descriptorSetMaxNum = 1;
        pool.storageBufferMaxNum = 2;
        REQUIRE(c.CreateDescriptorPool(ctx.device, &pool, &gpu.pool) == VriResult_Success);
        VriDescriptorSet* set = nullptr;
        REQUIRE(c.AllocateDescriptorSets(gpu.pool, gpu.layout, 0, &set, 1) == VriResult_Success);
        VriBufferViewDesc view {};
        view.buffer   = gpu.weights;
        view.viewType = VriDescriptorType_StorageBuffer;
        view.size     = packed.size();
        REQUIRE(c.CreateBufferView(ctx.device, &view, &gpu.weightsView) == VriResult_Success);
        view.buffer = gpu.output;
        view.size   = kOutputBytes;
        REQUIRE(c.CreateBufferView(ctx.device, &view, &gpu.outputView) == VriResult_Success);
        const VriDescriptor*                        descriptors[] = {gpu.weightsView, gpu.outputView};
        std::array<VriDescriptorRangeUpdateDesc, 2> updates {};
        for (size_t i = 0; i < updates.size(); ++i)
        {
            updates[i].descriptorNum = 1;
            updates[i].descriptors   = &descriptors[i];
        }
        c.UpdateDescriptorRanges(set, 0, static_cast<uint32_t>(updates.size()), updates.data());
        REQUIRE(c.CreateCommandAllocator(ctx.device, VriQueueType_Graphics, &gpu.allocator) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(gpu.allocator, &cmd) == VriResult_Success);
        REQUIRE(c.CreateFence(ctx.device, 0, &gpu.fence) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(ctx.device, VriQueueType_Graphics, 0, &queue) == VriResult_Success);
        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        c.CmdSetPipeline(cmd, gpu.pipeline);
        c.CmdSetPipelineLayout(cmd, gpu.layout);
        c.CmdSetDescriptorSet(cmd, 0, set);
        c.CmdSetConstants(cmd, 0, offsets.data(), sizeof(offsets));
        VriDispatchDesc dispatch {3, 1, 1};
        c.CmdDispatch(cmd, &dispatch);
        VriBufferBarrierDesc barrier {};
        barrier.buffer        = gpu.output;
        barrier.before.access = VriAccess_ShaderResourceStorageWrite;
        barrier.before.stages = VriPipelineStage_ComputeShader;
        barrier.after.access  = VriAccess_CopySourceRead;
        barrier.after.stages  = VriPipelineStage_Transfer;
        VriBarrierGroupDesc barriers {};
        barriers.buffers   = &barrier;
        barriers.bufferNum = 1;
        c.CmdBarrier(cmd, &barriers);
        VriBufferCopyDesc copy {};
        copy.size = kOutputBytes;
        c.CmdCopyBuffer(cmd, gpu.readback, gpu.output, &copy);
        REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
        VriFenceSubmitDesc signal {gpu.fence, 1};
        VriQueueSubmitDesc submit {};
        submit.commandBuffers   = &cmd;
        submit.commandBufferNum = 1;
        submit.signalFences     = &signal;
        submit.signalFenceNum   = 1;
        c.QueueSubmit(queue, &submit);
        c.Wait(gpu.fence, 1);
        const auto* actual = static_cast<const float*>(c.MapBuffer(gpu.readback, 0, kOutputBytes));
        REQUIRE(actual != nullptr);
        double maxError = 0;
        for (uint32_t sample = 0; sample < kSampleNum; ++sample)
        {
            std::array<double, 16> hidden {};
            for (int r = 0; r < 16; ++r)
            {
                double value = (r - 8) / 16.0;
                for (int col = 0; col < 8; ++col)
                    value +=
                        (((r * 3 + col * 5) % 9) - 4) / 16.0 * (static_cast<int>(sample % 9) - 4) * (col + 1) / 32.0;
                hidden[r] = std::max(value, 0.0);
            }
            for (int r = 0; r < 4; ++r)
            {
                double expected = (r - 2) / 8.0;
                for (int col = 0; col < 16; ++col)
                    expected += (((r * 5 + col * 3) % 7) - 3) / 16.0 * hidden[col];
                REQUIRE(std::isfinite(actual[sample * 4 + r]));
                maxError = std::max(maxError, std::abs(actual[sample * 4 + r] - expected));
            }
        }
        c.UnmapBuffer(gpu.readback);
        std::printf("CoopVec MLP: %s, 65 x (8 -> 16 -> 4), max error %.9f\n",
                    c.GetDeviceDesc(ctx.device)->adapter.name,
                    maxError);
        CHECK(maxError <= 0.002);
        CHECK(ctx.errors == 0);
    }
} // namespace

TEST_CASE("cooperative vector: opt-in, queries, conversion bounds and inference")
{
    if (!HasVulkan())
    {
        MESSAGE("Vulkan unavailable - skipping cooperative-vector device tests");
        return;
    }
    for (const VriBool validation : {VRI_FALSE, VRI_TRUE})
    {
        Context disabled;
        REQUIRE(disabled.Init(VRI_FALSE, validation) == VriResult_Success);
        CHECK(disabled.core.GetDeviceDesc(disabled.device)->hasCooperativeVector == VRI_FALSE);
        VriCooperativeVectorInterface unavailable {};
        CHECK(vriGetInterface(disabled.device, VRI_INTERFACE_COOPERATIVE_VECTOR, sizeof(unavailable), &unavailable) ==
              VriResult_Unsupported);
        Context ctx;
        REQUIRE(ctx.Init(VRI_TRUE, validation) == VriResult_Success);
        const auto* device = ctx.core.GetDeviceDesc(ctx.device);
        CHECK(bool(device->enabledFeatures & VriFeature_CooperativeVector) == bool(device->hasCooperativeVector));
        if (!device->hasCooperativeVector)
        {
            Context required;
            CHECK(required.Init(VRI_TRUE, validation, VRI_FALSE) == VriResult_Unsupported);
            CHECK(required.device == nullptr);
            CHECK(required.errors > 0);
            continue;
        }
        REQUIRE(ctx.neural.GetCooperativeVectorDeviceProperties != nullptr);
        VriCooperativeVectorDeviceProperties properties {};
        REQUIRE(ctx.neural.GetCooperativeVectorDeviceProperties(ctx.device, &properties) == VriResult_Success);
        CHECK(properties.maxCooperativeVectorComponents >= 16);
        CHECK((properties.cooperativeVectorSupportedStages & VriShaderStage_Compute) != 0);
        CHECK(ctx.neural.GetCooperativeVectorDeviceProperties(ctx.device, nullptr) == VriResult_InvalidArgument);
        CHECK(ctx.neural.GetCooperativeVectorDeviceProperties(nullptr, &properties) == VriResult_InvalidArgument);
        CHECK(ctx.neural.GetCooperativeVectorProperties(ctx.device, nullptr, nullptr) == VriResult_InvalidArgument);
        uint32_t cPropertyCount = 0;
        CHECK(vriTestCooperativeVectorC(
                  ctx.device, sizeof(ctx.neural), sizeof(VriConvertCooperativeVectorMatrixInfo), &cPropertyCount) == 0);
        auto combinations = Combinations(ctx);
        CHECK(cPropertyCount == combinations.size());
        std::printf("CoopVec capabilities: %zu tuples, max components %u, stages 0x%x\n",
                    combinations.size(),
                    properties.maxCooperativeVectorComponents,
                    properties.cooperativeVectorSupportedStages);
        CHECK(vriGetInterface(ctx.device, VRI_INTERFACE_COOPERATIVE_VECTOR, sizeof(ctx.neural) - 1, &unavailable) ==
              VriResult_InvalidArgument);
        uint32_t                       shortCount = 0;
        VriCooperativeVectorProperties sentinel {};
        sentinel.transpose = 123;
        CHECK(ctx.neural.GetCooperativeVectorProperties(ctx.device, &shortCount, &sentinel) ==
              VriResult_InvalidArgument);
        CHECK(shortCount == combinations.size());
        CHECK(sentinel.transpose == 123);
        const bool half = std::any_of(combinations.begin(), combinations.end(), [](const auto& c) {
            return c.inputType == VriComponentType_Float16 && c.inputInterpretation == VriComponentType_Float16 &&
                   c.matrixInterpretation == VriComponentType_Float16 &&
                   c.biasInterpretation == VriComponentType_Float16 && c.resultType == VriComponentType_Float16;
        });
        if (!half)
        {
            MESSAGE("FP16 inference tuple not supported by this adapter");
            continue;
        }
        auto   info   = MatrixDesc(16, 8);
        size_t size   = 0;
        info.pDstSize = &size;
        REQUIRE(ctx.neural.ConvertCooperativeVectorMatrix(ctx.device, &info) == VriResult_Success);
        std::vector<std::byte>    storage(size, std::byte {0x5a});
        std::array<uint16_t, 128> weights {};
        size_t                    shortSize = 1;
        info.pDstSize                       = &shortSize;
        info.pSrcData                       = weights.data();
        info.srcSize                        = sizeof(weights);
        info.pDstData                       = storage.data();
        CHECK(ctx.neural.ConvertCooperativeVectorMatrix(ctx.device, &info) == VriResult_InvalidArgument);
        CHECK(shortSize == size);
        CHECK(std::all_of(storage.begin(), storage.end(), [](auto value) { return value == std::byte {0x5a}; }));
        info.pDstSize = &size;
        info.srcSize  = 1;
        CHECK(ctx.neural.ConvertCooperativeVectorMatrix(ctx.device, &info) == VriResult_InvalidArgument);
        info.pSrcData = storage.data();
        info.srcSize  = storage.size();
        CHECK(ctx.neural.ConvertCooperativeVectorMatrix(ctx.device, &info) == VriResult_InvalidArgument);
        info.srcStride = 1;
        info.pSrcData  = nullptr;
        info.pDstData  = nullptr;
        CHECK(ctx.neural.ConvertCooperativeVectorMatrix(ctx.device, &info) == VriResult_InvalidArgument);
        for (const auto type : {VriComponentType_Float16, VriComponentType_Float8E4M3, VriComponentType_Float8E5M2})
        {
            if (std::any_of(combinations.begin(), combinations.end(), [type](const auto& c) {
                    return c.matrixInterpretation == type;
                }))
                CheckWeightConversion(ctx, type);
        }
        RunInference(ctx);
    }
}

TEST_CASE("cooperative vector: unsupported backend rejects a required feature before creation")
{
    VriDeviceCreationDesc desc  = vriDefaultDeviceCreationDesc();
    desc.coordinates.clipSpaceY = VriClipSpaceY_Up;
    desc.graphicsAPI            = VriGraphicsAPI_OpenGL;
    desc.enabledFeatures        = VriFeature_CooperativeVector;
    desc.bestEffort             = VRI_FALSE;
    VriDevice* device           = nullptr;
    CHECK(vriCreateDevice(&desc, &device) == VriResult_Unsupported);
    CHECK(device == nullptr);
}
