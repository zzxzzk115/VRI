// Interactive neural texture comparison, with an explicit window-free regression mode.
#include "../common/example_app.h"
#include "model_weights.h"
#include "shaders/examples/mlp_image_spv.h"

#include <vri/vri.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    constexpr float kErrorScale = 8.0f;

    void Check(VriResult result, const char* operation)
    {
        if (result != VriResult_Success)
            throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(result));
    }

    uint32_t ParseNumber(std::string_view text, uint32_t maximum)
    {
        uint32_t value          = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error != std::errc {} || end != text.data() + text.size() || value == 0 || value > maximum)
            throw std::runtime_error("Expected an integer in [1, " + std::to_string(maximum) + "]");
        return value;
    }

    struct Options
    {
        uint32_t              size    = 256;
        uint32_t              frames  = 0;
        bool                  offline = false;
        std::filesystem::path output  = "build/mlp.bmp";
        VriGraphicsAPI        api     = VriGraphicsAPI_Auto;
    };

    Options ParseOptions(int argc, char** argv)
    {
        Options options;
        bool    outputSpecified = false;
        if (const char* api = std::getenv("VRI_API"))
        {
            const std::string_view name(api);
            if (name == "vulkan")
                options.api = VriGraphicsAPI_Vulkan;
            else if (name == "d3d12")
                options.api = VriGraphicsAPI_D3D12;
            else if (name == "opengl")
                options.api = VriGraphicsAPI_OpenGL;
            else if (name == "metal")
                options.api = VriGraphicsAPI_Metal;
            else if (name == "webgpu")
                options.api = VriGraphicsAPI_WebGPU;
            else if (name == "software")
                options.api = VriGraphicsAPI_Software;
            else if (name != "auto")
                throw std::runtime_error("Unknown VRI_API: " + std::string(name));
        }

        if (const char* frames = std::getenv("VRI_MAX_FRAMES"))
            options.frames = ParseNumber(frames, 10000);
        for (int i = 1; i < argc; ++i)
        {
            const std::string_view argument(argv[i]);
            if (argument == "--offline")
            {
                options.offline = true;
                continue;
            }
            if (i + 1 == argc)
                throw std::runtime_error("Missing value for " + std::string(argument));
            if (argument == "--size")
                options.size = ParseNumber(argv[++i], 2048);
            else if (argument == "--frames")
                options.frames = ParseNumber(argv[++i], 10000);
            else if (argument == "--output")
            {
                options.output  = argv[++i];
                outputSpecified = true;
            }
            else
                throw std::runtime_error("Unknown argument: " + std::string(argument));
        }
        if (options.offline)
        {
            if (options.frames == 0)
                options.frames = 1;
            // VRI_CAPTURE names the presented window image in interactive mode.
            if (!outputSpecified)
            {
                if (const char* capture = std::getenv("VRI_CAPTURE"))
                    options.output = capture;
            }
        }
        return options;
    }

    struct Params
    {
        uint32_t firstMatrix  = 0;
        uint32_t firstBias    = 0;
        uint32_t secondMatrix = 0;
        uint32_t secondBias   = 0;
        uint32_t thirdMatrix  = 0;
        uint32_t thirdBias    = 0;
        uint32_t size         = 0;
        float    errorScale   = kErrorScale;
    };
    static_assert(sizeof(Params) == 32);

    std::array<double, 3> Reference(double x, double y)
    {
        return {0.5 + 0.4 * std::sin(2.4 * x + 1.1 * y),
                0.5 + 0.4 * std::sin(2.0 * y - 1.3 * x),
                0.5 + 0.4 * std::cos(2.7 * std::sqrt(x * x + y * y))};
    }

    std::array<double, 3> CpuInference(double x, double y)
    {
        // Independent FP32-weight reference with double accumulation includes the GPU's FP16 rounding error.
        std::array<double, 32> input {x, y};
        for (size_t layer = 0; layer < 3; ++layer)
        {
            std::array<double, 32> output {};
            const uint32_t         columns = vriex::mlp::kWidths[layer];
            const uint32_t         rows    = vriex::mlp::kWidths[layer + 1];
            for (uint32_t row = 0; row < rows; ++row)
            {
                double value = vriex::mlp::kWeights[vriex::mlp::kOffsets[layer * 2 + 1] + row];
                for (uint32_t column = 0; column < columns; ++column)
                    value +=
                        input[column] * vriex::mlp::kWeights[vriex::mlp::kOffsets[layer * 2] + row * columns + column];
                output[row] = layer < 2 ? std::max(value, 0.0) : std::clamp(value, 0.0, 1.0);
            }
            input = output;
        }
        return {input[0], input[1], input[2]};
    }

    class NeuralImage
    {
    public:
        NeuralImage()                              = default;
        NeuralImage(const NeuralImage&)            = delete;
        NeuralImage& operator=(const NeuralImage&) = delete;

        ~NeuralImage()
        {
            if (!m_device)
                return;
            if (m_core.DeviceWaitIdle)
            {
                // Includes partial initialization and failures after submission.
                m_core.DeviceWaitIdle(m_device);
                if (m_allocator)
                    m_core.DestroyCommandAllocator(m_allocator);
                if (m_fence)
                    m_core.DestroyFence(m_fence);
                if (m_pipeline)
                    m_core.DestroyPipeline(m_pipeline);
                if (m_pool)
                    m_core.DestroyDescriptorPool(m_pool);
                if (m_layout)
                    m_core.DestroyPipelineLayout(m_layout);
                for (auto* view : m_views)
                {
                    if (view)
                        m_core.DestroyDescriptor(view);
                }
                for (auto* buffer : {m_weights, m_predictions, m_valuesReadback, m_imageReadback})
                {
                    if (buffer)
                        m_core.DestroyBuffer(buffer);
                }
                if (m_image)
                    m_core.DestroyTexture(m_image);
            }
            if (m_ownsDevice)
                vriDestroyDevice(m_device);
        }

        bool Initialize(const Options& options, VriDevice* borrowedDevice)
        {
            m_device = borrowedDevice;
            if (!m_device)
            {
                VriCallbackInterface callback {};
                callback.userArg         = this;
                callback.MessageCallback = [](void* user, VriMessageSeverity severity, const char* message) {
                    if (severity == VriMessageSeverity_Error)
                        ++static_cast<NeuralImage*>(user)->m_errors;
                    const char* level = "INFO";
                    if (severity == VriMessageSeverity_Error)
                        level = "ERROR";
                    else if (severity == VriMessageSeverity_Warning)
                        level = "WARN";
                    std::fprintf(stderr, "[VRI][%s] %s\n", level, message);
                };
                VriDeviceCreationDesc device  = vriDefaultDeviceCreationDesc();
                device.coordinates.clipSpaceY = VriClipSpaceY_Up;
                device.graphicsAPI            = options.api;
                device.enableValidation       = VRI_TRUE;
                device.enabledFeatures        = VriFeature_CooperativeVector;
                device.bestEffort             = VRI_TRUE;
                device.callbackInterface      = &callback;
                Check(vriCreateDevice(&device, &m_device), "Create device");
                m_ownsDevice = true;
            }
            Check(vriGetInterface(m_device, VRI_INTERFACE_CORE, sizeof(m_core), &m_core), "Get core interface");
            const auto* desc = m_core.GetDeviceDesc(m_device);
            std::printf("[mlp] %s; API %d; cooperative vectors: %s\n",
                        desc->adapter.name,
                        desc->graphicsAPI,
                        desc->hasCooperativeVector ? "yes" : "no");
            if (!desc->hasCooperativeVector)
                return false;
            Check(vriGetInterface(m_device, VRI_INTERFACE_COOPERATIVE_VECTOR, sizeof(m_neural), &m_neural),
                  "Get cooperative vector interface");
            VriCooperativeVectorDeviceProperties properties {};
            Check(m_neural.GetCooperativeVectorDeviceProperties(m_device, &properties), "Query neural stages");
            if (!(properties.cooperativeVectorSupportedStages & VriShaderStage_Compute) ||
                properties.maxCooperativeVectorComponents < 32)
                return false;
            uint32_t count = 0;
            Check(m_neural.GetCooperativeVectorProperties(m_device, &count, nullptr), "Count neural type combinations");
            std::vector<VriCooperativeVectorProperties> types(count);
            Check(m_neural.GetCooperativeVectorProperties(m_device, &count, types.data()), "Query neural types");
            const bool fp16 = std::ranges::any_of(types, [](const auto& type) {
                return type.inputType == VriComponentType_Float16 &&
                       type.inputInterpretation == VriComponentType_Float16 &&
                       type.matrixInterpretation == VriComponentType_Float16 &&
                       type.biasInterpretation == VriComponentType_Float16 &&
                       type.resultType == VriComponentType_Float16;
            });
            if (!fp16)
                return false;
            if (!(m_core.GetFormatSupport(m_device, VriFormat_RGBA8_UNORM) & VriFormatSupport_StorageTexture))
                throw std::runtime_error("RGBA8_UNORM storage texture is not supported");
            m_params.size            = options.size;
            const uint32_t alignment = std::max(4u, desc->uploadBufferTextureRowAlignment);
            m_rowBytes               = (options.size * 3 * 4 + alignment - 1) / alignment * alignment;
            const auto packed        = PackWeights();
            CreateResources(packed);
            CreatePipeline();
            Check(m_core.GetQueue(m_device, VriQueueType_Graphics, 0, &m_queue), "Get queue");
            if (options.offline)
            {
                Check(m_core.CreateCommandAllocator(m_device, VriQueueType_Graphics, &m_allocator), "Create allocator");
                Check(m_core.CreateCommandBuffer(m_allocator, &m_commands), "Create command buffer");
                Check(m_core.CreateFence(m_device, 0, &m_fence), "Create completion fence");
            }
            return true;
        }

        void Render(uint32_t frames)
        {
            for (uint32_t frame = 0; frame < frames; ++frame)
            {
                m_core.ResetCommandAllocator(m_allocator);
                Check(m_core.BeginCommandBuffer(m_commands), "Begin commands");
                Record(m_commands);
                Check(m_core.EndCommandBuffer(m_commands), "End commands");
                const VriFenceSubmitDesc signal {m_fence, ++m_submission, VriPipelineStage_AllCommands};
                VriQueueSubmitDesc       submit {};
                submit.commandBuffers   = &m_commands;
                submit.commandBufferNum = 1;
                submit.signalFences     = &signal;
                submit.signalFenceNum   = 1;
                m_core.QueueSubmit(m_queue, &submit);
                m_core.Wait(m_fence, signal.value);
            }
        }

        // The caller owns command submission/completion; both hosts record the same inference/readback path.
        void Record(VriCommandBuffer* commands)
        {
            VriBufferBarrierDesc buffer {};
            buffer.buffer = m_predictions;
            if (m_recorded)
                buffer.before = {VriAccess_CopySourceRead, VriPipelineStage_Transfer};
            buffer.after = {VriAccess_ShaderResourceStorageWrite, VriPipelineStage_ComputeShader};
            VriTextureBarrierDesc image {};
            image.texture       = m_image;
            image.before.layout = m_recorded ? VriLayout_ShaderResource : VriLayout_Undefined;
            image.before.access = m_recorded ? VriAccess_ShaderResourceRead : VriAccess_None;
            image.before.stages = m_recorded ? VriPipelineStage_FragmentShader : VriPipelineStage_None;
            image.after         = {
                VriAccess_ShaderResourceStorageWrite, VriLayout_ShaderResourceStorage, VriPipelineStage_ComputeShader};
            image.aspect = VriImageAspect_Color;
            VriBarrierGroupDesc barriers {};
            barriers.buffers    = &buffer;
            barriers.bufferNum  = 1;
            barriers.textures   = &image;
            barriers.textureNum = 1;
            m_core.CmdBarrier(commands, &barriers);
            m_core.CmdSetPipeline(commands, m_pipeline);
            m_core.CmdSetPipelineLayout(commands, m_layout);
            m_core.CmdSetDescriptorSet(commands, 0, m_set);
            m_core.CmdSetConstants(commands, 0, &m_params, sizeof(m_params));
            const VriDispatchDesc dispatch {(m_params.size + 7) / 8, (m_params.size + 7) / 8, 1};
            m_core.CmdDispatch(commands, &dispatch);
            buffer.before = buffer.after;
            buffer.after  = {VriAccess_CopySourceRead, VriPipelineStage_Transfer};
            image.before  = image.after;
            image.after   = {VriAccess_CopySourceRead, VriLayout_CopySource, VriPipelineStage_Transfer};
            m_core.CmdBarrier(commands, &barriers);
            VriBufferCopyDesc valuesCopy {};
            valuesCopy.size = ValuesBytes();
            m_core.CmdCopyBuffer(commands, m_valuesReadback, m_predictions, &valuesCopy);
            VriBufferTextureCopyDesc imageCopy {};
            imageCopy.bufferRowLength  = m_rowBytes / 4;
            imageCopy.texture.width    = m_params.size * 3;
            imageCopy.texture.height   = m_params.size;
            imageCopy.texture.depth    = 1;
            imageCopy.texture.layerNum = 1;
            imageCopy.texture.aspect   = VriImageAspect_Color;
            m_core.CmdReadbackTextureToBuffer(commands, m_imageReadback, m_image, &imageCopy);
            image.before = image.after;
            image.after  = {VriAccess_ShaderResourceRead, VriLayout_ShaderResource, VriPipelineStage_FragmentShader};
            barriers.buffers   = nullptr;
            barriers.bufferNum = 0;
            m_core.CmdBarrier(commands, &barriers);
            m_recorded = true;
        }

        VriDescriptor* ImageView() const { return m_views[2]; }
        void           SetErrorScale(float value) { m_params.errorScale = value; }

        void VerifyAndSave(const std::filesystem::path& path)
        {
            std::vector<std::array<float, 4>> values(static_cast<size_t>(m_params.size) * m_params.size);
            std::vector<uint8_t>              pixels(static_cast<size_t>(m_rowBytes) * m_params.size);
            ReadBuffer(m_valuesReadback, std::as_writable_bytes(std::span(values)));
            ReadBuffer(m_imageReadback, std::as_writable_bytes(std::span(pixels)));
            double maxInferenceError = 0;
            double squaredFitError   = 0;
            int    maxPixelError     = 0;
            for (uint32_t y = 0; y < m_params.size; ++y)
            {
                for (uint32_t x = 0; x < m_params.size; ++x)
                {
                    const double px        = (static_cast<double>(x) + 0.5) / m_params.size * 2 - 1;
                    const double py        = (static_cast<double>(y) + 0.5) / m_params.size * 2 - 1;
                    const auto   reference = Reference(px, py);
                    const auto   cpu       = CpuInference(px, py);
                    const auto&  gpu       = values[static_cast<size_t>(y) * m_params.size + x];
                    if (!std::ranges::all_of(gpu, [](float value) { return std::isfinite(value); }) || gpu[3] != 1.0f)
                        throw std::runtime_error("Non-finite inference result or unwritten pixel");
                    for (size_t channel = 0; channel < 3; ++channel)
                    {
                        const double error = gpu[channel] - reference[channel];
                        maxInferenceError  = std::max(maxInferenceError, std::abs(gpu[channel] - cpu[channel]));
                        squaredFitError += error * error;
                        const std::array<double, 3> panels {
                            reference[channel],
                            gpu[channel],
                            std::clamp(std::abs(error) * m_params.errorScale, 0.0, 1.0)};
                        for (uint32_t panel = 0; panel < 3; ++panel)
                        {
                            const size_t index = static_cast<size_t>(y) * m_rowBytes + (x + panel * m_params.size) * 4;
                            const int    expected = static_cast<int>(std::lround(panels[panel] * 255));
                            maxPixelError = std::max(maxPixelError, std::abs(pixels[index + channel] - expected));
                            if (pixels[index + 3] != 255)
                                throw std::runtime_error("Invalid image alpha or unwritten pixel");
                        }
                    }
                }
            }
            const double rmse = std::sqrt(squaredFitError / static_cast<double>(values.size() * 3));
            std::printf(
                "[mlp] FP16 vs CPU max error %.7f (limit 0.01); fit RMSE %.7f (limit 0.03); pixel error %d (limit 1)\n",
                maxInferenceError,
                rmse,
                maxPixelError);
            if (maxInferenceError > 0.01 || rmse > 0.03 || maxPixelError > 1 || m_errors != 0)
                throw std::runtime_error("MLP readback verification failed");
            std::vector<uint8_t> bgra(values.size() * 3 * 4);
            for (uint32_t y = 0; y < m_params.size; ++y)
            {
                for (uint32_t x = 0; x < m_params.size * 3; ++x)
                {
                    const auto* source      = pixels.data() + static_cast<size_t>(y) * m_rowBytes + x * 4;
                    auto*       destination = bgra.data() + (static_cast<size_t>(y) * m_params.size * 3 + x) * 4;
                    destination[0]          = source[2];
                    destination[1]          = source[1];
                    destination[2]          = source[0];
                    destination[3]          = source[3];
                }
            }
            if (path.has_parent_path())
                std::filesystem::create_directories(path.parent_path());
            if (!WriteBmpBGRA(path.string().c_str(), bgra.data(), m_params.size * 3, m_params.size))
                throw std::runtime_error("Cannot write " + path.string());
            std::printf("[mlp] Saved %s: reference | MLP | absolute error x%g\n",
                        path.string().c_str(),
                        static_cast<double>(m_params.errorScale));
        }

    private:
        size_t ValuesBytes() const { return static_cast<size_t>(m_params.size) * m_params.size * sizeof(float) * 4; }

        std::vector<uint8_t> PackWeights()
        {
            std::vector<uint8_t> packed;
            const auto           append = [&](uint32_t rows, uint32_t columns, uint32_t sourceOffset, bool matrix) {
                VriConvertCooperativeVectorMatrixInfo info {};
                info.numRows          = rows;
                info.numColumns       = columns;
                info.srcComponentType = VriComponentType_Float32;
                info.dstComponentType = VriComponentType_Float16;
                info.srcLayout        = VriCooperativeVectorMatrixLayout_RowMajor;
                info.dstLayout        = matrix ? VriCooperativeVectorMatrixLayout_InferencingOptimal :
                                                           VriCooperativeVectorMatrixLayout_RowMajor;
                info.srcStride        = columns * sizeof(float);
                info.dstStride        = matrix ? 0 : columns * sizeof(uint16_t);
                info.srcSize          = static_cast<size_t>(rows) * columns * sizeof(float);
                info.pSrcData         = vriex::mlp::kWeights.data() + sourceOffset;
                size_t bytes          = 0;
                info.pDstSize         = &bytes;
                Check(m_neural.ConvertCooperativeVectorMatrix(m_device, &info), "Query converted weight size");
                // Cooperative-vector byte offsets must satisfy the shader's matrix/bias alignment.
                const size_t offset = (packed.size() + 63) & ~size_t {63};
                packed.resize(offset + bytes);
                info.pDstData = packed.data() + offset;
                Check(m_neural.ConvertCooperativeVectorMatrix(m_device, &info), "Convert weights");
                return static_cast<uint32_t>(offset);
            };
            m_params.firstMatrix  = append(32, 2, vriex::mlp::kOffsets[0], true);
            m_params.firstBias    = append(32, 1, vriex::mlp::kOffsets[1], false);
            m_params.secondMatrix = append(32, 32, vriex::mlp::kOffsets[2], true);
            m_params.secondBias   = append(32, 1, vriex::mlp::kOffsets[3], false);
            m_params.thirdMatrix  = append(3, 32, vriex::mlp::kOffsets[4], true);
            m_params.thirdBias    = append(3, 1, vriex::mlp::kOffsets[5], false);
            return packed;
        }

        void CreateResources(std::span<const uint8_t> packed)
        {
            const auto createBuffer =
                [&](size_t bytes, VriBufferUsageFlags usage, VriMemoryLocation memory, VriBuffer** buffer) {
                    VriBufferDesc desc {};
                    desc.size           = bytes;
                    desc.usage          = usage;
                    desc.memoryLocation = memory;
                    Check(m_core.CreateBuffer(m_device, &desc, buffer), "Create buffer");
                };
            createBuffer(packed.size(), VriBufferUsage_StorageBuffer, VriMemoryLocation_HostUpload, &m_weights);
            void* mapped = m_core.MapBuffer(m_weights, 0, packed.size());
            if (!mapped)
                throw std::runtime_error("Cannot map model weights");
            std::memcpy(mapped, packed.data(), packed.size());
            m_core.UnmapBuffer(m_weights);
            createBuffer(ValuesBytes(),
                         VriBufferUsage_StorageBuffer | VriBufferUsage_TransferSrc,
                         VriMemoryLocation_Device,
                         &m_predictions);
            createBuffer(ValuesBytes(), VriBufferUsage_TransferDst, VriMemoryLocation_HostReadback, &m_valuesReadback);
            createBuffer(static_cast<size_t>(m_rowBytes) * m_params.size,
                         VriBufferUsage_TransferDst,
                         VriMemoryLocation_HostReadback,
                         &m_imageReadback);
            VriTextureDesc image {};
            image.type      = VriTextureType_2D;
            image.format    = VriFormat_RGBA8_UNORM;
            image.width     = m_params.size * 3;
            image.height    = m_params.size;
            image.depth     = 1;
            image.mipNum    = 1;
            image.layerNum  = 1;
            image.sampleNum = 1;
            image.usage =
                VriTextureUsage_ShaderResourceStorage | VriTextureUsage_TransferSrc | VriTextureUsage_ShaderResource;
            image.memoryLocation = VriMemoryLocation_Device;
            Check(m_core.CreateTexture(m_device, &image, &m_image), "Create image");
            VriBufferViewDesc view {};
            view.buffer   = m_weights;
            view.viewType = VriDescriptorType_StorageBuffer;
            view.size     = packed.size();
            Check(m_core.CreateBufferView(m_device, &view, &m_views[0]), "Create weights view");
            view.buffer = m_predictions;
            view.size   = ValuesBytes();
            Check(m_core.CreateBufferView(m_device, &view, &m_views[1]), "Create predictions view");
            VriTextureViewDesc textureView {};
            textureView.texture  = m_image;
            textureView.viewType = VriTextureViewType_2D;
            textureView.aspect   = VriImageAspect_Color;
            Check(m_core.CreateTextureView(m_device, &textureView, &m_views[2]), "Create image view");
        }

        void CreatePipeline()
        {
            std::array<VriDescriptorRangeDesc, 3> ranges {};
            for (uint32_t i = 0; i < ranges.size(); ++i)
            {
                ranges[i].baseRegister   = i;
                ranges[i].descriptorType = i < 2 ? VriDescriptorType_StorageBuffer : VriDescriptorType_StorageTexture;
                ranges[i].descriptorNum  = 1;
                ranges[i].shaderStages   = VriShaderStage_Compute;
            }
            VriDescriptorSetDesc set {};
            set.ranges   = ranges.data();
            set.rangeNum = static_cast<uint32_t>(ranges.size());
            const VriPushConstantDesc constants {0, sizeof(m_params), VriShaderStage_Compute};
            VriPipelineLayoutDesc     layout {};
            layout.descriptorSets   = &set;
            layout.descriptorSetNum = 1;
            layout.pushConstants    = &constants;
            layout.pushConstantNum  = 1;
            Check(m_core.CreatePipelineLayout(m_device, &layout, &m_layout), "Create pipeline layout");
            VriComputePipelineDesc pipeline {};
            pipeline.pipelineLayout        = m_layout;
            pipeline.shader.stage          = VriShaderStage_Compute;
            pipeline.shader.bytecode       = g_mlpImageSpv;
            pipeline.shader.bytecodeSize   = sizeof(g_mlpImageSpv);
            pipeline.shader.entryPointName = "computeMain";
            Check(m_core.CreateComputePipeline(m_device, &pipeline, &m_pipeline), "Create compute pipeline");
            VriDescriptorPoolDesc pool {};
            pool.descriptorSetMaxNum  = 1;
            pool.storageBufferMaxNum  = 2;
            pool.storageTextureMaxNum = 1;
            Check(m_core.CreateDescriptorPool(m_device, &pool, &m_pool), "Create descriptor pool");
            Check(m_core.AllocateDescriptorSets(m_pool, m_layout, 0, &m_set, 1), "Allocate descriptor set");
            std::array<VriDescriptorRangeUpdateDesc, 3> updates {};
            for (size_t i = 0; i < updates.size(); ++i)
            {
                updates[i].descriptorNum = 1;
                updates[i].descriptors   = &m_views[i];
            }
            m_core.UpdateDescriptorRanges(m_set, 0, static_cast<uint32_t>(updates.size()), updates.data());
        }

        void ReadBuffer(VriBuffer* buffer, std::span<std::byte> output) const
        {
            const void* mapped = m_core.MapBuffer(buffer, 0, output.size());
            if (!mapped)
                throw std::runtime_error("Cannot map GPU readback");
            std::memcpy(output.data(), mapped, output.size());
            m_core.UnmapBuffer(buffer);
        }

        VriDevice*                    m_device = nullptr;
        VriCoreInterface              m_core {};
        VriCooperativeVectorInterface m_neural {};
        VriQueue*                     m_queue     = nullptr;
        VriCommandAllocator*          m_allocator = nullptr;
        VriCommandBuffer*             m_commands  = nullptr;
        VriFence*                     m_fence     = nullptr;
        VriPipelineLayout*            m_layout    = nullptr;
        VriPipeline*                  m_pipeline  = nullptr;
        VriDescriptorPool*            m_pool      = nullptr;
        VriDescriptorSet*             m_set       = nullptr;
        std::array<VriDescriptor*, 3> m_views {};
        VriBuffer*                    m_weights        = nullptr;
        VriBuffer*                    m_predictions    = nullptr;
        VriBuffer*                    m_valuesReadback = nullptr;
        VriBuffer*                    m_imageReadback  = nullptr;
        VriTexture*                   m_image          = nullptr;
        bool                          m_ownsDevice     = false; // Offline owns the device; preview borrows ExampleApp.
        bool                          m_recorded       = false;
        uint64_t                      m_submission     = 0;
        Params                        m_params {};
        uint32_t                      m_rowBytes = 0;
        int                           m_errors   = 0;
    };

    void StopPreview(vriex::ExampleApp& app, VriDescriptor* view)
    {
        // Retire callback borrows and GUI descriptor-cache references before destroying the neural image.
        app.onGui       = nullptr;
        app.onPreRender = nullptr;
        app.c.DeviceWaitIdle(app.dev);
        app.guiApi.FreeImguiTexture(app.gui, view);
    }

    bool RunPreview(const Options& options)
    {
        vriex::ExampleApp app;
        app.requestFeatures = VriFeature_CooperativeVector;
        app.Init("mlp", 1160, 520, false);
        bool supported = false;
        try
        {
            NeuralImage image;
            supported = image.Initialize(options, app.dev);
            if (supported)
            {
                float errorScale = kErrorScale;
                float zoom       = 1.0f;
                app.onGui        = [&] {
                    ImGui::TextUnformatted("GPU neural texture / 2 -> 32 -> 32 -> 3 / FP16 + ReLU");
                    ImGui::TextUnformatted("Pretrained weights: no Python or training at runtime.");
                    ImGui::SliderFloat("Error amplification", &errorScale, 1.0f, 32.0f, "%.1fx");
                    ImGui::SliderFloat("Preview zoom", &zoom, 1.0f, 4.0f, "%.1fx");
                    image.SetErrorScale(errorScale);
                    const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
                    const float side = std::max(1.0f, std::min((displaySize.x - 64.0f) / 3.0f, displaySize.y - 185.0f));
                    const std::array<const char*, 3> labels {
                        "Analytic reference", "FP16 neural prediction", "Absolute error"};
                    if (ImGui::BeginTable(
                            "comparison", 3, ImGuiTableFlags_SizingStretchSame, ImVec2(side * 3 + 24.0f, 0)))
                    {
                        const float minimum =
                            std::min(0.5f, 0.5f - 0.5f / zoom + 0.5f / static_cast<float>(options.size));
                        const float maximum = 1.0f - minimum;
                        for (int panel = 0; panel < 3; ++panel)
                        {
                            ImGui::TableNextColumn();
                            ImGui::TextUnformatted(labels[panel]);
                            const float column = static_cast<float>(panel);
                            ImGui::Image(reinterpret_cast<ImTextureID>(image.ImageView()),
                                         ImVec2(side, side),
                                         ImVec2((column + minimum) / 3.0f, minimum),
                                         ImVec2((column + maximum) / 3.0f, maximum));
                        }
                        ImGui::EndTable();
                    }
                };
                app.onPreRender = [&](VriCommandBuffer* commands) { image.Record(commands); };
                app.SetupCapture();
                if (options.frames != 0)
                    app.maxFrames = options.frames;
                try
                {
                    app.Run();
                    if (app.frameValue > 1)
                        image.VerifyAndSave(options.output);
                }
                catch (...)
                {
                    StopPreview(app, image.ImageView());
                    throw;
                }
                StopPreview(app, image.ImageView());
            }
        }
        catch (...)
        {
            app.Shutdown();
            throw;
        }
        app.Shutdown();
        return supported;
    }
} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto options   = ParseOptions(argc, argv);
        bool       supported = false;
        if (options.offline)
        {
            NeuralImage image;
            supported = image.Initialize(options, nullptr);
            if (supported)
            {
                image.Render(options.frames);
                image.VerifyAndSave(options.output);
            }
        }
        else
        {
            supported = RunPreview(options);
        }
        if (!supported)
        {
            std::puts("[mlp] SKIP: requires compute cooperative vectors with FP16 matrix/bias/input/output (exit 77)");
            return 77;
        }
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr,
                     "[mlp] %s\nUsage: example-mlp [--offline] [--size N] [--frames N] [--output image.bmp]\n",
                     error.what());
        return 1;
    }
    return 0;
}
