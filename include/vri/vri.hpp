/*
 * vri.hpp - header-only C++23 convenience layer over the VRI C API.
 *
 * Phase-0 stub: scoped enum aliases, a Result helper, and an RAII Device. The
 * full wrapper (RAII for every handle, std::expected returns, the cached core
 * interface table) is fleshed out alongside the Vulkan backend.
 */
#ifndef VRI_HPP
#define VRI_HPP

#include "vri.h"

#include <expected>
#include <utility>

namespace vri
{
    using Result       = ::VriResult;
    using GraphicsAPI  = ::VriGraphicsAPI;
    using Format       = ::VriFormat;
    using DeviceDesc   = ::VriDeviceDesc;
    using AdapterDesc  = ::VriAdapterDesc;
    using QueueType    = ::VriQueueType;
    using TextureType  = ::VriTextureType;
    using MemoryLocation = ::VriMemoryLocation;
    using DescriptorType = ::VriDescriptorType;
    using Layout       = ::VriLayout;
    // descriptor structs (used directly with the cached core interface)
    using BufferDesc           = ::VriBufferDesc;
    using TextureDesc          = ::VriTextureDesc;
    using SamplerDesc          = ::VriSamplerDesc;
    using PipelineLayoutDesc   = ::VriPipelineLayoutDesc;
    using GraphicsPipelineDesc = ::VriGraphicsPipelineDesc;
    using ComputePipelineDesc  = ::VriComputePipelineDesc;
    using AttachmentsDesc      = ::VriAttachmentsDesc;
    using BarrierGroupDesc     = ::VriBarrierGroupDesc;
    using QueueSubmitDesc      = ::VriQueueSubmitDesc;
    using CoreInterface        = ::VriCoreInterface;
    using SwapChainInterface   = ::VriSwapChainInterface;

    [[nodiscard]] inline bool succeeded(Result r) noexcept { return r == VriResult_Success; }

    // Minimal RAII owner for a VriDevice*. Expands into a full handle wrapper set later.
    class Device
    {
    public:
        Device() = default;
        explicit Device(VriDevice* d) noexcept : m_device(d) {}
        Device(const Device&)            = delete;
        Device& operator=(const Device&) = delete;
        Device(Device&& o) noexcept : m_device(std::exchange(o.m_device, nullptr)) {}
        Device& operator=(Device&& o) noexcept
        {
            if (this != &o)
            {
                reset();
                m_device = std::exchange(o.m_device, nullptr);
            }
            return *this;
        }
        ~Device() { reset(); }

        [[nodiscard]] VriDevice* get() const noexcept { return m_device; }
        explicit operator bool() const noexcept { return m_device != nullptr; }

        void reset() noexcept
        {
            if (m_device)
            {
                ::vriDestroyDevice(m_device);
                m_device = nullptr;
            }
        }

        [[nodiscard]] static std::expected<Device, Result> create(const VriDeviceCreationDesc& desc)
        {
            VriDevice* d = nullptr;
            const Result r = ::vriCreateDevice(&desc, &d);
            if (r != VriResult_Success)
                return std::unexpected(r);
            return Device(d);
        }

    private:
        VriDevice* m_device = nullptr;
    };
} // namespace vri

#endif /* VRI_HPP */
