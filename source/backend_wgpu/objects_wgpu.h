// objects_wgpu.h - concrete WebGPU objects behind the opaque VRI handles.
#pragma once

#include <vector>

#include <webgpu/webgpu.h>

#include <vri/vri.h>

namespace vri::wgpu
{
    class DeviceWGPU;

    struct QueueWGPU
    {
        DeviceWGPU* device;
        WGPUQueue   queue;
    };

    struct CommandAllocatorWGPU
    {
        DeviceWGPU* device; // WebGPU has no command pool; thin wrapper
    };

    struct CommandBufferWGPU
    {
        DeviceWGPU*            device;
        WGPUCommandEncoder     encoder;     // valid between Begin and End
        WGPURenderPassEncoder  pass;        // valid between BeginRendering/EndRendering
        WGPUComputePassEncoder computePass; // lazily begun for dispatch; ended on any encoder-level op
        WGPUCommandBuffer      finished;    // produced by End, consumed by Submit
        VriPipelineLayout*     boundLayout;
        WGPURenderPipeline     boundPipeline;
    };

    struct BufferWGPU
    {
        DeviceWGPU* device;
        WGPUBuffer  buffer;
        uint64_t    size;
        WGPUMapMode mapMode; // None / Read / Write (for MapBuffer)
    };

    struct TextureWGPU
    {
        DeviceWGPU*       device;
        WGPUTexture       texture;
        WGPUTextureFormat format;
        uint32_t          width;
        uint32_t          height;
        uint32_t          depth;
        uint32_t          mipNum;
        uint32_t          layerNum;
        uint32_t          texelSize; // bytes per pixel (uncompressed)
        bool              owned;
    };

    struct DescriptorWGPU
    {
        enum class Kind { TextureView, BufferView, Sampler } kind;
        DeviceWGPU*       device;
        WGPUTextureView   view;
        WGPUSampler       sampler;
        const BufferWGPU* buffer;
        uint64_t          bufferOffset;
        uint64_t          bufferRange;
    };

    struct RangeInfoWGPU
    {
        uint32_t          binding;
        VriDescriptorType type;
        uint32_t          count;
    };

    struct PipelineLayoutWGPU
    {
        DeviceWGPU*                            device;
        WGPUPipelineLayout                     layout;
        std::vector<WGPUBindGroupLayout>       bindGroupLayouts;
        std::vector<std::vector<RangeInfoWGPU>> setRanges; // [set][range]
    };

    struct DescriptorSetWGPU
    {
        DeviceWGPU*             device;
        const PipelineLayoutWGPU* layout;
        uint32_t                setIndex;
        WGPUBindGroup           bindGroup; // (re)created on UpdateDescriptorRanges
    };

    struct DescriptorPoolWGPU
    {
        DeviceWGPU*                    device; // WebGPU has no pool; owns the allocated sets
        std::vector<DescriptorSetWGPU*> sets;
    };

    struct PipelineWGPU
    {
        DeviceWGPU*        device;
        WGPURenderPipeline render;
        WGPUComputePipeline compute;
        bool               isCompute;
    };

    struct FenceWGPU
    {
        DeviceWGPU* device;
        uint64_t    value; // emulated timeline
    };

    inline VriQueue*           ToHandle(QueueWGPU* q)             { return reinterpret_cast<VriQueue*>(q); }
    inline VriCommandAllocator* ToHandle(CommandAllocatorWGPU* a) { return reinterpret_cast<VriCommandAllocator*>(a); }
    inline VriCommandBuffer*   ToHandle(CommandBufferWGPU* c)     { return reinterpret_cast<VriCommandBuffer*>(c); }
    inline VriBuffer*          ToHandle(BufferWGPU* b)            { return reinterpret_cast<VriBuffer*>(b); }
    inline VriTexture*         ToHandle(TextureWGPU* t)           { return reinterpret_cast<VriTexture*>(t); }
    inline VriDescriptor*      ToHandle(DescriptorWGPU* d)        { return reinterpret_cast<VriDescriptor*>(d); }
    inline VriPipelineLayout*  ToHandle(PipelineLayoutWGPU* p)    { return reinterpret_cast<VriPipelineLayout*>(p); }
    inline VriPipeline*        ToHandle(PipelineWGPU* p)          { return reinterpret_cast<VriPipeline*>(p); }
    inline VriFence*           ToHandle(FenceWGPU* f)             { return reinterpret_cast<VriFence*>(f); }
    inline VriDescriptorPool*  ToHandle(DescriptorPoolWGPU* p)    { return reinterpret_cast<VriDescriptorPool*>(p); }
    inline VriDescriptorSet*   ToHandle(DescriptorSetWGPU* s)     { return reinterpret_cast<VriDescriptorSet*>(s); }
} // namespace vri::wgpu
