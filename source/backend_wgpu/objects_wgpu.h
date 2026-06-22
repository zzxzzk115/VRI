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
        // Bounce buffers created to satisfy copyBufferToTexture's 256-byte bytesPerRow; held
        // until the GPU consumes them (released at the next BeginCommandBuffer).
        std::vector<WGPUBuffer> tempUploads;
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
        WGPUTextureFormat format = WGPUTextureFormat_Undefined; // for render-pass stencil-aspect detection
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
        // Push constants are emulated as a dynamic-offset uniform in a reserved bind group (group
        // == descriptorSetNum). pushBuffer is a RING: each CmdSetConstants writes the next slot and
        // binds it with a dynamic offset, so multiple per-draw push values in one pass each keep
        // their own value (a single fixed binding would collapse to the last queue-written value).
        // The buffer + bind group are created lazily in CmdSetConstants.
        bool          hasPush = false;
        uint32_t      pushGroup = 0;
        uint32_t      pushSize = 0;
        uint32_t      pushStride = 0; // per-slot stride (push size rounded up to the 256B dynamic-offset alignment)
        uint32_t      pushCursor = 0; // next ring slot (monotonic, wraps mod the slot count)
        WGPUBuffer    pushBuffer = nullptr;
        WGPUBindGroup pushBindGroup = nullptr;
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
        bool               stencilTest = false;     // set the stencil reference on bind
        uint32_t           stencilReference = 0;
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
