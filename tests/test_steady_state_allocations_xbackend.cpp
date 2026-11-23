// A frame of the commands a renderer records every frame makes no heap allocation once warmed up, on every backend
// that comes up: resetting the allocator, barriers, a render pass with a pipeline, a descriptor set, vertex and
// index buffers, viewport, scissor and draws, buffer and texture copies, an upload, a readback, a compute dispatch
// where compute exists, and the submission.
//
// This binary's global operator new counts the allocations made on the test's own thread while a frame is recorded
// and submitted. It counts what VRI's C++ code allocates; drivers and the Rust wgpu-native allocate through their
// own allocators and are not counted.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include "shaders/tests/compute_fill_dxbc.h" // g_computeFillDxbcCS
#include "shaders/tests/compute_fill_spv.h"  // g_computeFillSpv
#include "shaders/tests/compute_fill_wgsl.h" // g_computeFillWgsl
#include "shaders/tests/triangle_tex_dxbc.h" // g_triangleTexDxbcVS / g_triangleTexDxbcPS
#include "shaders/tests/triangle_tex_spv.h"  // g_triangleTexSpv
#include "shaders/tests/triangle_tex_wgsl.h" // g_triangleTexWgsl

#include <cstdint>
#include <cstdlib>
#include <new>
#include <string>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <stdlib.h> // posix_memalign
#endif

namespace
{
    thread_local bool     t_counting    = false;
    thread_local uint64_t t_allocations = 0;

    void* Allocate(std::size_t size)
    {
        if (t_counting)
            ++t_allocations;
        void* p = std::malloc(size ? size : 1);
        if (!p)
            std::abort();
        return p;
    }

    void* AllocateAligned(std::size_t size, std::align_val_t alignment)
    {
        if (t_counting)
            ++t_allocations;
        const std::size_t a = static_cast<std::size_t>(alignment);
#if defined(_WIN32)
        void* p = _aligned_malloc(size ? size : 1, a);
        if (!p)
            std::abort();
#else
        void* p = nullptr;
        if (posix_memalign(&p, a < sizeof(void*) ? sizeof(void*) : a, size ? size : 1) != 0)
            std::abort();
#endif
        return p;
    }

    void FreeAligned(void* p)
    {
#if defined(_WIN32)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }
} // namespace

void* operator new(std::size_t size) { return Allocate(size); }
void* operator new[](std::size_t size) { return Allocate(size); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept { return Allocate(size); }
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept { return Allocate(size); }
void* operator new(std::size_t size, std::align_val_t alignment) { return AllocateAligned(size, alignment); }
void* operator new[](std::size_t size, std::align_val_t alignment) { return AllocateAligned(size, alignment); }
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return AllocateAligned(size, alignment);
}
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return AllocateAligned(size, alignment);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { FreeAligned(p); }
void operator delete[](void* p, std::align_val_t) noexcept { FreeAligned(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { FreeAligned(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { FreeAligned(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { FreeAligned(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { FreeAligned(p); }

namespace
{
    constexpr uint32_t kSize = 64; // 256-byte RGBA8 rows, the copy pitch D3D12 and WebGPU require

    struct Shaders
    {
        const void* vs;
        size_t      vsSize;
        const void* ps;
        size_t      psSize;
        const void* cs;
        size_t      csSize;
    };

    VriTexture* Texture(const VriCoreInterface& c, VriDevice* dev, VriTextureUsageFlags usage, uint32_t mipNum = 1)
    {
        VriTextureDesc td {};
        td.type                    = VriTextureType_2D;
        td.format                  = VriFormat_RGBA8_UNORM;
        td.width                   = kSize;
        td.height                  = kSize;
        td.depth                   = 1;
        td.mipNum                  = mipNum;
        td.layerNum                = 1;
        td.sampleNum               = 1;
        td.usage                   = usage;
        td.memoryLocation          = VriMemoryLocation_Device;
        td.clearValue.color.f32[3] = 1.0f;
        VriTexture* t              = nullptr;
        REQUIRE(c.CreateTexture(dev, &td, &t) == VriResult_Success);
        return t;
    }

    VriBuffer* Buffer(const VriCoreInterface& c,
                      VriDevice*              dev,
                      uint64_t                size,
                      VriBufferUsageFlags     usage,
                      VriMemoryLocation       location)
    {
        VriBufferDesc bd {};
        bd.size           = size;
        bd.usage          = usage;
        bd.memoryLocation = location;
        VriBuffer* b      = nullptr;
        REQUIRE(c.CreateBuffer(dev, &bd, &b) == VriResult_Success);
        return b;
    }

    void Barrier(const VriCoreInterface& c,
                 VriCommandBuffer*       cmd,
                 VriTexture*             t,
                 VriAccessFlags          ba,
                 VriLayout               bl,
                 VriAccessFlags          aa,
                 VriLayout               al,
                 uint32_t                baseMip = 0,
                 uint32_t                mipNum  = 0)
    {
        VriTextureBarrierDesc tb {};
        tb.texture       = t;
        tb.before.access = ba;
        tb.before.layout = bl;
        tb.before.stages = VriPipelineStage_AllCommands;
        tb.after.access  = aa;
        tb.after.layout  = al;
        tb.after.stages  = VriPipelineStage_AllCommands;
        tb.aspect        = VriImageAspect_Color;
        tb.baseMip       = baseMip;
        tb.mipNum        = mipNum;
        VriBarrierGroupDesc g {};
        g.textures   = &tb;
        g.textureNum = 1;
        c.CmdBarrier(cmd, &g);
    }

    void Check(VriGraphicsAPI api, const char* name, const Shaders& shaders)
    {
        VriDeviceCreationDesc dc  = vriDefaultDeviceCreationDesc();
        dc.coordinates.clipSpaceY = VriClipSpaceY_Up;
        dc.graphicsAPI            = api;
        dc.bestEffort             = VRI_TRUE;
        VriDevice* dev            = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << std::string(name) << "] unavailable - skipped");
            return;
        }
        INFO(std::string(name));
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);
        const bool compute = c.GetDeviceDesc(dev)->hasComputeShader != VRI_FALSE;

        // Resources, all created before the measured frames.
        VriTexture* color = Texture(c, dev, VriTextureUsage_ColorAttachment | VriTextureUsage_TransferSrc);
        VriTexture* sampled =
            Texture(c, dev, VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst | VriTextureUsage_TransferSrc);
        VriTexture* copyTarget = Texture(c, dev, VriTextureUsage_TransferDst);
        // Two mips barriered one at a time: a barrier on part of a texture (D3D12 transitions it per subresource).
        VriTexture*    mipped     = Texture(c, dev, VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst, 2);
        const uint64_t imageBytes = static_cast<uint64_t>(kSize) * kSize * 4;
        VriBuffer*     upload = Buffer(c, dev, imageBytes, VriBufferUsage_TransferSrc, VriMemoryLocation_HostUpload);
        VriBuffer* readback   = Buffer(c, dev, imageBytes, VriBufferUsage_TransferDst, VriMemoryLocation_HostReadback);
        VriBuffer* vertices =
            Buffer(c,
                   dev,
                   256,
                   VriBufferUsage_VertexBuffer | VriBufferUsage_TransferDst | VriBufferUsage_TransferSrc,
                   VriMemoryLocation_Device);
        VriBuffer* indices =
            Buffer(c, dev, 256, VriBufferUsage_IndexBuffer | VriBufferUsage_TransferDst, VriMemoryLocation_Device);
        VriBuffer* storage =
            Buffer(c, dev, 256, VriBufferUsage_StorageBuffer | VriBufferUsage_TransferSrc, VriMemoryLocation_Device);

        VriTextureViewDesc cvd {};
        cvd.texture              = color;
        cvd.viewType             = VriTextureViewType_2D;
        cvd.aspect               = VriImageAspect_Color;
        VriDescriptor* colorView = nullptr;
        REQUIRE(c.CreateTextureView(dev, &cvd, &colorView) == VriResult_Success);
        VriTextureViewDesc svd = cvd;
        svd.texture            = sampled;
        VriDescriptor* texView = nullptr;
        REQUIRE(c.CreateTextureView(dev, &svd, &texView) == VriResult_Success);
        VriSamplerDesc smp {};
        smp.magFilter          = VriFilter_Nearest;
        smp.minFilter          = VriFilter_Nearest;
        smp.addressModeU       = VriAddressMode_ClampToEdge;
        smp.addressModeV       = VriAddressMode_ClampToEdge;
        smp.addressModeW       = VriAddressMode_ClampToEdge;
        smp.maxLod             = 1.0f;
        VriDescriptor* sampler = nullptr;
        REQUIRE(c.CreateSampler(dev, &smp, &sampler) == VriResult_Success);

        // Graphics: set 0 = {0: texture, 1: sampler}.
        VriDescriptorRangeDesc ranges[2] {};
        ranges[0].baseRegister   = 0;
        ranges[0].descriptorNum  = 1;
        ranges[0].descriptorType = VriDescriptorType_Texture;
        ranges[0].shaderStages   = VriShaderStage_Fragment;
        ranges[1].baseRegister   = 1;
        ranges[1].descriptorNum  = 1;
        ranges[1].descriptorType = VriDescriptorType_Sampler;
        ranges[1].shaderStages   = VriShaderStage_Fragment;
        VriDescriptorSetDesc setDesc {};
        setDesc.ranges   = ranges;
        setDesc.rangeNum = 2;
        VriPipelineLayoutDesc ld {};
        ld.descriptorSets         = &setDesc;
        ld.descriptorSetNum       = 1;
        VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);
        VriShaderDesc sh[2] {};
        sh[0].stage          = VriShaderStage_Vertex;
        sh[0].bytecode       = shaders.vs;
        sh[0].bytecodeSize   = shaders.vsSize;
        sh[0].entryPointName = "vertexMain";
        sh[1].stage          = VriShaderStage_Fragment;
        sh[1].bytecode       = shaders.ps;
        sh[1].bytecodeSize   = shaders.psSize;
        sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca {};
        ca.format         = VriFormat_RGBA8_UNORM;
        ca.colorWriteMask = VriColorWrite_RGBA;
        VriGraphicsPipelineDesc pd {};
        pd.pipelineLayout          = layout;
        pd.shaders                 = sh;
        pd.shaderNum               = 2;
        pd.inputAssembly.topology  = VriPrimitiveTopology_TriangleList;
        pd.rasterization.cullMode  = VriCullMode_None;
        pd.rasterization.lineWidth = 1.0f;
        pd.multisample.sampleNum   = 1;
        pd.outputMerger.colors     = &ca;
        pd.outputMerger.colorNum   = 1;
        VriPipeline* pipeline      = nullptr;
        REQUIRE(c.CreateGraphicsPipeline(dev, &pd, &pipeline) == VriResult_Success);
        VriDescriptorPoolDesc pool {};
        pool.descriptorSetMaxNum          = 2;
        pool.textureMaxNum                = 1;
        pool.samplerMaxNum                = 1;
        pool.storageBufferMaxNum          = 1;
        VriDescriptorPool* descriptorPool = nullptr;
        REQUIRE(c.CreateDescriptorPool(dev, &pool, &descriptorPool) == VriResult_Success);
        VriDescriptorSet* set = nullptr;
        REQUIRE(c.AllocateDescriptorSets(descriptorPool, layout, 0, &set, 1) == VriResult_Success);
        const VriDescriptor*         texDescs[1] = {texView};
        const VriDescriptor*         smpDescs[1] = {sampler};
        VriDescriptorRangeUpdateDesc updates[2] {};
        updates[0].descriptors   = texDescs;
        updates[0].descriptorNum = 1;
        updates[1].descriptors   = smpDescs;
        updates[1].descriptorNum = 1;
        c.UpdateDescriptorRanges(set, 0, 2, updates);

        // Compute: set 0 = {0: storage buffer}.
        VriPipelineLayout* computeLayout   = nullptr;
        VriPipeline*       computePipeline = nullptr;
        VriDescriptorSet*  computeSet      = nullptr;
        VriDescriptor*     storageView     = nullptr;
        if (compute)
        {
            VriDescriptorRangeDesc range {};
            range.descriptorNum  = 1;
            range.descriptorType = VriDescriptorType_StorageBuffer;
            range.shaderStages   = VriShaderStage_Compute;
            VriDescriptorSetDesc cset {};
            cset.ranges   = &range;
            cset.rangeNum = 1;
            VriPipelineLayoutDesc cld {};
            cld.descriptorSets   = &cset;
            cld.descriptorSetNum = 1;
            REQUIRE(c.CreatePipelineLayout(dev, &cld, &computeLayout) == VriResult_Success);
            VriComputePipelineDesc cpd {};
            cpd.pipelineLayout        = computeLayout;
            cpd.shader.stage          = VriShaderStage_Compute;
            cpd.shader.bytecode       = shaders.cs;
            cpd.shader.bytecodeSize   = shaders.csSize;
            cpd.shader.entryPointName = "computeMain";
            REQUIRE(c.CreateComputePipeline(dev, &cpd, &computePipeline) == VriResult_Success);
            REQUIRE(c.AllocateDescriptorSets(descriptorPool, computeLayout, 0, &computeSet, 1) == VriResult_Success);
            VriBufferViewDesc bv {};
            bv.buffer   = storage;
            bv.viewType = VriDescriptorType_StorageBuffer;
            bv.size     = 256;
            REQUIRE(c.CreateBufferView(dev, &bv, &storageView) == VriResult_Success);
            const VriDescriptor*         sbDescs[1] = {storageView};
            VriDescriptorRangeUpdateDesc sbUpdate {};
            sbUpdate.descriptors   = sbDescs;
            sbUpdate.descriptorNum = 1;
            c.UpdateDescriptorRanges(computeSet, 0, 1, &sbUpdate);
        }

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        VriBufferTextureCopyDesc imageRegion {};
        imageRegion.bufferRowLength  = kSize;
        imageRegion.texture.aspect   = VriImageAspect_Color;
        imageRegion.texture.layerNum = 1;
        imageRegion.texture.width    = kSize;
        imageRegion.texture.height   = kSize;
        imageRegion.texture.depth    = 1;

        uint64_t   value = 0;
        const auto frame = [&](bool first) {
            c.ResetCommandAllocator(alloc);
            REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
            // Upload into the sampled texture, then sample it in a render pass.
            Barrier(c,
                    cmd,
                    sampled,
                    first ? VriAccess_None : VriAccess_ShaderResourceRead,
                    first ? VriLayout_Undefined : VriLayout_ShaderResource,
                    VriAccess_CopyDestinationWrite,
                    VriLayout_CopyDestination);
            c.CmdUploadBufferToTexture(cmd, sampled, upload, &imageRegion);
            Barrier(c,
                    cmd,
                    sampled,
                    VriAccess_CopyDestinationWrite,
                    VriLayout_CopyDestination,
                    VriAccess_ShaderResourceRead,
                    VriLayout_ShaderResource);
            Barrier(c,
                    cmd,
                    color,
                    first ? VriAccess_None : VriAccess_CopySourceRead,
                    first ? VriLayout_Undefined : VriLayout_CopySource,
                    VriAccess_ColorAttachmentWrite,
                    VriLayout_ColorAttachment);
            VriAttachmentDesc rt {};
            rt.view                    = colorView;
            rt.loadOp                  = VriAttachmentLoadOp_Clear;
            rt.storeOp                 = VriAttachmentStoreOp_Store;
            rt.clearValue.color.f32[3] = 1.0f;
            VriAttachmentsDesc att {};
            att.colors            = &rt;
            att.colorNum          = 1;
            att.renderArea.width  = kSize;
            att.renderArea.height = kSize;
            att.layerNum          = 1;
            c.CmdBeginRendering(cmd, &att);
            const VriViewport viewport {0, 0, static_cast<float>(kSize), static_cast<float>(kSize), 0, 1};
            c.CmdSetViewports(cmd, &viewport, 1);
            const VriRect scissor {0, 0, kSize, kSize};
            c.CmdSetScissors(cmd, &scissor, 1);
            c.CmdSetPipeline(cmd, pipeline);
            c.CmdSetPipelineLayout(cmd, layout);
            c.CmdSetDescriptorSet(cmd, 0, set);
            const VriVertexBufferBinding vb {vertices, 0};
            c.CmdSetVertexBuffers(cmd, 0, &vb, 1);
            c.CmdSetIndexBuffer(cmd, indices, 0, VriIndexType_UInt16);
            VriDrawDesc draw {};
            draw.vertexNum   = 3;
            draw.instanceNum = 1;
            c.CmdDraw(cmd, &draw);
            VriDrawIndexedDesc drawIndexed {};
            drawIndexed.indexNum    = 3;
            drawIndexed.instanceNum = 1;
            c.CmdDrawIndexed(cmd, &drawIndexed);
            c.CmdEndRendering(cmd);
            Barrier(c,
                    cmd,
                    color,
                    VriAccess_ColorAttachmentWrite,
                    VriLayout_ColorAttachment,
                    VriAccess_CopySourceRead,
                    VriLayout_CopySource);
            Barrier(c,
                    cmd,
                    copyTarget,
                    first ? VriAccess_None : VriAccess_CopyDestinationWrite,
                    first ? VriLayout_Undefined : VriLayout_CopyDestination,
                    VriAccess_CopyDestinationWrite,
                    VriLayout_CopyDestination);
            c.CmdCopyTexture(cmd, copyTarget, color, nullptr);
            for (uint32_t mip = 0; mip < 2; ++mip)
            {
                Barrier(c,
                        cmd,
                        mipped,
                        first ? VriAccess_None : VriAccess_ShaderResourceRead,
                        first ? VriLayout_Undefined : VriLayout_ShaderResource,
                        VriAccess_CopyDestinationWrite,
                        VriLayout_CopyDestination,
                        mip,
                        1);
                Barrier(c,
                        cmd,
                        mipped,
                        VriAccess_CopyDestinationWrite,
                        VriLayout_CopyDestination,
                        VriAccess_ShaderResourceRead,
                        VriLayout_ShaderResource,
                        mip,
                        1);
            }
            c.CmdReadbackTextureToBuffer(cmd, readback, color, &imageRegion);
            VriBufferCopyDesc bufferCopy {};
            bufferCopy.size = 256;
            c.CmdCopyBuffer(cmd, indices, vertices, &bufferCopy);
            if (compute)
            {
                c.CmdSetPipeline(cmd, computePipeline);
                c.CmdSetPipelineLayout(cmd, computeLayout);
                c.CmdSetDescriptorSet(cmd, 0, computeSet);
                VriDispatchDesc dispatch {};
                dispatch.x = 1;
                dispatch.y = 1;
                dispatch.z = 1;
                c.CmdDispatch(cmd, &dispatch);
            }
            REQUIRE(c.EndCommandBuffer(cmd) == VriResult_Success);
            VriFenceSubmitDesc signal {};
            signal.fence = fence;
            signal.value = ++value;
            VriQueueSubmitDesc submit {};
            submit.commandBuffers   = &cmd;
            submit.commandBufferNum = 1;
            submit.signalFences     = &signal;
            submit.signalFenceNum   = 1;
            REQUIRE(c.QueueSubmit(queue, &submit) == VriResult_Success);
        };

        // Warm up: caches (framebuffers, descriptor tables, staging) fill in the first frames.
        for (int i = 0; i < 3; ++i)
        {
            frame(i == 0);
            c.Wait(fence, value);
        }
        for (int i = 0; i < 5; ++i)
        {
            t_allocations = 0;
            t_counting    = true;
            frame(false);
            t_counting = false;
            CAPTURE(i);
            CHECK(t_allocations == 0);
            c.Wait(fence, value);
        }

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        if (compute)
        {
            c.DestroyDescriptor(storageView);
            c.DestroyPipeline(computePipeline);
            c.DestroyPipelineLayout(computeLayout);
        }
        c.DestroyDescriptorPool(descriptorPool);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyDescriptor(sampler);
        c.DestroyDescriptor(texView);
        c.DestroyDescriptor(colorView);
        for (VriBuffer* b : {upload, readback, vertices, indices, storage})
            c.DestroyBuffer(b);
        for (VriTexture* t : {color, sampled, copyTarget, mipped})
            c.DestroyTexture(t);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("Steady-state frames make no heap allocation in VRI")
{
    const Shaders spirv {g_triangleTexSpv,
                         sizeof(g_triangleTexSpv),
                         g_triangleTexSpv,
                         sizeof(g_triangleTexSpv),
                         g_computeFillSpv,
                         sizeof(g_computeFillSpv)};
    const Shaders wgsl {g_triangleTexWgsl,
                        sizeof(g_triangleTexWgsl),
                        g_triangleTexWgsl,
                        sizeof(g_triangleTexWgsl),
                        g_computeFillWgsl,
                        sizeof(g_computeFillWgsl)};
    const Shaders dxbc {g_triangleTexDxbcVS,
                        sizeof(g_triangleTexDxbcVS),
                        g_triangleTexDxbcPS,
                        sizeof(g_triangleTexDxbcPS),
                        g_computeFillDxbcCS,
                        sizeof(g_computeFillDxbcCS)};
    Check(VriGraphicsAPI_Vulkan, "Vulkan", spirv);
    Check(VriGraphicsAPI_D3D12, "D3D12", dxbc);
    Check(VriGraphicsAPI_OpenGL, "OpenGL", spirv);
    Check(VriGraphicsAPI_WebGPU, "WebGPU", wgsl);
}
