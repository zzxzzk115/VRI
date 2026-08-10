// never-silent contract tests. VRI's README promises "Honest capabilities, never
// silent... it never silently no-ops": every core/extension entry point must either
// perform the operation correctly, or (when a backend genuinely can't) return an
// explicit VriResult_Unsupported / emit a diagnostic. A backend must never accept a
// call, do nothing, and hand back a wrong result with no signal.
//
// This file pins that contract as a regression net for the tiered-backend refactor:
//
//   * CmdCopyTexture and CmdDispatchIndirect are verified by *result* across every
//     available backend. The D3D12 backend currently ships them as empty `{}` no-ops
//     (core_d3d12.cpp CmdDispatchIndirect / CmdCopyTexture) with no diagnostic - a
//     silent wrong result. These tests therefore FAIL on D3D12 today (red) and turn
//     green once the Tier-1 DX12 gap is implemented. On a backend that legitimately
//     cannot perform the op (e.g. WebGL2 indirect dispatch), the never-silent clause
//     is satisfied instead by an emitted diagnostic - so the contract check passes
//     there without claiming false support.
//
//   * The interface-registry contract: a capability a device reports as *false* must
//     leave the matching optional interface un-queryable (Unsupported), never a
//     silently half-wired table. Generalizes the per-extension checks in
//     test_omm_d3d12.cpp / test_vrs_mtl.cpp across all available backends.
//
// Unavailable backends self-skip, so the file is stable under the CI backend matrix.
#include <doctest/doctest.h>

#include <vri/ext/vri_ext_meshshader.h>
#include <vri/ext/vri_ext_raytracing.h>
#include <vri/ext/vri_ext_vrs.h>
#include <vri/vri.h>

#include <cstdint>
#include <string>

#include "shaders/tests/compute_fill_dxbc.h" // g_computeFillDxbcCS (Direct3D 12)
#include "shaders/tests/compute_fill_spv.h"  // g_computeFillSpv    (Vulkan + OpenGL via SPIRV-Cross)
#include "shaders/tests/compute_fill_wgsl.h" // g_computeFillWgsl   (WebGPU)

namespace
{
    // Diagnostic sink: counts Error-severity messages so a test can tell "explicit
    // Unsupported diagnostic" apart from "silent no-op". Validation-layer misuse
    // messages would also land here, but these tests only issue valid calls, so any
    // Error observed around the probed op is the backend's own never-silent signal.
    struct MsgState
    {
        int errors = 0;
    };
    void VRI_CALL OnMessage(void* userArg, VriMessageSeverity severity, const char* /*message*/)
    {
        if (severity == VriMessageSeverity_Error)
            static_cast<MsgState*>(userArg)->errors++;
    }

    const char* ApiName(VriGraphicsAPI api)
    {
        switch (api)
        {
            case VriGraphicsAPI_Vulkan:
                return "Vulkan";
            case VriGraphicsAPI_WebGPU:
                return "WebGPU";
            case VriGraphicsAPI_OpenGL:
                return "OpenGL";
            case VriGraphicsAPI_D3D12:
                return "D3D12";
            case VriGraphicsAPI_Metal:
                return "Metal";
            default:
                return "?";
        }
    }

    // Result of probing one backend: whether the device came up, whether the op
    // produced the correct GPU result, and whether the backend emitted a diagnostic.
    struct ContractProbe
    {
        bool ran           = false; // backend available and device created
        bool skipped       = false; // op not applicable on this backend (e.g. no compute)
        bool resultCorrect = false; // op produced the expected bytes
        bool sawDiagnostic = false; // backend emitted an Error diagnostic for the op
    };

    // ---- CmdCopyTexture: texture -> texture copy, verified by readback ----------
    ContractProbe RunCopyTexture(VriGraphicsAPI api)
    {
        ContractProbe        p;
        MsgState             msg;
        VriCallbackInterface cb {};
        cb.userArg         = &msg;
        cb.MessageCallback = OnMessage;

        VriDeviceCreationDesc dc {};
        dc.graphicsAPI       = api;
        dc.enableValidation  = VRI_TRUE;
        dc.bestEffort        = VRI_TRUE;
        dc.callbackInterface = &cb;
        VriDevice* dev       = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return p; // ran == false -> caller skips
        p.ran = true;

        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        constexpr uint32_t kW = 64, kH = 64; // 64*4 = 256B rows (WebGPU copy alignment)
        const uint64_t     bytes = static_cast<uint64_t>(kW) * kH * 4;

        auto makeTex = [&](VriTexture** out) {
            VriTextureDesc td {};
            td.type           = VriTextureType_2D;
            td.format         = VriFormat_RGBA8_UNORM;
            td.width          = kW;
            td.height         = kH;
            td.depth          = 1;
            td.mipNum         = 1;
            td.layerNum       = 1;
            td.sampleNum      = 1;
            td.usage          = VriTextureUsage_TransferSrc | VriTextureUsage_TransferDst;
            td.memoryLocation = VriMemoryLocation_Device;
            REQUIRE(c.CreateTexture(dev, &td, out) == VriResult_Success);
        };
        VriTexture* src = nullptr;
        VriTexture* dst = nullptr;
        makeTex(&src);
        makeTex(&dst);

        // solid-blue staging upload source
        VriBufferDesc sb {};
        sb.size            = bytes;
        sb.usage           = VriBufferUsage_TransferSrc;
        sb.memoryLocation  = VriMemoryLocation_HostUpload;
        VriBuffer* staging = nullptr;
        REQUIRE(c.CreateBuffer(dev, &sb, &staging) == VriResult_Success);
        {
            uint8_t* m = static_cast<uint8_t*>(c.MapBuffer(staging, 0, bytes));
            REQUIRE(m != nullptr);
            for (uint64_t i = 0; i < bytes; i += 4)
            {
                m[i + 0] = 0;
                m[i + 1] = 0;
                m[i + 2] = 255;
                m[i + 3] = 255;
            }
            c.UnmapBuffer(staging);
        }
        VriBufferDesc rb {};
        rb.size             = bytes;
        rb.usage            = VriBufferUsage_TransferDst;
        rb.memoryLocation   = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rb, &readback) == VriResult_Success);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        auto barrier = [&](VriTexture*           t,
                           VriAccessFlags        ba,
                           VriLayout             bl,
                           VriPipelineStageFlags bs,
                           VriAccessFlags        aa,
                           VriLayout             al,
                           VriPipelineStageFlags as) {
            VriTextureBarrierDesc tb {};
            tb.texture       = t;
            tb.before.access = ba;
            tb.before.layout = bl;
            tb.before.stages = bs;
            tb.after.access  = aa;
            tb.after.layout  = al;
            tb.after.stages  = as;
            tb.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc g {};
            g.textures   = &tb;
            g.textureNum = 1;
            c.CmdBarrier(cmd, &g);
        };

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);

        // staging -> src, then src to copy-source
        barrier(src,
                VriAccess_None,
                VriLayout_Undefined,
                VriPipelineStage_None,
                VriAccess_CopyDestinationWrite,
                VriLayout_CopyDestination,
                VriPipelineStage_Transfer);
        VriBufferTextureCopyDesc up {};
        up.bufferOffset     = 0;
        up.texture.aspect   = VriImageAspect_Color;
        up.texture.layerNum = 1;
        up.texture.width    = kW;
        up.texture.height   = kH;
        c.CmdUploadBufferToTexture(cmd, src, staging, &up);
        barrier(src,
                VriAccess_CopyDestinationWrite,
                VriLayout_CopyDestination,
                VriPipelineStage_Transfer,
                VriAccess_CopySourceRead,
                VriLayout_CopySource,
                VriPipelineStage_Transfer);
        // dst to copy-destination
        barrier(dst,
                VriAccess_None,
                VriLayout_Undefined,
                VriPipelineStage_None,
                VriAccess_CopyDestinationWrite,
                VriLayout_CopyDestination,
                VriPipelineStage_Transfer);

        // ---- the operation under contract ----
        const int          errBefore = msg.errors;
        VriTextureCopyDesc region {};
        region.src.aspect   = VriImageAspect_Color;
        region.src.layerNum = 1;
        region.dst.aspect   = VriImageAspect_Color;
        region.dst.layerNum = 1;
        c.CmdCopyTexture(cmd, dst, src, &region);

        // dst -> copy-source, readback
        barrier(dst,
                VriAccess_CopyDestinationWrite,
                VriLayout_CopyDestination,
                VriPipelineStage_Transfer,
                VriAccess_CopySourceRead,
                VriLayout_CopySource,
                VriPipelineStage_Transfer);
        VriBufferTextureCopyDesc tc {};
        tc.texture.aspect   = VriImageAspect_Color;
        tc.texture.layerNum = 1;
        c.CmdReadbackTextureToBuffer(cmd, readback, dst, &tc);
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
        p.sawDiagnostic = msg.errors > errBefore;

        const uint8_t* px = static_cast<const uint8_t*>(c.MapBuffer(readback, 0, bytes));
        REQUIRE(px != nullptr);
        const uint32_t o = ((kH / 2) * kW + (kW / 2)) * 4;
        p.resultCorrect  = (px[o + 0] == 0 && px[o + 1] == 0 && px[o + 2] == 255);
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(staging);
        c.DestroyTexture(dst);
        c.DestroyTexture(src);
        vriDestroyDevice(dev);
        return p;
    }

    // ---- CmdDispatchIndirect: group count read from a buffer, verified by result -
    ContractProbe RunDispatchIndirect(VriGraphicsAPI api, const void* shader, size_t shaderSize)
    {
        ContractProbe        p;
        MsgState             msg;
        VriCallbackInterface cb {};
        cb.userArg         = &msg;
        cb.MessageCallback = OnMessage;

        VriDeviceCreationDesc dc {};
        dc.graphicsAPI       = api;
        dc.enableValidation  = VRI_TRUE;
        dc.bestEffort        = VRI_TRUE;
        dc.callbackInterface = &cb;
        VriDevice* dev       = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
            return p;
        p.ran = true;

        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        if (c.GetDeviceDesc(dev)->hasComputeShader == VRI_FALSE)
        {
            p.skipped = true; // compute-unsupported contract is covered by test_compute_xbackend
            vriDestroyDevice(dev);
            return p;
        }

        VriQueue* queue = nullptr;
        REQUIRE(c.GetQueue(dev, VriQueueType_Graphics, 0, &queue) == VriResult_Success);

        constexpr uint32_t kCount = 64; // one 64-wide workgroup -> out[i] = i + 100
        const uint64_t     sbSize = kCount * sizeof(uint32_t);

        VriBufferDesc sbd {};
        sbd.size           = sbSize;
        sbd.usage          = VriBufferUsage_StorageBuffer | VriBufferUsage_TransferSrc;
        sbd.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* storage = nullptr;
        REQUIRE(c.CreateBuffer(dev, &sbd, &storage) == VriResult_Success);
        VriBufferDesc rbd {};
        rbd.size            = sbSize;
        rbd.usage           = VriBufferUsage_TransferDst;
        rbd.memoryLocation  = VriMemoryLocation_HostReadback;
        VriBuffer* readback = nullptr;
        REQUIRE(c.CreateBuffer(dev, &rbd, &readback) == VriResult_Success);

        // indirect args {x,y,z} = {1,1,1}, uploaded to a device-local indirect buffer
        const uint32_t args[3] = {1u, 1u, 1u};
        VriBufferDesc  ibd {};
        ibd.size            = sizeof(args);
        ibd.usage           = VriBufferUsage_IndirectBuffer | VriBufferUsage_TransferDst;
        ibd.memoryLocation  = VriMemoryLocation_Device;
        VriBuffer* indirect = nullptr;
        REQUIRE(c.CreateBuffer(dev, &ibd, &indirect) == VriResult_Success);
        VriBufferDesc asd {};
        asd.size           = sizeof(args);
        asd.usage          = VriBufferUsage_TransferSrc;
        asd.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* argStg  = nullptr;
        REQUIRE(c.CreateBuffer(dev, &asd, &argStg) == VriResult_Success);
        {
            void* m = c.MapBuffer(argStg, 0, sizeof(args));
            REQUIRE(m != nullptr);
            for (size_t i = 0; i < sizeof(args); ++i)
                static_cast<uint8_t*>(m)[i] = reinterpret_cast<const uint8_t*>(args)[i];
            c.UnmapBuffer(argStg);
        }

        VriDescriptorRangeDesc range {};
        range.baseRegister   = 0;
        range.descriptorNum  = 1;
        range.descriptorType = VriDescriptorType_StorageBuffer;
        range.shaderStages   = VriShaderStage_Compute;
        VriDescriptorSetDesc setDesc {};
        setDesc.registerSpace = 0;
        setDesc.ranges        = &range;
        setDesc.rangeNum      = 1;
        VriPipelineLayoutDesc ld {};
        ld.descriptorSets         = &setDesc;
        ld.descriptorSetNum       = 1;
        VriPipelineLayout* layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        VriComputePipelineDesc cpd {};
        cpd.pipelineLayout        = layout;
        cpd.shader.stage          = VriShaderStage_Compute;
        cpd.shader.bytecode       = shader;
        cpd.shader.bytecodeSize   = shaderSize;
        cpd.shader.entryPointName = "computeMain";
        VriPipeline* pipeline     = nullptr;
        REQUIRE(c.CreateComputePipeline(dev, &cpd, &pipeline) == VriResult_Success);

        VriDescriptorPoolDesc pdsc {};
        pdsc.descriptorSetMaxNum = 1;
        pdsc.storageBufferMaxNum = 1;
        VriDescriptorPool* pool  = nullptr;
        REQUIRE(c.CreateDescriptorPool(dev, &pdsc, &pool) == VriResult_Success);
        VriDescriptorSet* set = nullptr;
        REQUIRE(c.AllocateDescriptorSets(pool, layout, 0, &set, 1) == VriResult_Success);
        VriBufferViewDesc bv {};
        bv.buffer             = storage;
        bv.viewType           = VriDescriptorType_StorageBuffer;
        bv.offset             = 0;
        bv.size               = sbSize;
        VriDescriptor* sbView = nullptr;
        REQUIRE(c.CreateBufferView(dev, &bv, &sbView) == VriResult_Success);
        const VriDescriptor*         descs[1] = {sbView};
        VriDescriptorRangeUpdateDesc upd {};
        upd.descriptors    = descs;
        upd.descriptorNum  = 1;
        upd.baseDescriptor = 0;
        c.UpdateDescriptorRanges(set, 0, 1, &upd);

        VriCommandAllocator* alloc = nullptr;
        REQUIRE(c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc) == VriResult_Success);
        VriCommandBuffer* cmd = nullptr;
        REQUIRE(c.CreateCommandBuffer(alloc, &cmd) == VriResult_Success);
        VriFence* fence = nullptr;
        REQUIRE(c.CreateFence(dev, 0, &fence) == VriResult_Success);

        REQUIRE(c.BeginCommandBuffer(cmd) == VriResult_Success);
        // stage the indirect args
        VriBufferCopyDesc acp {};
        acp.size = sizeof(args);
        c.CmdCopyBuffer(cmd, indirect, argStg, &acp);
        {
            VriBufferBarrierDesc bb {};
            bb.buffer        = indirect;
            bb.before.access = VriAccess_CopyDestinationWrite;
            bb.before.stages = VriPipelineStage_Transfer;
            bb.after.access  = VriAccess_IndirectBufferRead;
            bb.after.stages  = VriPipelineStage_DrawIndirect;
            VriBarrierGroupDesc g {};
            g.buffers   = &bb;
            g.bufferNum = 1;
            c.CmdBarrier(cmd, &g);
        }

        c.CmdSetPipeline(cmd, pipeline);
        c.CmdSetPipelineLayout(cmd, layout);
        c.CmdSetDescriptorSet(cmd, 0, set);

        // ---- the operation under contract ----
        const int errBefore = msg.errors;
        c.CmdDispatchIndirect(cmd, indirect, 0);

        {
            VriBufferBarrierDesc bb {};
            bb.buffer        = storage;
            bb.before.access = VriAccess_ShaderResourceStorageWrite;
            bb.before.stages = VriPipelineStage_ComputeShader;
            bb.after.access  = VriAccess_CopySourceRead;
            bb.after.stages  = VriPipelineStage_Transfer;
            VriBarrierGroupDesc g {};
            g.buffers   = &bb;
            g.bufferNum = 1;
            c.CmdBarrier(cmd, &g);
        }
        VriBufferCopyDesc copy {};
        copy.size = sbSize;
        c.CmdCopyBuffer(cmd, readback, storage, &copy);
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
        p.sawDiagnostic = msg.errors > errBefore;

        const uint32_t* px = static_cast<const uint32_t*>(c.MapBuffer(readback, 0, rbd.size));
        REQUIRE(px != nullptr);
        bool allOk = true;
        for (uint32_t i = 0; i < kCount; ++i)
            if (px[i] != i + 100u)
            {
                allOk = false;
                break;
            }
        p.resultCorrect = allOk;
        c.UnmapBuffer(readback);

        c.DeviceWaitIdle(dev);
        c.DestroyFence(fence);
        c.DestroyCommandAllocator(alloc);
        c.DestroyDescriptor(sbView);
        c.DestroyDescriptorPool(pool);
        c.DestroyPipeline(pipeline);
        c.DestroyPipelineLayout(layout);
        c.DestroyBuffer(argStg);
        c.DestroyBuffer(indirect);
        c.DestroyBuffer(readback);
        c.DestroyBuffer(storage);
        vriDestroyDevice(dev);
        return p;
    }

    // Assert the never-silent contract for one backend probe. Universal rule (all
    // tiers): the op either produced the right result OR the backend told us it
    // couldn't (a diagnostic). Additionally, DX12 is a Tier-1 backend targeted at
    // Vulkan parity, so an empty no-op there is a bug, not an allowed subset: it must
    // actually perform the op. That strict check is what fails today for the two
    // known D3D12 silent no-ops and turns green once they are implemented.
    void CheckContract(VriGraphicsAPI api, const ContractProbe& p, const char* op)
    {
        // std::string, not bare const char*: doctest stringifies string literals but
        // prints a const char* variable as its pointer address.
        const std::string tag = std::string(ApiName(api)) + " " + op;
        if (!p.ran)
        {
            const std::string m = tag + " unavailable - skipped";
            MESSAGE(m);
            return;
        }
        if (p.skipped)
        {
            const std::string m = tag + " prerequisite unsupported - skipped";
            MESSAGE(m);
            return;
        }
        INFO(tag << ": resultCorrect=" << p.resultCorrect << " sawDiagnostic=" << p.sawDiagnostic);
        CHECK((p.resultCorrect || p.sawDiagnostic)); // never silent
        if (api == VriGraphicsAPI_D3D12)
            CHECK(p.resultCorrect); // Tier-1: must perform it, not silently no-op
    }
} // namespace

TEST_CASE("never-silent: CmdCopyTexture copies (verified by readback) or diagnoses - never a silent no-op")
{
    CheckContract(VriGraphicsAPI_Vulkan, RunCopyTexture(VriGraphicsAPI_Vulkan), "CmdCopyTexture");
    CheckContract(VriGraphicsAPI_WebGPU, RunCopyTexture(VriGraphicsAPI_WebGPU), "CmdCopyTexture");
    CheckContract(VriGraphicsAPI_OpenGL, RunCopyTexture(VriGraphicsAPI_OpenGL), "CmdCopyTexture");
    CheckContract(VriGraphicsAPI_D3D12, RunCopyTexture(VriGraphicsAPI_D3D12), "CmdCopyTexture");
    CheckContract(VriGraphicsAPI_Metal, RunCopyTexture(VriGraphicsAPI_Metal), "CmdCopyTexture");
}

TEST_CASE("never-silent: CmdDispatchIndirect runs the dispatch (verified by result) or diagnoses")
{
    CheckContract(VriGraphicsAPI_Vulkan,
                  RunDispatchIndirect(VriGraphicsAPI_Vulkan, g_computeFillSpv, sizeof(g_computeFillSpv)),
                  "CmdDispatchIndirect");
    CheckContract(VriGraphicsAPI_WebGPU,
                  RunDispatchIndirect(VriGraphicsAPI_WebGPU, g_computeFillWgsl, sizeof(g_computeFillWgsl)),
                  "CmdDispatchIndirect");
    CheckContract(VriGraphicsAPI_OpenGL,
                  RunDispatchIndirect(VriGraphicsAPI_OpenGL, g_computeFillSpv, sizeof(g_computeFillSpv)),
                  "CmdDispatchIndirect");
    CheckContract(VriGraphicsAPI_D3D12,
                  RunDispatchIndirect(VriGraphicsAPI_D3D12, g_computeFillDxbcCS, sizeof(g_computeFillDxbcCS)),
                  "CmdDispatchIndirect");
}

// ---- interface-registry contract: capability=false => interface not queryable -----
// A device must not silently claim partial support: if it reports an optional
// capability as false, the matching optional interface must be unqueryable
// (vriGetInterface != Success), i.e. an explicit Unsupported rather than a
// half-wired table the app could call into and get wrong results from. Generalizes
// test_omm_d3d12.cpp / test_vrs_mtl.cpp across every available backend.
namespace
{
    void CheckInterfaceContract(VriGraphicsAPI api)
    {
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI         = api;
        dc.enableValidation    = VRI_TRUE;
        dc.bestEffort          = VRI_TRUE;
        VriDevice*        dev  = nullptr;
        const std::string name = ApiName(api);
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            const std::string m = name + " unavailable - skipped";
            MESSAGE(m);
            return;
        }
        struct Guard
        {
            VriDevice* d;
            ~Guard() { vriDestroyDevice(d); }
        } guard {dev};

        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        const VriDeviceDesc* d = c.GetDeviceDesc(dev);

        // Each entry: reported capability, and the optional interface it gates. When
        // the capability is false, the interface must NOT be queryable.
        {
            VriShadingRateInterface vrs {};
            const bool              q = vriGetInterface(dev, VRI_INTERFACE_VRS, sizeof(vrs), &vrs) == VriResult_Success;
            INFO(name << " VRS: hasVariableShadingRate=" << d->hasVariableShadingRate << " queryable=" << q);
            if (d->hasVariableShadingRate == VRI_FALSE)
                CHECK(q == false);
        }
        {
            VriMeshShaderInterface mesh {};
            const bool q = vriGetInterface(dev, VRI_INTERFACE_MESHSHADER, sizeof(mesh), &mesh) == VriResult_Success;
            INFO(name << " MeshShader: hasMeshShader=" << d->hasMeshShader << " queryable=" << q);
            if (d->hasMeshShader == VRI_FALSE)
                CHECK(q == false);
        }
        {
            VriRayTracingInterface rt {};
            const bool q = vriGetInterface(dev, VRI_INTERFACE_RAYTRACING, sizeof(rt), &rt) == VriResult_Success;
            INFO(name << " RayTracing: hasRayTracing=" << d->hasRayTracing << " queryable=" << q);
            if (d->hasRayTracing == VRI_FALSE)
                CHECK(q == false);
        }
    }
} // namespace

TEST_CASE("never-silent: a capability reported false leaves its optional interface unqueryable")
{
    CheckInterfaceContract(VriGraphicsAPI_Vulkan);
    CheckInterfaceContract(VriGraphicsAPI_WebGPU);
    CheckInterfaceContract(VriGraphicsAPI_OpenGL);
    CheckInterfaceContract(VriGraphicsAPI_D3D12);
    CheckInterfaceContract(VriGraphicsAPI_Metal);
}
