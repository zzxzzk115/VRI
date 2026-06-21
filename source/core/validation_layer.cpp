// validation_layer.cpp - VRI Validation (NRI-style), enabled by
// VriDeviceCreationDesc::enableValidation. Like the Vulkan validation layers, it
// sits between the app and the backend: a per-device VriCoreInterface whose
// control-plane functions check preconditions (capability gating, command-buffer
// lifecycle, render-pass scope, null handles) and then forward to the real
// backend. When validation is off, the app gets the raw backend table (zero cost).
//
// Design: only the context-carrying handles are wrapped (Device, CommandAllocator,
// CommandBuffer, Queue, Fence) so the validating functions can find the device +
// real table. Resource handles (Buffer/Texture/Descriptor/Pipeline/...) stay real,
// so descs that embed them (attachments, barriers, vertex bindings) need no
// unwrapping; only QueueSubmit unwraps its wrapped command-buffer/fence arrays.
// Violations are reported via the device MessageCallback and the offending call is
// suppressed (not forwarded) so a misuse can't crash the backend under validation.

#include <vri/vri.h>
#include <vri/ext/vri_ext_swapchain.h>

#include "device_base.h"

#include <cstdio>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace vri::core
{
    namespace
    {
        struct DeviceVal;

        struct QueueVal     { DeviceVal* dev; VriQueue* real; };
        struct AllocatorVal { DeviceVal* dev; VriCommandAllocator* real; };
        struct FenceVal     { DeviceVal* dev; VriFence* real; };
        struct SwapChainVal { DeviceVal* dev; VriSwapChain* real; };
        struct CmdBufVal
        {
            DeviceVal* dev;
            VriCommandBuffer* real;
            bool recording = false;
            bool insideRenderPass = false;
        };

        struct DeviceVal
        {
            VriDevice*           real;     // the backend device handle
            VriCoreInterface     core;     // the backend's real core table
            VriDeviceDesc        desc;     // cached capabilities
            VriCallbackInterface cb;       // app message sink (may be empty)
            bool                 hasCb;
            VriCoreInterface     table;    // the validating table handed to the app
            VriSwapChainInterface swap;    // the backend's real swapchain table (if supported)
            bool                 hasSwap = false;
            VriShadingRateInterface vrs;   // the backend's real VRS table (if supported)
            VriMeshShaderInterface  mesh;  // the backend's real mesh-shader table (if supported)
            // wrappers without an explicit destroy entry point, freed at device teardown
            std::vector<QueueVal*>   queues;
            std::vector<CmdBufVal*>  cmds;
        };

        std::unordered_set<const void*> g_valDevices;
        std::mutex                      g_mutex;

        void Msg(DeviceVal* d, VriMessageSeverity sev, const char* m)
        {
            if (d->hasCb && d->cb.MessageCallback)
                d->cb.MessageCallback(d->cb.userArg, sev, m);
            else
                std::fprintf(stderr, "[VRI/Validation] %s\n", m);
        }
        void Err(DeviceVal* d, const char* m) { Msg(d, VriMessageSeverity_Error, m); }

        inline DeviceVal*    DV(VriDevice* h)           { return reinterpret_cast<DeviceVal*>(h); }
        inline DeviceVal*    DV(const VriDevice* h)      { return reinterpret_cast<DeviceVal*>(const_cast<VriDevice*>(h)); }
        inline AllocatorVal* AV(VriCommandAllocator* h)  { return reinterpret_cast<AllocatorVal*>(h); }
        inline CmdBufVal*    CV(VriCommandBuffer* h)     { return reinterpret_cast<CmdBufVal*>(h); }
        inline QueueVal*     QV(VriQueue* h)             { return reinterpret_cast<QueueVal*>(h); }
        inline FenceVal*     FV(VriFence* h)             { return reinterpret_cast<FenceVal*>(h); }
        inline SwapChainVal* SV(VriSwapChain* h)         { return reinterpret_cast<SwapChainVal*>(h); }

        // ---- queries -------------------------------------------------------
        const VriDeviceDesc* VRI_CALL GetDeviceDesc(const VriDevice* device) { return DV(device)->core.GetDeviceDesc(DV(device)->real); }
        VriFormatSupportFlags VRI_CALL GetFormatSupport(const VriDevice* device, VriFormat f) { return DV(device)->core.GetFormatSupport(DV(device)->real, f); }
        VriResult VRI_CALL GetQueue(VriDevice* device, VriQueueType type, uint32_t index, VriQueue** outQueue)
        {
            DeviceVal* d = DV(device);
            VriQueue* real = nullptr;
            VriResult r = d->core.GetQueue(d->real, type, index, &real);
            if (r != VriResult_Success) return r;
            QueueVal* q = new QueueVal{d, real};
            d->queues.push_back(q);
            *outQueue = reinterpret_cast<VriQueue*>(q);
            return VriResult_Success;
        }

        // ---- command allocation / lifecycle --------------------------------
        VriResult VRI_CALL CreateCommandAllocator(VriDevice* device, VriQueueType type, VriCommandAllocator** out)
        {
            DeviceVal* d = DV(device);
            VriCommandAllocator* real = nullptr;
            VriResult r = d->core.CreateCommandAllocator(d->real, type, &real);
            if (r != VriResult_Success) return r;
            *out = reinterpret_cast<VriCommandAllocator*>(new AllocatorVal{d, real});
            return VriResult_Success;
        }
        void VRI_CALL ResetCommandAllocator(VriCommandAllocator* a) { AV(a)->dev->core.ResetCommandAllocator(AV(a)->real); }
        void VRI_CALL DestroyCommandAllocator(VriCommandAllocator* a) { if (!a) return; AllocatorVal* v = AV(a); v->dev->core.DestroyCommandAllocator(v->real); delete v; }
        VriResult VRI_CALL CreateCommandBuffer(VriCommandAllocator* allocator, VriCommandBuffer** out)
        {
            AllocatorVal* a = AV(allocator);
            VriCommandBuffer* real = nullptr;
            VriResult r = a->dev->core.CreateCommandBuffer(a->real, &real);
            if (r != VriResult_Success) return r;
            CmdBufVal* cv = new CmdBufVal{a->dev, real};
            a->dev->cmds.push_back(cv);
            *out = reinterpret_cast<VriCommandBuffer*>(cv);
            return VriResult_Success;
        }
        VriResult VRI_CALL BeginCommandBuffer(VriCommandBuffer* cmd)
        {
            CmdBufVal* c = CV(cmd);
            if (c->recording) Err(c->dev, "BeginCommandBuffer called on a command buffer already recording");
            c->recording = true; c->insideRenderPass = false;
            return c->dev->core.BeginCommandBuffer(c->real);
        }
        VriResult VRI_CALL EndCommandBuffer(VriCommandBuffer* cmd)
        {
            CmdBufVal* c = CV(cmd);
            if (!c->recording) Err(c->dev, "EndCommandBuffer called on a command buffer that is not recording");
            if (c->insideRenderPass) Err(c->dev, "EndCommandBuffer called inside a render pass (missing CmdEndRendering)");
            c->recording = false;
            return c->dev->core.EndCommandBuffer(c->real);
        }

        // ---- command-buffer scope helpers ----------------------------------
        bool InsideOk(CmdBufVal* c, const char* fn)
        {
            if (!c->recording) { Err(c->dev, fn); return false; }
            if (!c->insideRenderPass) { Err(c->dev, fn); return false; }
            return true;
        }
        bool OutsideOk(CmdBufVal* c, const char* fn)
        {
            if (!c->recording) { Err(c->dev, fn); return false; }
            if (c->insideRenderPass) { Err(c->dev, fn); return false; }
            return true;
        }
        // Shared by graphics (inside a render pass) and compute (outside): only the
        // recording state is required.
        bool RecordingOk(CmdBufVal* c, const char* fn)
        {
            if (!c->recording) { Err(c->dev, fn); return false; }
            return true;
        }

        // ---- resources (device-first wrappers; return real handles) --------
        VriResult VRI_CALL CreateBuffer(VriDevice* device, const VriBufferDesc* d, VriBuffer** out) { return DV(device)->core.CreateBuffer(DV(device)->real, d, out); }
        VriResult VRI_CALL CreateTexture(VriDevice* device, const VriTextureDesc* d, VriTexture** out) { return DV(device)->core.CreateTexture(DV(device)->real, d, out); }
        void VRI_CALL GetBufferMemoryDesc(const VriDevice* device, const VriBufferDesc* d, VriMemoryLocation l, VriMemoryDesc* o) { DV(device)->core.GetBufferMemoryDesc(DV(device)->real, d, l, o); }
        void VRI_CALL GetTextureMemoryDesc(const VriDevice* device, const VriTextureDesc* d, VriMemoryLocation l, VriMemoryDesc* o) { DV(device)->core.GetTextureMemoryDesc(DV(device)->real, d, l, o); }
        VriResult VRI_CALL AllocateMemory(VriDevice* device, const VriMemoryDesc* d, VriMemory** out) { return DV(device)->core.AllocateMemory(DV(device)->real, d, out); }
        VriResult VRI_CALL BindBufferMemory(VriDevice* device, VriBuffer* b, VriMemory* m, uint64_t off) { return DV(device)->core.BindBufferMemory(DV(device)->real, b, m, off); }
        VriResult VRI_CALL BindTextureMemory(VriDevice* device, VriTexture* t, VriMemory* m, uint64_t off) { return DV(device)->core.BindTextureMemory(DV(device)->real, t, m, off); }
        VriResult VRI_CALL CreateBufferView(VriDevice* device, const VriBufferViewDesc* d, VriDescriptor** out) { return DV(device)->core.CreateBufferView(DV(device)->real, d, out); }
        VriResult VRI_CALL CreateTextureView(VriDevice* device, const VriTextureViewDesc* d, VriDescriptor** out) { return DV(device)->core.CreateTextureView(DV(device)->real, d, out); }
        VriResult VRI_CALL CreateSampler(VriDevice* device, const VriSamplerDesc* d, VriDescriptor** out) { return DV(device)->core.CreateSampler(DV(device)->real, d, out); }
        VriResult VRI_CALL CreatePipelineLayout(VriDevice* device, const VriPipelineLayoutDesc* d, VriPipelineLayout** out) { return DV(device)->core.CreatePipelineLayout(DV(device)->real, d, out); }
        VriResult VRI_CALL CreateGraphicsPipeline(VriDevice* device, const VriGraphicsPipelineDesc* d, VriPipeline** out)
        {
            DeviceVal* dv = DV(device);
            for (uint32_t i = 0; i < d->shaderNum; ++i)
            {
                const VriShaderStageBits st = d->shaders[i].stage;
                if (st == VriShaderStage_Geometry && !dv->desc.hasGeometryShader)
                {
                    Err(dv, "CreateGraphicsPipeline uses a geometry shader but device.hasGeometryShader is false (no geometry stage on this backend, e.g. WebGPU/WebGL2)");
                    return VriResult_Unsupported;
                }
                if ((st == VriShaderStage_TessControl || st == VriShaderStage_TessEval) && !dv->desc.hasTessellation)
                {
                    Err(dv, "CreateGraphicsPipeline uses a tessellation shader but device.hasTessellation is false (no tessellation stage on this backend, e.g. WebGPU/WebGL2)");
                    return VriResult_Unsupported;
                }
            }
            return dv->core.CreateGraphicsPipeline(dv->real, d, out);
        }
        VriResult VRI_CALL CreateComputePipeline(VriDevice* device, const VriComputePipelineDesc* d, VriPipeline** out)
        {
            DeviceVal* dv = DV(device);
            if (!dv->desc.hasComputeShader)
            {
                Err(dv, "CreateComputePipeline called but device.hasComputeShader is false (compute unsupported on this backend, e.g. WebGL2)");
                return VriResult_Unsupported;
            }
            return dv->core.CreateComputePipeline(dv->real, d, out);
        }
        VriResult VRI_CALL CreateDescriptorPool(VriDevice* device, const VriDescriptorPoolDesc* d, VriDescriptorPool** out) { return DV(device)->core.CreateDescriptorPool(DV(device)->real, d, out); }
        VriResult VRI_CALL CreateFence(VriDevice* device, uint64_t initial, VriFence** out)
        {
            DeviceVal* d = DV(device);
            VriFence* real = nullptr;
            VriResult r = d->core.CreateFence(d->real, initial, &real);
            if (r != VriResult_Success) return r;
            *out = reinterpret_cast<VriFence*>(new FenceVal{d, real});
            return VriResult_Success;
        }
        void VRI_CALL DestroyFence(VriFence* fence) { if (!fence) return; FenceVal* v = FV(fence); v->dev->core.DestroyFence(v->real); delete v; }
        uint64_t VRI_CALL GetFenceValue(VriFence* fence) { return FV(fence)->dev->core.GetFenceValue(FV(fence)->real); }
        void VRI_CALL Wait(VriFence* fence, uint64_t value) { FV(fence)->dev->core.Wait(FV(fence)->real, value); }
        void VRI_CALL DeviceWaitIdle(VriDevice* device) { DV(device)->core.DeviceWaitIdle(DV(device)->real); }

        // ---- command recording (scope-checked, forward to real) ------------
        void VRI_CALL CmdBeginRendering(VriCommandBuffer* cmd, const VriAttachmentsDesc* a)
        {
            CmdBufVal* c = CV(cmd);
            if (!OutsideOk(c, "CmdBeginRendering called outside command recording or inside an active render pass")) return;
            c->insideRenderPass = true;
            c->dev->core.CmdBeginRendering(c->real, a);
        }
        void VRI_CALL CmdEndRendering(VriCommandBuffer* cmd)
        {
            CmdBufVal* c = CV(cmd);
            if (!c->insideRenderPass) { Err(c->dev, "CmdEndRendering called without an active render pass"); return; }
            c->insideRenderPass = false;
            c->dev->core.CmdEndRendering(c->real);
        }
        void VRI_CALL CmdSetViewports(VriCommandBuffer* cmd, const VriViewport* v, uint32_t n) { CmdBufVal* c = CV(cmd); if (!InsideOk(c, "CmdSetViewports outside a render pass")) return; c->dev->core.CmdSetViewports(c->real, v, n); }
        void VRI_CALL CmdSetScissors(VriCommandBuffer* cmd, const VriRect* r, uint32_t n) { CmdBufVal* c = CV(cmd); if (!InsideOk(c, "CmdSetScissors outside a render pass")) return; c->dev->core.CmdSetScissors(c->real, r, n); }
        void VRI_CALL CmdSetPipelineLayout(VriCommandBuffer* cmd, VriPipelineLayout* l) { CmdBufVal* c = CV(cmd); if (!RecordingOk(c, "CmdSetPipelineLayout outside command recording")) return; c->dev->core.CmdSetPipelineLayout(c->real, l); }
        void VRI_CALL CmdSetPipeline(VriCommandBuffer* cmd, VriPipeline* p) { CmdBufVal* c = CV(cmd); if (!RecordingOk(c, "CmdSetPipeline outside command recording")) return; c->dev->core.CmdSetPipeline(c->real, p); }
        void VRI_CALL CmdSetDescriptorSet(VriCommandBuffer* cmd, uint32_t i, const VriDescriptorSet* s) { CmdBufVal* c = CV(cmd); if (!RecordingOk(c, "CmdSetDescriptorSet outside command recording")) return; c->dev->core.CmdSetDescriptorSet(c->real, i, s); }
        void VRI_CALL CmdSetConstants(VriCommandBuffer* cmd, uint32_t i, const void* data, uint32_t size) { CmdBufVal* c = CV(cmd); if (!RecordingOk(c, "CmdSetConstants outside command recording")) return; c->dev->core.CmdSetConstants(c->real, i, data, size); }
        void VRI_CALL CmdSetVertexBuffers(VriCommandBuffer* cmd, uint32_t s, const VriVertexBufferBinding* b, uint32_t n) { CmdBufVal* c = CV(cmd); if (!InsideOk(c, "CmdSetVertexBuffers outside a render pass")) return; c->dev->core.CmdSetVertexBuffers(c->real, s, b, n); }
        void VRI_CALL CmdSetIndexBuffer(VriCommandBuffer* cmd, VriBuffer* b, uint64_t off, VriIndexType t) { CmdBufVal* c = CV(cmd); if (!InsideOk(c, "CmdSetIndexBuffer outside a render pass")) return; c->dev->core.CmdSetIndexBuffer(c->real, b, off, t); }
        void VRI_CALL CmdDraw(VriCommandBuffer* cmd, const VriDrawDesc* d) { CmdBufVal* c = CV(cmd); if (!InsideOk(c, "CmdDraw outside a render pass")) return; c->dev->core.CmdDraw(c->real, d); }
        void VRI_CALL CmdDrawIndexed(VriCommandBuffer* cmd, const VriDrawIndexedDesc* d) { CmdBufVal* c = CV(cmd); if (!InsideOk(c, "CmdDrawIndexed outside a render pass")) return; c->dev->core.CmdDrawIndexed(c->real, d); }
        void VRI_CALL CmdDrawIndirect(VriCommandBuffer* cmd, VriBuffer* b, uint64_t off, uint32_t n, uint32_t s) { CmdBufVal* c = CV(cmd); if (!InsideOk(c, "CmdDrawIndirect outside a render pass")) return; c->dev->core.CmdDrawIndirect(c->real, b, off, n, s); }
        void VRI_CALL CmdDispatch(VriCommandBuffer* cmd, const VriDispatchDesc* d) { CmdBufVal* c = CV(cmd); if (!OutsideOk(c, "CmdDispatch must be outside a render pass")) return; c->dev->core.CmdDispatch(c->real, d); }
        void VRI_CALL CmdDispatchIndirect(VriCommandBuffer* cmd, VriBuffer* b, uint64_t off) { CmdBufVal* c = CV(cmd); if (!OutsideOk(c, "CmdDispatchIndirect must be outside a render pass")) return; c->dev->core.CmdDispatchIndirect(c->real, b, off); }
        void VRI_CALL CmdBarrier(VriCommandBuffer* cmd, const VriBarrierGroupDesc* g) { CmdBufVal* c = CV(cmd); if (!OutsideOk(c, "CmdBarrier must be outside a render pass")) return; c->dev->core.CmdBarrier(c->real, g); }
        void VRI_CALL CmdCopyBuffer(VriCommandBuffer* cmd, VriBuffer* dst, VriBuffer* src, const VriBufferCopyDesc* r) { CmdBufVal* c = CV(cmd); if (!OutsideOk(c, "CmdCopyBuffer must be outside a render pass")) return; c->dev->core.CmdCopyBuffer(c->real, dst, src, r); }
        void VRI_CALL CmdCopyTexture(VriCommandBuffer* cmd, VriTexture* dst, VriTexture* src, const VriTextureCopyDesc* r) { CmdBufVal* c = CV(cmd); if (!OutsideOk(c, "CmdCopyTexture must be outside a render pass")) return; c->dev->core.CmdCopyTexture(c->real, dst, src, r); }
        void VRI_CALL CmdUploadBufferToTexture(VriCommandBuffer* cmd, VriTexture* dst, VriBuffer* src, const VriBufferTextureCopyDesc* r) { CmdBufVal* c = CV(cmd); if (!OutsideOk(c, "CmdUploadBufferToTexture must be outside a render pass")) return; c->dev->core.CmdUploadBufferToTexture(c->real, dst, src, r); }
        void VRI_CALL CmdReadbackTextureToBuffer(VriCommandBuffer* cmd, VriBuffer* dst, VriTexture* src, const VriBufferTextureCopyDesc* r) { CmdBufVal* c = CV(cmd); if (!OutsideOk(c, "CmdReadbackTextureToBuffer must be outside a render pass")) return; c->dev->core.CmdReadbackTextureToBuffer(c->real, dst, src, r); }
        void VRI_CALL CmdBeginDebugGroup(VriCommandBuffer* cmd, const char* n) { CmdBufVal* c = CV(cmd); c->dev->core.CmdBeginDebugGroup(c->real, n); }
        void VRI_CALL CmdEndDebugGroup(VriCommandBuffer* cmd) { CmdBufVal* c = CV(cmd); c->dev->core.CmdEndDebugGroup(c->real); }

        // ---- submission (unwrap command buffers + fences) ------------------
        void VRI_CALL QueueSubmit(VriQueue* queue, const VriQueueSubmitDesc* submit)
        {
            QueueVal* q = QV(queue);
            DeviceVal* d = q->dev;
            std::vector<VriCommandBuffer*> cmds(submit->commandBufferNum);
            for (uint32_t i = 0; i < submit->commandBufferNum; ++i)
                cmds[i] = CV(submit->commandBuffers[i])->real;
            std::vector<VriFenceSubmitDesc> waits(submit->waitFenceNum);
            for (uint32_t i = 0; i < submit->waitFenceNum; ++i) { waits[i] = submit->waitFences[i]; waits[i].fence = FV(submit->waitFences[i].fence)->real; }
            std::vector<VriFenceSubmitDesc> signals(submit->signalFenceNum);
            for (uint32_t i = 0; i < submit->signalFenceNum; ++i) { signals[i] = submit->signalFences[i]; signals[i].fence = FV(submit->signalFences[i].fence)->real; }

            VriQueueSubmitDesc real = *submit;
            real.commandBuffers = cmds.data();
            real.waitFences = waits.empty() ? nullptr : waits.data();
            real.signalFences = signals.empty() ? nullptr : signals.data();
            d->core.QueueSubmit(q->real, &real);
        }
        void VRI_CALL QueueWaitIdle(VriQueue* queue) { QV(queue)->dev->core.QueueWaitIdle(QV(queue)->real); }

        // ---- swapchain (unwrap device/queue/fences; wrap the swapchain handle) ----
        // The backbuffer textures returned by GetSwapChainTextures are real backend
        // handles - the app feeds them back through the (forwarding) core wrappers, so
        // they need no wrapping here.
        VriResult VRI_CALL CreateSwapChain(VriDevice* device, const VriSwapChainDesc* desc, VriSwapChain** out)
        {
            DeviceVal* d = DV(device);
            VriSwapChainDesc real = *desc;
            real.queue = desc->queue ? QV(desc->queue)->real : nullptr; // unwrap the present queue
            VriSwapChain* rsc = nullptr;
            const VriResult r = d->swap.CreateSwapChain(d->real, &real, &rsc);
            if (r != VriResult_Success) return r;
            *out = reinterpret_cast<VriSwapChain*>(new SwapChainVal{d, rsc});
            return VriResult_Success;
        }
        void VRI_CALL DestroySwapChain(VriSwapChain* sc)
        {
            if (!sc) return;
            SwapChainVal* v = SV(sc);
            v->dev->swap.DestroySwapChain(v->real);
            delete v;
        }
        VriResult VRI_CALL GetSwapChainTextures(VriSwapChain* sc, VriTexture** out, uint32_t* io) { return SV(sc)->dev->swap.GetSwapChainTextures(SV(sc)->real, out, io); }
        VriResult VRI_CALL AcquireNextTexture(VriSwapChain* sc, VriFence* fence, uint64_t value, uint32_t* outIndex)
        {
            SwapChainVal* v = SV(sc);
            return v->dev->swap.AcquireNextTexture(v->real, fence ? FV(fence)->real : nullptr, value, outIndex);
        }
        VriResult VRI_CALL Present(VriSwapChain* sc, VriFence* fence, uint64_t value)
        {
            SwapChainVal* v = SV(sc);
            return v->dev->swap.Present(v->real, fence ? FV(fence)->real : nullptr, value);
        }
        VriResult VRI_CALL Resize(VriSwapChain* sc, uint32_t w, uint32_t h) { return SV(sc)->dev->swap.Resize(SV(sc)->real, w, h); }

        // ---- extension interfaces taking a wrapped command buffer ----------
        // These forward to the backend's real table after unwrapping cmd->real.
        // (Buffers/textures/pipelines are not wrapped by validation, so they pass
        // through untouched.)
        void VRI_CALL CmdSetShadingRate(VriCommandBuffer* cmd, const VriShadingRateDesc* desc)
        {
            CmdBufVal* c = CV(cmd);
            if (!RecordingOk(c, "CmdSetShadingRate")) return;
            c->dev->vrs.CmdSetShadingRate(c->real, desc);
        }
        void VRI_CALL CmdDrawMeshTasks(VriCommandBuffer* cmd, uint32_t x, uint32_t y, uint32_t z)
        {
            CmdBufVal* c = CV(cmd);
            if (!RecordingOk(c, "CmdDrawMeshTasks")) return;
            c->dev->mesh.CmdDrawMeshTasks(c->real, x, y, z);
        }
        void VRI_CALL CmdDrawMeshTasksIndirect(VriCommandBuffer* cmd, VriBuffer* buffer, uint64_t offset, uint32_t drawNum, uint32_t stride)
        {
            CmdBufVal* c = CV(cmd);
            if (!RecordingOk(c, "CmdDrawMeshTasksIndirect")) return;
            c->dev->mesh.CmdDrawMeshTasksIndirect(c->real, buffer, offset, drawNum, stride);
        }

        void BuildTable(DeviceVal* d)
        {
            VriCoreInterface t = d->core; // start from the real table (resource plane passes through)
            // control plane -> validating wrappers
            t.GetDeviceDesc = GetDeviceDesc; t.GetFormatSupport = GetFormatSupport; t.GetQueue = GetQueue;
            t.CreateCommandAllocator = CreateCommandAllocator; t.ResetCommandAllocator = ResetCommandAllocator; t.DestroyCommandAllocator = DestroyCommandAllocator;
            t.CreateCommandBuffer = CreateCommandBuffer; t.BeginCommandBuffer = BeginCommandBuffer; t.EndCommandBuffer = EndCommandBuffer;
            t.CreateBuffer = CreateBuffer; t.CreateTexture = CreateTexture;
            t.GetBufferMemoryDesc = GetBufferMemoryDesc; t.GetTextureMemoryDesc = GetTextureMemoryDesc; t.AllocateMemory = AllocateMemory; t.BindBufferMemory = BindBufferMemory; t.BindTextureMemory = BindTextureMemory;
            t.CreateBufferView = CreateBufferView; t.CreateTextureView = CreateTextureView; t.CreateSampler = CreateSampler;
            t.CreatePipelineLayout = CreatePipelineLayout; t.CreateGraphicsPipeline = CreateGraphicsPipeline; t.CreateComputePipeline = CreateComputePipeline;
            t.CreateDescriptorPool = CreateDescriptorPool;
            t.CreateFence = CreateFence; t.DestroyFence = DestroyFence; t.GetFenceValue = GetFenceValue; t.Wait = Wait;
            t.CmdBeginRendering = CmdBeginRendering; t.CmdEndRendering = CmdEndRendering; t.CmdSetViewports = CmdSetViewports; t.CmdSetScissors = CmdSetScissors;
            t.CmdSetPipelineLayout = CmdSetPipelineLayout; t.CmdSetPipeline = CmdSetPipeline; t.CmdSetDescriptorSet = CmdSetDescriptorSet; t.CmdSetConstants = CmdSetConstants;
            t.CmdSetVertexBuffers = CmdSetVertexBuffers; t.CmdSetIndexBuffer = CmdSetIndexBuffer;
            t.CmdDraw = CmdDraw; t.CmdDrawIndexed = CmdDrawIndexed; t.CmdDrawIndirect = CmdDrawIndirect; t.CmdDispatch = CmdDispatch; t.CmdDispatchIndirect = CmdDispatchIndirect;
            t.CmdBarrier = CmdBarrier; t.CmdCopyBuffer = CmdCopyBuffer; t.CmdCopyTexture = CmdCopyTexture; t.CmdUploadBufferToTexture = CmdUploadBufferToTexture; t.CmdReadbackTextureToBuffer = CmdReadbackTextureToBuffer;
            t.CmdBeginDebugGroup = CmdBeginDebugGroup; t.CmdEndDebugGroup = CmdEndDebugGroup;
            t.QueueSubmit = QueueSubmit; t.QueueWaitIdle = QueueWaitIdle; t.DeviceWaitIdle = DeviceWaitIdle;
            // DestroyBuffer/MapBuffer/UnmapBuffer/GetBufferDeviceAddress/DestroyTexture/FreeMemory/
            // DestroyDescriptor/DestroyPipelineLayout/DestroyPipeline/ResetDescriptorPool/
            // DestroyDescriptorPool/AllocateDescriptorSets/UpdateDescriptorRanges/SetDebugName:
            // resource plane, pass straight through (handles are real).
            d->table = t;
        }
    } // namespace

    // ---- entry points used by vri_entry.cpp --------------------------------
    VriDevice* WrapValidationDevice(VriDevice* realDevice, const VriDeviceCreationDesc& desc)
    {
        DeviceVal* d = new DeviceVal{};
        d->real = realDevice;
        if (reinterpret_cast<DeviceBase*>(realDevice)->GetInterface(VRI_INTERFACE_CORE, sizeof(d->core), &d->core) != VriResult_Success)
        {
            delete d;
            return realDevice; // can't validate without the core table; fall back
        }
        d->desc = *d->core.GetDeviceDesc(realDevice);
        d->hasCb = desc.callbackInterface != nullptr;
        if (d->hasCb) d->cb = *desc.callbackInterface;
        BuildTable(d);
        {
            std::lock_guard<std::mutex> lk(g_mutex);
            g_valDevices.insert(d);
        }
        return reinterpret_cast<VriDevice*>(d);
    }

    bool IsValidationDevice(const VriDevice* device)
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        return g_valDevices.find(device) != g_valDevices.end();
    }

    VriResult ValidationGetInterface(const VriDevice* device, const char* name, size_t size, void* out)
    {
        DeviceVal* d = DV(device);
        auto nameIs = [&](const char* want) {
            for (const char* p = name, *q = want; ; ++p, ++q) { if (*p != *q) return false; if (*p == '\0') return true; }
        };
        // Core + swapchain are validated (they take wrapped handles that must be
        // unwrapped before reaching the backend). Every other interface forwards to the
        // real device untouched.
        if (nameIs(VRI_INTERFACE_CORE))
        {
            if (size != sizeof(VriCoreInterface)) return VriResult_InvalidArgument;
            *static_cast<VriCoreInterface*>(out) = d->table;
            return VriResult_Success;
        }
        if (nameIs(VRI_INTERFACE_SWAPCHAIN))
        {
            if (size != sizeof(VriSwapChainInterface)) return VriResult_InvalidArgument;
            // Cache the backend's real swapchain table, then hand back a wrapper that
            // unwraps the device/queue/fences (the app holds wrapped handles).
            const VriResult r = reinterpret_cast<DeviceBase*>(d->real)->GetInterface(VRI_INTERFACE_SWAPCHAIN, sizeof(d->swap), &d->swap);
            if (r != VriResult_Success) return r;
            d->hasSwap = true;
            static const VriSwapChainInterface wrapped = {
                CreateSwapChain, DestroySwapChain, GetSwapChainTextures, AcquireNextTexture, Present, Resize,
            };
            *static_cast<VriSwapChainInterface*>(out) = wrapped;
            return VriResult_Success;
        }
        if (nameIs(VRI_INTERFACE_VRS))
        {
            if (size != sizeof(VriShadingRateInterface)) return VriResult_InvalidArgument;
            const VriResult r = reinterpret_cast<DeviceBase*>(d->real)->GetInterface(VRI_INTERFACE_VRS, sizeof(d->vrs), &d->vrs);
            if (r != VriResult_Success) return r;
            static const VriShadingRateInterface wrapped = { CmdSetShadingRate };
            *static_cast<VriShadingRateInterface*>(out) = wrapped;
            return VriResult_Success;
        }
        if (nameIs(VRI_INTERFACE_MESHSHADER))
        {
            if (size != sizeof(VriMeshShaderInterface)) return VriResult_InvalidArgument;
            const VriResult r = reinterpret_cast<DeviceBase*>(d->real)->GetInterface(VRI_INTERFACE_MESHSHADER, sizeof(d->mesh), &d->mesh);
            if (r != VriResult_Success) return r;
            static const VriMeshShaderInterface wrapped = { CmdDrawMeshTasks, CmdDrawMeshTasksIndirect };
            *static_cast<VriMeshShaderInterface*>(out) = wrapped;
            return VriResult_Success;
        }
        return reinterpret_cast<DeviceBase*>(d->real)->GetInterface(name, size, out);
    }

    void DestroyValidationDevice(VriDevice* device)
    {
        DeviceVal* d = DV(device);
        {
            std::lock_guard<std::mutex> lk(g_mutex);
            g_valDevices.erase(d);
        }
        delete reinterpret_cast<DeviceBase*>(d->real); // destroy the backend device
        for (QueueVal* q : d->queues) delete q;
        for (CmdBufVal* c : d->cmds) delete c;
        delete d;
    }
} // namespace vri::core
