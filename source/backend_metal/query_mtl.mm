// Metal GPU execution ranges sample real encoder stages. Arbitrary bottom-of-pipe
// timestamp points cannot be represented by an unrelated auxiliary blit encoder.
#include "query_mtl.h"
#include "device_mtl.h"
#include "objects_mtl.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>

namespace vri::mtl
{
    namespace
    {
        constexpr uint32_t kSampleCapacity = 1024;
        constexpr uint64_t kInvalidTimestamp = UINT64_MAX;
        inline CommandBufferMTL* CB(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferMTL*>(h); }
        inline BufferMTL* Buf(VriBuffer* h) { return reinterpret_cast<BufferMTL*>(h); }
        inline QueryPoolMTL* QP(VriQueryPool* h) { return reinterpret_cast<QueryPoolMTL*>(h); }
        inline DeviceMTL* Dev(VriDevice* h) { return reinterpret_cast<DeviceMTL*>(h); }

        id<MTLCounterSet> TimestampCounterSet(id<MTLDevice> device)
        {
            for (id<MTLCounterSet> set in [device counterSets])
                if ([[set name] isEqualToString:MTLCommonCounterSetTimestamp]) return set;
            return nil;
        }

        void Fail(CommandBufferMTL* c, const char* message)
        {
            c->timestampsValid = false;
            c->device->ReportError(message);
        }

        std::optional<TimestampSampleMTL> NewSamples(CommandBufferMTL* c, uint32_t count)
        {
            if (c->timestampDepth == 0 || !c->timestampsValid) return std::nullopt;
            if (c->timestampChunk < c->timestampChunks.size() &&
                c->timestampChunks[c->timestampChunk].used + count > kSampleCapacity) ++c->timestampChunk;
            if (c->timestampChunk == c->timestampChunks.size())
            {
                TimestampChunkMTL chunk;
                chunk.resolved.resize(kSampleCapacity);
                MTLCounterSampleBufferDescriptor* desc = [[MTLCounterSampleBufferDescriptor alloc] init];
                desc.counterSet = TimestampCounterSet(c->device->Device());
                desc.storageMode = MTLStorageModeShared;
                desc.sampleCount = kSampleCapacity;
                NSError* error = nil;
                chunk.buffer = [c->device->Device() newCounterSampleBufferWithDescriptor:desc error:&error];
                [desc release];
                if (!chunk.buffer)
                {
                    Fail(c, "Create Metal execution-range counter buffer failed");
                    return std::nullopt;
                }
                c->timestampChunks.push_back(std::move(chunk));
            }
            auto& chunk = c->timestampChunks[c->timestampChunk];
            TimestampSampleMTL sample {c->timestampChunk, chunk.used, count};
            chunk.used += count;
            c->timestampSamples.push_back(sample);
            return sample;
        }

        template<class Descriptor>
        void AttachEncoderSamples(CommandBufferMTL* c, Descriptor* desc)
        {
            if (auto sample = NewSamples(c, 2))
            {
                auto* attachment = desc.sampleBufferAttachments[0];
                attachment.sampleBuffer = c->timestampChunks[sample->chunk].buffer;
                attachment.startOfEncoderSampleIndex = sample->first;
                attachment.endOfEncoderSampleIndex = sample->first + 1;
            }
        }

        // A scope boundary must coincide with an encoder boundary. Ending an existing aux
        // encoder creates no extra GPU work; the next real command opens its normal encoder.
        bool CloseAux(CommandBufferMTL* c)
        {
            if (c->renderEnc)
            {
                Fail(c, "Metal TimestampRange boundaries must be outside a render pass");
                return false;
            }
            if (c->blitEnc) { [c->blitEnc endEncoding]; [c->blitEnc release]; c->blitEnc = nil; }
            if (c->computeEnc) { [c->computeEnc endEncoding]; [c->computeEnc release]; c->computeEnc = nil; }
            return true;
        }

        VriResult VRI_CALL CreateQueryPool(VriDevice* device, const VriQueryPoolDesc* desc, VriQueryPool** out)
        {
            if (!desc || !out || !desc->queryCount) return VriResult_InvalidArgument;
            if (desc->type != VriQueryType_Occlusion && desc->type != VriQueryType_TimestampRange)
                return VriResult_Unsupported;
            DeviceMTL* d = Dev(device);
            if (desc->type == VriQueryType_TimestampRange && !d->Desc().hasTimestampRangeQueries)
                return VriResult_Unsupported;
            QueryPoolMTL* q = new QueryPoolMTL{};
            q->device = d; q->type = desc->type; q->count = desc->queryCount;
            if (desc->type == VriQueryType_TimestampRange) q->ranges.resize(desc->queryCount);
            else
            {
                q->results = [d->Device() newBufferWithLength:uint64_t(desc->queryCount) * sizeof(uint64_t)
                                                     options:MTLResourceStorageModeShared];
                if (!q->results) { delete q; return VriResult_OutOfMemory; }
            }
            *out = reinterpret_cast<VriQueryPool*>(q);
            return VriResult_Success;
        }

        void VRI_CALL DestroyQueryPool(VriQueryPool* pool)
        {
            if (!pool) return;
            auto* q = QP(pool);
            if (q->results) [q->results release];
            delete q;
        }
        uint32_t VRI_CALL GetQuerySize(const VriQueryPool* pool)
        {
            return reinterpret_cast<const QueryPoolMTL*>(pool)->type == VriQueryType_TimestampRange ?
                sizeof(VriTimestampRange) : sizeof(uint64_t);
        }
        void VRI_CALL CmdResetQueries(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t offset, uint32_t num)
        {
            auto* q = QP(pool);
            if (offset > q->count || num > q->count - offset) { Fail(CB(cmd), "Metal query reset exceeds pool"); return; }
            if (q->type == VriQueryType_TimestampRange)
                std::fill(q->ranges.begin() + offset, q->ranges.begin() + offset + num, TimestampRangeMTL{});
        }
        void VRI_CALL CmdWriteTimestamp(VriCommandBuffer* cmd, VriQueryPool*, uint32_t)
        {
            Fail(CB(cmd), "Metal has encoder-stage counters; use TimestampRange instead of timestamp points");
        }
        void VRI_CALL CmdBeginQuery(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index)
        {
            auto* c = CB(cmd); auto* q = QP(pool);
            if (index >= q->count) { Fail(c, "Metal query index exceeds pool"); return; }
            if (q->type == VriQueryType_TimestampRange)
            {
                if (!CloseAux(c)) return;
                q->ranges[index] = {c, uint32_t(c->timestampSamples.size()), 0, false};
                ++c->timestampDepth;
                return;
            }
            if (!q->results || !c->renderEnc) return;
            if (c->visNextSlot >= kMtlMaxOcclusionSlots) { c->device->ReportError("Metal occlusion slots exceeded"); return; }
            const uint32_t slot = c->visNextSlot++;
            c->occMarks.push_back({q, index, slot});
            [c->renderEnc setVisibilityResultMode:MTLVisibilityResultModeCounting offset:slot * sizeof(uint64_t)];
        }
        void VRI_CALL CmdEndQuery(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t index)
        {
            auto* c = CB(cmd); auto* q = QP(pool);
            if (index >= q->count) { Fail(c, "Metal query index exceeds pool"); return; }
            if (q->type == VriQueryType_TimestampRange)
            {
                auto& range = q->ranges[index];
                if (range.commands != c || range.closed || !c->timestampDepth)
                { Fail(c, "Unmatched Metal TimestampRange end"); return; }
                if (!CloseAux(c)) return;
                range.end = uint32_t(c->timestampSamples.size()); range.closed = true;
                --c->timestampDepth;
                return;
            }
            if (q->results && c->renderEnc) [c->renderEnc setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
        }
        void VRI_CALL CmdCopyQueries(VriCommandBuffer* cmd, VriQueryPool* pool, uint32_t offset, uint32_t num,
                                     VriBuffer* destination, uint64_t destinationOffset)
        {
            @autoreleasepool
            {
                auto* c = CB(cmd); auto* q = QP(pool); auto* buffer = Buf(destination);
                if (offset > q->count || num > q->count - offset) { Fail(c, "Metal query copy exceeds pool"); return; }
                const uint64_t stride = GetQuerySize(pool);
                if (destinationOffset > buffer->size || uint64_t(num) * stride > buffer->size - destinationOffset)
                { Fail(c, "Metal query copy exceeds destination"); return; }
                if (q->type == VriQueryType_TimestampRange)
                {
                    auto* data = static_cast<char*>([buffer->buffer contents]);
                    if (!data || destinationOffset % alignof(VriTimestampRange))
                    { Fail(c, "Metal TimestampRange needs an aligned HostReadback destination"); return; }
                    for (uint32_t i = 0; i < num; ++i)
                    {
                        const auto& range = q->ranges[offset + i];
                        auto* dst = reinterpret_cast<VriTimestampRange*>(data + destinationOffset + i * stride);
                        if (range.commands != c || !range.closed)
                        { Fail(c, "Copy of incomplete Metal TimestampRange"); *dst = {kInvalidTimestamp, kInvalidTimestamp, kInvalidTimestamp}; continue; }
                        c->timestampResolves.push_back({range.first, range.end, dst});
                    }
                    return;
                }
                if (!q->results || !c->visBuffer) return;
                if (c->computeEnc) { [c->computeEnc endEncoding]; [c->computeEnc release]; c->computeEnc = nil; }
                if (!c->blitEnc)
                {
                    MTLBlitPassDescriptor* desc = [MTLBlitPassDescriptor blitPassDescriptor];
                    AttachTimestampSamples(c, desc);
                    c->blitEnc = [[c->cmd blitCommandEncoderWithDescriptor:desc] retain];
                }
                for (uint32_t k = 0; k < num; ++k)
                    for (const auto& mark : c->occMarks)
                        if (mark.pool == q && mark.index == offset + k)
                        {
                            [c->blitEnc copyFromBuffer:c->visBuffer sourceOffset:mark.visSlot * sizeof(uint64_t)
                                             toBuffer:buffer->buffer destinationOffset:destinationOffset + k * sizeof(uint64_t)
                                                 size:sizeof(uint64_t)];
                            break;
                        }
            }
        }
        VriResult VRI_CALL GetCalibratedTimestamps(VriDevice* device, VriCalibratedTimestamps* out)
        {
            if (!out) return VriResult_InvalidArgument;
            MTLTimestamp cpu = 0, gpu = 0;
            [Dev(device)->Device() sampleTimestamps:&cpu gpuTimestamp:&gpu];
            out->gpuTimestamp = gpu; out->cpuTimestamp = cpu;
            return VriResult_Success;
        }
        const VriQueryInterface g_queryMTL = {CreateQueryPool, DestroyQueryPool, GetQuerySize, CmdResetQueries,
            CmdWriteTimestamp, CmdBeginQuery, CmdEndQuery, CmdCopyQueries, GetCalibratedTimestamps};
    }

    void AttachTimestampSamples(CommandBufferMTL* c, MTLRenderPassDescriptor* desc)
    {
        if (auto sample = NewSamples(c, 4))
        {
            auto* attachment = desc.sampleBufferAttachments[0];
            attachment.sampleBuffer = c->timestampChunks[sample->chunk].buffer;
            attachment.startOfVertexSampleIndex = sample->first;
            attachment.endOfVertexSampleIndex = sample->first + 1;
            attachment.startOfFragmentSampleIndex = sample->first + 2;
            attachment.endOfFragmentSampleIndex = sample->first + 3;
        }
    }
    void AttachTimestampSamples(CommandBufferMTL* c, MTLComputePassDescriptor* d) { AttachEncoderSamples(c, d); }
    void AttachTimestampSamples(CommandBufferMTL* c, MTLBlitPassDescriptor* d) { AttachEncoderSamples(c, d); }
    void AttachTimestampSamples(CommandBufferMTL* c, MTLAccelerationStructurePassDescriptor* d) { AttachEncoderSamples(c, d); }

    void ResolveTimestampRanges(CommandBufferMTL* c, id<MTLCommandBuffer> native)
    {
        // Counter buffers retain cells not written by an unused stage. Reject zero/error
        // values and stale cells from previous submissions using this command buffer's
        // execution window. GPUStart/EndTime use the host monotonic clock; calibrate it
        // into the timestamp counter's clock domain instead of assuming identical epochs.
        MTLTimestamp cpu = 0, gpu = 0;
        [c->device->Device() sampleTimestamps:&cpu gpuTimestamp:&gpu];
        const int64_t offset = int64_t(gpu) - int64_t(cpu);
        const uint64_t firstTick = uint64_t(int64_t(native.GPUStartTime * 1e9) + offset);
        const uint64_t lastTick = uint64_t(int64_t(native.GPUEndTime * 1e9) + offset);
        constexpr uint64_t calibrationSlack = 2000; // Clock-pair/readout rounding, not a timing correction.
        if (native.status != MTLCommandBufferStatusCompleted)
            Fail(c, "Metal command buffer failed during GPU execution-range sampling");
        for (auto& chunk : c->timestampChunks)
        {
            if (!chunk.used) continue;
            NSData* data = [chunk.buffer resolveCounterRange:NSMakeRange(0, chunk.used)];
            if (!data || data.length != chunk.used * sizeof(uint64_t))
            { Fail(c, "Resolve Metal encoder counter samples failed"); continue; }
            std::memcpy(chunk.resolved.data(), data.bytes, data.length);
        }
        const auto valid = [&](uint64_t tick)
        {
            return tick != 0 && tick != MTLCounterErrorValue && tick + calibrationSlack >= firstTick &&
                   tick <= lastTick + calibrationSlack;
        };
        for (const auto& resolve : c->timestampResolves)
        {
            auto& intervals = c->timestampIntervals;
            intervals.clear();
            for (uint32_t i = resolve.first; i < resolve.end; ++i)
            {
                const auto& sample = c->timestampSamples[i];
                const auto& chunk = c->timestampChunks[sample.chunk];
                for (uint32_t j = 0; j < sample.count; j += 2)
                {
                    const uint64_t begin = chunk.resolved[sample.first + j];
                    const uint64_t end = chunk.resolved[sample.first + j + 1];
                    if (begin == MTLCounterErrorValue || end == MTLCounterErrorValue)
                    { Fail(c, "Metal stage timestamp counter reported an error"); continue; }
                    if (!valid(begin) && !valid(end)) continue;
                    if (!valid(begin) || !valid(end) || end < begin)
                    { Fail(c, "Incomplete Metal stage counter pair"); continue; }
                    intervals.emplace_back(begin, end);
                }
            }
            if (!c->timestampsValid)
                *resolve.destination = {kInvalidTimestamp, kInvalidTimestamp, kInvalidTimestamp};
            else if (intervals.empty()) *resolve.destination = {0, 0, 0};
            else
            {
                std::sort(intervals.begin(), intervals.end());
                const uint64_t begin = intervals.front().first;
                uint64_t last = begin;
                uint64_t active = 0;
                for (const auto& interval : intervals)
                {
                    const uint64_t first = std::max(last, interval.first);
                    if (interval.second > first) active += interval.second - first;
                    last = std::max(last, interval.second);
                }
                *resolve.destination = {begin, last, active};
            }
        }
    }
    void DestroyTimestampSamples(CommandBufferMTL* c)
    {
        for (auto& chunk : c->timestampChunks) [chunk.buffer release];
        c->timestampChunks.clear();
    }
    const VriQueryInterface* GetQueryInterfaceMTL() { return &g_queryMTL; }
}
