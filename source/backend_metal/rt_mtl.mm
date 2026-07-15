// rt_mtl.mm - Metal acceleration structures for inline ray query.
//
// BLAS = MTLPrimitiveAccelerationStructure (triangles); TLAS =
// MTLInstanceAccelerationStructure. The app drives this with the Vulkan/DXR
// instance layout (VkAccelerationStructureInstanceKHR), which CmdBuild translates
// into Metal's MTLAccelerationStructureInstanceDescriptor + an instanced-AS array
// (Metal references BLASes by array index, not by device address). Builds run on a
// MTLAccelerationStructureCommandEncoder. The RT *pipeline* half (raygen/SBT/
// TraceRays) is Unsupported - Metal has no equivalent; ray query runs in compute.

#include "rt_mtl.h"
#include "conversions_mtl.h"
#include "device_mtl.h"
#include "objects_mtl.h"

#import <Metal/Metal.h>

#include <cstring>

namespace vri::mtl
{
    namespace
    {
        inline DeviceMTL*               Dev(VriDevice* h)               { return reinterpret_cast<DeviceMTL*>(h); }
        inline CommandBufferMTL*        CB(VriCommandBuffer* h)         { return reinterpret_cast<CommandBufferMTL*>(h); }
        inline BufferMTL*               Buf(VriBuffer* h)               { return reinterpret_cast<BufferMTL*>(h); }
        inline AccelerationStructureMTL* AS(VriAccelerationStructure* h) { return reinterpret_cast<AccelerationStructureMTL*>(h); }

        MTLAttributeFormat ToAttrFormat(VriFormat f)
        {
            switch (f)
            {
                case VriFormat_RGB32_SFLOAT:  return MTLAttributeFormatFloat3;
                case VriFormat_RG32_SFLOAT:   return MTLAttributeFormatFloat2;
                case VriFormat_RGBA32_SFLOAT: return MTLAttributeFormatFloat4;
                case VriFormat_RGBA16_SFLOAT: return MTLAttributeFormatHalf4;
                case VriFormat_RG16_SFLOAT:   return MTLAttributeFormatHalf2;
                default:                      return MTLAttributeFormatFloat3;
            }
        }

        MTLAccelerationStructureUsage ToAsUsage(VriAccelerationStructureBuildFlags f)
        {
            MTLAccelerationStructureUsage u = MTLAccelerationStructureUsageNone;
            if (f & VriAccelerationStructureBuild_AllowUpdate)     u |= MTLAccelerationStructureUsageRefit;
            if (f & VriAccelerationStructureBuild_PreferFastBuild) u |= MTLAccelerationStructureUsagePreferFastBuild;
            return u; // PreferFastTrace == Metal default (None)
        }

        // Build the Metal descriptor for an AS (geometry + buffers known at create time).
        // For TLAS only instanceCount is set here; the instance buffer + instanced-AS array
        // are filled at build time (TranslateInstances).
        MTLAccelerationStructureDescriptor* MakeDescriptor(const VriAccelerationStructureDesc* desc)
        {
            if (desc->type == VriAccelerationStructureType_TopLevel)
            {
                MTLInstanceAccelerationStructureDescriptor* d = [MTLInstanceAccelerationStructureDescriptor descriptor];
                d.usage = ToAsUsage(desc->flags);
                uint32_t instances = 0;
                for (uint32_t i = 0; i < desc->geometryCount; ++i)
                    if (desc->geometries[i].type == VriAsGeometryType_Instances)
                        instances += desc->geometries[i].instances.instanceCount;
                d.instanceCount = instances;
                return [d retain];
            }

            NSMutableArray* geoms = [NSMutableArray array];
            for (uint32_t i = 0; i < desc->geometryCount; ++i)
            {
                const VriAsGeometryDesc& g = desc->geometries[i];
                if (g.type == VriAsGeometryType_Aabbs)
                {
                    // Procedural primitives: Metal bounding-box geometry (custom
                    // intersection runs in the ray-query / intersection function).
                    const VriAsAabbsDesc& a = g.aabbs;
                    MTLAccelerationStructureBoundingBoxGeometryDescriptor* box =
                        [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
                    box.boundingBoxBuffer = Buf(a.buffer)->buffer;
                    box.boundingBoxBufferOffset = a.offset;
                    box.boundingBoxStride = a.stride ? a.stride : 24; // 6 floats
                    box.boundingBoxCount = a.count;
                    box.opaque = (g.flags & VriAsGeometry_Opaque) ? YES : NO;
                    [geoms addObject:box];
                    continue;
                }
                if (g.type != VriAsGeometryType_Triangles)
                    continue;
                const VriAsTrianglesDesc& t = g.triangles;
                MTLAccelerationStructureTriangleGeometryDescriptor* tri = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
                tri.vertexBuffer = Buf(t.vertexBuffer)->buffer;
                tri.vertexBufferOffset = t.vertexOffset;
                tri.vertexStride = t.vertexStride;
                tri.vertexFormat = ToAttrFormat(t.vertexFormat);
                tri.triangleCount = t.indexBuffer ? (t.indexCount / 3) : (t.vertexCount / 3);
                if (t.indexBuffer)
                {
                    tri.indexBuffer = Buf(t.indexBuffer)->buffer;
                    tri.indexBufferOffset = t.indexOffset;
                    tri.indexType = t.indexType == VriIndexType_UInt16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
                }
                tri.opaque = (g.flags & VriAsGeometry_Opaque) ? YES : NO;
                [geoms addObject:tri];
            }
            MTLPrimitiveAccelerationStructureDescriptor* d = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
            d.usage = ToAsUsage(desc->flags);
            d.geometryDescriptors = geoms;
            return [d retain];
        }

        VriResult VRI_CALL CreateAccelerationStructure(VriDevice* device, const VriAccelerationStructureDesc* desc, VriAccelerationStructure** out)
        {
            DeviceMTL* d = Dev(device);
            MTLAccelerationStructureDescriptor* mdesc = MakeDescriptor(desc);
            MTLAccelerationStructureSizes sizes = [d->Device() accelerationStructureSizesWithDescriptor:mdesc];

            id<MTLAccelerationStructure> as = [d->Device() newAccelerationStructureWithSize:sizes.accelerationStructureSize];
            if (!as) { [mdesc release]; return VriResult_OutOfMemory; }
            id<MTLBuffer> scratch = [d->Device() newBufferWithLength:(sizes.buildScratchBufferSize ? sizes.buildScratchBufferSize : 1)
                                                            options:MTLResourceStorageModePrivate];

            AccelerationStructureMTL* a = new AccelerationStructureMTL{};
            a->device = d;
            a->as = as;
            a->scratch = scratch;
            a->descriptor = mdesc; // owned
            a->type = desc->type;
            *out = ToHandle(a);
            return VriResult_Success;
        }

        void VRI_CALL DestroyAccelerationStructure(VriAccelerationStructure* as)
        {
            if (!as) return;
            AccelerationStructureMTL* a = AS(as);
            if (a->as) [a->as release];
            if (a->scratch) [a->scratch release];
            if (a->descriptor) [a->descriptor release];
            if (a->instanceBuffer) [a->instanceBuffer release];
            if (a->instancedAS) [a->instancedAS release];
            delete a;
        }

        // Metal has no AS device address; hand back the object pointer as a stable token.
        // CmdBuild (TLAS) casts it back to recover the BLAS's id<MTLAccelerationStructure>.
        uint64_t VRI_CALL GetAccelerationStructureDeviceAddress(const VriAccelerationStructure* as)
        {
            return reinterpret_cast<uint64_t>(as);
        }

        VriResult VRI_CALL CreateAccelerationStructureDescriptor(VriDevice* device, VriAccelerationStructure* as, VriDescriptor** out)
        {
            DescriptorMTL* v = new DescriptorMTL{};
            v->kind = DescriptorMTL::Kind::AccelStruct;
            v->device = Dev(device);
            v->accel = AS(as)->as;
            v->accelObj = AS(as); // so binding can make the TLAS's referenced BLASes resident (useResource)
            *out = reinterpret_cast<VriDescriptor*>(v);
            return VriResult_Success;
        }

        VriResult VRI_CALL CreateRayTracingPipeline(VriDevice*, const VriRayTracingPipelineDesc*, VriPipeline**)
        {
            return VriResult_Unsupported; // Metal has no DXR-style RT pipeline; use ray query in compute
        }
        VriResult VRI_CALL GetShaderGroupHandles(VriPipeline*, uint32_t, uint32_t, size_t, void*)
        {
            return VriResult_Unsupported;
        }

        // The app's instance buffer is in VkAccelerationStructureInstanceKHR layout; translate
        // it into Metal's MTLAccelerationStructureInstanceDescriptor + the instanced-AS array
        // (Metal references each BLAS by index into that array, not by device address).
        void TranslateInstances(DeviceMTL* d, AccelerationStructureMTL* a, const VriAsInstancesDesc& in,
                                MTLInstanceAccelerationStructureDescriptor* tlasDesc)
        {
            struct VkInstance { float transform[12]; uint32_t idAndMask; uint32_t sbtAndFlags; uint64_t blasRef; };
            const auto* src = reinterpret_cast<const VkInstance*>(static_cast<const uint8_t*>([Buf(in.instanceBuffer)->buffer contents]) + in.offset);

            id<MTLBuffer> dst = [d->Device() newBufferWithLength:(sizeof(MTLAccelerationStructureInstanceDescriptor) * (in.instanceCount ? in.instanceCount : 1))
                                                        options:MTLResourceStorageModeShared];
            auto* out = static_cast<MTLAccelerationStructureInstanceDescriptor*>([dst contents]);
            NSMutableArray* asArray = [[NSMutableArray alloc] init];

            for (uint32_t i = 0; i < in.instanceCount; ++i)
            {
                const VkInstance& s = src[i];
                MTLAccelerationStructureInstanceDescriptor& o = out[i];
                // VK transform is row-major 3x4; Metal's is 4 columns of float3 (column c, row r = transform[r*4+c]).
                for (int c = 0; c < 4; ++c)
                {
                    o.transformationMatrix.columns[c].x = s.transform[0 * 4 + c];
                    o.transformationMatrix.columns[c].y = s.transform[1 * 4 + c];
                    o.transformationMatrix.columns[c].z = s.transform[2 * 4 + c];
                }
                const uint32_t flags = s.sbtAndFlags >> 24;
                MTLAccelerationStructureInstanceOptions opt = MTLAccelerationStructureInstanceOptionNone;
                if (flags & 0x1u) opt |= MTLAccelerationStructureInstanceOptionDisableTriangleCulling;          // VK facing-cull-disable
                if (flags & 0x4u) opt |= MTLAccelerationStructureInstanceOptionOpaque;                           // VK force-opaque
                if (flags & 0x8u) opt |= MTLAccelerationStructureInstanceOptionNonOpaque;                        // VK force-non-opaque
                o.options = opt;
                o.mask = s.idAndMask >> 24;
                o.intersectionFunctionTableOffset = 0; // ray query: no SBT
                // Map the BLAS token to an index into instancedAccelerationStructures (dedup).
                AccelerationStructureMTL* blas = reinterpret_cast<AccelerationStructureMTL*>(s.blasRef);
                NSUInteger idx = blas ? [asArray indexOfObject:blas->as] : NSNotFound;
                if (idx == NSNotFound && blas) { idx = asArray.count; [asArray addObject:blas->as]; }
                o.accelerationStructureIndex = (uint32_t)(idx == NSNotFound ? 0 : idx);
            }

            if (a->instanceBuffer) [a->instanceBuffer release];
            if (a->instancedAS) [a->instancedAS release];
            a->instanceBuffer = dst;       // keep alive for the GPU build
            a->instancedAS = asArray;      // ditto
            tlasDesc.instanceDescriptorBuffer = dst;
            tlasDesc.instanceDescriptorBufferOffset = 0;
            tlasDesc.instanceCount = in.instanceCount;
            tlasDesc.instancedAccelerationStructures = asArray;
        }

        void VRI_CALL CmdBuildAccelerationStructure(VriCommandBuffer* cmd, const VriBuildAccelerationStructureDesc* desc)
        {
            CommandBufferMTL* c = CB(cmd);
            AccelerationStructureMTL* a = AS(desc->dst);

            // Close any open blit/compute encoder before the AS-build encoder (one at a time).
            if (c->blitEnc)    { [c->blitEnc endEncoding];    [c->blitEnc release];    c->blitEnc = nil; }
            if (c->computeEnc) { [c->computeEnc endEncoding]; [c->computeEnc release]; c->computeEnc = nil; }

            if (a->type == VriAccelerationStructureType_TopLevel)
            {
                auto* tlasDesc = (MTLInstanceAccelerationStructureDescriptor*)a->descriptor;
                for (uint32_t i = 0; i < desc->geometry->geometryCount; ++i)
                    if (desc->geometry->geometries[i].type == VriAsGeometryType_Instances)
                    { TranslateInstances(c->device, a, desc->geometry->geometries[i].instances, tlasDesc); break; }
            }

            id<MTLAccelerationStructureCommandEncoder> enc = [c->cmd accelerationStructureCommandEncoder];
            [enc buildAccelerationStructure:a->as descriptor:a->descriptor scratchBuffer:a->scratch scratchBufferOffset:0];
            [enc endEncoding];
        }

        void VRI_CALL CmdTraceRays(VriCommandBuffer*, const VriDispatchRaysDesc*) {} // no RT pipeline on Metal

        const VriRayTracingInterface g_rtMTL = {
            CreateAccelerationStructure,
            DestroyAccelerationStructure,
            GetAccelerationStructureDeviceAddress,
            CreateAccelerationStructureDescriptor,
            CreateRayTracingPipeline,
            GetShaderGroupHandles,
            CmdBuildAccelerationStructure,
            CmdTraceRays,
        };
    } // namespace

    const VriRayTracingInterface* GetRayTracingInterfaceMTL() { return &g_rtMTL; }
} // namespace vri::mtl
