// rt_mtl.h - Metal hardware ray tracing (inline ray query subset).
//
// Metal has no DXR/VK-style ray-tracing pipeline (raygen/miss/hit + SBT +
// TraceRays), so only the acceleration-structure half of VriRayTracingInterface
// is real here; the pipeline entry points return Unsupported. Ray query itself
// runs in a normal compute shader (SPIRV-Cross lowers OpRayQuery* to
// metal::raytracing), bound via the core interface.
#pragma once

#import <Metal/Metal.h>

#include <vri/vri.h>
#include <vri/ext/vri_ext_raytracing.h>

namespace vri::mtl
{
    class DeviceMTL;

    struct AccelerationStructureMTL
    {
        DeviceMTL*                     device;
        id<MTLAccelerationStructure>   as;
        id<MTLBuffer>                  scratch;
        MTLAccelerationStructureDescriptor* descriptor; // retained; rebuilt-into on CmdBuild
        VriAccelerationStructureType   type;
        // TLAS build-time state (translated from the app's VK-layout instance buffer).
        id<MTLBuffer>                  instanceBuffer;   // Metal-layout instance descriptors
        NSMutableArray*                instancedAS;      // id<MTLAccelerationStructure> referenced by instances
    };

    inline VriAccelerationStructure* ToHandle(AccelerationStructureMTL* a) { return reinterpret_cast<VriAccelerationStructure*>(a); }

    const VriRayTracingInterface* GetRayTracingInterfaceMTL();
} // namespace vri::mtl
