// rt_d3d12.cpp - DXR ray tracing via ID3D12Device5 / ID3D12GraphicsCommandList4.
//
// Acceleration structures own a result buffer + build scratch (sized from
// GetRaytracingAccelerationStructurePrebuildInfo). CmdBuild emits a UAV barrier so a
// dependent build (BLAS->TLAS) or DispatchRays sees the result. The RT pipeline is a
// state object (DXIL libraries + hit groups + shader/pipeline config + global root
// signature). The TLAS binds as an acceleration-structure SRV (see UpdateDescriptorRanges).
#include "rt_d3d12.h"

#include "conversions_d3d12.h"
#include "device_d3d12.h"
#include "objects_d3d12.h"

#include <deque>
#include <string>
#include <vector>

namespace vri::d3d12
{
    namespace
    {
        DeviceD3D12*                Dev(VriDevice* h) { return reinterpret_cast<DeviceD3D12*>(h); }
        CommandBufferD3D12*         CB(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferD3D12*>(h); }
        BufferD3D12*                BUF(VriBuffer* h) { return reinterpret_cast<BufferD3D12*>(h); }
        PipelineD3D12*              Pipe(VriPipeline* h) { return reinterpret_cast<PipelineD3D12*>(h); }
        AccelerationStructureD3D12* AS(VriAccelerationStructure* h)
        {
            return reinterpret_cast<AccelerationStructureD3D12*>(h);
        }

        std::wstring Widen(const char* s)
        {
            std::wstring w;
            if (s)
                while (*s)
                    w.push_back(static_cast<wchar_t>(*s++));
            return w;
        }

        ComPtr<ID3D12Resource> MakeUavBuffer(DeviceD3D12* d, UINT64 size, D3D12_RESOURCE_STATES state)
        {
            D3D12_HEAP_PROPERTIES hp = {};
            hp.Type                  = D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC rd   = {};
            rd.Dimension             = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width                 = size ? size : 1;
            rd.Height                = 1;
            rd.DepthOrArraySize      = 1;
            rd.MipLevels             = 1;
            rd.SampleDesc.Count      = 1;
            rd.Layout                = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            rd.Flags                 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            ComPtr<ID3D12Resource> res;
            d->Device()->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&res));
            return res;
        }

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS ToBuildFlags(VriAccelerationStructureBuildFlags f)
        {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS r =
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;
            if (f & VriAccelerationStructureBuild_AllowUpdate)
                r |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
            if (f & VriAccelerationStructureBuild_AllowCompaction)
                r |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
            if (f & VriAccelerationStructureBuild_PreferFastTrace)
                r |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            if (f & VriAccelerationStructureBuild_PreferFastBuild)
                r |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
            if (f & VriAccelerationStructureBuild_LowMemory)
                r |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY;
            return r;
        }

        MicromapD3D12* MM(VriMicromap* h) { return reinterpret_cast<MicromapD3D12*>(h); }

        // Geometry build storage: the geometry descs plus stable backing for the nested
        // triangles/linkage descs an OMM geometry points at (must outlive the build call).
        struct GeomStore
        {
            std::vector<D3D12_RAYTRACING_GEOMETRY_DESC>            geoms;
            std::deque<D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC>   tris; // for OMM geometries
            std::deque<D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC> links;
        };

        // Build the D3D12 geometry descs from the VRI desc. With withAddr the buffer GPU
        // VAs are filled (for build); without, only formats/counts (for sizing).
        void BuildGeoms(const VriAccelerationStructureDesc* desc, bool withAddr, GeomStore& gs)
        {
            for (uint32_t i = 0; i < desc->geometryCount; ++i)
            {
                const VriAsGeometryDesc&       g  = desc->geometries[i];
                D3D12_RAYTRACING_GEOMETRY_DESC vg = {};
                vg.Flags = (g.flags & VriAsGeometry_Opaque) ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE :
                                                              D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
                if (g.type == VriAsGeometryType_Triangles)
                {
                    const VriAsTrianglesDesc&                t   = g.triangles;
                    D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC tri = {};
                    tri.VertexFormat                             = ToDxgiVertexFormat(t.vertexFormat);
                    tri.VertexCount                              = t.vertexCount;
                    tri.VertexBuffer.StrideInBytes               = t.vertexStride;
                    tri.IndexFormat = t.indexBuffer ? (t.indexType == VriIndexType_UInt16 ? DXGI_FORMAT_R16_UINT :
                                                                                            DXGI_FORMAT_R32_UINT) :
                                                      DXGI_FORMAT_UNKNOWN;
                    tri.IndexCount  = t.indexBuffer ? t.indexCount : 0;
                    if (withAddr)
                    {
                        tri.VertexBuffer.StartAddress =
                            BUF(t.vertexBuffer)->resource->GetGPUVirtualAddress() + t.vertexOffset;
                        if (t.indexBuffer)
                            tri.IndexBuffer = BUF(t.indexBuffer)->resource->GetGPUVirtualAddress() + t.indexOffset;
                        if (t.transformBuffer)
                            tri.Transform3x4 =
                                BUF(t.transformBuffer)->resource->GetGPUVirtualAddress() + t.transformOffset;
                    }
                    if (t.micromap) // attach an opacity micromap (OMM-triangles geometry)
                    {
                        gs.tris.push_back(tri);
                        D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC link = {};
                        link.OpacityMicromapArray                       = MM(t.micromap)->address;
                        link.OpacityMicromapIndexFormat =
                            t.ommIndexType == VriIndexType_UInt16 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;
                        if (withAddr && t.ommIndexBuffer)
                        {
                            link.OpacityMicromapIndexBuffer.StartAddress =
                                BUF(t.ommIndexBuffer)->resource->GetGPUVirtualAddress() + t.ommIndexOffset;
                            link.OpacityMicromapIndexBuffer.StrideInBytes =
                                t.ommIndexType == VriIndexType_UInt16 ? 2 : 4;
                        }
                        gs.links.push_back(link);
                        vg.Type                     = D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES;
                        vg.OmmTriangles.pTriangles  = &gs.tris.back();
                        vg.OmmTriangles.pOmmLinkage = &gs.links.back();
                    }
                    else
                    {
                        vg.Type      = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
                        vg.Triangles = tri;
                    }
                }
                else // instances (TLAS)
                {
                    vg.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES; // unused for TLAS inputs
                }
                gs.geoms.push_back(vg);
            }
        }

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE ToAsType(VriAccelerationStructureType t)
        {
            return t == VriAccelerationStructureType_TopLevel ?
                       D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL :
                       D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        }

        // Fill build inputs for sizing/build. For TLAS, instance data lives in a buffer.
        void FillInputs(const VriAccelerationStructureDesc*                   desc,
                        bool                                                  withAddr,
                        GeomStore&                                            gs,
                        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& inputs)
        {
            inputs             = {};
            inputs.Type        = ToAsType(desc->type);
            inputs.Flags       = ToBuildFlags(desc->flags);
            inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            if (desc->type == VriAccelerationStructureType_TopLevel)
            {
                const VriAsInstancesDesc& in = desc->geometries[0].instances;
                inputs.NumDescs              = in.instanceCount;
                if (withAddr)
                    inputs.InstanceDescs = BUF(in.instanceBuffer)->resource->GetGPUVirtualAddress() + in.offset;
            }
            else
            {
                BuildGeoms(desc, withAddr, gs);
                inputs.NumDescs       = static_cast<UINT>(gs.geoms.size());
                inputs.pGeometryDescs = gs.geoms.data();
            }
        }

        VriResult VRI_CALL CreateAccelerationStructure(VriDevice*                          device,
                                                       const VriAccelerationStructureDesc* desc,
                                                       VriAccelerationStructure**          out)
        {
            DeviceD3D12*          d = Dev(device);
            ComPtr<ID3D12Device5> dev5;
            if (FAILED(d->Device()->QueryInterface(IID_PPV_ARGS(&dev5))))
            {
                d->ReportError("ID3D12Device5 unavailable");
                return VriResult_Unsupported;
            }

            GeomStore                                            gs;
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs;
            FillInputs(desc, false, gs, inputs);

            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
            dev5->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);

            AccelerationStructureD3D12* a = new AccelerationStructureD3D12 {};
            a->device                     = d;
            a->type                       = inputs.Type;
            a->result =
                MakeUavBuffer(d, info.ResultDataMaxSizeInBytes, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
            a->scratch = MakeUavBuffer(d, info.ScratchDataSizeInBytes, D3D12_RESOURCE_STATE_COMMON);
            if (!a->result || !a->scratch)
            {
                delete a;
                d->ReportError("AS buffer alloc failed");
                return VriResult_OutOfMemory;
            }
            a->address = a->result->GetGPUVirtualAddress();
            *out       = ToHandle(a);
            return VriResult_Success;
        }

        void VRI_CALL DestroyAccelerationStructure(VriAccelerationStructure* as)
        {
            if (as)
                delete AS(as);
        }
        uint64_t VRI_CALL GetAccelerationStructureDeviceAddress(const VriAccelerationStructure* as)
        {
            return reinterpret_cast<const AccelerationStructureD3D12*>(as)->address;
        }

        VriResult VRI_CALL CreateAccelerationStructureDescriptor(VriDevice*                device,
                                                                 VriAccelerationStructure* as,
                                                                 VriDescriptor**           out)
        {
            DescriptorD3D12* v = new DescriptorD3D12 {};
            v->device          = Dev(device);
            v->accelAddress    = AS(as)->address;
            *out               = reinterpret_cast<VriDescriptor*>(v);
            return VriResult_Success;
        }

        void VRI_CALL CmdBuildAccelerationStructure(VriCommandBuffer*                        cmd,
                                                    const VriBuildAccelerationStructureDesc* desc)
        {
            CommandBufferD3D12*                c = CB(cmd);
            AccelerationStructureD3D12*        a = AS(desc->dst);
            ComPtr<ID3D12GraphicsCommandList4> list4;
            if (FAILED(c->list.As(&list4)))
                return;

            GeomStore                                            gs;
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs;
            FillInputs(desc->geometry, true, gs, inputs);

            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd = {};
            bd.DestAccelerationStructureData                      = a->result->GetGPUVirtualAddress();
            bd.Inputs                                             = inputs;
            bd.ScratchAccelerationStructureData                   = a->scratch->GetGPUVirtualAddress();
            list4->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);

            // Make the built AS visible to a dependent build (BLAS->TLAS) or trace.
            D3D12_RESOURCE_BARRIER uav = {};
            uav.Type                   = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav.UAV.pResource          = a->result.Get();
            c->list->ResourceBarrier(1, &uav);
        }

        VriResult VRI_CALL CreateRayTracingPipeline(VriDevice*                       device,
                                                    const VriRayTracingPipelineDesc* desc,
                                                    VriPipeline**                    out)
        {
            DeviceD3D12*          d = Dev(device);
            ComPtr<ID3D12Device5> dev5;
            if (FAILED(d->Device()->QueryInterface(IID_PPV_ARGS(&dev5))))
            {
                d->ReportError("ID3D12Device5 unavailable");
                return VriResult_Unsupported;
            }
            PipelineLayoutD3D12* layout = reinterpret_cast<PipelineLayoutD3D12*>(desc->pipelineLayout);

            // Storage that must outlive CreateStateObject (subobjects point into it).
            std::deque<std::wstring>            names; // export + hit-group names
            std::deque<D3D12_EXPORT_DESC>       exports;
            std::deque<D3D12_DXIL_LIBRARY_DESC> libs;
            std::deque<D3D12_HIT_GROUP_DESC>    hitGroups;
            std::vector<D3D12_STATE_SUBOBJECT>  subs;

            // One DXIL library per shader, exporting its entry-point name.
            std::vector<const wchar_t*> exportNames(desc->shaderNum);
            for (uint32_t i = 0; i < desc->shaderNum; ++i)
            {
                names.push_back(Widen(desc->shaders[i].entryPointName ? desc->shaders[i].entryPointName : "main"));
                exportNames[i]       = names.back().c_str();
                D3D12_EXPORT_DESC ed = {};
                ed.Name              = exportNames[i];
                ed.Flags             = D3D12_EXPORT_FLAG_NONE;
                exports.push_back(ed);
                D3D12_DXIL_LIBRARY_DESC ld     = {};
                ld.DXILLibrary.pShaderBytecode = desc->shaders[i].bytecode;
                ld.DXILLibrary.BytecodeLength  = desc->shaders[i].bytecodeSize;
                ld.NumExports                  = 1;
                ld.pExports                    = &exports.back();
                libs.push_back(ld);
                D3D12_STATE_SUBOBJECT so = {};
                so.Type                  = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
                so.pDesc                 = &libs.back();
                subs.push_back(so);
            }

            // Groups -> export name (general) or a hit-group subobject.
            std::vector<std::wstring> groupNames(desc->groupNum);
            for (uint32_t i = 0; i < desc->groupNum; ++i)
            {
                const VriShaderGroupDesc& g = desc->groups[i];
                if (g.type == VriShaderGroupType_General)
                {
                    groupNames[i] = Widen(desc->shaders[g.generalShader].entryPointName);
                }
                else // triangles/procedural hit group
                {
                    names.push_back(L"vriHitGroup" + std::to_wstring(i));
                    groupNames[i]           = names.back();
                    D3D12_HIT_GROUP_DESC hg = {};
                    hg.HitGroupExport       = names.back().c_str();
                    hg.Type                 = g.type == VriShaderGroupType_ProceduralHitGroup ?
                                                  D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE :
                                                  D3D12_HIT_GROUP_TYPE_TRIANGLES;
                    if (g.closestHitShader != VRI_SHADER_UNUSED)
                    {
                        names.push_back(Widen(desc->shaders[g.closestHitShader].entryPointName));
                        hg.ClosestHitShaderImport = names.back().c_str();
                    }
                    if (g.anyHitShader != VRI_SHADER_UNUSED)
                    {
                        names.push_back(Widen(desc->shaders[g.anyHitShader].entryPointName));
                        hg.AnyHitShaderImport = names.back().c_str();
                    }
                    if (g.intersectionShader != VRI_SHADER_UNUSED)
                    {
                        names.push_back(Widen(desc->shaders[g.intersectionShader].entryPointName));
                        hg.IntersectionShaderImport = names.back().c_str();
                    }
                    hitGroups.push_back(hg);
                    D3D12_STATE_SUBOBJECT so = {};
                    so.Type                  = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
                    so.pDesc                 = &hitGroups.back();
                    subs.push_back(so);
                }
            }

            D3D12_RAYTRACING_SHADER_CONFIG shaderCfg = {};
            shaderCfg.MaxPayloadSizeInBytes          = 16; // float3 color (+pad)
            shaderCfg.MaxAttributeSizeInBytes        = 8;  // barycentrics
            {
                D3D12_STATE_SUBOBJECT so = {};
                so.Type                  = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
                so.pDesc                 = &shaderCfg;
                subs.push_back(so);
            }

            // PIPELINE_CONFIG1 carries the flags; use ALLOW_OPACITY_MICROMAPS when OMM
            // was enabled at device creation so a BLAS with OMM-triangles can be traced.
            D3D12_RAYTRACING_PIPELINE_CONFIG1 pipeCfg = {};
            pipeCfg.MaxTraceRecursionDepth            = desc->maxRecursionDepth ? desc->maxRecursionDepth : 1;
            if (d->EnabledFeatures() & VriFeature_OpacityMicromap)
                pipeCfg.Flags = D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_OPACITY_MICROMAPS;
            {
                D3D12_STATE_SUBOBJECT so = {};
                so.Type                  = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG1;
                so.pDesc                 = &pipeCfg;
                subs.push_back(so);
            }

            D3D12_GLOBAL_ROOT_SIGNATURE grs = {};
            grs.pGlobalRootSignature        = layout->rootSig.Get();
            {
                D3D12_STATE_SUBOBJECT so = {};
                so.Type                  = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
                so.pDesc                 = &grs;
                subs.push_back(so);
            }

            D3D12_STATE_OBJECT_DESC sod = {};
            sod.Type                    = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
            sod.NumSubobjects           = static_cast<UINT>(subs.size());
            sod.pSubobjects             = subs.data();

            PipelineD3D12* p = new PipelineD3D12 {};
            p->device        = d;
            p->rootSig       = layout->rootSig.Get();
            p->isRt          = true;
            p->isCompute     = true; // descriptor sets bind on the compute root
            if (FAILED(dev5->CreateStateObject(&sod, IID_PPV_ARGS(&p->stateObject))))
            {
                delete p;
                d->ReportError("CreateStateObject (RT) failed");
                return VriResult_Failure;
            }
            if (FAILED(p->stateObject.As(&p->stateProps)))
            {
                delete p;
                d->ReportError("ID3D12StateObjectProperties query failed");
                return VriResult_Failure;
            }
            p->groupNames.assign(groupNames.begin(), groupNames.end());
            *out = ToHandle(p);
            return VriResult_Success;
        }

        VriResult VRI_CALL
        GetShaderGroupHandles(VriPipeline* pipeline, uint32_t firstGroup, uint32_t groupNum, size_t dstSize, void* dst)
        {
            PipelineD3D12* p      = Pipe(pipeline);
            const UINT     idSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
            if (dstSize < static_cast<size_t>(groupNum) * idSize)
                return VriResult_InvalidArgument;
            uint8_t* out = static_cast<uint8_t*>(dst);
            for (uint32_t i = 0; i < groupNum; ++i)
            {
                const uint32_t gi = firstGroup + i;
                if (gi >= p->groupNames.size())
                    return VriResult_InvalidArgument;
                const void* id = p->stateProps->GetShaderIdentifier(p->groupNames[gi].c_str());
                if (!id)
                    return VriResult_Failure;
                memcpy(out + static_cast<size_t>(i) * idSize, id, idSize);
            }
            return VriResult_Success;
        }

        void VRI_CALL CmdTraceRays(VriCommandBuffer* cmd, const VriDispatchRaysDesc* desc)
        {
            CommandBufferD3D12*                c = CB(cmd);
            ComPtr<ID3D12GraphicsCommandList4> list4;
            if (FAILED(c->list.As(&list4)))
                return;
            auto va = [](const VriStridedBufferRegion& r) -> D3D12_GPU_VIRTUAL_ADDRESS {
                return r.buffer ? BUF(r.buffer)->resource->GetGPUVirtualAddress() + r.offset : 0;
            };
            D3D12_DISPATCH_RAYS_DESC dr               = {};
            dr.RayGenerationShaderRecord.StartAddress = va(desc->raygen);
            dr.RayGenerationShaderRecord.SizeInBytes  = desc->raygen.size;
            dr.MissShaderTable.StartAddress           = va(desc->miss);
            dr.MissShaderTable.SizeInBytes            = desc->miss.size;
            dr.MissShaderTable.StrideInBytes          = desc->miss.stride;
            dr.HitGroupTable.StartAddress             = va(desc->hit);
            dr.HitGroupTable.SizeInBytes              = desc->hit.size;
            dr.HitGroupTable.StrideInBytes            = desc->hit.stride;
            if (desc->callable.buffer)
            {
                dr.CallableShaderTable.StartAddress  = va(desc->callable);
                dr.CallableShaderTable.SizeInBytes   = desc->callable.size;
                dr.CallableShaderTable.StrideInBytes = desc->callable.stride;
            }
            dr.Width  = desc->width;
            dr.Height = desc->height;
            dr.Depth  = desc->depth ? desc->depth : 1;
            list4->DispatchRays(&dr);
        }

        const VriRayTracingInterface g_rtD3D12 = {
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

    const VriRayTracingInterface* GetRayTracingInterfaceD3D12() { return &g_rtD3D12; }
} // namespace vri::d3d12
