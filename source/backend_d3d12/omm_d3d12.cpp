// omm_d3d12.cpp - opacity micromaps via DXR 1.2 (Raytracing Tier 1.2).
//
// An OMM "array" is built through the acceleration-structure build path with type
// OPACITY_MICROMAP_ARRAY (ID3D12Device5 prebuild sizing + ID3D12GraphicsCommandList4
// BuildRaytracingAccelerationStructure). It owns a result buffer + scratch. The OMM is
// linked to a BLAS triangle geometry in rt_d3d12's BuildGeoms (OMM_TRIANGLES geometry).
#include "omm_d3d12.h"

#include "device_d3d12.h"
#include "objects_d3d12.h"

namespace vri::d3d12
{
    namespace
    {
        DeviceD3D12*        Dev(VriDevice* h) { return reinterpret_cast<DeviceD3D12*>(h); }
        CommandBufferD3D12* CB(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferD3D12*>(h); }
        BufferD3D12*        BUF(VriBuffer* h) { return reinterpret_cast<BufferD3D12*>(h); }
        MicromapD3D12*      MM(VriMicromap* h) { return reinterpret_cast<MicromapD3D12*>(h); }

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

        D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT ToOmmFormat(VriOpacityMicromapFormat f)
        {
            return f == VriOpacityMicromapFormat_4State ? D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT_OC1_4_STATE :
                                                          D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT_OC1_2_STATE;
        }

        VriResult VRI_CALL CreateMicromap(VriDevice* device, const VriMicromapDesc* desc, VriMicromap** out)
        {
            DeviceD3D12*          d = Dev(device);
            ComPtr<ID3D12Device5> dev5;
            if (FAILED(d->Device()->QueryInterface(IID_PPV_ARGS(&dev5))))
            {
                d->ReportError("ID3D12Device5 unavailable");
                return VriResult_Unsupported;
            }

            D3D12_RAYTRACING_OPACITY_MICROMAP_HISTOGRAM_ENTRY hist = {};
            hist.Count                                             = desc->triangleCount;
            hist.SubdivisionLevel                                  = desc->subdivisionLevel;
            hist.Format                                            = ToOmmFormat(desc->format);

            D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC ommDesc = {};
            ommDesc.NumOmmHistogramEntries                       = 1;
            ommDesc.pOmmHistogram                                = &hist;
            // Prebuild sizing reads only the histogram, but the debug layer still rejects null
            // InputBuffer/PerOmmDescs when the histogram declares OMMs. The real data buffers only
            // arrive at build time, so point both at a throwaway buffer (large enough for the desc
            // array) purely to provide valid non-null addresses for the sizing query.
            const UINT64 descBytes =
                static_cast<UINT64>(desc->triangleCount) * sizeof(D3D12_RAYTRACING_OPACITY_MICROMAP_DESC);
            ComPtr<ID3D12Resource> prebuildDummy =
                MakeUavBuffer(d, descBytes ? descBytes : 1, D3D12_RESOURCE_STATE_COMMON);
            if (prebuildDummy)
            {
                ommDesc.InputBuffer               = prebuildDummy->GetGPUVirtualAddress();
                ommDesc.PerOmmDescs.StartAddress  = prebuildDummy->GetGPUVirtualAddress();
                ommDesc.PerOmmDescs.StrideInBytes = sizeof(D3D12_RAYTRACING_OPACITY_MICROMAP_DESC);
            }

            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
            inputs.Type                      = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY;
            inputs.Flags                     = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            inputs.DescsLayout               = D3D12_ELEMENTS_LAYOUT_ARRAY;
            inputs.NumDescs                  = desc->triangleCount;
            inputs.pOpacityMicromapArrayDesc = &ommDesc;

            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
            dev5->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);

            MicromapD3D12* m    = new MicromapD3D12 {};
            m->device           = d;
            m->format           = ToOmmFormat(desc->format);
            m->subdivisionLevel = desc->subdivisionLevel;
            m->triangleCount    = desc->triangleCount;
            m->result =
                MakeUavBuffer(d, info.ResultDataMaxSizeInBytes, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
            m->scratch = MakeUavBuffer(d, info.ScratchDataSizeInBytes, D3D12_RESOURCE_STATE_COMMON);
            if (!m->result || !m->scratch)
            {
                delete m;
                d->ReportError("OMM buffer alloc failed");
                return VriResult_OutOfMemory;
            }
            m->address = m->result->GetGPUVirtualAddress();
            *out       = ToHandle(m);
            return VriResult_Success;
        }

        void VRI_CALL DestroyMicromap(VriMicromap* micromap)
        {
            if (micromap)
                delete MM(micromap);
        }

        void VRI_CALL CmdBuildMicromap(VriCommandBuffer* cmd, const VriBuildMicromapDesc* desc)
        {
            CommandBufferD3D12*                c = CB(cmd);
            MicromapD3D12*                     m = MM(desc->dst);
            ComPtr<ID3D12GraphicsCommandList4> list4;
            if (FAILED(c->list.As(&list4)))
                return;

            D3D12_RAYTRACING_OPACITY_MICROMAP_HISTOGRAM_ENTRY hist = {};
            hist.Count                                             = m->triangleCount;
            hist.SubdivisionLevel                                  = m->subdivisionLevel;
            hist.Format                                            = m->format;

            D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC ommDesc = {};
            ommDesc.NumOmmHistogramEntries                       = 1;
            ommDesc.pOmmHistogram                                = &hist;
            ommDesc.InputBuffer = BUF(desc->dataBuffer)->resource->GetGPUVirtualAddress() + desc->dataOffset;
            ommDesc.PerOmmDescs.StartAddress =
                BUF(desc->triangleBuffer)->resource->GetGPUVirtualAddress() + desc->triangleOffset;
            ommDesc.PerOmmDescs.StrideInBytes = sizeof(D3D12_RAYTRACING_OPACITY_MICROMAP_DESC);

            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
            inputs.Type                      = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY;
            inputs.Flags                     = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            inputs.DescsLayout               = D3D12_ELEMENTS_LAYOUT_ARRAY;
            inputs.NumDescs                  = m->triangleCount;
            inputs.pOpacityMicromapArrayDesc = &ommDesc;

            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC bd = {};
            bd.DestAccelerationStructureData                      = m->result->GetGPUVirtualAddress();
            bd.Inputs                                             = inputs;
            bd.ScratchAccelerationStructureData                   = m->scratch->GetGPUVirtualAddress();
            list4->BuildRaytracingAccelerationStructure(&bd, 0, nullptr);

            D3D12_RESOURCE_BARRIER uav = {};
            uav.Type                   = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            uav.UAV.pResource          = m->result.Get();
            c->list->ResourceBarrier(1, &uav);
        }

        const VriOpacityMicromapInterface g_ommD3D12 = {
            CreateMicromap,
            DestroyMicromap,
            CmdBuildMicromap,
        };
    } // namespace

    const VriOpacityMicromapInterface* GetOpacityMicromapInterfaceD3D12() { return &g_ommD3D12; }
} // namespace vri::d3d12
