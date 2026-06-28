// pipeline_cache_d3d12.cpp - VriPipelineCacheInterface for D3D12, backed by ID3D12PipelineLibrary.
// The library stores/loads PSOs by a stable name derived from the pipeline definition (see
// core_d3d12.cpp GraphicsPsoName), is serialized via Serialize/GetSerializedSize, and reseeded via
// CreatePipelineLibrary. A stale/foreign seed (DRIVER_VERSION_MISMATCH / ADAPTER_NOT_FOUND) is
// detected and the cache starts empty. Adapters without pipeline-library support report Unsupported.

#include "pipeline_cache_d3d12.h"
#include "device_d3d12.h"
#include "objects_d3d12.h"

namespace vri::d3d12
{
    namespace
    {
        inline DeviceD3D12* Dev(VriDevice* h) { return reinterpret_cast<DeviceD3D12*>(h); }

        VriResult VRI_CALL CreatePipelineCache(VriDevice*         device,
                                               const void*        initialData,
                                               size_t             initialSize,
                                               VriPipelineCache** out)
        {
            if (!out)
                return VriResult_InvalidArgument;
            DeviceD3D12*          d = Dev(device);
            ComPtr<ID3D12Device1> dev1;
            if (FAILED(d->Device()->QueryInterface(IID_PPV_ARGS(&dev1))))
                return VriResult_Unsupported; // pipeline libraries need ID3D12Device1

            PipelineCacheD3D12* p = new PipelineCacheD3D12 {};
            p->device             = d;
            if (initialData && initialSize)
                p->seed.assign(static_cast<const uint8_t*>(initialData),
                               static_cast<const uint8_t*>(initialData) + initialSize);
            // The blob must stay alive for the library's lifetime (held in p->seed).
            HRESULT hr = dev1->CreatePipelineLibrary(
                p->seed.empty() ? nullptr : p->seed.data(), p->seed.size(), IID_PPV_ARGS(&p->lib));
            if (FAILED(hr) && !p->seed.empty())
            {
                // Stale / foreign blob: discard the seed and start with an empty library.
                p->seed.clear();
                hr = dev1->CreatePipelineLibrary(nullptr, 0, IID_PPV_ARGS(&p->lib));
            }
            if (FAILED(hr))
            {
                // DXGI_ERROR_UNSUPPORTED on adapters without pipeline-library support (e.g. WARP).
                delete p;
                return VriResult_Unsupported;
            }
            *out = ToHandle(p);
            return VriResult_Success;
        }

        void VRI_CALL DestroyPipelineCache(VriPipelineCache* h)
        {
            if (h)
                delete PipeCache(h);
        }

        VriResult VRI_CALL GetPipelineCacheData(VriPipelineCache* h, void* data, size_t* size)
        {
            if (!h || !size)
                return VriResult_InvalidArgument;
            PipelineCacheD3D12* p  = PipeCache(h);
            const SIZE_T        sz = p->lib->GetSerializedSize();
            if (!data) // size query
            {
                *size = sz;
                return VriResult_Success;
            }
            if (*size < sz)
                return VriResult_InvalidArgument; // buffer too small
            if (FAILED(p->lib->Serialize(data, sz)))
                return VriResult_Failure;
            *size = sz;
            return VriResult_Success;
        }

        const VriPipelineCacheInterface g_pipelineCacheD3D12 = {
            CreatePipelineCache,
            DestroyPipelineCache,
            GetPipelineCacheData,
        };
    } // namespace

    const VriPipelineCacheInterface* GetPipelineCacheInterfaceD3D12() { return &g_pipelineCacheD3D12; }
} // namespace vri::d3d12
