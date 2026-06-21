// vrs_d3d12.cpp - Variable Rate Shading via ID3D12GraphicsCommandList5::RSSetShadingRate.
//
// Per-draw (pipeline) shading rate set dynamically before a draw; VRS Tier 1 covers
// this. The two combiners (per-primitive, shading-rate-image) mirror the VK model.
#include "vrs_d3d12.h"

#include "device_d3d12.h"
#include "objects_d3d12.h"

namespace vri::d3d12
{
    namespace
    {
        CommandBufferD3D12* CB(VriCommandBuffer* h) { return reinterpret_cast<CommandBufferD3D12*>(h); }

        D3D12_SHADING_RATE ToD3DShadingRate(VriShadingRate r)
        {
            switch (r)
            {
                case VriShadingRate_1x1: return D3D12_SHADING_RATE_1X1;
                case VriShadingRate_1x2: return D3D12_SHADING_RATE_1X2;
                case VriShadingRate_2x1: return D3D12_SHADING_RATE_2X1;
                case VriShadingRate_2x2: return D3D12_SHADING_RATE_2X2;
                case VriShadingRate_2x4: return D3D12_SHADING_RATE_2X4;
                case VriShadingRate_4x2: return D3D12_SHADING_RATE_4X2;
                case VriShadingRate_4x4: return D3D12_SHADING_RATE_4X4;
                default:                 return D3D12_SHADING_RATE_1X1;
            }
        }

        D3D12_SHADING_RATE_COMBINER ToD3DCombiner(VriShadingRateCombiner c)
        {
            switch (c)
            {
                case VriShadingRateCombiner_Keep:    return D3D12_SHADING_RATE_COMBINER_PASSTHROUGH;
                case VriShadingRateCombiner_Replace: return D3D12_SHADING_RATE_COMBINER_OVERRIDE;
                case VriShadingRateCombiner_Min:     return D3D12_SHADING_RATE_COMBINER_MIN;
                case VriShadingRateCombiner_Max:     return D3D12_SHADING_RATE_COMBINER_MAX;
                case VriShadingRateCombiner_Sum:     return D3D12_SHADING_RATE_COMBINER_SUM;
                default:                             return D3D12_SHADING_RATE_COMBINER_PASSTHROUGH;
            }
        }

        void VRI_CALL CmdSetShadingRate(VriCommandBuffer* cmd, const VriShadingRateDesc* desc)
        {
            if (!desc) return;
            CommandBufferD3D12* c = CB(cmd);
            Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList5> list5;
            if (FAILED(c->list.As(&list5)))
                return;
            const D3D12_SHADING_RATE_COMBINER combiners[2] = {
                ToD3DCombiner(desc->primitiveCombiner),
                ToD3DCombiner(desc->attachmentCombiner),
            };
            list5->RSSetShadingRate(ToD3DShadingRate(desc->shadingRate), combiners);
        }

        const VriShadingRateInterface g_vrsD3D12 = {
            CmdSetShadingRate,
        };
    } // namespace

    const VriShadingRateInterface* GetShadingRateInterfaceD3D12() { return &g_vrsD3D12; }
} // namespace vri::d3d12
