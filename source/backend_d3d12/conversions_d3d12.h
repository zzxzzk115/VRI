// conversions_d3d12.h - VRI enum -> D3D12/DXGI mappings.
#pragma once

#include <d3d12.h>
#include <dxgi.h>

#include <vri/vri.h>

namespace vri::d3d12
{
    struct DxgiFormatInfo
    {
        DXGI_FORMAT format;
        uint32_t    texelSize; // bytes per texel (for readback row math)
    };

    inline DxgiFormatInfo ToDxgiFormat(VriFormat f)
    {
        switch (f)
        {
            case VriFormat_RGBA8_UNORM:  return {DXGI_FORMAT_R8G8B8A8_UNORM, 4};
            case VriFormat_RGBA8_SRGB:   return {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 4};
            case VriFormat_BGRA8_UNORM:  return {DXGI_FORMAT_B8G8R8A8_UNORM, 4};
            case VriFormat_RG8_UNORM:    return {DXGI_FORMAT_R8G8_UNORM, 2};
            case VriFormat_R8_UNORM:     return {DXGI_FORMAT_R8_UNORM, 1};
            case VriFormat_RGBA16_SFLOAT:return {DXGI_FORMAT_R16G16B16A16_FLOAT, 8};
            case VriFormat_RGBA32_SFLOAT:return {DXGI_FORMAT_R32G32B32A32_FLOAT, 16};
            case VriFormat_RG32_SFLOAT:  return {DXGI_FORMAT_R32G32_FLOAT, 8};
            case VriFormat_RGB32_SFLOAT: return {DXGI_FORMAT_R32G32B32_FLOAT, 12};
            case VriFormat_R32_SFLOAT:   return {DXGI_FORMAT_R32_FLOAT, 4};
            case VriFormat_D32_SFLOAT:   return {DXGI_FORMAT_D32_FLOAT, 4};
            case VriFormat_D24_UNORM_S8_UINT: return {DXGI_FORMAT_D24_UNORM_S8_UINT, 4};
            default:                     return {DXGI_FORMAT_R8G8B8A8_UNORM, 4};
        }
    }
} // namespace vri::d3d12
