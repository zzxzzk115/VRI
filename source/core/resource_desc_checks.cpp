#include "resource_desc_checks.h"

#include "format_traits.h"

#include <algorithm>
#include <bit>
#include <cstdint>

namespace vri::core
{
    namespace
    {
        const char* TypeName(VriTextureType type)
        {
            switch (type)
            {
                case VriTextureType_1D:
                    return "1D";
                case VriTextureType_1DArray:
                    return "1DArray";
                case VriTextureType_2D:
                    return "2D";
                case VriTextureType_2DArray:
                    return "2DArray";
                case VriTextureType_3D:
                    return "3D";
                case VriTextureType_Cube:
                    return "Cube";
                case VriTextureType_CubeArray:
                    return "CubeArray";
                default:
                    return "unknown";
            }
        }

        std::string Message(const char* function, const std::string& reason, const char* field)
        {
            return std::string(function) + ": " + reason + " (" + field + ")";
        }

        std::string Number(uint64_t value) { return std::to_string(value); }
    } // namespace

    std::string CheckTextureDesc(const char* function, const VriTextureDesc& desc)
    {
        const std::string type = TypeName(desc.type);

        // Every extent is at least 1: the backends used to turn 0 into 1, or pass it on to fail later.
        if (desc.width == 0)
            return Message(function, "a texture is at least 1 texel wide, not 0", "width");
        if (desc.height == 0)
            return Message(function, "a texture is at least 1 texel high, not 0", "height");
        if (desc.depth == 0)
            return Message(function, "a texture is at least 1 texel deep, not 0", "depth");
        if (desc.layerNum == 0)
            return Message(function, "a texture has at least 1 layer, not 0", "layerNum");
        if (desc.mipNum == 0)
            return Message(function, "a texture has at least 1 mip, not 0", "mipNum");
        if (desc.sampleNum == 0)
            return Message(function, "a texture has at least 1 sample, not 0", "sampleNum");

        // The layer count matches the type, so no backend has to reinterpret it.
        switch (desc.type)
        {
            case VriTextureType_1D:
            case VriTextureType_2D:
                if (desc.layerNum > 1)
                    return Message(function,
                                   "a " + type + " texture has 1 layer, not " + Number(desc.layerNum) + "; use " +
                                       type + "Array for more",
                                   "layerNum");
                break;
            case VriTextureType_Cube:
                if (desc.layerNum != 6)
                    return Message(function, "a Cube has 6 layers, not " + Number(desc.layerNum), "layerNum");
                break;
            case VriTextureType_CubeArray:
                if (desc.layerNum % 6 != 0)
                    return Message(function,
                                   "a CubeArray has a non-zero multiple of 6 layers, not " + Number(desc.layerNum),
                                   "layerNum");
                break;
            case VriTextureType_3D:
                if (desc.layerNum != 1)
                    return Message(function,
                                   "a 3D texture has 1 layer, not " + Number(desc.layerNum) + "; its slices are depth",
                                   "layerNum");
                break;
            default:
                break;
        }
        if (desc.type != VriTextureType_3D && desc.depth != 1)
            return Message(
                function, "only a 3D texture is deeper than 1, not " + Number(desc.depth) + " for a " + type, "depth");
        if ((desc.type == VriTextureType_1D || desc.type == VriTextureType_1DArray) && desc.height != 1)
            return Message(function, "a " + type + " texture is 1 texel high, not " + Number(desc.height), "height");

        // The full chain ends at 1x1x1: floor(log2(largest extent)) + 1 levels.
        uint32_t largest = std::max(desc.width, desc.height);
        if (desc.type == VriTextureType_3D)
            largest = std::max(largest, desc.depth);
        const uint32_t maxMips = static_cast<uint32_t>(std::bit_width(largest));
        if (desc.mipNum > maxMips)
            return Message(function,
                           "a texture " + Number(largest) + " texels across has at most " + Number(maxMips) +
                               " mips, not " + Number(desc.mipNum),
                           "mipNum");

        // Attachment usages follow the format's kind.
        if (FormatIsDepthStencil(desc.format))
        {
            if (desc.usage & VriTextureUsage_ColorAttachment)
                return Message(
                    function, "a depth/stencil format cannot be a color attachment", "usage ColorAttachment");
            if (desc.usage & VriTextureUsage_ShaderResourceStorage)
                return Message(
                    function, "a depth/stencil format cannot be a storage texture", "usage ShaderResourceStorage");
        }
        else if (desc.usage & VriTextureUsage_DepthStencilAttachment)
        {
            return Message(
                function, "a color format cannot be a depth/stencil attachment", "usage DepthStencilAttachment");
        }
        return {};
    }

    std::string CheckTextureCopy(const VriTexture* dst, const VriTexture* src, const VriTextureCopyDesc* region)
    {
        if (dst != src)
            return {};
        const VriTextureRegionDesc from = region ? region->src : VriTextureRegionDesc {};
        const VriTextureRegionDesc to   = region ? region->dst : VriTextureRegionDesc {};
        // The source region's layer count applies to both ends.
        const uint32_t layerNum = from.layerNum ? from.layerNum : 1;
        const uint32_t first    = (std::max)(from.baseLayer, to.baseLayer);
        if (from.mip != to.mip || first >= (std::min)(from.baseLayer, to.baseLayer) + layerNum)
            return {};
        return "CmdCopyTexture: the source and destination are the same subresource of one texture (mip " +
               Number(from.mip) + ", layer " + Number(first) +
               "); a copy within one texture must be between different mips or layers";
    }

    std::string CheckBufferDesc(const char* function, const VriBufferDesc& desc)
    {
        if (desc.size == 0)
            return Message(function, "a buffer is at least 1 byte, not 0", "size");
        if (desc.structureStride != 0 && desc.size % desc.structureStride != 0)
            return Message(function,
                           "size " + Number(desc.size) + " is not a multiple of structureStride " +
                               Number(desc.structureStride),
                           "size");
        return {};
    }
} // namespace vri::core
