#include <vri/vri.h>

#include <cmath>
#include <cstring>
#include <numbers>

namespace vri::core
{
    bool ValidConventions(const VriCoordinateConventions& c)
    {
        return (c.clipSpaceY == VriClipSpaceY_Down || c.clipSpaceY == VriClipSpaceY_Up) &&
               (c.clipDepthRange == VriClipDepthRange_ZeroToOne ||
                c.clipDepthRange == VriClipDepthRange_NegativeOneToOne) &&
               (c.textureOrigin == VriTextureOrigin_TopLeft || c.textureOrigin == VriTextureOrigin_BottomLeft) &&
               (c.matrixLayout == VriMatrixLayout_ColumnMajor || c.matrixLayout == VriMatrixLayout_RowMajor) &&
               (c.vectorConvention == VriVectorConvention_ColumnVector ||
                c.vectorConvention == VriVectorConvention_RowVector) &&
               (c.handedness == VriHandedness_RightHanded || c.handedness == VriHandedness_LeftHanded);
    }
} // namespace vri::core

namespace
{
    uint32_t MatrixIndex(const VriCoordinateConventions& conventions, uint32_t row, uint32_t column)
    {
        // Work in column-vector algebra; a row-vector matrix is its transpose.
        const bool columnStorage = conventions.matrixLayout == VriMatrixLayout_ColumnMajor;
        const bool columnVectors = conventions.vectorConvention == VriVectorConvention_ColumnVector;
        return columnStorage == columnVectors ? column * 4 + row : row * 4 + column;
    }

    const VriDeviceDesc* DeviceProperties(const VriDevice* device)
    {
        VriCoreInterface core {};
        if (vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(core), &core) != VriResult_Success)
        {
            return nullptr;
        }
        return core.GetDeviceDesc(device);
    }
} // namespace

extern "C"
{
    VriResult VRI_CALL vriGetConventionPreset(VriConventionPreset preset, VriCoordinateConventions* out)
    {
        if (!out || preset < VriConventionPreset_Vulkan || preset > VriConventionPreset_Metal)
        {
            return VriResult_InvalidArgument;
        }
        VriCoordinateConventions conventions {};
        if (preset != VriConventionPreset_Vulkan)
        {
            conventions.clipSpaceY = VriClipSpaceY_Up;
        }
        if (preset == VriConventionPreset_Direct3D)
        {
            conventions.handedness = VriHandedness_LeftHanded;
        }
        *out = conventions;
        return VriResult_Success;
    }

    VriResult VRI_CALL vriGetCoordinateConventions(const VriDevice* device, VriCoordinateConventions* out)
    {
        if (!out)
        {
            return VriResult_InvalidArgument;
        }
        const VriDeviceDesc* properties = DeviceProperties(device);
        if (!properties)
        {
            return VriResult_InvalidArgument;
        }
        *out = properties->coordinates;
        return VriResult_Success;
    }

    VriResult VRI_CALL vriGetCoordinateTransform(const VriDevice*                device,
                                                 const VriCoordinateConventions* source,
                                                 VriCoordinateTransform*         out)
    {
        if (!out)
        {
            return VriResult_InvalidArgument;
        }
        const VriDeviceDesc* properties = DeviceProperties(device);
        if (!properties)
        {
            return VriResult_InvalidArgument;
        }
        const auto& conventions = source ? *source : properties->coordinates;
        if (!vri::core::ValidConventions(conventions))
        {
            return VriResult_InvalidArgument;
        }
        const bool vulkan =
            properties->graphicsAPI == VriGraphicsAPI_Vulkan || properties->graphicsAPI == VriGraphicsAPI_Software;
        // Vulkan implements the requested Y convention with viewport height.
        // The other rasterizers use their existing portable Y-up shader contract.
        const VriClipSpaceY    rasterY = vulkan ? properties->coordinates.clipSpaceY : VriClipSpaceY_Up;
        VriCoordinateTransform transform {};
        transform.clipYScale = conventions.clipSpaceY == rasterY ? 1.0f : -1.0f;
        transform.clipZScale = conventions.clipDepthRange == VriClipDepthRange_ZeroToOne ? 1.0f : 0.5f;
        transform.clipZBias  = conventions.clipDepthRange == VriClipDepthRange_ZeroToOne ? 0.0f : 0.5f;
        transform.uvYScale   = conventions.textureOrigin == VriTextureOrigin_TopLeft ? 1.0f : -1.0f;
        transform.uvYBias    = conventions.textureOrigin == VriTextureOrigin_TopLeft ? 0.0f : 1.0f;
        *out                 = transform;
        return VriResult_Success;
    }

    VriResult VRI_CALL vriMakePerspectiveProjection(const VriCoordinateConventions*     conventions,
                                                    const VriPerspectiveProjectionDesc* desc,
                                                    float                               out[16])
    {
        if (!conventions || !desc || !out || !vri::core::ValidConventions(*conventions) ||
            !std::isfinite(desc->verticalFovRadians) || desc->verticalFovRadians <= 0.0f ||
            desc->verticalFovRadians >= std::numbers::pi_v<float> || !std::isfinite(desc->aspectRatio) ||
            desc->aspectRatio <= 0.0f || !std::isfinite(desc->nearPlane) || desc->nearPlane <= 0.0f ||
            !std::isfinite(desc->farPlane) || desc->farPlane <= desc->nearPlane ||
            (desc->reversedDepth != VRI_FALSE && desc->reversedDepth != VRI_TRUE))
        {
            return VriResult_InvalidArgument;
        }
        const float nearDepth = conventions->clipDepthRange == VriClipDepthRange_ZeroToOne ? 0.0f : -1.0f;
        const float z0        = desc->reversedDepth ? 1.0f : nearDepth;
        const float z1        = desc->reversedDepth ? nearDepth : 1.0f;
        const float sign      = conventions->handedness == VriHandedness_RightHanded ? -1.0f : 1.0f;
        const float y         = 1.0f / std::tan(desc->verticalFovRadians * 0.5f);
        const float depth     = desc->farPlane - desc->nearPlane;
        float       matrix[16] {};
        matrix[MatrixIndex(*conventions, 0, 0)] = y / desc->aspectRatio;
        matrix[MatrixIndex(*conventions, 1, 1)] = conventions->clipSpaceY == VriClipSpaceY_Down ? -y : y;
        matrix[MatrixIndex(*conventions, 2, 2)] = sign * (desc->farPlane * z1 - desc->nearPlane * z0) / depth;
        matrix[MatrixIndex(*conventions, 2, 3)] = desc->nearPlane * desc->farPlane * (z0 - z1) / depth;
        matrix[MatrixIndex(*conventions, 3, 2)] = sign;
        std::memcpy(out, matrix, sizeof(matrix));
        return VriResult_Success;
    }

    VriResult VRI_CALL vriTransformProjection(const VriCoordinateConventions* conventions,
                                              const VriCoordinateTransform*   transform,
                                              float                           matrix[16])
    {
        if (!conventions || !transform || !matrix || !vri::core::ValidConventions(*conventions) ||
            !std::isfinite(transform->clipYScale) || !std::isfinite(transform->clipZScale) ||
            !std::isfinite(transform->clipZBias))
        {
            return VriResult_InvalidArgument;
        }
        for (uint32_t column = 0; column < 4; ++column)
        {
            matrix[MatrixIndex(*conventions, 1, column)] *= transform->clipYScale;
            const uint32_t zIndex = MatrixIndex(*conventions, 2, column);
            matrix[zIndex]        = matrix[zIndex] * transform->clipZScale +
                             matrix[MatrixIndex(*conventions, 3, column)] * transform->clipZBias;
        }
        return VriResult_Success;
    }
}
