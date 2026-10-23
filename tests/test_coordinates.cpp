#include <vri/vri.h>

#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace
{
    std::array<float, 4>
    Project(const VriCoordinateConventions& c, const float matrix[16], const std::array<float, 4>& position)
    {
        std::array<float, 4> output {};
        for (uint32_t row = 0; row < 4; ++row)
        {
            for (uint32_t column = 0; column < 4; ++column)
            {
                uint32_t r   = row;
                uint32_t col = column;
                if (c.vectorConvention == VriVectorConvention_RowVector)
                {
                    r   = column;
                    col = row;
                }
                const uint32_t index = c.matrixLayout == VriMatrixLayout_ColumnMajor ? col * 4 + r : r * 4 + col;
                output[row] += matrix[index] * position[column];
            }
        }
        return output;
    }
} // namespace

TEST_CASE("coordinate presets are editable values and creation rejects stale ABI")
{
    const VriDeviceCreationDesc initial = vriDefaultDeviceCreationDesc();
    CHECK(initial.structSize == sizeof(initial));
    CHECK(initial.coordinates.clipSpaceY == VriClipSpaceY_Down);
    VriCoordinateConventions direct3D {};
    REQUIRE(vriGetConventionPreset(VriConventionPreset_Direct3D, &direct3D) == VriResult_Success);
    CHECK(direct3D.clipSpaceY == VriClipSpaceY_Up);
    CHECK(direct3D.handedness == VriHandedness_LeftHanded);
    direct3D.textureOrigin = VriTextureOrigin_BottomLeft;
    CHECK(initial.coordinates.textureOrigin == VriTextureOrigin_TopLeft);
    CHECK(vriGetConventionPreset(VriConventionPreset_MaxEnum, &direct3D) == VriResult_InvalidArgument);
    CHECK(vriGetCoordinateConventions(nullptr, &direct3D) == VriResult_InvalidArgument);
    VriDeviceCreationDesc invalid = initial;
    invalid.structSize            = 0;
    VriDevice* device             = reinterpret_cast<VriDevice*>(static_cast<uintptr_t>(1));
    CHECK(vriCreateDevice(&invalid, &device) == VriResult_InvalidArgument);
    CHECK(device == nullptr);
    invalid                        = initial;
    invalid.coordinates.clipSpaceY = VriClipSpaceY_MaxEnum;
    CHECK(vriCreateDevice(&invalid, &device) == VriResult_InvalidArgument);
    CHECK(device == nullptr);
}

TEST_CASE("projection policies preserve near far and off-center terms for every matrix convention")
{
    for (uint32_t bits = 0; bits < 64; ++bits)
    {
        CAPTURE(bits);
        VriCoordinateConventions c {};
        c.clipSpaceY       = bits & 1 ? VriClipSpaceY_Up : VriClipSpaceY_Down;
        c.clipDepthRange   = bits & 2 ? VriClipDepthRange_NegativeOneToOne : VriClipDepthRange_ZeroToOne;
        c.handedness       = bits & 4 ? VriHandedness_LeftHanded : VriHandedness_RightHanded;
        c.matrixLayout     = bits & 8 ? VriMatrixLayout_RowMajor : VriMatrixLayout_ColumnMajor;
        c.vectorConvention = bits & 16 ? VriVectorConvention_RowVector : VriVectorConvention_ColumnVector;
        VriPerspectiveProjectionDesc desc {1.1f, 1.5f, 0.25f, 40.0f, bits & 32 ? VRI_TRUE : VRI_FALSE};
        float                        matrix[16] {};
        REQUIRE(vriMakePerspectiveProjection(&c, &desc, matrix) == VriResult_Success);
        const float sign      = c.handedness == VriHandedness_RightHanded ? -1.0f : 1.0f;
        const auto  near      = Project(c, matrix, {0, 0, sign * desc.nearPlane, 1});
        const auto  far       = Project(c, matrix, {0, 0, sign * desc.farPlane, 1});
        const float nearDepth = c.clipDepthRange == VriClipDepthRange_ZeroToOne ? 0.0f : -1.0f;
        CHECK(near[2] / near[3] == doctest::Approx(desc.reversedDepth ? 1.0f : nearDepth).epsilon(0.00001));
        CHECK(far[2] / far[3] == doctest::Approx(desc.reversedDepth ? nearDepth : 1.0f).epsilon(0.00001));
        const auto up = Project(c, matrix, {0, 0.1f, sign, 1});
        CHECK((up[1] > 0) == (c.clipSpaceY == VriClipSpaceY_Up));

        // An arbitrary asymmetric matrix catches partial-row/sign fixes that only
        // happen to work for a centered projection.
        for (uint32_t i = 0; i < 16; ++i)
        {
            matrix[i] = 0.1f * static_cast<float>(i + 1);
        }
        const auto             before = Project(c, matrix, {0.4f, -0.2f, 0.7f, 1});
        VriCoordinateTransform transform {-1.0f, 0.5f, 0.5f, -1.0f, 1.0f};
        REQUIRE(vriTransformProjection(&c, &transform, matrix) == VriResult_Success);
        const auto after = Project(c, matrix, {0.4f, -0.2f, 0.7f, 1});
        CHECK(after[0] == doctest::Approx(before[0]));
        CHECK(after[1] == doctest::Approx(-before[1]));
        CHECK(after[2] == doctest::Approx(0.5f * (before[2] + before[3])));
        CHECK(after[3] == doctest::Approx(before[3]));
    }
    VriCoordinateConventions     conventions {};
    VriPerspectiveProjectionDesc invalid {0, 1, 0.1f, 10, VRI_FALSE};
    float                        output[16];
    CHECK(vriMakePerspectiveProjection(&conventions, &invalid, output) == VriResult_InvalidArgument);
    invalid.verticalFovRadians = std::numeric_limits<float>::quiet_NaN();
    CHECK(vriMakePerspectiveProjection(&conventions, &invalid, output) == VriResult_InvalidArgument);
}
