#include <vri/vri.h>

_Static_assert(sizeof(VriClipSpaceY) == sizeof(int32_t), "Coordinate enums must remain 32-bit");
_Static_assert(sizeof(VriCoordinateConventions) == 24, "Convention layout must remain fixed");
_Static_assert(sizeof(VriCoordinateTransform) == 20, "Transform layout must remain fixed");

int vriTestCoordinatesC(VriDevice* device, size_t creationSize, size_t conventionsSize)
{
    VriCoordinateConventions     conventions = {0};
    VriCoordinateTransform       transform   = {0};
    VriPerspectiveProjectionDesc projection  = {1.0f, 1.5f, 0.1f, 100.0f, VRI_FALSE};
    VriDeviceCreationDesc        creation    = vriDefaultDeviceCreationDesc();
    float                        matrix[16];
    if (creationSize != sizeof(creation) || conventionsSize != sizeof(conventions) ||
        vriGetCoordinateConventions(device, &conventions) != VriResult_Success ||
        vriGetCoordinateTransform(device, NULL, &transform) != VriResult_Success ||
        vriMakePerspectiveProjection(&conventions, &projection, matrix) != VriResult_Success ||
        vriTransformProjection(&conventions, &transform, matrix) != VriResult_Success)
    {
        return __LINE__;
    }
    return 0;
}
