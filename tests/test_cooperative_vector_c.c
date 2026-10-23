#include <vri/vri.h>

_Static_assert(sizeof(VriResult) == sizeof(int32_t), "Result width must be stable with short enums");
_Static_assert(sizeof(VriComponentType) == sizeof(int32_t), "Component width must be stable with short enums");

/* The device is borrowed from the C++ fixture. All interface calls and descriptor reads here are C. */
int vriTestCooperativeVectorC(VriDevice* device, size_t tableSize, size_t infoSize, uint32_t* propertyCount)
{
    VriCoreInterface                     core       = {0};
    VriCooperativeVectorInterface        neural     = {0};
    VriCooperativeVectorDeviceProperties properties = {0};
    VriCooperativeVectorProperties       first      = {0};
    uint32_t                             capacity   = 1;
    VriResult                            result;
    const VriDeviceDesc*                 description;

    if (tableSize != sizeof(neural) || infoSize != sizeof(VriConvertCooperativeVectorMatrixInfo))
    {
        return __LINE__;
    }
    if (vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(core), &core) != VriResult_Success ||
        vriGetInterface(device, VRI_INTERFACE_COOPERATIVE_VECTOR, sizeof(neural), &neural) != VriResult_Success)
    {
        return __LINE__;
    }
    description = core.GetDeviceDesc(device);
    if (!description || !description->hasCooperativeVector ||
        !(description->enabledFeatures & VriFeature_CooperativeVector))
    {
        return __LINE__;
    }
    if (neural.GetCooperativeVectorDeviceProperties(device, &properties) != VriResult_Success ||
        properties.maxCooperativeVectorComponents == 0 ||
        !(properties.cooperativeVectorSupportedStages & VriShaderStage_Compute))
    {
        return __LINE__;
    }
    if (neural.GetCooperativeVectorProperties(device, propertyCount, NULL) != VriResult_Success || *propertyCount == 0)
    {
        return __LINE__;
    }
    first.transpose = 123;
    result          = neural.GetCooperativeVectorProperties(device, &capacity, &first);
    if (*propertyCount > 1)
    {
        if (result != VriResult_InvalidArgument || capacity != *propertyCount || first.transpose != 123)
        {
            return __LINE__;
        }
    }
    else if (result != VriResult_Success || capacity != 1 || first.transpose > 1)
    {
        return __LINE__;
    }
    return 0;
}
