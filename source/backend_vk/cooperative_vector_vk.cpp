#include "cooperative_vector_vk.h"
#include "conversions_vk.h"
#include "device_vk.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

namespace vri::vk
{
    namespace
    {
        struct Component
        {
            VriComponentType   type;
            VkComponentTypeKHR native;
            size_t             bytes;
        };

        constexpr Component kComponents[] = {
            {VriComponentType_Float16, VK_COMPONENT_TYPE_FLOAT16_KHR, 2},
            {VriComponentType_Float32, VK_COMPONENT_TYPE_FLOAT32_KHR, 4},
            {VriComponentType_Float64, VK_COMPONENT_TYPE_FLOAT64_KHR, 8},
            {VriComponentType_Sint8, VK_COMPONENT_TYPE_SINT8_KHR, 1},
            {VriComponentType_Sint16, VK_COMPONENT_TYPE_SINT16_KHR, 2},
            {VriComponentType_Sint32, VK_COMPONENT_TYPE_SINT32_KHR, 4},
            {VriComponentType_Sint64, VK_COMPONENT_TYPE_SINT64_KHR, 8},
            {VriComponentType_Uint8, VK_COMPONENT_TYPE_UINT8_KHR, 1},
            {VriComponentType_Uint16, VK_COMPONENT_TYPE_UINT16_KHR, 2},
            {VriComponentType_Uint32, VK_COMPONENT_TYPE_UINT32_KHR, 4},
            {VriComponentType_Uint64, VK_COMPONENT_TYPE_UINT64_KHR, 8},
            {VriComponentType_Sint8Packed, VK_COMPONENT_TYPE_SINT8_PACKED_NV, 0},
            {VriComponentType_Uint8Packed, VK_COMPONENT_TYPE_UINT8_PACKED_NV, 0},
            {VriComponentType_Float8E4M3, VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT, 1},
            {VriComponentType_Float8E5M2, VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT, 1},
            {VriComponentType_Bfloat16, VK_COMPONENT_TYPE_BFLOAT16_KHR, 2},
        };

        VriComponentType FromVk(VkComponentTypeKHR type)
        {
            for (const auto& component : kComponents)
            {
                if (component.native == type)
                    return component.type;
            }
            return VriComponentType_MaxEnum;
        }

        DeviceVK* Dev(VriDevice* device) { return reinterpret_cast<DeviceVK*>(device); }

        VriResult VRI_CALL GetCooperativeVectorDeviceProperties(VriDevice*                            device,
                                                                VriCooperativeVectorDeviceProperties* out)
        {
            if (!device || !out)
                return VriResult_InvalidArgument;
            *out = Dev(device)->CooperativeVectorDeviceProperties();
            return VriResult_Success;
        }

        VriResult VRI_CALL GetCooperativeVectorProperties(VriDevice*                      device,
                                                          uint32_t*                       count,
                                                          VriCooperativeVectorProperties* out)
        {
            if (!device || !count)
                return VriResult_InvalidArgument;
            const auto& combinations = Dev(device)->CooperativeVectorCombinations();
            const auto  capacity     = out ? *count : 0;
            *count                   = static_cast<uint32_t>(combinations.size());
            if (!out)
                return VriResult_Success;
            if (capacity < *count)
                return VriResult_InvalidArgument;
            std::copy(combinations.begin(), combinations.end(), out);
            return VriResult_Success;
        }

        bool ValidLayout(VriCooperativeVectorMatrixLayout layout,
                         size_t                           stride,
                         size_t                           elementSize,
                         uint32_t                         rows,
                         uint32_t                         columns)
        {
            if (layout == VriCooperativeVectorMatrixLayout_InferencingOptimal)
                return true;
            if (layout != VriCooperativeVectorMatrixLayout_RowMajor &&
                layout != VriCooperativeVectorMatrixLayout_ColumnMajor)
                return false;
            const uint32_t elements = layout == VriCooperativeVectorMatrixLayout_RowMajor ? columns : rows;
            const uint32_t count    = layout == VriCooperativeVectorMatrixLayout_RowMajor ? rows : columns;
            return stride % elementSize == 0 && stride >= static_cast<size_t>(elements) * elementSize &&
                   stride <= std::numeric_limits<size_t>::max() / count;
        }

        VriResult VRI_CALL ConvertCooperativeVectorMatrix(VriDevice*                                   device,
                                                          const VriConvertCooperativeVectorMatrixInfo* pInfo)
        {
            if (!device || !pInfo || !pInfo->pDstSize || (!pInfo->pSrcData && pInfo->pDstData))
                return VriResult_InvalidArgument;
            DeviceVK&  d     = *Dev(device);
            const auto limit = d.CooperativeVectorDeviceProperties().maxCooperativeVectorComponents;
            if (!pInfo->numRows || !pInfo->numColumns || pInfo->numRows > limit || pInfo->numColumns > limit)
                return VriResult_InvalidArgument;
            const auto* srcComponent = std::find_if(std::begin(kComponents), std::end(kComponents), [&](const auto& c) {
                return c.type == pInfo->srcComponentType;
            });
            const auto* dstComponent = std::find_if(std::begin(kComponents), std::end(kComponents), [&](const auto& c) {
                return c.type == pInfo->dstComponentType;
            });
            if (srcComponent == std::end(kComponents) || dstComponent == std::end(kComponents) ||
                !srcComponent->bytes || !dstComponent->bytes)
                return VriResult_InvalidArgument;
            if (!ValidLayout(
                    pInfo->srcLayout, pInfo->srcStride, srcComponent->bytes, pInfo->numRows, pInfo->numColumns) ||
                !ValidLayout(
                    pInfo->dstLayout, pInfo->dstStride, dstComponent->bytes, pInfo->numRows, pInfo->numColumns))
                return VriResult_InvalidArgument;
            const auto& combinations        = d.CooperativeVectorCombinations();
            const auto  matrixTypeSupported = [&](VriComponentType type) {
                return type == VriComponentType_Float32 ||
                       std::any_of(combinations.begin(), combinations.end(), [&](const auto& c) {
                           return c.matrixInterpretation == type;
                       });
            };
            if (!matrixTypeSupported(srcComponent->type) || !matrixTypeSupported(dstComponent->type))
                return VriResult_Unsupported;
            const auto float8 = [](VriComponentType type) {
                return type == VriComponentType_Float8E4M3 || type == VriComponentType_Float8E5M2;
            };
            const auto float16or32 = [](VriComponentType type) {
                return type == VriComponentType_Float16 || type == VriComponentType_Float32;
            };
            if (srcComponent->type != dstComponent->type &&
                !(float16or32(srcComponent->type) && float16or32(dstComponent->type)) &&
                !(float16or32(srcComponent->type) && float8(dstComponent->type)) &&
                !(float8(srcComponent->type) && float16or32(dstComponent->type)))
                return VriResult_Unsupported;
            if (float8(dstComponent->type) && pInfo->dstLayout != VriCooperativeVectorMatrixLayout_InferencingOptimal)
                return VriResult_InvalidArgument;

            VkConvertCooperativeVectorMatrixInfoNV info {VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV};
            info.srcComponentType                             = srcComponent->native;
            info.dstComponentType                             = dstComponent->native;
            info.numRows                                      = pInfo->numRows;
            info.numColumns                                   = pInfo->numColumns;
            const VkCooperativeVectorMatrixLayoutNV layouts[] = {
                VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV,
                VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_COLUMN_MAJOR_NV,
                VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV};
            info.srcLayout     = layouts[pInfo->srcLayout];
            info.srcStride     = pInfo->srcStride;
            info.dstLayout     = layouts[pInfo->dstLayout];
            info.dstStride     = pInfo->dstStride;
            size_t required    = 0;
            info.pDstSize      = &required;
            const auto convert = d.Ext().convertCooperativeVectorMatrix;
            if (convert(d.Device(), &info) != VK_SUCCESS)
                return VriResult_Failure;
            const auto capacity = pInfo->pDstData ? *pInfo->pDstSize : 0;
            *pInfo->pDstSize    = required;
            if (!pInfo->pDstData)
                return VriResult_Success;
            if (capacity < required)
                return VriResult_InvalidArgument;

            size_t sourceRequired = 0;
            if (pInfo->srcLayout == VriCooperativeVectorMatrixLayout_InferencingOptimal)
            {
                auto sourceQuery             = info;
                sourceQuery.dstComponentType = info.srcComponentType;
                sourceQuery.dstLayout        = info.srcLayout;
                sourceQuery.dstStride        = info.srcStride;
                sourceQuery.pDstSize         = &sourceRequired;
                if (convert(d.Device(), &sourceQuery) != VK_SUCCESS)
                    return VriResult_Failure;
            }
            else
            {
                const bool rowMajor = pInfo->srcLayout == VriCooperativeVectorMatrixLayout_RowMajor;
                const auto count    = rowMajor ? pInfo->numRows : pInfo->numColumns;
                const auto elements = rowMajor ? pInfo->numColumns : pInfo->numRows;
                sourceRequired      = (count - 1) * pInfo->srcStride + elements * srcComponent->bytes;
            }
            if (pInfo->srcSize < sourceRequired)
                return VriResult_InvalidArgument;
            const auto srcAddress = reinterpret_cast<uintptr_t>(pInfo->pSrcData);
            const auto dstAddress = reinterpret_cast<uintptr_t>(pInfo->pDstData);
            const bool overlaps   = srcAddress <= dstAddress ? dstAddress - srcAddress < pInfo->srcSize :
                                                               srcAddress - dstAddress < required;
            if (overlaps)
                return VriResult_InvalidArgument;
            info.srcSize             = pInfo->srcSize;
            info.srcData.hostAddress = pInfo->pSrcData;
            info.dstData.hostAddress = pInfo->pDstData;
            info.pDstSize            = pInfo->pDstSize;
            const VkResult result    = convert(d.Device(), &info);
            if (result == VK_ERROR_OUT_OF_HOST_MEMORY)
                return VriResult_OutOfMemory;
            return result == VK_SUCCESS ? VriResult_Success : VriResult_Failure;
        }

        const VriCooperativeVectorInterface kInterface = {GetCooperativeVectorDeviceProperties,
                                                          GetCooperativeVectorProperties,
                                                          ConvertCooperativeVectorMatrix};
    } // namespace

    VriResult DeviceVK::InitializeCooperativeVectors()
    {
        if (!m_ext.getCooperativeVectorProperties || !m_ext.convertCooperativeVectorMatrix)
        {
            ReportError("cooperative vector extension entry points are unavailable");
            return VriResult_Failure;
        }
        VkPhysicalDeviceCooperativeVectorPropertiesNV properties {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_PROPERTIES_NV};
        VkPhysicalDeviceProperties2 query {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        query.pNext = &properties;
        vkGetPhysicalDeviceProperties2(m_physicalDevice, &query);
        m_cooperativeVectorDeviceProperties.maxCooperativeVectorComponents = properties.maxCooperativeVectorComponents;
        for (uint32_t stage = VriShaderStage_Vertex; stage <= VriShaderStage_Task; stage <<= 1)
        {
            if (properties.cooperativeVectorSupportedStages & ToVkShaderStage(static_cast<VriShaderStageBits>(stage)))
                m_cooperativeVectorDeviceProperties.cooperativeVectorSupportedStages |= stage;
        }
        uint32_t count = 0;
        auto     get   = m_ext.getCooperativeVectorProperties;
        if (get(m_physicalDevice, &count, nullptr) != VK_SUCCESS || !count)
        {
            ReportError("query cooperative vector combinations failed");
            return VriResult_Failure;
        }
        std::vector<VkCooperativeVectorPropertiesNV> native(count);
        for (auto& combination : native)
            combination.sType = VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV;
        if (get(m_physicalDevice, &count, native.data()) != VK_SUCCESS)
        {
            ReportError("read cooperative vector combinations failed");
            return VriResult_Failure;
        }
        m_cooperativeVectorCombinations.reserve(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            const auto&                    c = native[i];
            VriCooperativeVectorProperties mapped {FromVk(c.inputType),
                                                   FromVk(c.inputInterpretation),
                                                   FromVk(c.matrixInterpretation),
                                                   FromVk(c.biasInterpretation),
                                                   FromVk(c.resultType),
                                                   c.transpose};
            if (mapped.inputType == VriComponentType_MaxEnum ||
                mapped.inputInterpretation == VriComponentType_MaxEnum ||
                mapped.matrixInterpretation == VriComponentType_MaxEnum ||
                mapped.biasInterpretation == VriComponentType_MaxEnum || mapped.resultType == VriComponentType_MaxEnum)
            {
                ReportError("driver reported an unknown cooperative vector component type");
                return VriResult_Unsupported;
            }
            m_cooperativeVectorCombinations.push_back(mapped);
        }
        return VriResult_Success;
    }

    const VriCooperativeVectorInterface* GetCooperativeVectorInterfaceVK() { return &kInterface; }
} // namespace vri::vk
