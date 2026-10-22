/*
 * Cooperative-vector inference for small shader networks. Enable
 * VriFeature_CooperativeVector, check hasCooperativeVector, then query this table.
 * No network object or inference runtime is required: shaders use Slang CoopVec.
 * The first implementation is Vulkan VK_NV_cooperative_vector; other backends
 * return Unsupported. Training and GPU-side matrix conversion are not exposed.
 */
#ifndef VRI_EXT_COOPERATIVE_VECTOR_H
#define VRI_EXT_COOPERATIVE_VECTOR_H

#include "../vri_base.h"
#include "../vri_flags.h"
#include "../vri_handles.h"

VRI_EXTERN_C_BEGIN

typedef enum VriComponentType
{
    VriComponentType_Float16 = 0,
    VriComponentType_Float32,
    VriComponentType_Float64,
    VriComponentType_Sint8,
    VriComponentType_Sint16,
    VriComponentType_Sint32,
    VriComponentType_Sint64,
    VriComponentType_Uint8,
    VriComponentType_Uint16,
    VriComponentType_Uint32,
    VriComponentType_Uint64,
    VriComponentType_Sint8Packed,
    VriComponentType_Uint8Packed,
    VriComponentType_Float8E4M3,
    VriComponentType_Float8E5M2,
    VriComponentType_Bfloat16,
    VriComponentType_MaxEnum = 0x7fffffff
} VriComponentType;

typedef enum VriCooperativeVectorMatrixLayout
{
    VriCooperativeVectorMatrixLayout_RowMajor = 0,
    VriCooperativeVectorMatrixLayout_ColumnMajor,
    VriCooperativeVectorMatrixLayout_InferencingOptimal,
    VriCooperativeVectorMatrixLayout_MaxEnum = 0x7fffffff
} VriCooperativeVectorMatrixLayout;

typedef struct VriCooperativeVectorDeviceProperties
{
    /* Other stage features must also be enabled, e.g. ray tracing. */
    VriShaderStageFlags cooperativeVectorSupportedStages;
    uint32_t            maxCooperativeVectorComponents;
} VriCooperativeVectorDeviceProperties;

/* A complete supported tuple, not independently combinable type lists.
 * inputType is the shader vector's element type; inputInterpretation describes
 * packed values. Transpose is only allowed with an optimal matrix layout. */
typedef struct VriCooperativeVectorProperties
{
    VriComponentType inputType;
    VriComponentType inputInterpretation;
    VriComponentType matrixInterpretation;
    VriComponentType biasInterpretation;
    VriComponentType resultType;
    VriBool transpose; /* Supports transposing optimal-layout matrices; non-transposed use is always allowed. */
} VriCooperativeVectorProperties;

/* Host addresses are borrowed only for the conversion call. GPU conversion is
 * a separate command and is intentionally not part of this host-only interface. */
typedef struct VriConvertCooperativeVectorMatrixInfo
{
    size_t                           srcSize;
    const void*                      pSrcData;
    size_t*                          pDstSize;
    void*                            pDstData;
    VriComponentType                 srcComponentType;
    VriComponentType                 dstComponentType;
    uint32_t                         numRows;
    uint32_t                         numColumns;
    VriCooperativeVectorMatrixLayout srcLayout;
    size_t                           srcStride; /* Bytes between rows/columns; ignored for optimal layout. */
    VriCooperativeVectorMatrixLayout dstLayout;
    size_t                           dstStride;
} VriConvertCooperativeVectorMatrixInfo;

typedef struct VriCooperativeVectorInterface
{
    VriResult(VRI_CALL* GetCooperativeVectorDeviceProperties)(VriDevice*                            device,
                                                              VriCooperativeVectorDeviceProperties* pProperties);

    /* NULL pProperties queries the required count. Otherwise *pPropertyCount is
     * capacity. Insufficient capacity returns InvalidArgument, updates the count
     * and writes no entries. Properties are complete, supported type tuples. */
    VriResult(VRI_CALL* GetCooperativeVectorProperties)(VriDevice*                      device,
                                                        uint32_t*                       pPropertyCount,
                                                        VriCooperativeVectorProperties* pProperties);

    /* NULL pDstData queries the byte size; pSrcData may then also be NULL.
     * Otherwise *pDstSize is capacity. Insufficient capacity returns
     * InvalidArgument, updates *pDstSize and writes no data. Source/destination
     * must not overlap. Supports same-type layout conversion, Float32/Float16
     * conversion and Float32/Float16 <-> supported Float8 conversion.
     * Optimal bytes are device/driver-specific; convert portable weights on load. */
    VriResult(VRI_CALL* ConvertCooperativeVectorMatrix)(VriDevice*                                   device,
                                                        const VriConvertCooperativeVectorMatrixInfo* pInfo);
} VriCooperativeVectorInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_COOPERATIVE_VECTOR_H */
