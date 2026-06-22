/*
 * vri_enums.h - plain (non-flag) enumerations. Flag sets live in vri_flags.h,
 * formats in vri_format.h.
 *
 * Naming: VriThing_Value. Every enum carries _MaxEnum to fix the underlying
 * width across compilers. Vocabulary aligns with libvultra's RHI enums.
 */
#ifndef VRI_ENUMS_H
#define VRI_ENUMS_H

#include "vri_base.h"

VRI_EXTERN_C_BEGIN

/* Which graphics API a device is created against. */
typedef enum VriGraphicsAPI
{
    VriGraphicsAPI_None = 0,
    VriGraphicsAPI_Vulkan,
    VriGraphicsAPI_WebGPU,
    VriGraphicsAPI_OpenGL,
    VriGraphicsAPI_OpenGLES,
    VriGraphicsAPI_D3D11,
    VriGraphicsAPI_D3D12,
    VriGraphicsAPI_Metal,
    // Let VRI pick the best available backend for the platform (tries a platform-ordered
    // list of the compiled-in backends until one creates a device). Query the actual
    // choice afterwards via VriDeviceDesc::graphicsAPI.
    VriGraphicsAPI_Auto,
    VriGraphicsAPI_MaxEnum = 0x7fffffff
} VriGraphicsAPI;

typedef enum VriQueueType
{
    VriQueueType_Graphics = 0,
    VriQueueType_Compute,
    VriQueueType_Transfer,
    VriQueueType_Count,
    VriQueueType_MaxEnum = 0x7fffffff
} VriQueueType;

typedef enum VriAdapterType
{
    VriAdapterType_Unknown = 0,
    VriAdapterType_Discrete,
    VriAdapterType_Integrated,
    VriAdapterType_Software,
    VriAdapterType_MaxEnum = 0x7fffffff
} VriAdapterType;

/* ---- resources -------------------------------------------------------- */

typedef enum VriTextureType
{
    VriTextureType_1D = 0,
    VriTextureType_1DArray,
    VriTextureType_2D,
    VriTextureType_2DArray,
    VriTextureType_3D,
    VriTextureType_Cube,
    VriTextureType_CubeArray,
    VriTextureType_MaxEnum = 0x7fffffff
} VriTextureType;

/* View dimension for a created texture view / descriptor. */
typedef enum VriTextureViewType
{
    VriTextureViewType_1D = 0,
    VriTextureViewType_1DArray,
    VriTextureViewType_2D,
    VriTextureViewType_2DArray,
    VriTextureViewType_3D,
    VriTextureViewType_Cube,
    VriTextureViewType_CubeArray,
    VriTextureViewType_MaxEnum = 0x7fffffff
} VriTextureViewType;

/* Where a memory allocation lives / how it is accessed by the host. */
typedef enum VriMemoryLocation
{
    VriMemoryLocation_Device = 0,    /* GPU-only, fastest */
    VriMemoryLocation_DeviceUpload,  /* GPU-local, host-writable (ReBAR) */
    VriMemoryLocation_HostUpload,    /* host->device staging */
    VriMemoryLocation_HostReadback,  /* device->host readback */
    /* Create the resource UNBOUND: CreateBuffer/CreateTexture allocate no
       memory; the caller drives the explicit GetMemoryDesc -> AllocateMemory ->
       BindBufferMemory/BindTextureMemory path (enables aliasing / pooling). */
    VriMemoryLocation_Undefined,
    VriMemoryLocation_MaxEnum = 0x7fffffff
} VriMemoryLocation;

typedef enum VriIndexType
{
    VriIndexType_UInt16 = 0,
    VriIndexType_UInt32,
    VriIndexType_MaxEnum = 0x7fffffff
} VriIndexType;

/* ---- pipeline state --------------------------------------------------- */

typedef enum VriPrimitiveTopology
{
    VriPrimitiveTopology_PointList = 0,
    VriPrimitiveTopology_LineList,
    VriPrimitiveTopology_LineStrip,
    VriPrimitiveTopology_TriangleList,
    VriPrimitiveTopology_TriangleStrip,
    VriPrimitiveTopology_PatchList, /* tessellation control points (see VriTessellationDesc) */
    VriPrimitiveTopology_MaxEnum = 0x7fffffff
} VriPrimitiveTopology;

typedef enum VriPolygonMode
{
    VriPolygonMode_Fill = 0,
    VriPolygonMode_Line,
    VriPolygonMode_Point,
    VriPolygonMode_MaxEnum = 0x7fffffff
} VriPolygonMode;

typedef enum VriCullMode
{
    VriCullMode_None = 0,
    VriCullMode_Front,
    VriCullMode_Back,
    VriCullMode_MaxEnum = 0x7fffffff
} VriCullMode;

typedef enum VriFrontFace
{
    VriFrontFace_CounterClockwise = 0,
    VriFrontFace_Clockwise,
    VriFrontFace_MaxEnum = 0x7fffffff
} VriFrontFace;

typedef enum VriCompareOp
{
    VriCompareOp_Never = 0,
    VriCompareOp_Less,
    VriCompareOp_Equal,
    VriCompareOp_LessOrEqual,
    VriCompareOp_Greater,
    VriCompareOp_NotEqual,
    VriCompareOp_GreaterOrEqual,
    VriCompareOp_Always,
    VriCompareOp_MaxEnum = 0x7fffffff
} VriCompareOp;

typedef enum VriStencilOp
{
    VriStencilOp_Keep = 0,
    VriStencilOp_Zero,
    VriStencilOp_Replace,
    VriStencilOp_IncrementAndClamp,
    VriStencilOp_DecrementAndClamp,
    VriStencilOp_Invert,
    VriStencilOp_IncrementAndWrap,
    VriStencilOp_DecrementAndWrap,
    VriStencilOp_MaxEnum = 0x7fffffff
} VriStencilOp;

typedef enum VriBlendOp
{
    VriBlendOp_Add = 0,
    VriBlendOp_Subtract,
    VriBlendOp_ReverseSubtract,
    VriBlendOp_Min,
    VriBlendOp_Max,
    VriBlendOp_MaxEnum = 0x7fffffff
} VriBlendOp;

typedef enum VriBlendFactor
{
    VriBlendFactor_Zero = 0,
    VriBlendFactor_One,
    VriBlendFactor_SrcColor,
    VriBlendFactor_OneMinusSrcColor,
    VriBlendFactor_DstColor,
    VriBlendFactor_OneMinusDstColor,
    VriBlendFactor_SrcAlpha,
    VriBlendFactor_OneMinusSrcAlpha,
    VriBlendFactor_DstAlpha,
    VriBlendFactor_OneMinusDstAlpha,
    VriBlendFactor_ConstantColor,
    VriBlendFactor_OneMinusConstantColor,
    VriBlendFactor_ConstantAlpha,
    VriBlendFactor_OneMinusConstantAlpha,
    VriBlendFactor_SrcAlphaSaturate,
    VriBlendFactor_Src1Color,
    VriBlendFactor_OneMinusSrc1Color,
    VriBlendFactor_Src1Alpha,
    VriBlendFactor_OneMinusSrc1Alpha,
    VriBlendFactor_MaxEnum = 0x7fffffff
} VriBlendFactor;

/* Resource layout for barriers / attachments (explicit, Vulkan-like). */
typedef enum VriLayout
{
    VriLayout_Undefined = 0,
    VriLayout_General,
    VriLayout_ColorAttachment,
    VriLayout_DepthStencilAttachment,
    VriLayout_DepthStencilReadOnly,
    VriLayout_ShaderResource,       /* sampled / read-only */
    VriLayout_ShaderResourceStorage,/* UAV / storage image */
    VriLayout_CopySource,
    VriLayout_CopyDestination,
    VriLayout_Present,
    VriLayout_MaxEnum = 0x7fffffff
} VriLayout;

typedef enum VriAttachmentLoadOp
{
    VriAttachmentLoadOp_Load = 0,
    VriAttachmentLoadOp_Clear,
    VriAttachmentLoadOp_DontCare,
    VriAttachmentLoadOp_MaxEnum = 0x7fffffff
} VriAttachmentLoadOp;

typedef enum VriAttachmentStoreOp
{
    VriAttachmentStoreOp_Store = 0,
    VriAttachmentStoreOp_DontCare,
    VriAttachmentStoreOp_MaxEnum = 0x7fffffff
} VriAttachmentStoreOp;

/* ---- sampler ---------------------------------------------------------- */

typedef enum VriFilter
{
    VriFilter_Nearest = 0,
    VriFilter_Linear,
    VriFilter_MaxEnum = 0x7fffffff
} VriFilter;

typedef enum VriMipmapMode
{
    VriMipmapMode_Nearest = 0,
    VriMipmapMode_Linear,
    VriMipmapMode_MaxEnum = 0x7fffffff
} VriMipmapMode;

typedef enum VriAddressMode
{
    VriAddressMode_Repeat = 0,
    VriAddressMode_MirroredRepeat,
    VriAddressMode_ClampToEdge,
    VriAddressMode_ClampToBorder,
    VriAddressMode_MirrorClampToEdge,
    VriAddressMode_MaxEnum = 0x7fffffff
} VriAddressMode;

typedef enum VriBorderColor
{
    VriBorderColor_FloatTransparentBlack = 0,
    VriBorderColor_FloatOpaqueBlack,
    VriBorderColor_FloatOpaqueWhite,
    VriBorderColor_IntTransparentBlack,
    VriBorderColor_IntOpaqueBlack,
    VriBorderColor_IntOpaqueWhite,
    VriBorderColor_MaxEnum = 0x7fffffff
} VriBorderColor;

/* ---- descriptors ------------------------------------------------------ */

typedef enum VriDescriptorType
{
    VriDescriptorType_Sampler = 0,
    VriDescriptorType_Texture,            /* sampled image */
    VriDescriptorType_StorageTexture,     /* storage image / UAV */
    VriDescriptorType_ConstantBuffer,     /* uniform buffer */
    VriDescriptorType_StructuredBuffer,   /* read-only structured / SRV */
    VriDescriptorType_StorageBuffer,      /* RW structured / SSBO / UAV */
    VriDescriptorType_AccelerationStructure,
    VriDescriptorType_MaxEnum = 0x7fffffff
} VriDescriptorType;

typedef enum VriVertexStepRate
{
    VriVertexStepRate_PerVertex = 0,
    VriVertexStepRate_PerInstance,
    VriVertexStepRate_MaxEnum = 0x7fffffff
} VriVertexStepRate;

VRI_EXTERN_C_END

#endif /* VRI_ENUMS_H */
