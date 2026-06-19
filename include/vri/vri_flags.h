/*
 * vri_flags.h - bitflag enumerations.
 *
 * Each flag set is a `uint32_t` (or `uint64_t` for sync stages/access) typedef
 * plus a `*Bits` enum of individual bits. Bit layouts mirror libvultra's RHI
 * (buffer_usage / image_usage / shader_type / pipeline_stage / access /
 * image_aspect) so libvultra can adopt VRI with a near-identity mapping.
 */
#ifndef VRI_FLAGS_H
#define VRI_FLAGS_H

#include "vri_base.h"

VRI_EXTERN_C_BEGIN

/* ---- buffer usage ----------------------------------------------------- */
typedef uint32_t VriBufferUsageFlags;
typedef enum VriBufferUsageBits
{
    VriBufferUsage_None              = 0,
    VriBufferUsage_TransferSrc       = 1u << 0,
    VriBufferUsage_TransferDst       = 1u << 1,
    VriBufferUsage_VertexBuffer      = 1u << 2,
    VriBufferUsage_IndexBuffer       = 1u << 3,
    VriBufferUsage_ConstantBuffer    = 1u << 4,  /* uniform buffer */
    VriBufferUsage_StorageBuffer     = 1u << 5,  /* SSBO / UAV */
    VriBufferUsage_IndirectBuffer    = 1u << 6,
    VriBufferUsage_ShaderDeviceAddress = 1u << 7,
    VriBufferUsage_AccelerationBuildInput = 1u << 8,
    VriBufferUsage_AccelerationStorage    = 1u << 9,
    VriBufferUsage_ShaderBindingTable     = 1u << 10,
    VriBufferUsage_MaxEnum           = 0x7fffffff
} VriBufferUsageBits;

/* ---- texture usage ---------------------------------------------------- */
typedef uint32_t VriTextureUsageFlags;
typedef enum VriTextureUsageBits
{
    VriTextureUsage_None                  = 0,
    VriTextureUsage_TransferSrc           = 1u << 0,
    VriTextureUsage_TransferDst           = 1u << 1,
    VriTextureUsage_ShaderResource        = 1u << 2,  /* sampled */
    VriTextureUsage_ShaderResourceStorage = 1u << 3,  /* storage image / UAV */
    VriTextureUsage_ColorAttachment       = 1u << 4,
    VriTextureUsage_DepthStencilAttachment = 1u << 5,
    VriTextureUsage_MaxEnum               = 0x7fffffff
} VriTextureUsageBits;

/* ---- shader stages ---------------------------------------------------- */
typedef uint32_t VriShaderStageFlags;
typedef enum VriShaderStageBits
{
    VriShaderStage_None         = 0,
    VriShaderStage_Vertex       = 1u << 0,
    VriShaderStage_TessControl  = 1u << 1,
    VriShaderStage_TessEval     = 1u << 2,
    VriShaderStage_Geometry     = 1u << 3,
    VriShaderStage_Fragment     = 1u << 4,
    VriShaderStage_Compute      = 1u << 5,
    VriShaderStage_RayGen       = 1u << 6,
    VriShaderStage_Miss         = 1u << 7,
    VriShaderStage_ClosestHit   = 1u << 8,
    VriShaderStage_AnyHit       = 1u << 9,
    VriShaderStage_Intersection = 1u << 10,
    VriShaderStage_Callable     = 1u << 11,
    VriShaderStage_Mesh         = 1u << 12,
    VriShaderStage_Task         = 1u << 13,
    VriShaderStage_AllGraphics  = 0x0000001f,
    VriShaderStage_MaxEnum      = 0x7fffffff
} VriShaderStageBits;

/* ---- pipeline stages (synchronization) -------------------------------- */
typedef uint64_t VriPipelineStageFlags;
typedef enum VriPipelineStageBits
{
    VriPipelineStage_None                = 0,
    VriPipelineStage_DrawIndirect        = 1ull << 0,
    VriPipelineStage_VertexInput         = 1ull << 1,
    VriPipelineStage_VertexShader        = 1ull << 2,
    VriPipelineStage_TessControlShader   = 1ull << 3,
    VriPipelineStage_TessEvalShader      = 1ull << 4,
    VriPipelineStage_GeometryShader      = 1ull << 5,
    VriPipelineStage_FragmentShader      = 1ull << 6,
    VriPipelineStage_EarlyFragmentTests  = 1ull << 7,
    VriPipelineStage_LateFragmentTests   = 1ull << 8,
    VriPipelineStage_ColorAttachmentOutput = 1ull << 9,
    VriPipelineStage_ComputeShader       = 1ull << 10,
    VriPipelineStage_RayTracingShader    = 1ull << 11,
    VriPipelineStage_AccelerationStructureBuild = 1ull << 12,
    VriPipelineStage_Transfer            = 1ull << 13,
    VriPipelineStage_MeshShader          = 1ull << 14,
    VriPipelineStage_TaskShader          = 1ull << 15,
    VriPipelineStage_AllGraphics         = 1ull << 16,
    VriPipelineStage_AllCommands         = 1ull << 17,
    VriPipelineStage_MaxEnum             = 0x7fffffffffffffffull
} VriPipelineStageBits;

/* ---- access ----------------------------------------------------------- */
typedef uint64_t VriAccessFlags;
typedef enum VriAccessBits
{
    VriAccess_None                       = 0,
    VriAccess_IndexBufferRead            = 1ull << 0,
    VriAccess_VertexBufferRead           = 1ull << 1,
    VriAccess_IndirectBufferRead         = 1ull << 2,
    VriAccess_ConstantBufferRead         = 1ull << 3,
    VriAccess_ShaderResourceRead         = 1ull << 4,  /* sampled / readonly storage */
    VriAccess_ShaderResourceStorageRead  = 1ull << 5,
    VriAccess_ShaderResourceStorageWrite = 1ull << 6,
    VriAccess_ColorAttachmentRead        = 1ull << 7,
    VriAccess_ColorAttachmentWrite       = 1ull << 8,
    VriAccess_DepthStencilAttachmentRead = 1ull << 9,
    VriAccess_DepthStencilAttachmentWrite = 1ull << 10,
    VriAccess_CopySourceRead             = 1ull << 11,
    VriAccess_CopyDestinationWrite       = 1ull << 12,
    VriAccess_AccelerationStructureRead  = 1ull << 13,
    VriAccess_AccelerationStructureWrite = 1ull << 14,
    VriAccess_MaxEnum                    = 0x7fffffffffffffffull
} VriAccessBits;

/* ---- image aspect ----------------------------------------------------- */
typedef uint32_t VriImageAspectFlags;
typedef enum VriImageAspectBits
{
    VriImageAspect_None    = 0,
    VriImageAspect_Color   = 1u << 0,
    VriImageAspect_Depth   = 1u << 1,
    VriImageAspect_Stencil = 1u << 2,
    VriImageAspect_All     = 0x7u,
    VriImageAspect_MaxEnum = 0x7fffffff
} VriImageAspectBits;

/* ---- color write mask ------------------------------------------------- */
typedef uint32_t VriColorWriteFlags;
typedef enum VriColorWriteBits
{
    VriColorWrite_None    = 0,
    VriColorWrite_R       = 1u << 0,
    VriColorWrite_G       = 1u << 1,
    VriColorWrite_B       = 1u << 2,
    VriColorWrite_A       = 1u << 3,
    VriColorWrite_RGBA    = 0xf,
    VriColorWrite_MaxEnum = 0x7fffffff
} VriColorWriteBits;

VRI_EXTERN_C_END

#endif /* VRI_FLAGS_H */
