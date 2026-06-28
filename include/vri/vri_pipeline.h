/*
 * vri_pipeline.h - shader, vertex input, fixed-function state, and
 * graphics/compute pipeline descriptors.
 */
#ifndef VRI_PIPELINE_H
#define VRI_PIPELINE_H

#include "vri_base.h"
#include "vri_handles.h"
#include "vri_enums.h"
#include "vri_flags.h"
#include "vri_format.h"

VRI_EXTERN_C_BEGIN

/* ---- shader ----------------------------------------------------------- */
typedef struct VriShaderDesc
{
    VriShaderStageBits stage;          /* exactly one stage */
    const void*        bytecode;       /* SPIR-V / DXBC / DXIL / WGSL / MSL ... */
    size_t             bytecodeSize;
    const char*        entryPointName; /* NULL => "main" */
} VriShaderDesc;

/* ---- vertex input ----------------------------------------------------- */
typedef struct VriVertexAttributeDesc
{
    VriFormat   format;
    uint32_t    offset;
    uint32_t    streamIndex;   /* index into the streams array */
    const char* semanticName;  /* D3D semantic (e.g. "POSITION"); NULL elsewhere */
    uint32_t    semanticIndex;
} VriVertexAttributeDesc;

typedef struct VriVertexStreamDesc
{
    uint32_t          stride;
    uint32_t          bindingSlot;
    VriVertexStepRate stepRate;
} VriVertexStreamDesc;

typedef struct VriVertexInputDesc
{
    const VriVertexAttributeDesc* attributes;
    uint32_t                      attributeNum;
    const VriVertexStreamDesc*    streams;
    uint32_t                      streamNum;
} VriVertexInputDesc;

/* ---- input assembly --------------------------------------------------- */
typedef struct VriInputAssemblyDesc
{
    VriPrimitiveTopology topology;
    VriBool              primitiveRestart;
} VriInputAssemblyDesc;

/* ---- rasterization ---------------------------------------------------- */
typedef struct VriDepthBiasDesc
{
    float constant;
    float clamp;
    float slope;
} VriDepthBiasDesc;

typedef struct VriRasterizationDesc
{
    VriPolygonMode   polygonMode;
    VriCullMode      cullMode;
    VriFrontFace     frontFace;
    VriBool          depthClamp;
    VriDepthBiasDesc depthBias;
    float            lineWidth;
    /* Conservative rasterization: rasterize a primitive if it touches a pixel at all
       (over-estimation). Requires VriDeviceDesc::hasConservativeRaster. */
    VriBool          conservativeRaster;
} VriRasterizationDesc;

/* ---- depth / stencil -------------------------------------------------- */
typedef struct VriStencilOpDesc
{
    VriStencilOp failOp;
    VriStencilOp passOp;
    VriStencilOp depthFailOp;
    VriCompareOp compareOp;
    uint32_t     compareMask;
    uint32_t     writeMask;
    uint32_t     reference;
} VriStencilOpDesc;

typedef struct VriDepthStencilDesc
{
    VriBool          depthTest;
    VriBool          depthWrite;
    VriCompareOp     depthCompareOp;
    VriBool          stencilTest;
    VriStencilOpDesc front;
    VriStencilOpDesc back;
} VriDepthStencilDesc;

/* ---- blend / output merger -------------------------------------------- */
typedef struct VriBlendDesc
{
    VriBool        enable;
    VriBlendFactor srcColor;
    VriBlendFactor dstColor;
    VriBlendOp     colorOp;
    VriBlendFactor srcAlpha;
    VriBlendFactor dstAlpha;
    VriBlendOp     alphaOp;
} VriBlendDesc;

typedef struct VriColorAttachmentDesc
{
    VriFormat          format;
    VriBlendDesc       blend;
    VriColorWriteFlags colorWriteMask;
} VriColorAttachmentDesc;

typedef struct VriOutputMergerDesc
{
    const VriColorAttachmentDesc* colors;
    uint32_t                      colorNum;
    VriFormat                     depthStencilFormat; /* Unknown = none */
    uint32_t                      viewMask;           /* multiview; 0 = single view. Must match the
                                                       * render's VriAttachmentsDesc::viewMask. */
} VriOutputMergerDesc;

typedef struct VriMultisampleDesc
{
    uint32_t sampleNum; /* 1 = disabled */
} VriMultisampleDesc;

/* ---- tessellation ----------------------------------------------------- */
typedef struct VriTessellationDesc
{
    uint32_t patchControlPoints; /* 0 = no tessellation; vertices per patch otherwise */
} VriTessellationDesc;

/* ---- pipelines -------------------------------------------------------- */
typedef struct VriGraphicsPipelineDesc
{
    VriPipelineLayout*          pipelineLayout;
    const VriShaderDesc*        shaders;
    uint32_t                    shaderNum;
    VriVertexInputDesc          vertexInput;
    VriInputAssemblyDesc        inputAssembly;
    VriTessellationDesc         tessellation;
    VriRasterizationDesc        rasterization;
    VriMultisampleDesc          multisample;
    VriDepthStencilDesc         depthStencil;
    VriOutputMergerDesc         outputMerger;
    /* Optional warm-creation cache from VRI_INTERFACE_PIPELINE_CACHE; NULL = none. */
    VriPipelineCache*           pipelineCache;
} VriGraphicsPipelineDesc;

typedef struct VriComputePipelineDesc
{
    VriPipelineLayout* pipelineLayout;
    VriShaderDesc      shader;
    /* Optional warm-creation cache from VRI_INTERFACE_PIPELINE_CACHE; NULL = none. */
    VriPipelineCache*  pipelineCache;
} VriComputePipelineDesc;

VRI_EXTERN_C_END

#endif /* VRI_PIPELINE_H */
