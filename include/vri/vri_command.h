/*
 * vri_command.h - descriptors for command recording: viewports, attachments,
 * draws/dispatches, barriers, and copies.
 */
#ifndef VRI_COMMAND_H
#define VRI_COMMAND_H

#include "vri_base.h"
#include "vri_handles.h"
#include "vri_enums.h"
#include "vri_flags.h"

VRI_EXTERN_C_BEGIN

typedef struct VriViewport
{
    float x;
    float y;
    float width;
    float height;
    float minDepth;
    float maxDepth;
} VriViewport;

typedef struct VriRect
{
    int32_t  x;
    int32_t  y;
    uint32_t width;
    uint32_t height;
} VriRect;

/* ---- clear values ----------------------------------------------------- */
typedef union VriClearColor
{
    float    f32[4];
    uint32_t u32[4];
    int32_t  i32[4];
} VriClearColor;

typedef struct VriClearDepthStencil
{
    float    depth;
    uint32_t stencil;
} VriClearDepthStencil;

typedef union VriClearValue
{
    VriClearColor        color;
    VriClearDepthStencil depthStencil;
} VriClearValue;

/* ---- dynamic rendering attachments ------------------------------------ */
typedef struct VriAttachmentDesc
{
    VriDescriptor*       view;     /* a texture view descriptor */
    VriAttachmentLoadOp  loadOp;
    VriAttachmentStoreOp storeOp;
    VriClearValue        clearValue;
} VriAttachmentDesc;

typedef struct VriAttachmentsDesc
{
    const VriAttachmentDesc* colors;
    uint32_t                 colorNum;
    const VriAttachmentDesc* depth;    /* NULL = none */
    const VriAttachmentDesc* stencil;  /* NULL = none */
    VriRect                  renderArea;
    uint32_t                 layerNum;
    uint32_t                 viewMask; /* multiview; 0 = single view */
} VriAttachmentsDesc;

/* ---- vertex / index bindings ------------------------------------------ */
typedef struct VriVertexBufferBinding
{
    VriBuffer* buffer;
    uint64_t   offset;
} VriVertexBufferBinding;

/* ---- draw / dispatch -------------------------------------------------- */
typedef struct VriDrawDesc
{
    uint32_t vertexNum;
    uint32_t instanceNum;
    uint32_t baseVertex;
    uint32_t baseInstance;
} VriDrawDesc;

typedef struct VriDrawIndexedDesc
{
    uint32_t indexNum;
    uint32_t instanceNum;
    uint32_t baseIndex;
    int32_t  vertexOffset;
    uint32_t baseInstance;
} VriDrawIndexedDesc;

typedef struct VriDispatchDesc
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
} VriDispatchDesc;

/* ---- barriers (explicit stages/access[/layout]) ----------------------- */
typedef struct VriAccessStage
{
    VriAccessFlags        access;
    VriPipelineStageFlags stages;
} VriAccessStage;

typedef struct VriAccessLayoutStage
{
    VriAccessFlags        access;
    VriLayout             layout;
    VriPipelineStageFlags stages;
} VriAccessLayoutStage;

typedef struct VriBufferBarrierDesc
{
    VriBuffer*     buffer;
    VriAccessStage before;
    VriAccessStage after;
} VriBufferBarrierDesc;

typedef struct VriTextureBarrierDesc
{
    VriTexture*          texture;
    VriAccessLayoutStage before;
    VriAccessLayoutStage after;
    VriImageAspectFlags  aspect;
    uint32_t             baseMip;
    uint32_t             mipNum;   /* 0 = all */
    uint32_t             baseLayer;
    uint32_t             layerNum; /* 0 = all */
} VriTextureBarrierDesc;

typedef struct VriBarrierGroupDesc
{
    const VriBufferBarrierDesc*  buffers;
    uint32_t                     bufferNum;
    const VriTextureBarrierDesc* textures;
    uint32_t                     textureNum;
} VriBarrierGroupDesc;

/* ---- copies ----------------------------------------------------------- */
typedef struct VriBufferCopyDesc
{
    uint64_t srcOffset;
    uint64_t dstOffset;
    uint64_t size;
} VriBufferCopyDesc;

typedef struct VriTextureRegionDesc
{
    uint32_t            mip;
    uint32_t            baseLayer;
    uint32_t            layerNum;
    VriImageAspectFlags aspect;
    int32_t             x;
    int32_t             y;
    int32_t             z;
    uint32_t            width;  /* 0 = full extent */
    uint32_t            height;
    uint32_t            depth;
} VriTextureRegionDesc;

typedef struct VriTextureCopyDesc
{
    VriTextureRegionDesc src;
    VriTextureRegionDesc dst;
} VriTextureCopyDesc;

/* Buffer<->texture transfer (aligned with libvultra BufferImageCopy). */
typedef struct VriBufferTextureCopyDesc
{
    uint64_t             bufferOffset;
    uint32_t             bufferRowLength;   /* in texels; 0 = tightly packed */
    uint32_t             bufferImageHeight; /* in rows; 0 = tightly packed */
    VriTextureRegionDesc texture;
} VriBufferTextureCopyDesc;

VRI_EXTERN_C_END

#endif /* VRI_COMMAND_H */
