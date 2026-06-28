/*
 * vri_ext_imgui.h - a built-in Dear ImGui renderer, queried via
 * vriGetInterface(device, VRI_INTERFACE_IMGUI, ...).
 *
 * VRI does NOT depend on or link Dear ImGui. The renderer is written against VRI's own core
 * interface, so a single implementation draws ImGui on every backend (Vulkan/D3D12/WebGPU/OpenGL)
 * with no per-backend imgui_impl_*. The application owns the ImGui *environment* (context, input,
 * NewFrame/Render) and each frame translates ImDrawData into the neutral VriImguiDrawData below --
 * so no ImGui types ever cross into VRI.
 *
 * Per frame: build a VriImguiDrawData from ImGui's draw data (concatenate the draw lists into one
 * vertex + one index buffer and make the per-command offsets global), then:
 *   1. UploadImguiData(data)             -- stages the geometry; call BEFORE acquiring the
 *                                           swapchain image (the host map can yield on WebGPU).
 *   2. CmdCopyImguiData(cmd)             -- records the staging->device copy; OUTSIDE the pass.
 *   3. CmdDrawImgui(cmd, data)           -- records the UI draws; INSIDE the render pass.
 *
 * Only the font atlas is textured (set GetImguiFontView() as ImGui's default texture id);
 * arbitrary per-command image textures are a planned addition.
 */
#ifndef VRI_EXT_IMGUI_H
#define VRI_EXT_IMGUI_H

#include "../vri_base.h"
#include "../vri_handles.h"
#include "../vri_format.h"

VRI_EXTERN_C_BEGIN

typedef struct VriImgui VriImgui;

/* Byte-compatible with ImDrawVert (pos float2, uv float2, packed-RGBA8 color). */
typedef struct VriImguiVertex
{
    float    position[2];
    float    uv[2];
    uint32_t color;
} VriImguiVertex;

/* One scissored indexed draw (an ImDrawCmd with its offsets made global). */
typedef struct VriImguiDrawCommand
{
    float    clipRect[4];  /* minX, minY, maxX, maxY in display coordinates */
    uint32_t indexCount;
    uint32_t indexOffset;  /* into the flattened index buffer */
    int32_t  vertexOffset; /* into the flattened vertex buffer */
} VriImguiDrawCommand;

/* A flattened ImGui frame: all draw lists concatenated, per-command offsets global. */
typedef struct VriImguiDrawData
{
    const VriImguiVertex*      vertices;
    uint32_t                   vertexCount;
    const void*                indices;   /* 16- or 32-bit elements (see indexSize) */
    uint32_t                   indexCount;
    uint32_t                   indexSize; /* sizeof(ImDrawIdx): 2 or 4 */
    const VriImguiDrawCommand* commands;
    uint32_t                   commandCount;
    float                      displayPos[2];  /* ImDrawData::DisplayPos */
    float                      displaySize[2]; /* ImDrawData::DisplaySize */
    uint32_t                   framebufferWidth;
    uint32_t                   framebufferHeight;
} VriImguiDrawData;

typedef struct VriImguiDesc
{
    VriQueue*   uploadQueue; /* a Graphics queue, for the one-shot font-atlas upload */
    VriFormat   colorFormat; /* the render target / swapchain color format */
    VriFormat   depthFormat; /* the pass's depth attachment format (VriFormat_Unknown if none) */
    const void* fontAtlas;   /* RGBA8 pixels from io.Fonts->GetTexDataAsRGBA32 */
    uint32_t    fontWidth;
    uint32_t    fontHeight;
} VriImguiDesc;

typedef struct VriImguiInterface
{
    VriResult (VRI_CALL *CreateImgui)(VriDevice* device, const VriImguiDesc* desc, VriImgui** outImgui);
    void      (VRI_CALL *DestroyImgui)(VriImgui* imgui);
    /* The font-atlas texture view; set it as ImGui's default texture id (io.Fonts->SetTexID). */
    VriDescriptor* (VRI_CALL *GetImguiFontView)(VriImgui* imgui);

    /* Stage the frame's geometry (host map; call before acquiring the swapchain image). */
    void (VRI_CALL *UploadImguiData)(VriImgui* imgui, const VriImguiDrawData* data);
    /* Record the staging->device copy + barriers; call OUTSIDE the render pass. */
    void (VRI_CALL *CmdCopyImguiData)(VriCommandBuffer* cmd, VriImgui* imgui);
    /* Record the UI draws; call INSIDE the render pass. */
    void (VRI_CALL *CmdDrawImgui)(VriCommandBuffer* cmd, VriImgui* imgui, const VriImguiDrawData* data);
} VriImguiInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_IMGUI_H */
