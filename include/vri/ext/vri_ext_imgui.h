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
 * Textures: each VriImguiDrawCommand carries the texture it samples (VriImguiDrawCommand::textureView,
 * a VriDescriptor* texture view), so user images and ImGui 1.92's dynamic textures both work, not just
 * the font atlas. The host owns the textures (created with the core interface) and routes them through
 * ImGui's texture id: set ImTextureData::SetTexID()/io.Fonts->SetTexID() to the VriDescriptor* view,
 * then copy ImDrawCmd::GetTexID() into each command's textureView. The renderer caches one descriptor
 * set per distinct view; tell it to drop that cache entry with FreeImguiTexture before destroying a
 * view (ImGui 1.92 WantDestroy), since a freed pointer can be reused by a later texture. A null
 * textureView falls back to the font atlas (GetImguiFontView), so the simple single-font path is
 * unchanged. For ImGui 1.92's dynamic font atlas, drive the texture lifecycle yourself and leave
 * VriImguiDesc::fontAtlas null (no built-in font texture); see example-cube.
 */
#ifndef VRI_EXT_IMGUI_H
#define VRI_EXT_IMGUI_H

#include "../vri_base.h"
#include "../vri_handles.h"
#include "../vri_format.h"

VRI_EXTERN_C_BEGIN

typedef struct VriImgui VriImgui;
typedef struct VriImguiViewport VriImguiViewport; /* one detached window's geometry (multi-viewport) */

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
    float          clipRect[4];  /* minX, minY, maxX, maxY in display coordinates */
    uint32_t       indexCount;
    uint32_t       indexOffset;  /* into the flattened index buffer */
    int32_t        vertexOffset; /* into the flattened vertex buffer */
    /* Texture this command samples, from ImDrawCmd::GetTexID(); null = the font atlas. */
    VriDescriptor* textureView;
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
    /* Optional built-in font atlas: RGBA8 pixels from io.Fonts->GetTexDataAsRGBA32 (the legacy
       single-font path). Leave null to manage every texture yourself via ImGui 1.92's dynamic
       texture lifecycle + per-command textureView; GetImguiFontView() then returns null. */
    const void* fontAtlas;
    uint32_t    fontWidth;
    uint32_t    fontHeight;
} VriImguiDesc;

typedef struct VriImguiInterface
{
    VriResult (VRI_CALL *CreateImgui)(VriDevice* device, const VriImguiDesc* desc, VriImgui** outImgui);
    void      (VRI_CALL *DestroyImgui)(VriImgui* imgui);
    /* The built-in font-atlas texture view (null if VriImguiDesc::fontAtlas was null); set it as
       ImGui's default texture id (io.Fonts->SetTexID) for the simple single-font path. */
    VriDescriptor* (VRI_CALL *GetImguiFontView)(VriImgui* imgui);

    /* Stage the frame's geometry (host map; call before acquiring the swapchain image). */
    void (VRI_CALL *UploadImguiData)(VriImgui* imgui, const VriImguiDrawData* data);
    /* Record the staging->device copy + barriers; call OUTSIDE the render pass. */
    void (VRI_CALL *CmdCopyImguiData)(VriCommandBuffer* cmd, VriImgui* imgui);
    /* Record the UI draws; call INSIDE the render pass. */
    void (VRI_CALL *CmdDrawImgui)(VriCommandBuffer* cmd, VriImgui* imgui, const VriImguiDrawData* data);

    /* Drop the cached descriptor set for a texture view before the host destroys it (ImGui 1.92
       WantDestroy). Required for correctness when textures churn: a freed VriDescriptor* address can
       be reused by a later texture, and a stale cache entry would bind the wrong (destroyed) texture. */
    void (VRI_CALL *FreeImguiTexture)(VriImgui* imgui, VriDescriptor* textureView);

    /* Multi-viewport (detached OS windows). Each platform window ImGui spawns needs its own geometry
       buffers so their staging doesn't clobber the main window's (the renderer's Upload/Draw are
       single-buffered per viewport). Create one VriImguiViewport per window (shares the VriImgui's
       pipeline/font/texture cache), then drive it with the *To calls exactly like the main window's
       Upload/CmdCopy/CmdDraw. Typically wired into ImGuiPlatformIO's Renderer_* callbacks. */
    VriImguiViewport* (VRI_CALL *CreateImguiViewport)(VriImgui* imgui);
    void (VRI_CALL *DestroyImguiViewport)(VriImguiViewport* viewport);
    void (VRI_CALL *UploadImguiDataTo)(VriImguiViewport* viewport, const VriImguiDrawData* data);
    void (VRI_CALL *CmdCopyImguiDataTo)(VriCommandBuffer* cmd, VriImguiViewport* viewport);
    void (VRI_CALL *CmdDrawImguiTo)(VriCommandBuffer* cmd, VriImgui* imgui, VriImguiViewport* viewport,
                                    const VriImguiDrawData* data);
} VriImguiInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_IMGUI_H */
