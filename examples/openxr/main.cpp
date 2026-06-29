// example-openxr - VRI single-pass-stereo rendering driven by OpenXR.
//
// VRI does NOT depend on OpenXR. This example links the OpenXR loader and bridges the two through
// VRI_INTERFACE_INTEROP:
//
//   1. Create an XrInstance (XR_KHR_vulkan_enable) and pick the HMD system.
//   2. Ask OpenXR which Vulkan instance/device extensions it requires and pass them straight to
//      vriCreateDevice -- so VRI builds a Vulkan device OpenXR is happy to render with.
//   3. Read VRI's native Vulkan handles back (GetDeviceNativeHandles) to fill the
//      XrGraphicsBindingVulkanKHR and create the session.
//   4. Create ONE stereo swapchain (a 2-layer array), wrap each swapchain VkImage as a VriTexture,
//      and render BOTH eyes in a single pass with VRI multiview (viewMask = 0b11). SV_ViewID tints
//      the left eye red and the right eye green.
//
// Build: xmake f --vri_build_examples=y --vri_build_openxr=y -y && xmake build example-openxr
// Run:   needs an OpenXR runtime (a headset, or a runtime simulator). Not run in CI.
//
// This uses XR_KHR_vulkan_enable (v1): VRI creates the Vulkan instance/device itself from the
// extension set OpenXR asks for. That works with runtimes that accept v1 (SteamVR, Monado, ...).
// The newer XR_KHR_vulkan_enable2 instead has OpenXR create the VkInstance/VkDevice for you, which
// some runtimes (e.g. the Meta XR Simulator) require; supporting it means VRI adopting an
// externally-created instance/device through VriDeviceCreationDesc::nativeCreateInfo (declared but
// not yet implemented in the Vulkan backend) -- a documented follow-up.

#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <vulkan/vulkan.h> // only for the Vulkan types in the OpenXR<->Vulkan structs; no vk calls

#include <vri/vri.h>

#include "shaders/examples/openxr_stereo_spv.h" // g_openxrStereoSpv

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{
    bool xrOk(XrResult r, const char* what)
    {
        if (XR_SUCCEEDED(r))
            return true;
        std::fprintf(stderr, "[openxr] %s failed (XrResult %d)\n", what, int(r));
        return false;
    }
    bool vriOk(VriResult r, const char* what)
    {
        if (r == VriResult_Success)
            return true;
        std::fprintf(stderr, "[vri] %s failed (VriResult %d)\n", what, int(r));
        return false;
    }

    // OpenXR returns required extensions as one space-separated string; split into NUL-terminated names.
    std::vector<std::string> splitExtensions(const std::string& s)
    {
        std::vector<std::string> out;
        std::string              cur;
        for (char ch : s)
        {
            if (ch == ' ')
            {
                if (!cur.empty())
                    out.push_back(cur), cur.clear();
            }
            else
                cur.push_back(ch);
        }
        if (!cur.empty())
            out.push_back(cur);
        return out;
    }

    VriFormat toVriFormat(int64_t vk)
    {
        switch (vk)
        {
            case VK_FORMAT_R8G8B8A8_SRGB:
                return VriFormat_RGBA8_SRGB;
            case VK_FORMAT_R8G8B8A8_UNORM:
                return VriFormat_RGBA8_UNORM;
            case VK_FORMAT_B8G8R8A8_SRGB:
                return VriFormat_BGRA8_SRGB;
            case VK_FORMAT_B8G8R8A8_UNORM:
                return VriFormat_BGRA8_UNORM;
            default:
                return VriFormat_Unknown;
        }
    }
} // namespace

int main()
{
    // ---- 1. OpenXR instance + system ------------------------------------------------------------
    XrInstance xrInstance = XR_NULL_HANDLE;
    {
        const char*          exts[] = {XR_KHR_VULKAN_ENABLE_EXTENSION_NAME};
        XrInstanceCreateInfo ci {XR_TYPE_INSTANCE_CREATE_INFO};
        std::strcpy(ci.applicationInfo.applicationName, "vri-openxr");
        ci.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
        ci.enabledExtensionCount      = 1;
        ci.enabledExtensionNames      = exts;
        if (!xrOk(xrCreateInstance(&ci, &xrInstance), "xrCreateInstance"))
        {
            std::fprintf(stderr, "[openxr] no OpenXR runtime available - is one installed/active?\n");
            return 1;
        }
    }

    XrSystemId systemId = XR_NULL_SYSTEM_ID;
    {
        XrSystemGetInfo gi {XR_TYPE_SYSTEM_GET_INFO};
        gi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
        if (!xrOk(xrGetSystem(xrInstance, &gi, &systemId), "xrGetSystem"))
            return 1;
    }

    // KHR Vulkan-enable entry points are extension functions -> resolve them dynamically.
    auto load = [&](const char* name, auto& fp) {
        xrGetInstanceProcAddr(xrInstance, name, reinterpret_cast<PFN_xrVoidFunction*>(&fp));
    };
    PFN_xrGetVulkanGraphicsRequirementsKHR xrGetVulkanGraphicsRequirementsKHR = nullptr;
    PFN_xrGetVulkanInstanceExtensionsKHR   xrGetVulkanInstanceExtensionsKHR   = nullptr;
    PFN_xrGetVulkanDeviceExtensionsKHR     xrGetVulkanDeviceExtensionsKHR     = nullptr;
    PFN_xrGetVulkanGraphicsDeviceKHR       xrGetVulkanGraphicsDeviceKHR       = nullptr;
    load("xrGetVulkanGraphicsRequirementsKHR", xrGetVulkanGraphicsRequirementsKHR);
    load("xrGetVulkanInstanceExtensionsKHR", xrGetVulkanInstanceExtensionsKHR);
    load("xrGetVulkanDeviceExtensionsKHR", xrGetVulkanDeviceExtensionsKHR);
    load("xrGetVulkanGraphicsDeviceKHR", xrGetVulkanGraphicsDeviceKHR);
    if (!xrGetVulkanGraphicsRequirementsKHR || !xrGetVulkanInstanceExtensionsKHR || !xrGetVulkanDeviceExtensionsKHR ||
        !xrGetVulkanGraphicsDeviceKHR)
    {
        std::fprintf(stderr, "[openxr] runtime is missing XR_KHR_vulkan_enable entry points\n");
        return 1;
    }

    // Must be called before creating the Vulkan device (spec requirement), even if we only log it.
    XrGraphicsRequirementsVulkanKHR vkReq {XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
    xrGetVulkanGraphicsRequirementsKHR(xrInstance, systemId, &vkReq);

    // ---- 2. ask OpenXR which Vulkan extensions it needs, hand them to VRI -----------------------
    auto queryExts = [&](auto fn) {
        uint32_t n = 0;
        fn(xrInstance, systemId, 0, &n, nullptr);
        std::string s(n, '\0');
        fn(xrInstance, systemId, n, &n, s.data());
        if (!s.empty() && s.back() == '\0')
            s.pop_back();
        return splitExtensions(s);
    };
    const std::vector<std::string> instExtStr = queryExts(xrGetVulkanInstanceExtensionsKHR);
    const std::vector<std::string> devExtStr  = queryExts(xrGetVulkanDeviceExtensionsKHR);
    std::vector<const char*>       instExts, devExts;
    for (const std::string& e : instExtStr)
        instExts.push_back(e.c_str());
    for (const std::string& e : devExtStr)
        devExts.push_back(e.c_str());

    VriDeviceCreationDesc dc {};
    dc.graphicsAPI                  = VriGraphicsAPI_Vulkan;
    dc.enableValidation             = VRI_TRUE;
    dc.bestEffort                   = VRI_TRUE;
    dc.requiredInstanceExtensions   = instExts.data();
    dc.requiredInstanceExtensionNum = static_cast<uint32_t>(instExts.size());
    dc.requiredDeviceExtensions     = devExts.data();
    dc.requiredDeviceExtensionNum   = static_cast<uint32_t>(devExts.size());
    VriDevice* dev                  = nullptr;
    if (!vriOk(vriCreateDevice(&dc, &dev), "vriCreateDevice"))
        return 1;

    VriCoreInterface    c {};
    VriInteropInterface interop {};
    if (!vriOk(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c), "get core") ||
        !vriOk(vriGetInterface(dev, VRI_INTERFACE_INTEROP, sizeof(interop), &interop), "get interop"))
        return 1;
    if (c.GetDeviceDesc(dev)->hasMultiview == VRI_FALSE)
    {
        std::fprintf(stderr, "[vri] device has no multiview - cannot do single-pass stereo\n");
        return 1;
    }

    // ---- 3. native handles -> XrGraphicsBindingVulkanKHR -> session -----------------------------
    VriDeviceNativeHandles nh {};
    if (!vriOk(interop.GetDeviceNativeHandles(dev, &nh), "GetDeviceNativeHandles"))
        return 1;

    VkPhysicalDevice xrPhysical = VK_NULL_HANDLE;
    xrGetVulkanGraphicsDeviceKHR(xrInstance, systemId, static_cast<VkInstance>(nh.u.vulkan.instance), &xrPhysical);
    if (xrPhysical != static_cast<VkPhysicalDevice>(nh.u.vulkan.physicalDevice))
        std::fprintf(stderr,
                     "[openxr] warning: VRI picked a different physical device than OpenXR wants; "
                     "pass a matching adapterIndex on a multi-GPU machine\n");

    XrGraphicsBindingVulkanKHR binding {XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR};
    binding.instance         = static_cast<VkInstance>(nh.u.vulkan.instance);
    binding.physicalDevice   = static_cast<VkPhysicalDevice>(nh.u.vulkan.physicalDevice);
    binding.device           = static_cast<VkDevice>(nh.u.vulkan.device);
    binding.queueFamilyIndex = nh.u.vulkan.graphicsQueueFamilyIndex;
    binding.queueIndex       = nh.u.vulkan.graphicsQueueIndex;

    XrSession session = XR_NULL_HANDLE;
    {
        XrSessionCreateInfo si {XR_TYPE_SESSION_CREATE_INFO};
        si.next     = &binding;
        si.systemId = systemId;
        if (!xrOk(xrCreateSession(xrInstance, &si, &session), "xrCreateSession"))
            return 1;
    }

    XrSpace appSpace = XR_NULL_HANDLE;
    {
        XrReferenceSpaceCreateInfo sci {XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
        sci.referenceSpaceType                 = XR_REFERENCE_SPACE_TYPE_LOCAL;
        sci.poseInReferenceSpace.orientation.w = 1.0f;
        if (!xrOk(xrCreateReferenceSpace(session, &sci, &appSpace), "xrCreateReferenceSpace"))
            return 1;
    }

    // ---- 4. stereo view config + a single 2-layer swapchain ------------------------------------
    const XrViewConfigurationType viewConfig = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    uint32_t                      viewCount  = 0;
    xrEnumerateViewConfigurationViews(xrInstance, systemId, viewConfig, 0, &viewCount, nullptr);
    std::vector<XrViewConfigurationView> viewCfgs(viewCount, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
    xrEnumerateViewConfigurationViews(xrInstance, systemId, viewConfig, viewCount, &viewCount, viewCfgs.data());
    if (viewCount != 2)
    {
        std::fprintf(stderr, "[openxr] expected 2 stereo views, got %u\n", viewCount);
        return 1;
    }
    const uint32_t w = viewCfgs[0].recommendedImageRectWidth;
    const uint32_t h = viewCfgs[0].recommendedImageRectHeight;

    uint32_t fmtCount = 0;
    xrEnumerateSwapchainFormats(session, 0, &fmtCount, nullptr);
    std::vector<int64_t> formats(fmtCount);
    xrEnumerateSwapchainFormats(session, fmtCount, &fmtCount, formats.data());
    int64_t   chosenVk = 0;
    VriFormat chosen   = VriFormat_Unknown;
    for (int64_t f : formats)
        if (toVriFormat(f) != VriFormat_Unknown)
        {
            chosenVk = f;
            chosen   = toVriFormat(f);
            break;
        }
    if (chosen == VriFormat_Unknown)
    {
        std::fprintf(stderr, "[openxr] no supported RGBA8/BGRA8 swapchain format\n");
        return 1;
    }

    XrSwapchain swapchain = XR_NULL_HANDLE;
    {
        XrSwapchainCreateInfo sci {XR_TYPE_SWAPCHAIN_CREATE_INFO};
        sci.usageFlags  = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        sci.format      = chosenVk;
        sci.sampleCount = 1;
        sci.width       = w;
        sci.height      = h;
        sci.faceCount   = 1;
        sci.arraySize   = 2; // <-- both eyes in one array image: single-pass multiview
        sci.mipCount    = 1;
        if (!xrOk(xrCreateSwapchain(session, &sci, &swapchain), "xrCreateSwapchain"))
            return 1;
    }

    uint32_t imgCount = 0;
    xrEnumerateSwapchainImages(swapchain, 0, &imgCount, nullptr);
    std::vector<XrSwapchainImageVulkanKHR> xrImages(imgCount, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
    xrEnumerateSwapchainImages(
        swapchain, imgCount, &imgCount, reinterpret_cast<XrSwapchainImageBaseHeader*>(xrImages.data()));

    // Wrap each swapchain VkImage as a VriTexture + a 2D-array view spanning both eye-layers.
    std::vector<VriTexture*>    wrapped(imgCount, nullptr);
    std::vector<VriDescriptor*> views(imgCount, nullptr);
    for (uint32_t i = 0; i < imgCount; ++i)
    {
        VriWrapTextureDesc wd {};
        wd.nativeTexture  = xrImages[i].image;
        wd.desc.type      = VriTextureType_2D;
        wd.desc.format    = chosen;
        wd.desc.width     = w;
        wd.desc.height    = h;
        wd.desc.depth     = 1;
        wd.desc.mipNum    = 1;
        wd.desc.layerNum  = 2;
        wd.desc.sampleNum = 1;
        wd.desc.usage     = VriTextureUsage_ColorAttachment;
        if (!vriOk(interop.WrapTexture(dev, &wd, &wrapped[i]), "WrapTexture"))
            return 1;
        VriTextureViewDesc vd {};
        vd.texture  = wrapped[i];
        vd.viewType = VriTextureViewType_2DArray;
        vd.format   = chosen;
        vd.aspect   = VriImageAspect_Color;
        vd.layerNum = 2;
        if (!vriOk(c.CreateTextureView(dev, &vd, &views[i]), "CreateTextureView"))
            return 1;
    }

    // ---- VRI pipeline (multiview) + command resources ------------------------------------------
    VriQueue* queue = nullptr;
    c.GetQueue(dev, VriQueueType_Graphics, 0, &queue);

    VriPipelineLayoutDesc ld {};
    VriPipelineLayout*    layout = nullptr;
    c.CreatePipelineLayout(dev, &ld, &layout);

    VriShaderDesc sh[2] {};
    sh[0].stage          = VriShaderStage_Vertex;
    sh[1].stage          = VriShaderStage_Fragment;
    sh[0].entryPointName = "vertexMain";
    sh[1].entryPointName = "fragmentMain";
    sh[0].bytecode = sh[1].bytecode = g_openxrStereoSpv;
    sh[0].bytecodeSize = sh[1].bytecodeSize = sizeof(g_openxrStereoSpv);

    VriColorAttachmentDesc ca {};
    ca.format         = chosen;
    ca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc pd {};
    pd.pipelineLayout          = layout;
    pd.shaders                 = sh;
    pd.shaderNum               = 2;
    pd.inputAssembly.topology  = VriPrimitiveTopology_TriangleList;
    pd.rasterization.cullMode  = VriCullMode_None;
    pd.rasterization.lineWidth = 1.0f;
    pd.multisample.sampleNum   = 1;
    pd.outputMerger.colors     = &ca;
    pd.outputMerger.colorNum   = 1;
    pd.outputMerger.viewMask   = 0x3; // 0b11: both eyes
    VriPipeline* pipeline      = nullptr;
    if (!vriOk(c.CreateGraphicsPipeline(dev, &pd, &pipeline), "CreateGraphicsPipeline"))
        return 1;

    VriCommandAllocator* alloc = nullptr;
    c.CreateCommandAllocator(dev, VriQueueType_Graphics, &alloc);
    VriCommandBuffer* cmd = nullptr;
    c.CreateCommandBuffer(alloc, &cmd);
    VriFence* fence = nullptr;
    c.CreateFence(dev, 0, &fence);
    uint64_t fenceValue = 0;

    std::printf("[vri-openxr] session up: %ux%u per eye, %u swapchain image(s), format=%d. "
                "Put on the headset - left eye red, right eye green.\n",
                w,
                h,
                imgCount,
                int(chosen));

    // ---- 5. event + frame loop -----------------------------------------------------------------
    bool running = false; // session state >= READY
    bool quit    = false;
    while (!quit)
    {
        XrEventDataBuffer ev {XR_TYPE_EVENT_DATA_BUFFER};
        while (xrPollEvent(xrInstance, &ev) == XR_SUCCESS)
        {
            if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED)
            {
                const auto* ss = reinterpret_cast<const XrEventDataSessionStateChanged*>(&ev);
                if (ss->state == XR_SESSION_STATE_READY)
                {
                    XrSessionBeginInfo bi {XR_TYPE_SESSION_BEGIN_INFO};
                    bi.primaryViewConfigurationType = viewConfig;
                    xrBeginSession(session, &bi);
                    running = true;
                }
                else if (ss->state == XR_SESSION_STATE_STOPPING)
                {
                    xrEndSession(session);
                    running = false;
                }
                else if (ss->state == XR_SESSION_STATE_EXITING || ss->state == XR_SESSION_STATE_LOSS_PENDING)
                    quit = true;
            }
            else if (ev.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING)
                quit = true;
            ev = {XR_TYPE_EVENT_DATA_BUFFER};
        }
        if (!running)
            continue;

        // -- frame timing --
        XrFrameState    fs {XR_TYPE_FRAME_STATE};
        XrFrameWaitInfo fwi {XR_TYPE_FRAME_WAIT_INFO};
        xrWaitFrame(session, &fwi, &fs);
        XrFrameBeginInfo fbi {XR_TYPE_FRAME_BEGIN_INFO};
        xrBeginFrame(session, &fbi);

        std::vector<XrCompositionLayerProjectionView> projViews(2, {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW});
        XrCompositionLayerProjection                  layer {XR_TYPE_COMPOSITION_LAYER_PROJECTION};
        bool                                          haveLayer = false;

        if (fs.shouldRender)
        {
            // Locate the two eye views (poses + fovs) for the projection layer.
            XrViewState         vs {XR_TYPE_VIEW_STATE};
            uint32_t            got = 0;
            std::vector<XrView> xrViews(2, {XR_TYPE_VIEW});
            XrViewLocateInfo    li {XR_TYPE_VIEW_LOCATE_INFO};
            li.viewConfigurationType = viewConfig;
            li.displayTime           = fs.predictedDisplayTime;
            li.space                 = appSpace;
            xrLocateViews(session, &li, &vs, 2, &got, xrViews.data());

            // Acquire the swapchain image (a 2-layer array) and render both eyes in one pass.
            uint32_t idx = 0;
            xrAcquireSwapchainImage(swapchain, nullptr, &idx);
            XrSwapchainImageWaitInfo wi {XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
            wi.timeout = XR_INFINITE_DURATION;
            xrWaitSwapchainImage(swapchain, &wi);

            c.ResetCommandAllocator(alloc);
            c.BeginCommandBuffer(cmd);
            VriTextureBarrierDesc b {};
            b.texture       = wrapped[idx];
            b.before.layout = VriLayout_Undefined;
            b.before.stages = VriPipelineStage_None;
            b.after.access  = VriAccess_ColorAttachmentWrite;
            b.after.layout  = VriLayout_ColorAttachment;
            b.after.stages  = VriPipelineStage_ColorAttachmentOutput;
            b.aspect        = VriImageAspect_Color;
            VriBarrierGroupDesc bg {};
            bg.textures   = &b;
            bg.textureNum = 1;
            c.CmdBarrier(cmd, &bg);

            VriAttachmentDesc rt {};
            rt.view                    = views[idx];
            rt.loadOp                  = VriAttachmentLoadOp_Clear;
            rt.storeOp                 = VriAttachmentStoreOp_Store;
            rt.clearValue.color.f32[0] = 0.05f;
            rt.clearValue.color.f32[1] = 0.05f;
            rt.clearValue.color.f32[2] = 0.08f;
            rt.clearValue.color.f32[3] = 1.0f;
            VriAttachmentsDesc att {};
            att.colors            = &rt;
            att.colorNum          = 1;
            att.renderArea.width  = w;
            att.renderArea.height = h;
            att.layerNum          = 1;
            att.viewMask          = 0x3; // both eyes in one pass
            c.CmdBeginRendering(cmd, &att);
            VriViewport vp {0, 0, float(w), float(h), 0, 1};
            c.CmdSetViewports(cmd, &vp, 1);
            VriRect sc {0, 0, w, h};
            c.CmdSetScissors(cmd, &sc, 1);
            c.CmdSetPipeline(cmd, pipeline);
            VriDrawDesc draw {};
            draw.vertexNum   = 3;
            draw.instanceNum = 1;
            c.CmdDraw(cmd, &draw);
            c.CmdEndRendering(cmd);
            c.EndCommandBuffer(cmd);

            VriFenceSubmitDesc sig {};
            sig.fence = fence;
            sig.value = ++fenceValue;
            VriQueueSubmitDesc sub {};
            sub.commandBuffers   = &cmd;
            sub.commandBufferNum = 1;
            sub.signalFences     = &sig;
            sub.signalFenceNum   = 1;
            c.QueueSubmit(queue, &sub);
            c.Wait(fence, fenceValue);

            xrReleaseSwapchainImage(swapchain, nullptr);

            for (uint32_t eye = 0; eye < 2; ++eye)
            {
                projViews[eye].pose                      = xrViews[eye].pose;
                projViews[eye].fov                       = xrViews[eye].fov;
                projViews[eye].subImage.swapchain        = swapchain;
                projViews[eye].subImage.imageRect.offset = {0, 0};
                projViews[eye].subImage.imageRect.extent = {int32_t(w), int32_t(h)};
                projViews[eye].subImage.imageArrayIndex  = eye; // eye -> array layer
            }
            layer.space     = appSpace;
            layer.viewCount = 2;
            layer.views     = projViews.data();
            haveLayer       = true;
        }

        XrFrameEndInfo fei {XR_TYPE_FRAME_END_INFO};
        fei.displayTime                              = fs.predictedDisplayTime;
        fei.environmentBlendMode                     = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        const XrCompositionLayerBaseHeader* layers[] = {reinterpret_cast<XrCompositionLayerBaseHeader*>(&layer)};
        fei.layerCount                               = haveLayer ? 1u : 0u;
        fei.layers                                   = haveLayer ? layers : nullptr;
        xrEndFrame(session, &fei);
    }

    // ---- cleanup -------------------------------------------------------------------------------
    c.DeviceWaitIdle(dev);
    for (uint32_t i = 0; i < imgCount; ++i)
    {
        c.DestroyDescriptor(views[i]);
        c.DestroyTexture(wrapped[i]); // borrowed image: frees the view/wrapper, not OpenXR's VkImage
    }
    c.DestroyFence(fence);
    c.DestroyCommandAllocator(alloc);
    c.DestroyPipeline(pipeline);
    c.DestroyPipelineLayout(layout);
    xrDestroySwapchain(swapchain);
    xrDestroySpace(appSpace);
    xrDestroySession(session);
    vriDestroyDevice(dev);
    xrDestroyInstance(xrInstance);
    std::printf("[vri-openxr] clean exit\n");
    return 0;
}
