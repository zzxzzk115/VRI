/*
 * c_clean_check.c - compiled as C to guarantee the public headers stay
 * C-clean (no C++-only constructs leak into the public headers). Exercises a
 * broad slice of the API surface; existence + successful C compilation is the
 * test (the function is never called).
 */
#include <vri/vri.h>
#include <vri/integration/vri_native.h>

int vri_c_clean_check(void)
{
    VriDeviceCreationDesc deviceDesc;
    VriDevice* device = NULL;
    VriResult r;
    VriCoreInterface core;
    VriSwapChainInterface swapchain;
    VriInteropInterface interop;
    VriAdapterDesc adapters[4];
    uint32_t count;

    /* device creation, incl. the OpenXR-style required-extensions seam */
    const char* instExts[1];
    instExts[0] = "VK_KHR_surface";
    deviceDesc.graphicsAPI = VriGraphicsAPI_Vulkan;
    deviceDesc.adapterIndex = 0;
    deviceDesc.enableValidation = VRI_FALSE;
    deviceDesc.enabledFeatures = VriFeature_None;
    deviceDesc.bestEffort = VRI_TRUE;
    deviceDesc.requiredInstanceExtensions = instExts;
    deviceDesc.requiredInstanceExtensionNum = 1;
    deviceDesc.requiredDeviceExtensions = NULL;
    deviceDesc.requiredDeviceExtensionNum = 0;
    deviceDesc.nativeCreateInfo = NULL;
    deviceDesc.allocationCallbacks = NULL;
    deviceDesc.callbackInterface = NULL;

    count = vriEnumerateAdapters(VriGraphicsAPI_Vulkan, adapters, 4);
    (void)count;

    /* fill some real descriptor structs to prove they are POD-constructible in C */
    {
        VriBufferDesc bufferDesc;
        VriTextureDesc textureDesc;
        VriGraphicsPipelineDesc pipelineDesc;
        VriBarrierGroupDesc barrierGroup;
        VriTextureBarrierDesc texBarrier;
        VriAttachmentsDesc attachments;
        VriAttachmentDesc color;
        VriViewport viewport;
        VriWindowHandle window;

        bufferDesc.size = 1024;
        bufferDesc.structureStride = 0;
        bufferDesc.usage = VriBufferUsage_VertexBuffer | VriBufferUsage_TransferDst;

        textureDesc.type = VriTextureType_2D;
        textureDesc.format = VriFormat_RGBA8_UNORM;
        textureDesc.width = 256;
        textureDesc.height = 256;
        textureDesc.depth = 1;
        textureDesc.mipNum = 1;
        textureDesc.layerNum = 1;
        textureDesc.sampleNum = 1;
        textureDesc.usage = VriTextureUsage_ColorAttachment | VriTextureUsage_ShaderResource;

        pipelineDesc.pipelineLayout = NULL;
        pipelineDesc.shaders = NULL;
        pipelineDesc.shaderNum = 0;
        pipelineDesc.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
        pipelineDesc.rasterization.cullMode = VriCullMode_Back;
        pipelineDesc.rasterization.frontFace = VriFrontFace_CounterClockwise;
        pipelineDesc.rasterization.polygonMode = VriPolygonMode_Fill;
        pipelineDesc.depthStencil.depthTest = VRI_TRUE;
        pipelineDesc.depthStencil.depthCompareOp = VriCompareOp_LessOrEqual;
        pipelineDesc.outputMerger.colorNum = 0;
        pipelineDesc.outputMerger.depthStencilFormat = VriFormat_D32_SFLOAT;

        texBarrier.texture = NULL;
        texBarrier.before.access = VriAccess_None;
        texBarrier.before.layout = VriLayout_Undefined;
        texBarrier.before.stages = VriPipelineStage_None;
        texBarrier.after.access = VriAccess_ColorAttachmentWrite;
        texBarrier.after.layout = VriLayout_ColorAttachment;
        texBarrier.after.stages = VriPipelineStage_ColorAttachmentOutput;
        texBarrier.aspect = VriImageAspect_Color;
        texBarrier.baseMip = 0;
        texBarrier.mipNum = 0;
        texBarrier.baseLayer = 0;
        texBarrier.layerNum = 0;

        barrierGroup.buffers = NULL;
        barrierGroup.bufferNum = 0;
        barrierGroup.textures = &texBarrier;
        barrierGroup.textureNum = 1;

        color.view = NULL;
        color.loadOp = VriAttachmentLoadOp_Clear;
        color.storeOp = VriAttachmentStoreOp_Store;
        color.clearValue.color.f32[0] = 0.0f;

        attachments.colors = &color;
        attachments.colorNum = 1;
        attachments.depth = NULL;
        attachments.stencil = NULL;
        attachments.renderArea.x = 0;
        attachments.renderArea.y = 0;
        attachments.renderArea.width = 256;
        attachments.renderArea.height = 256;
        attachments.layerNum = 1;
        attachments.viewMask = 0;

        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = 256.0f;
        viewport.height = 256.0f;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        window = vriWindowHandleWin32(NULL, NULL);
        (void)window;
        (void)bufferDesc;
        (void)textureDesc;
        (void)pipelineDesc;
        (void)barrierGroup;
        (void)attachments;
        (void)viewport;
    }

    r = vriCreateDevice(&deviceDesc, &device);
    if (r == VriResult_Success && device != NULL)
    {
        if (vriGetInterface(device, VRI_INTERFACE_CORE, sizeof(core), &core) == VriResult_Success)
            (void)core.GetDeviceDesc;
        if (vriGetInterface(device, VRI_INTERFACE_SWAPCHAIN, sizeof(swapchain), &swapchain) == VriResult_Success)
            (void)swapchain.CreateSwapChain;
        if (vriGetInterface(device, VRI_INTERFACE_INTEROP, sizeof(interop), &interop) == VriResult_Success)
            (void)interop.GetDeviceNativeHandles;
        vriDestroyDevice(device);
    }

    return (int)r;
}
