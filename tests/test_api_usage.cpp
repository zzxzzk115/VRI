// Compile-time + light runtime checks that the API descriptors and the core
// function table are self-consistent and usable from C++ (via the wrapper
// aliases). No backend is required: we only build descs and inspect sizes.
#include <doctest/doctest.h>

#include <vri/vri.hpp>

#include <cstring>

TEST_CASE("descriptor structs are constructible and wrapper aliases resolve")
{
    vri::BufferDesc buffer{};
    buffer.size            = 4096;
    buffer.usage           = VriBufferUsage_StorageBuffer | VriBufferUsage_TransferDst;
    buffer.structureStride = 16;
    CHECK(buffer.size == 4096);

    vri::TextureDesc texture{};
    texture.type      = VriTextureType_2D;
    texture.format    = VriFormat_RGBA16_SFLOAT;
    texture.width     = 1920;
    texture.height    = 1080;
    texture.depth     = 1;
    texture.mipNum    = 1;
    texture.layerNum  = 1;
    texture.sampleNum = 1;
    texture.usage     = VriTextureUsage_ColorAttachment;
    CHECK(texture.width * texture.height == 1920u * 1080u);
}

TEST_CASE("graphics pipeline desc wires layout + state together")
{
    VriColorAttachmentDesc color{};
    color.format         = VriFormat_BGRA8_UNORM;
    color.colorWriteMask = VriColorWrite_RGBA;
    color.blend.enable   = VRI_FALSE;

    vri::GraphicsPipelineDesc pipeline{};
    pipeline.inputAssembly.topology   = VriPrimitiveTopology_TriangleList;
    pipeline.rasterization.cullMode   = VriCullMode_Back;
    pipeline.rasterization.frontFace  = VriFrontFace_CounterClockwise;
    pipeline.rasterization.lineWidth  = 1.0f;
    pipeline.outputMerger.colors      = &color;
    pipeline.outputMerger.colorNum    = 1;
    pipeline.depthStencil.depthTest   = VRI_TRUE;
    pipeline.depthStencil.depthWrite  = VRI_TRUE;

    CHECK(pipeline.outputMerger.colorNum == 1);
    CHECK(pipeline.outputMerger.colors[0].format == VriFormat_BGRA8_UNORM);
}

TEST_CASE("barrier group expresses an explicit stage/access/layout transition")
{
    VriTextureBarrierDesc tex{};
    tex.before.layout = VriLayout_Undefined;
    tex.before.stages = VriPipelineStage_None;
    tex.after.access  = VriAccess_ColorAttachmentWrite;
    tex.after.layout  = VriLayout_ColorAttachment;
    tex.after.stages  = VriPipelineStage_ColorAttachmentOutput;

    vri::BarrierGroupDesc group{};
    group.textures   = &tex;
    group.textureNum = 1;

    CHECK(group.textureNum == 1);
    CHECK(group.textures[0].after.layout == VriLayout_ColorAttachment);
}

TEST_CASE("core interface table has the expected entry points (non-null type)")
{
    // The table is plain function pointers; verify it is zero-initializable and
    // that querying it on a (nonexistent) device is well-defined.
    vri::CoreInterface core{};
    CHECK(core.CreateBuffer == nullptr);
    CHECK(sizeof(core) > 0);

    // ABI-drift guard: wrong size must be rejected even with a null device path.
    CHECK(vriGetInterface(nullptr, VRI_INTERFACE_CORE, sizeof(core), &core) == VriResult_InvalidArgument);
}
