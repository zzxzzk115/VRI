// Deferred shading: the classic two-pass G-buffer technique, cross-backend.
//
// Pass 1 (geometry / MRT) draws the whole scene - a textured floor + a grid of colored cubes -
// ONCE into three offscreen targets at the same time: world position (RGBA16F), surface normal
// (RGBA8, encoded), and albedo+specular (RGBA8). Pass 2 (the swapchain pass) is a single
// fullscreen triangle that reads those three targets back and shades every screen pixel against
// SIX moving point lights. The whole point of deferred: lighting cost scales with screen pixels
// and light count, not with how many (overdrawn) fragments the geometry produced - so the six
// lights are essentially free no matter how dense the scene gets.
//
// Builds on examples/offscreen (multiple render targets + the ColorAttachment->ShaderResource
// barrier dance) and examples/pbrbasic (per-instance object data via a vertex stream, so it runs
// on every backend incl. the web - no push constants needed). An ImGui combo switches the
// composite output to any single G-buffer plane (position / normal / albedo / specular) so you
// can see exactly what the geometry pass wrote. VRI_API / ?backend force a backend; the floor
// texture is reused from the cube example (no asset duplication). Shared host scaffolding lives
// in examples/common/example_app.h.
#include "../common/example_app.h"

#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include "../cube/mat4.h"
#include "../cube/ktx.h"

#include "tests/shaders/deferred_mrt_spv.h"    // g_deferredMrtSpv    (Vulkan + OpenGL)
#include "tests/shaders/deferred_mrt_wgsl.h"   // g_deferredMrtWgsl   (WebGPU)
#include "tests/shaders/deferred_mrt_dxbc.h"   // g_deferredMrtDxbcVS / PS (D3D12)
#include "tests/shaders/deferred_light_spv.h"  // g_deferredLightSpv
#include "tests/shaders/deferred_light_wgsl.h" // g_deferredLightWgsl
#include "tests/shaders/deferred_light_dxbc.h" // g_deferredLightDxbcVS / PS

namespace
{
    constexpr uint32_t kWidth = 640, kHeight = 480; // width*4 is 256-aligned (readback pitch)
    constexpr int kGrid = 6;          // kGrid x kGrid cubes
    constexpr float kSpacing = 3.0f;
    constexpr int kLightCount = 6;

    // G-buffer plane formats. Position needs range+precision (a float target); the normal is
    // encoded into [0,1] and the albedo+spec packs fine into plain RGBA8 (both universally
    // renderable). One float target keeps the example portable while staying a "fat" G-buffer.
    constexpr VriFormat kPositionFormat = VriFormat_RGBA16_SFLOAT;
    constexpr VriFormat kNormalFormat = VriFormat_RGBA8_UNORM;
    constexpr VriFormat kAlbedoFormat = VriFormat_RGBA8_UNORM;

    struct Vertex { float px, py, pz; float nx, ny, nz; float u, v; };
    struct Instance { float a[4]; float b[4]; }; // a: offset.xyz + uvScale, b: tint.rgb + specular

    // Unit cube centered at the origin, with per-face normals + uvs (24 verts, 36 indices).
    void BuildCube(std::vector<Vertex>& v, std::vector<uint16_t>& idx)
    {
        const float h = 0.5f;
        struct Face { float n[3]; float t[3]; float b[3]; }; // normal + the face's two in-plane axes
        const Face faces[6] = {
            {{ 0, 0, 1}, { 1, 0, 0}, { 0, 1, 0}}, // +Z
            {{ 0, 0,-1}, {-1, 0, 0}, { 0, 1, 0}}, // -Z
            {{ 1, 0, 0}, { 0, 0,-1}, { 0, 1, 0}}, // +X
            {{-1, 0, 0}, { 0, 0, 1}, { 0, 1, 0}}, // -X
            {{ 0, 1, 0}, { 1, 0, 0}, { 0, 0,-1}}, // +Y
            {{ 0,-1, 0}, { 1, 0, 0}, { 0, 0, 1}}, // -Y
        };
        const float quv[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
        const float qs[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}; // corner signs in (t, b)
        for (const Face& f : faces)
        {
            const uint16_t base = static_cast<uint16_t>(v.size());
            for (int c = 0; c < 4; ++c)
            {
                Vertex vert{};
                vert.px = (f.n[0] + f.t[0] * qs[c][0] + f.b[0] * qs[c][1]) * h;
                vert.py = (f.n[1] + f.t[1] * qs[c][0] + f.b[1] * qs[c][1]) * h;
                vert.pz = (f.n[2] + f.t[2] * qs[c][0] + f.b[2] * qs[c][1]) * h;
                vert.nx = f.n[0]; vert.ny = f.n[1]; vert.nz = f.n[2];
                vert.u = quv[c][0]; vert.v = quv[c][1];
                v.push_back(vert);
            }
            idx.push_back(base); idx.push_back(uint16_t(base + 1)); idx.push_back(uint16_t(base + 2));
            idx.push_back(base); idx.push_back(uint16_t(base + 2)); idx.push_back(uint16_t(base + 3));
        }
    }

    // A large floor quad in the XZ plane at y=0, normal +Y, uv 0..1 (tiled via the instance's uvScale).
    void BuildFloor(std::vector<Vertex>& v, std::vector<uint16_t>& idx, float size)
    {
        const float s = size * 0.5f;
        v.push_back({-s, 0, -s, 0, 1, 0, 0, 0});
        v.push_back({ s, 0, -s, 0, 1, 0, 1, 0});
        v.push_back({ s, 0,  s, 0, 1, 0, 1, 1});
        v.push_back({-s, 0,  s, 0, 1, 0, 0, 1});
        idx = {0, 1, 2, 0, 2, 3};
    }

    // GPU layout (must match deferred_light.slang's Ubo: cbuffer 16-byte rules, all float4/int4).
    // Parallel position/color arrays (not an array of structs) - an array-of-structs UBO
    // transpiles to ESSL that WebGL2/ANGLE rejects; flat float4 arrays are portable.
    struct LightUbo { float lightPosition[kLightCount][4]; float lightColor[kLightCount][4]; float viewPos[4]; int32_t params[4]; };

    bool g_gInit = false;      // G-buffer color targets written at least once (layout tracking)
    bool g_gDepthInit = false;

    VriTexture* MakeAttachment(vriex::ExampleApp& app, VriFormat fmt, VriTextureUsageFlags usage, VriImageAspectFlags aspect, VriDescriptor** outView, VriClearValue clear = {})
    {
        VriTextureDesc td{}; td.type = VriTextureType_2D; td.format = fmt; td.width = kWidth; td.height = kHeight; td.depth = 1;
        td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1; td.usage = usage; td.memoryLocation = VriMemoryLocation_Device;
        td.clearValue = clear; // match the render-pass clear (G-buffers clear to 0; depth to 1.0)
        VriTexture* t = nullptr; if (app.c.CreateTexture(app.dev, &td, &t) != VriResult_Success) app.Fail("G-buffer CreateTexture failed");
        VriTextureViewDesc vd{}; vd.texture = t; vd.viewType = VriTextureViewType_2D; vd.format = VriFormat_Unknown; vd.aspect = aspect;
        if (app.c.CreateTextureView(app.dev, &vd, outView) != VriResult_Success) app.Fail("G-buffer view failed");
        return t;
    }
} // namespace

int main(int, char**)
{
    // ---- geometry: a cube mesh + a floor quad, both {position, normal, uv} ----
    std::vector<Vertex> cubeV; std::vector<uint16_t> cubeI; BuildCube(cubeV, cubeI);
    std::vector<Vertex> floorV; std::vector<uint16_t> floorI; BuildFloor(floorV, floorI, kGrid * kSpacing + 6.0f);
    const uint32_t cubeIndexNum = static_cast<uint32_t>(cubeI.size());
    const uint32_t floorIndexNum = static_cast<uint32_t>(floorI.size());

    // cube instances: a grid resting on the floor (bottom at y=0 => center y=0.5), colored
    std::vector<Instance> cubeInstances;
    const float half = (kGrid - 1) * kSpacing * 0.5f;
    const float palette[6][3] = {
        {0.90f, 0.30f, 0.25f}, {0.30f, 0.75f, 0.40f}, {0.25f, 0.45f, 0.95f},
        {0.95f, 0.80f, 0.25f}, {0.70f, 0.35f, 0.85f}, {0.25f, 0.80f, 0.85f}};
    for (int z = 0; z < kGrid; ++z)
        for (int x = 0; x < kGrid; ++x)
        {
            Instance in{};
            in.a[0] = x * kSpacing - half; in.a[1] = 0.5f; in.a[2] = z * kSpacing - half; in.a[3] = 1.0f; // uvScale 1
            const float* col = palette[(x + z) % 6];
            in.b[0] = col[0]; in.b[1] = col[1]; in.b[2] = col[2];
            in.b[3] = 0.25f + 0.75f * float(x) / float(kGrid - 1); // specular varies across X
            cubeInstances.push_back(in);
        }
    const uint32_t cubeInstanceNum = static_cast<uint32_t>(cubeInstances.size());

    // floor: one instance, neutral tint, uv tiled 8x, low specular
    Instance floorInst{}; floorInst.a[0] = 0; floorInst.a[1] = 0; floorInst.a[2] = 0; floorInst.a[3] = 8.0f;
    floorInst.b[0] = 0.55f; floorInst.b[1] = 0.55f; floorInst.b[2] = 0.58f; floorInst.b[3] = 0.2f;

    static vriex::ExampleApp app;
    app.Init("deferred", kWidth, kHeight, /*hasDepth*/ false); // swapchain pass is the fullscreen lighting composite
    VriCoreInterface& c = app.c;
    const bool useWgsl = app.useWgsl, useDxbc = app.useDxbc;

    // ---- floor texture (reused from the cube example) ----
    KtxImage tex;
#if defined(__EMSCRIPTEN__)
    if (!LoadKtxRgba("/metalplate01_rgba.ktx", tex) && !LoadKtxRgba("metalplate01_rgba.ktx", tex))
        app.Fail("failed to load metalplate01_rgba.ktx");
#else
    {
        const char* base = SDL_GetBasePath();
        std::string path = (base ? std::string(base) : std::string()) + "metalplate01_rgba.ktx";
        if (!LoadKtxRgba(path.c_str(), tex) && !LoadKtxRgba("metalplate01_rgba.ktx", tex))
            app.Fail("failed to load metalplate01_rgba.ktx");
    }
#endif

    auto makeDeviceBuf = [&](uint64_t size, VriBufferUsageFlags usage, const void* src) {
        VriBufferDesc bd{}; bd.size = size; bd.usage = usage | VriBufferUsage_TransferDst; bd.memoryLocation = VriMemoryLocation_Device;
        VriBuffer* b = nullptr; c.CreateBuffer(app.dev, &bd, &b);
        VriBufferDesc sd{}; sd.size = size; sd.usage = VriBufferUsage_TransferSrc; sd.memoryLocation = VriMemoryLocation_HostUpload;
        VriBuffer* s = nullptr; c.CreateBuffer(app.dev, &sd, &s);
        std::memcpy(c.MapBuffer(s, 0, size), src, size); c.UnmapBuffer(s);
        return std::pair<VriBuffer*, VriBuffer*>(b, s);
    };
    const uint64_t cubeVBytes = cubeV.size() * sizeof(Vertex), cubeIBytes = cubeI.size() * sizeof(uint16_t);
    const uint64_t floorVBytes = floorV.size() * sizeof(Vertex), floorIBytes = floorI.size() * sizeof(uint16_t);
    const uint64_t cubeInstBytes = cubeInstances.size() * sizeof(Instance), floorInstBytes = sizeof(Instance);
    auto [cubeVB, cubeVStg] = makeDeviceBuf(cubeVBytes, VriBufferUsage_VertexBuffer, cubeV.data());
    auto [cubeIB, cubeIStg] = makeDeviceBuf(cubeIBytes, VriBufferUsage_IndexBuffer, cubeI.data());
    auto [floorVB, floorVStg] = makeDeviceBuf(floorVBytes, VriBufferUsage_VertexBuffer, floorV.data());
    auto [floorIB, floorIStg] = makeDeviceBuf(floorIBytes, VriBufferUsage_IndexBuffer, floorI.data());
    auto [cubeInstB, cubeInstStg] = makeDeviceBuf(cubeInstBytes, VriBufferUsage_VertexBuffer, cubeInstances.data());
    auto [floorInstB, floorInstStg] = makeDeviceBuf(floorInstBytes, VriBufferUsage_VertexBuffer, &floorInst);

    // ---- constant buffers: camera view-projection (MRT pass) + lights (lighting pass) ----
    VriBufferDesc cbd{}; cbd.size = sizeof(Mat4); cbd.usage = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst; cbd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* camUbo = nullptr; c.CreateBuffer(app.dev, &cbd, &camUbo);
    VriBufferDesc csd{}; csd.size = sizeof(Mat4); csd.usage = VriBufferUsage_TransferSrc; csd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* camStg = nullptr; c.CreateBuffer(app.dev, &csd, &camStg);
    VriBufferViewDesc cbv{}; cbv.buffer = camUbo; cbv.viewType = VriDescriptorType_ConstantBuffer; cbv.offset = 0; cbv.size = sizeof(Mat4);
    VriDescriptor* camView = nullptr; c.CreateBufferView(app.dev, &cbv, &camView);

    VriBufferDesc lbd{}; lbd.size = sizeof(LightUbo); lbd.usage = VriBufferUsage_ConstantBuffer | VriBufferUsage_TransferDst; lbd.memoryLocation = VriMemoryLocation_Device;
    VriBuffer* lightUbo = nullptr; c.CreateBuffer(app.dev, &lbd, &lightUbo);
    VriBufferDesc lsd{}; lsd.size = sizeof(LightUbo); lsd.usage = VriBufferUsage_TransferSrc; lsd.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* lightStg = nullptr; c.CreateBuffer(app.dev, &lsd, &lightStg);
    VriBufferViewDesc lbv{}; lbv.buffer = lightUbo; lbv.viewType = VriDescriptorType_ConstantBuffer; lbv.offset = 0; lbv.size = sizeof(LightUbo);
    VriDescriptor* lightView = nullptr; c.CreateBufferView(app.dev, &lbv, &lightView);

    // ---- floor texture (device-local) + staging + samplers ----
    VriTextureDesc ttd{}; ttd.type = VriTextureType_2D; ttd.format = VriFormat_RGBA8_UNORM; ttd.width = tex.width; ttd.height = tex.height; ttd.depth = 1;
    ttd.mipNum = 1; ttd.layerNum = 1; ttd.sampleNum = 1; ttd.usage = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst; ttd.memoryLocation = VriMemoryLocation_Device;
    VriTexture* texture = nullptr; c.CreateTexture(app.dev, &ttd, &texture);
    VriTextureViewDesc tvd{}; tvd.texture = texture; tvd.viewType = VriTextureViewType_2D; tvd.format = VriFormat_Unknown; tvd.aspect = VriImageAspect_Color;
    VriDescriptor* texView = nullptr; c.CreateTextureView(app.dev, &tvd, &texView);
    VriBufferDesc stg{}; stg.size = tex.rgba.size(); stg.usage = VriBufferUsage_TransferSrc; stg.memoryLocation = VriMemoryLocation_HostUpload;
    VriBuffer* staging = nullptr; c.CreateBuffer(app.dev, &stg, &staging);
    std::memcpy(c.MapBuffer(staging, 0, tex.rgba.size()), tex.rgba.data(), tex.rgba.size()); c.UnmapBuffer(staging);

    VriSamplerDesc rsmp{}; rsmp.magFilter = VriFilter_Linear; rsmp.minFilter = VriFilter_Linear; rsmp.mipmapMode = VriMipmapMode_Nearest;
    rsmp.addressModeU = VriAddressMode_Repeat; rsmp.addressModeV = VriAddressMode_Repeat; rsmp.addressModeW = VriAddressMode_Repeat; rsmp.maxLod = 1.0f;
    VriDescriptor* repeatSampler = nullptr; c.CreateSampler(app.dev, &rsmp, &repeatSampler);
    // The lighting pass samples the G-buffer 1:1 (no filtering): nearest + clamp.
    VriSamplerDesc gsmp{}; gsmp.magFilter = VriFilter_Nearest; gsmp.minFilter = VriFilter_Nearest; gsmp.mipmapMode = VriMipmapMode_Nearest;
    gsmp.addressModeU = VriAddressMode_ClampToEdge; gsmp.addressModeV = VriAddressMode_ClampToEdge; gsmp.addressModeW = VriAddressMode_ClampToEdge; gsmp.maxLod = 1.0f;
    VriDescriptor* gbufferSampler = nullptr; c.CreateSampler(app.dev, &gsmp, &gbufferSampler);

    // ---- the G-buffer: 3 color targets (sampled later) + a depth target ----
    VriDescriptor* gPosView = nullptr; VriDescriptor* gNormalView = nullptr; VriDescriptor* gAlbedoView = nullptr; VriDescriptor* gDepthView = nullptr;
    VriTexture* gPos = MakeAttachment(app, kPositionFormat, VriTextureUsage_ColorAttachment | VriTextureUsage_ShaderResource, VriImageAspect_Color, &gPosView);
    VriTexture* gNormal = MakeAttachment(app, kNormalFormat, VriTextureUsage_ColorAttachment | VriTextureUsage_ShaderResource, VriImageAspect_Color, &gNormalView);
    VriTexture* gAlbedo = MakeAttachment(app, kAlbedoFormat, VriTextureUsage_ColorAttachment | VriTextureUsage_ShaderResource, VriImageAspect_Color, &gAlbedoView);
    VriClearValue gDepthClear{}; gDepthClear.depthStencil.depth = 1.0f;
    VriTexture* gDepth = MakeAttachment(app, app.depthFormat, VriTextureUsage_DepthStencilAttachment, VriImageAspect_Depth, &gDepthView, gDepthClear);

    // ---- MRT (geometry) pipeline: CB@0 (vertex), Texture@1 + Sampler@2 (fragment); 3 colors + depth ----
    VriDescriptorRangeDesc mr[3]{};
    mr[0].baseRegister = 0; mr[0].descriptorNum = 1; mr[0].descriptorType = VriDescriptorType_ConstantBuffer; mr[0].shaderStages = VriShaderStage_Vertex;
    mr[1].baseRegister = 1; mr[1].descriptorNum = 1; mr[1].descriptorType = VriDescriptorType_Texture;        mr[1].shaderStages = VriShaderStage_Fragment;
    mr[2].baseRegister = 2; mr[2].descriptorNum = 1; mr[2].descriptorType = VriDescriptorType_Sampler;        mr[2].shaderStages = VriShaderStage_Fragment;
    VriDescriptorSetDesc msd{}; msd.registerSpace = 0; msd.ranges = mr; msd.rangeNum = 3;
    VriPipelineLayoutDesc mld{}; mld.descriptorSets = &msd; mld.descriptorSetNum = 1;
    VriPipelineLayout* mrtLayout = nullptr; c.CreatePipelineLayout(app.dev, &mld, &mrtLayout);

    VriShaderDesc msh[2]{};
    msh[0].stage = VriShaderStage_Vertex;   msh[0].entryPointName = "vertexMain";
    msh[1].stage = VriShaderStage_Fragment; msh[1].entryPointName = "fragmentMain";
    if (useDxbc) { msh[0].bytecode = g_deferredMrtDxbcVS; msh[0].bytecodeSize = sizeof(g_deferredMrtDxbcVS); msh[1].bytecode = g_deferredMrtDxbcPS; msh[1].bytecodeSize = sizeof(g_deferredMrtDxbcPS); }
    else { msh[0].bytecode = msh[1].bytecode = useWgsl ? static_cast<const void*>(g_deferredMrtWgsl) : static_cast<const void*>(g_deferredMrtSpv);
           msh[0].bytecodeSize = msh[1].bytecodeSize = useWgsl ? sizeof(g_deferredMrtWgsl) : sizeof(g_deferredMrtSpv); }

    VriVertexAttributeDesc mattrs[5]{};
    mattrs[0].format = VriFormat_RGB32_SFLOAT;  mattrs[0].offset = 0;  mattrs[0].streamIndex = 0; // position
    mattrs[1].format = VriFormat_RGB32_SFLOAT;  mattrs[1].offset = 12; mattrs[1].streamIndex = 0; // normal
    mattrs[2].format = VriFormat_RG32_SFLOAT;   mattrs[2].offset = 24; mattrs[2].streamIndex = 0; // uv
    mattrs[3].format = VriFormat_RGBA32_SFLOAT; mattrs[3].offset = 0;  mattrs[3].streamIndex = 1; // instA
    mattrs[4].format = VriFormat_RGBA32_SFLOAT; mattrs[4].offset = 16; mattrs[4].streamIndex = 1; // instB
    VriVertexStreamDesc mstreams[2]{};
    mstreams[0].stride = sizeof(Vertex);   mstreams[0].bindingSlot = 0; mstreams[0].stepRate = VriVertexStepRate_PerVertex;
    mstreams[1].stride = sizeof(Instance); mstreams[1].bindingSlot = 1; mstreams[1].stepRate = VriVertexStepRate_PerInstance;

    VriColorAttachmentDesc gca[3]{};
    gca[0].format = kPositionFormat; gca[0].colorWriteMask = VriColorWrite_RGBA;
    gca[1].format = kNormalFormat;   gca[1].colorWriteMask = VriColorWrite_RGBA;
    gca[2].format = kAlbedoFormat;   gca[2].colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc mpd{};
    mpd.pipelineLayout = mrtLayout; mpd.shaders = msh; mpd.shaderNum = 2;
    mpd.vertexInput.attributes = mattrs; mpd.vertexInput.attributeNum = 5; mpd.vertexInput.streams = mstreams; mpd.vertexInput.streamNum = 2;
    mpd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
    mpd.rasterization.cullMode = VriCullMode_None; mpd.rasterization.frontFace = VriFrontFace_CounterClockwise; mpd.rasterization.lineWidth = 1.0f;
    mpd.multisample.sampleNum = 1;
    mpd.depthStencil.depthTest = VRI_TRUE; mpd.depthStencil.depthWrite = VRI_TRUE; mpd.depthStencil.depthCompareOp = VriCompareOp_Less;
    mpd.outputMerger.colors = gca; mpd.outputMerger.colorNum = 3; mpd.outputMerger.depthStencilFormat = app.depthFormat;
    VriPipeline* mrtPipeline = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &mpd, &mrtPipeline) != VriResult_Success) app.Fail("MRT CreateGraphicsPipeline failed");

    // ---- lighting pipeline: CB@0 + Texture@1..3 + Sampler@4 (fragment); 1 swapchain color, no depth ----
    // (CB at register 0 so the D3D12 root CBV lands on b0, matching Slang - see deferred_light.slang.)
    VriDescriptorRangeDesc lr[5]{};
    lr[0].baseRegister = 0; lr[0].descriptorNum = 1; lr[0].descriptorType = VriDescriptorType_ConstantBuffer; lr[0].shaderStages = VriShaderStage_Fragment;
    lr[1].baseRegister = 1; lr[1].descriptorNum = 1; lr[1].descriptorType = VriDescriptorType_Texture;        lr[1].shaderStages = VriShaderStage_Fragment;
    lr[2].baseRegister = 2; lr[2].descriptorNum = 1; lr[2].descriptorType = VriDescriptorType_Texture;        lr[2].shaderStages = VriShaderStage_Fragment;
    lr[3].baseRegister = 3; lr[3].descriptorNum = 1; lr[3].descriptorType = VriDescriptorType_Texture;        lr[3].shaderStages = VriShaderStage_Fragment;
    lr[4].baseRegister = 4; lr[4].descriptorNum = 1; lr[4].descriptorType = VriDescriptorType_Sampler;        lr[4].shaderStages = VriShaderStage_Fragment;
    VriDescriptorSetDesc lsetd{}; lsetd.registerSpace = 0; lsetd.ranges = lr; lsetd.rangeNum = 5;
    VriPipelineLayoutDesc lld{}; lld.descriptorSets = &lsetd; lld.descriptorSetNum = 1;
    VriPipelineLayout* lightLayout = nullptr; c.CreatePipelineLayout(app.dev, &lld, &lightLayout);

    VriShaderDesc lsh[2]{};
    lsh[0].stage = VriShaderStage_Vertex;   lsh[0].entryPointName = "vertexMain";
    lsh[1].stage = VriShaderStage_Fragment; lsh[1].entryPointName = "fragmentMain";
    if (useDxbc) { lsh[0].bytecode = g_deferredLightDxbcVS; lsh[0].bytecodeSize = sizeof(g_deferredLightDxbcVS); lsh[1].bytecode = g_deferredLightDxbcPS; lsh[1].bytecodeSize = sizeof(g_deferredLightDxbcPS); }
    else { lsh[0].bytecode = lsh[1].bytecode = useWgsl ? static_cast<const void*>(g_deferredLightWgsl) : static_cast<const void*>(g_deferredLightSpv);
           lsh[0].bytecodeSize = lsh[1].bytecodeSize = useWgsl ? sizeof(g_deferredLightWgsl) : sizeof(g_deferredLightSpv); }

    VriColorAttachmentDesc lca{}; lca.format = app.swapFormat; lca.colorWriteMask = VriColorWrite_RGBA;
    VriGraphicsPipelineDesc lpd{};
    lpd.pipelineLayout = lightLayout; lpd.shaders = lsh; lpd.shaderNum = 2;
    lpd.inputAssembly.topology = VriPrimitiveTopology_TriangleList;
    lpd.rasterization.cullMode = VriCullMode_None; lpd.rasterization.lineWidth = 1.0f;
    lpd.multisample.sampleNum = 1; lpd.outputMerger.colors = &lca; lpd.outputMerger.colorNum = 1;
    VriPipeline* lightPipeline = nullptr;
    if (c.CreateGraphicsPipeline(app.dev, &lpd, &lightPipeline) != VriResult_Success) app.Fail("lighting CreateGraphicsPipeline failed");

    // ---- descriptor sets ----
    VriDescriptorPoolDesc pdsc{}; pdsc.descriptorSetMaxNum = 2; pdsc.constantBufferMaxNum = 2; pdsc.textureMaxNum = 4; pdsc.samplerMaxNum = 2;
    VriDescriptorPool* pool = nullptr; c.CreateDescriptorPool(app.dev, &pdsc, &pool);
    VriDescriptorSet* mrtSet = nullptr; c.AllocateDescriptorSets(pool, mrtLayout, 0, &mrtSet, 1);
    VriDescriptorSet* lightSet = nullptr; c.AllocateDescriptorSets(pool, lightLayout, 0, &lightSet, 1);
    { const VriDescriptor* a[1] = {camView}; const VriDescriptor* b[1] = {texView}; const VriDescriptor* s[1] = {repeatSampler};
      VriDescriptorRangeUpdateDesc u[3]{}; u[0].descriptors=a; u[0].descriptorNum=1; u[1].descriptors=b; u[1].descriptorNum=1; u[2].descriptors=s; u[2].descriptorNum=1; c.UpdateDescriptorRanges(mrtSet, 0, 3, u); }
    { const VriDescriptor* a[1] = {lightView}; const VriDescriptor* b[1] = {gPosView}; const VriDescriptor* d[1] = {gNormalView}; const VriDescriptor* e[1] = {gAlbedoView}; const VriDescriptor* s[1] = {gbufferSampler};
      VriDescriptorRangeUpdateDesc u[5]{}; u[0].descriptors=a; u[0].descriptorNum=1; u[1].descriptors=b; u[1].descriptorNum=1; u[2].descriptors=d; u[2].descriptorNum=1; u[3].descriptors=e; u[3].descriptorNum=1; u[4].descriptors=s; u[4].descriptorNum=1; c.UpdateDescriptorRanges(lightSet, 0, 5, u); }

    // ---- one-time upload: all geometry buffers + the floor texture (fence value 1) ----
    {
        VriCommandBuffer* cmd = app.cmd;
        c.BeginCommandBuffer(cmd);
        struct Up { VriBuffer* dst; VriBuffer* src; uint64_t size; VriAccessFlags access; };
        const Up ups[6] = {
            {cubeVB, cubeVStg, cubeVBytes, VriAccess_VertexBufferRead},
            {cubeIB, cubeIStg, cubeIBytes, VriAccess_IndexBufferRead},
            {floorVB, floorVStg, floorVBytes, VriAccess_VertexBufferRead},
            {floorIB, floorIStg, floorIBytes, VriAccess_IndexBufferRead},
            {cubeInstB, cubeInstStg, cubeInstBytes, VriAccess_VertexBufferRead},
            {floorInstB, floorInstStg, floorInstBytes, VriAccess_VertexBufferRead},
        };
        VriBufferBarrierDesc gb[6]{};
        for (int i = 0; i < 6; ++i)
        {
            VriBufferCopyDesc cp{}; cp.size = ups[i].size; c.CmdCopyBuffer(cmd, ups[i].dst, ups[i].src, &cp);
            gb[i].buffer = ups[i].dst; gb[i].before.access = VriAccess_CopyDestinationWrite; gb[i].before.stages = VriPipelineStage_Transfer;
            gb[i].after.access = ups[i].access; gb[i].after.stages = VriPipelineStage_VertexInput;
        }
        VriBarrierGroupDesc gbg{}; gbg.buffers = gb; gbg.bufferNum = 6; c.CmdBarrier(cmd, &gbg);

        VriTextureBarrierDesc tb{}; tb.texture = texture; tb.before.layout = VriLayout_Undefined; tb.before.stages = VriPipelineStage_None;
        tb.after.access = VriAccess_CopyDestinationWrite; tb.after.layout = VriLayout_CopyDestination; tb.after.stages = VriPipelineStage_Transfer; tb.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc g0{}; g0.textures = &tb; g0.textureNum = 1; c.CmdBarrier(cmd, &g0);
        VriBufferTextureCopyDesc up{}; up.texture.aspect = VriImageAspect_Color; up.texture.layerNum = 1; up.texture.width = tex.width; up.texture.height = tex.height;
        c.CmdUploadBufferToTexture(cmd, texture, staging, &up);
        VriTextureBarrierDesc tb2{}; tb2.texture = texture; tb2.before.access = VriAccess_CopyDestinationWrite; tb2.before.layout = VriLayout_CopyDestination; tb2.before.stages = VriPipelineStage_Transfer;
        tb2.after.access = VriAccess_ShaderResourceRead; tb2.after.layout = VriLayout_ShaderResource; tb2.after.stages = VriPipelineStage_FragmentShader; tb2.aspect = VriImageAspect_Color;
        VriBarrierGroupDesc g1{}; g1.textures = &tb2; g1.textureNum = 1; c.CmdBarrier(cmd, &g1);
        c.EndCommandBuffer(cmd);
        VriFenceSubmitDesc sig{}; sig.fence = app.fence; sig.value = 1;
        VriQueueSubmitDesc sub{}; sub.commandBuffers = &cmd; sub.commandBufferNum = 1; sub.signalFences = &sig; sub.signalFenceNum = 1;
        c.QueueSubmit(app.queue, &sub); c.Wait(app.fence, 1);
    }

    static float orbit = 1.0f, t = 0.0f, lightT = 0.0f; static bool animate = true; static int debugTarget = 0;
    app.onUpdate = [camStg, lightStg](uint64_t) {
        t += 0.2f * orbit * app.dt;
        if (animate) lightT += 0.6f * app.dt;
        const float R = kGrid * kSpacing + 12.0f;
        const float eye[3] = {std::sin(t) * R, R * 0.55f, std::cos(t) * R}, ctr[3] = {0, 0.5f, 0}, up[3] = {0, 1, 0};

        Mat4 vp = Transpose(Mul(Perspective(0.9f, float(kWidth) / float(kHeight), 0.5f, 300.0f), LookAt(eye, ctr, up)));
        std::memcpy(app.c.MapBuffer(camStg, 0, sizeof(Mat4)), &vp, sizeof(Mat4)); app.c.UnmapBuffer(camStg);

        LightUbo lu{};
        const float lightColors[kLightCount][3] = {
            {1.0f, 1.0f, 1.0f}, {1.0f, 0.25f, 0.15f}, {0.2f, 1.0f, 0.3f},
            {0.25f, 0.4f, 1.0f}, {1.0f, 0.85f, 0.2f}, {1.0f, 0.3f, 0.9f}};
        const float ring = kGrid * kSpacing * 0.45f;
        for (int i = 0; i < kLightCount; ++i)
        {
            const float a = lightT + float(i) * (6.2831853f / kLightCount);
            const float rad = ring * (0.6f + 0.4f * std::sin(lightT * 0.7f + float(i)));
            lu.lightPosition[i][0] = std::cos(a) * rad;
            lu.lightPosition[i][1] = 3.0f + 1.5f * std::sin(lightT + float(i) * 1.3f);
            lu.lightPosition[i][2] = std::sin(a) * rad;
            lu.lightPosition[i][3] = 1.0f;
            lu.lightColor[i][0] = lightColors[i][0]; lu.lightColor[i][1] = lightColors[i][1]; lu.lightColor[i][2] = lightColors[i][2];
            lu.lightColor[i][3] = 20.0f; // radius (attenuation strength)
        }
        lu.viewPos[0] = eye[0]; lu.viewPos[1] = eye[1]; lu.viewPos[2] = eye[2];
        lu.params[0] = debugTarget;
        std::memcpy(app.c.MapBuffer(lightStg, 0, sizeof(LightUbo)), &lu, sizeof(LightUbo)); app.c.UnmapBuffer(lightStg);
    };
    app.onGui = [] {
        ImGui::SliderFloat("orbit", &orbit, 0.0f, 4.0f);
        ImGui::Checkbox("animate lights", &animate);
        const char* targets[] = {"Final", "Position", "Normal", "Albedo", "Specular"};
        ImGui::Combo("G-buffer", &debugTarget, targets, 5);
        ImGui::Text("%d point lights, 1 lighting pass", kLightCount);
    };

    // onPreRender records the ENTIRE geometry (MRT) pass: refresh the constant buffers, fill the
    // three G-buffer targets, then transition them to sampleable for the lighting pass.
    app.onPreRender = [=](VriCommandBuffer* cmd) {
        // refresh both constant buffers from their staging copies
        VriBufferCopyDesc ccp{}; ccp.size = sizeof(Mat4); c.CmdCopyBuffer(cmd, camUbo, camStg, &ccp);
        VriBufferCopyDesc lcp{}; lcp.size = sizeof(LightUbo); c.CmdCopyBuffer(cmd, lightUbo, lightStg, &lcp);
        VriBufferBarrierDesc ub[2]{};
        ub[0].buffer = camUbo; ub[0].before.access = VriAccess_CopyDestinationWrite; ub[0].before.stages = VriPipelineStage_Transfer;
        ub[0].after.access = VriAccess_ConstantBufferRead; ub[0].after.stages = VriPipelineStage_VertexShader;
        ub[1].buffer = lightUbo; ub[1].before.access = VriAccess_CopyDestinationWrite; ub[1].before.stages = VriPipelineStage_Transfer;
        ub[1].after.access = VriAccess_ConstantBufferRead; ub[1].after.stages = VriPipelineStage_FragmentShader;
        VriBarrierGroupDesc ubg{}; ubg.buffers = ub; ubg.bufferNum = 2; c.CmdBarrier(cmd, &ubg);

        // G-buffer targets -> writable (3 colors + depth)
        VriTextureBarrierDesc tb[4]{};
        tb[0].texture = gPos; tb[0].before.layout = g_gInit ? VriLayout_ShaderResource : VriLayout_Undefined; tb[0].before.stages = g_gInit ? VriPipelineStage_FragmentShader : VriPipelineStage_None;
        tb[0].after.access = VriAccess_ColorAttachmentWrite; tb[0].after.layout = VriLayout_ColorAttachment; tb[0].after.stages = VriPipelineStage_ColorAttachmentOutput; tb[0].aspect = VriImageAspect_Color;
        tb[1] = tb[0]; tb[1].texture = gNormal;
        tb[2] = tb[0]; tb[2].texture = gAlbedo;
        tb[3].texture = gDepth; tb[3].before.layout = g_gDepthInit ? VriLayout_DepthStencilAttachment : VriLayout_Undefined; tb[3].before.stages = VriPipelineStage_None;
        tb[3].after.access = VriAccess_DepthStencilAttachmentWrite; tb[3].after.layout = VriLayout_DepthStencilAttachment; tb[3].after.stages = VriPipelineStage_EarlyFragmentTests; tb[3].aspect = VriImageAspect_Depth;
        VriBarrierGroupDesc gw{}; gw.textures = tb; gw.textureNum = 4; c.CmdBarrier(cmd, &gw);
        g_gInit = true; g_gDepthInit = true;

        VriAttachmentDesc colors[3]{};
        colors[0].view = gPosView;    colors[0].loadOp = VriAttachmentLoadOp_Clear; colors[0].storeOp = VriAttachmentStoreOp_Store; // position clears to 0
        colors[1].view = gNormalView; colors[1].loadOp = VriAttachmentLoadOp_Clear; colors[1].storeOp = VriAttachmentStoreOp_Store;
        colors[2].view = gAlbedoView; colors[2].loadOp = VriAttachmentLoadOp_Clear; colors[2].storeOp = VriAttachmentStoreOp_Store; // albedo+spec clears to 0
        VriAttachmentDesc depthRT{}; depthRT.view = gDepthView; depthRT.loadOp = VriAttachmentLoadOp_Clear; depthRT.storeOp = VriAttachmentStoreOp_DontCare; depthRT.clearValue.depthStencil.depth = 1.0f;
        VriAttachmentsDesc att{}; att.colors = colors; att.colorNum = 3; att.depth = &depthRT; att.renderArea.width = kWidth; att.renderArea.height = kHeight; att.layerNum = 1;
        c.CmdBeginRendering(cmd, &att);
        VriViewport vp{0, 0, float(kWidth), float(kHeight), 0, 1}; c.CmdSetViewports(cmd, &vp, 1);
        VriRect sc{0, 0, kWidth, kHeight}; c.CmdSetScissors(cmd, &sc, 1);
        c.CmdSetPipeline(cmd, mrtPipeline);
        c.CmdSetPipelineLayout(cmd, mrtLayout);
        c.CmdSetDescriptorSet(cmd, 0, mrtSet);
        // draw the floor (1 instance)
        VriVertexBufferBinding fvb[2]{}; fvb[0].buffer = floorVB; fvb[1].buffer = floorInstB; c.CmdSetVertexBuffers(cmd, 0, fvb, 2);
        c.CmdSetIndexBuffer(cmd, floorIB, 0, VriIndexType_UInt16);
        VriDrawIndexedDesc fdi{}; fdi.indexNum = floorIndexNum; fdi.instanceNum = 1; c.CmdDrawIndexed(cmd, &fdi);
        // draw the cube grid (N instances)
        VriVertexBufferBinding cvb[2]{}; cvb[0].buffer = cubeVB; cvb[1].buffer = cubeInstB; c.CmdSetVertexBuffers(cmd, 0, cvb, 2);
        c.CmdSetIndexBuffer(cmd, cubeIB, 0, VriIndexType_UInt16);
        VriDrawIndexedDesc cdi{}; cdi.indexNum = cubeIndexNum; cdi.instanceNum = cubeInstanceNum; c.CmdDrawIndexed(cmd, &cdi);
        c.CmdEndRendering(cmd);

        // G-buffer colors -> sampleable by the lighting pass
        VriTextureBarrierDesc ts[3]{};
        ts[0].texture = gPos; ts[0].before.access = VriAccess_ColorAttachmentWrite; ts[0].before.layout = VriLayout_ColorAttachment; ts[0].before.stages = VriPipelineStage_ColorAttachmentOutput;
        ts[0].after.access = VriAccess_ShaderResourceRead; ts[0].after.layout = VriLayout_ShaderResource; ts[0].after.stages = VriPipelineStage_FragmentShader; ts[0].aspect = VriImageAspect_Color;
        ts[1] = ts[0]; ts[1].texture = gNormal;
        ts[2] = ts[0]; ts[2].texture = gAlbedo;
        VriBarrierGroupDesc gs{}; gs.textures = ts; gs.textureNum = 3; c.CmdBarrier(cmd, &gs);
    };

    app.onRecord = [=](VriCommandBuffer* cmd) {
        c.CmdSetPipeline(cmd, lightPipeline);
        c.CmdSetPipelineLayout(cmd, lightLayout);
        c.CmdSetDescriptorSet(cmd, 0, lightSet);
        VriDrawDesc d{}; d.vertexNum = 3; d.instanceNum = 1; c.CmdDraw(cmd, &d);
    };

    app.SetupCapture();
    app.Run();

#if !defined(__EMSCRIPTEN__)
    c.DestroyDescriptorPool(pool);
    c.DestroyPipeline(lightPipeline); c.DestroyPipelineLayout(lightLayout);
    c.DestroyPipeline(mrtPipeline); c.DestroyPipelineLayout(mrtLayout);
    c.DestroyDescriptor(gPosView); c.DestroyDescriptor(gNormalView); c.DestroyDescriptor(gAlbedoView); c.DestroyDescriptor(gDepthView);
    c.DestroyTexture(gPos); c.DestroyTexture(gNormal); c.DestroyTexture(gAlbedo); c.DestroyTexture(gDepth);
    c.DestroyDescriptor(gbufferSampler); c.DestroyDescriptor(repeatSampler);
    c.DestroyDescriptor(texView); c.DestroyTexture(texture); c.DestroyBuffer(staging);
    c.DestroyDescriptor(lightView); c.DestroyBuffer(lightStg); c.DestroyBuffer(lightUbo);
    c.DestroyDescriptor(camView); c.DestroyBuffer(camStg); c.DestroyBuffer(camUbo);
    c.DestroyBuffer(floorInstStg); c.DestroyBuffer(floorInstB); c.DestroyBuffer(cubeInstStg); c.DestroyBuffer(cubeInstB);
    c.DestroyBuffer(floorIStg); c.DestroyBuffer(floorIB); c.DestroyBuffer(floorVStg); c.DestroyBuffer(floorVB);
    c.DestroyBuffer(cubeIStg); c.DestroyBuffer(cubeIB); c.DestroyBuffer(cubeVStg); c.DestroyBuffer(cubeVB);
    app.Shutdown();
#endif
    return 0;
}
