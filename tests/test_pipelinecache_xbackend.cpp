// Pipeline-cache round-trip (VRI_INTERFACE_PIPELINE_CACHE). Create an empty cache, build a
// pipeline through it (populating the cache), serialize the cache, seed a fresh cache with the
// blob, and build another pipeline from it. Also feed a garbage blob to confirm a stale/foreign
// seed is ignored rather than failing. Backends without a cache concept (OpenGL/WebGPU) report
// Unsupported and the test self-skips; so does an adapter that lacks pipeline-library support (e.g.
// the WARP software adapter in CI). Exercised on Vulkan (VkPipelineCache) + D3D12
// (ID3D12PipelineLibrary).
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <vector>

#include "shaders/common/triangle_spv.h" // g_triangleSpv (vertex + fragment, no descriptors)
#if defined(_WIN32)
#include "shaders/common/triangle_dxbc.h" // g_triangleDxbcVS / g_triangleDxbcPS (D3D12)
#endif

namespace
{
    struct Shaders
    {
        const void* vs;
        size_t      vsSize;
        const void* ps;
        size_t      psSize;
    };

    // Build a minimal vertex-id triangle pipeline through `cache`; returns true on success.
    bool MakePipeline(const VriCoreInterface& c,
                      VriDevice*              dev,
                      VriPipelineLayout*      layout,
                      VriPipelineCache*       cache,
                      const Shaders&          s)
    {
        VriShaderDesc sh[2] {};
        sh[0].stage          = VriShaderStage_Vertex;
        sh[0].bytecode       = s.vs;
        sh[0].bytecodeSize   = s.vsSize;
        sh[0].entryPointName = "vertexMain";
        sh[1].stage          = VriShaderStage_Fragment;
        sh[1].bytecode       = s.ps;
        sh[1].bytecodeSize   = s.psSize;
        sh[1].entryPointName = "fragmentMain";
        VriColorAttachmentDesc ca {};
        ca.format         = VriFormat_RGBA8_UNORM;
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
        pd.pipelineCache           = cache;
        VriPipeline* pipeline      = nullptr;
        if (c.CreateGraphicsPipeline(dev, &pd, &pipeline) != VriResult_Success)
            return false;
        c.DestroyPipeline(pipeline);
        return true;
    }

    void Check(VriGraphicsAPI api, const char* name, const Shaders& s)
    {
        VriDeviceCreationDesc dc {};
        dc.graphicsAPI      = api;
        dc.enableValidation = VRI_TRUE;
        dc.bestEffort       = VRI_TRUE;
        VriDevice* dev      = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << name << "] unavailable - skipped");
            return;
        }

        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
        VriPipelineCacheInterface pc {};
        if (vriGetInterface(dev, VRI_INTERFACE_PIPELINE_CACHE, sizeof(pc), &pc) != VriResult_Success)
        {
            MESSAGE("[" << name << "] no pipeline cache - skipped");
            vriDestroyDevice(dev);
            return;
        }

        VriPipelineLayoutDesc ld {};
        VriPipelineLayout*    layout = nullptr;
        REQUIRE(c.CreatePipelineLayout(dev, &ld, &layout) == VriResult_Success);

        // 1) empty cache -> build a pipeline through it -> the cache now has content.
        VriPipelineCache* cache = nullptr;
        const VriResult   cr    = pc.CreatePipelineCache(dev, nullptr, 0, &cache);
        if (cr == VriResult_Unsupported) // adapter without pipeline-library support (e.g. WARP)
        {
            MESSAGE("[" << name << "] pipeline cache unsupported on this adapter - skipped");
            c.DestroyPipelineLayout(layout);
            vriDestroyDevice(dev);
            return;
        }
        REQUIRE(cr == VriResult_Success);
        REQUIRE(cache != nullptr);
        CHECK(MakePipeline(c, dev, layout, cache, s));

        // 2) serialize (size query, then fetch).
        size_t sz = 0;
        REQUIRE(pc.GetPipelineCacheData(cache, nullptr, &sz) == VriResult_Success);
        CHECK(sz > 0); // at least the cache header
        std::vector<uint8_t> blob(sz);
        size_t               got = sz;
        REQUIRE(pc.GetPipelineCacheData(cache, blob.data(), &got) == VriResult_Success);
        CHECK(got == sz);

        // 3) seed a fresh cache with the blob and build another pipeline from it.
        VriPipelineCache* warm = nullptr;
        REQUIRE(pc.CreatePipelineCache(dev, blob.data(), blob.size(), &warm) == VriResult_Success);
        CHECK(MakePipeline(c, dev, layout, warm, s));

        // 4) a garbage seed must be ignored, not fatal (empty cache returned).
        const uint8_t     junk[16] = {0xDE, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        VriPipelineCache* bogus    = nullptr;
        CHECK(pc.CreatePipelineCache(dev, junk, sizeof(junk), &bogus) == VriResult_Success);
        if (bogus)
            pc.DestroyPipelineCache(bogus);

        pc.DestroyPipelineCache(warm);
        pc.DestroyPipelineCache(cache);
        c.DestroyPipelineLayout(layout);
        c.DeviceWaitIdle(dev);
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("Pipeline cache: serialize + reseed round-trip")
{
    const Shaders spv {g_triangleSpv, sizeof(g_triangleSpv), g_triangleSpv, sizeof(g_triangleSpv)};
    Check(VriGraphicsAPI_Vulkan, "Vulkan", spv);
    Check(VriGraphicsAPI_OpenGL, "OpenGL", spv);
#if defined(_WIN32)
    const Shaders dxbc {g_triangleDxbcVS, sizeof(g_triangleDxbcVS), g_triangleDxbcPS, sizeof(g_triangleDxbcPS)};
    Check(VriGraphicsAPI_D3D12, "D3D12", dxbc);
#endif
}
