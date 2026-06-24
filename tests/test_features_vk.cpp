// Phase-3 feature/extension system (Vulkan): adapter enumeration, optional
// feature query + enable (with best-effort vs fail-fast), and the interop
// interface (native handles + WrapTexture). Skips gracefully without a device.
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <cstdint>
#include <cstring>

namespace
{
    bool VulkanAvailable()
    {
        VriAdapterDesc adapters[8];
        return vriEnumerateAdapters(VriGraphicsAPI_Vulkan, adapters, 8) > 0;
    }

    struct Ctx
    {
        VriDevice* device = nullptr;
        VriCoreInterface core{};
        ~Ctx() { if (device) vriDestroyDevice(device); }
    };

    bool Init(Ctx& ctx, uint64_t features, VriBool bestEffort)
    {
        VriDeviceCreationDesc desc{};
        desc.graphicsAPI = VriGraphicsAPI_Vulkan;
        desc.enableValidation = VRI_TRUE;
        desc.enabledFeatures = features;
        desc.bestEffort = bestEffort;
        if (vriCreateDevice(&desc, &ctx.device) != VriResult_Success)
            return false;
        return vriGetInterface(ctx.device, VRI_INTERFACE_CORE, sizeof(ctx.core), &ctx.core) == VriResult_Success;
    }
} // namespace

TEST_CASE("Vulkan: adapter enumeration reports at least one device")
{
    if (!VulkanAvailable()) { MESSAGE("Vulkan unavailable - skipping"); return; }

    const uint32_t count = vriEnumerateAdapters(VriGraphicsAPI_Vulkan, nullptr, 0);
    CHECK(count >= 1);

    VriAdapterDesc adapters[8];
    const uint32_t filled = vriEnumerateAdapters(VriGraphicsAPI_Vulkan, adapters, 8);
    CHECK(filled >= 1);
    CHECK(adapters[0].name[0] != '\0');
}

TEST_CASE("Vulkan: best-effort feature request never exceeds the granted set")
{
    if (!VulkanAvailable()) { MESSAGE("Vulkan unavailable - skipping"); return; }

    Ctx ctx;
    const uint64_t requested = VriFeature_Bindless | VriFeature_RayTracing | VriFeature_MeshShader;
    REQUIRE(Init(ctx, requested, VRI_TRUE));

    const VriDeviceDesc* desc = ctx.core.GetDeviceDesc(ctx.device);
    // granted features must be a subset of what we asked for
    CHECK((desc->enabledFeatures & ~requested) == 0u);
    // consistency between the bitset and the convenience bools
    CHECK(((desc->enabledFeatures & VriFeature_Bindless) != 0) == (desc->hasBindless != VRI_FALSE));
    CHECK(((desc->enabledFeatures & VriFeature_RayTracing) != 0) == (desc->hasRayTracing != VRI_FALSE));
}

TEST_CASE("Vulkan: fail-fast (bestEffort=false) rejects an unsupported feature")
{
    if (!VulkanAvailable()) { MESSAGE("Vulkan unavailable - skipping"); return; }

    // The fail-fast path reports *why* it rejected the feature; capture that via a callback (instead
    // of letting it fall to the default stderr sink) and confirm the diagnostic is actually delivered.
    struct Captured { int count = 0; } captured;
    VriCallbackInterface cb{};
    cb.userArg = &captured;
    cb.MessageCallback = [](void* userArg, VriMessageSeverity, const char*) { ++static_cast<Captured*>(userArg)->count; };

    VriDeviceCreationDesc desc{};
    desc.graphicsAPI = VriGraphicsAPI_Vulkan;
    desc.enabledFeatures = VriFeature_LowLatency; // not implemented -> must fail when not best-effort
    desc.bestEffort = VRI_FALSE;
    desc.callbackInterface = &cb;

    VriDevice* device = nullptr;
    CHECK(vriCreateDevice(&desc, &device) == VriResult_Unsupported);
    CHECK(device == nullptr);
    CHECK(captured.count >= 1); // the rejection reason was reported to the app, not swallowed
}

TEST_CASE("Vulkan: interop interface exposes native handles and wraps textures")
{
    if (!VulkanAvailable()) { MESSAGE("Vulkan unavailable - skipping"); return; }

    Ctx ctx;
    REQUIRE(Init(ctx, VriFeature_None, VRI_TRUE));
    const VriCoreInterface& c = ctx.core;

    VriInteropInterface interop{};
    REQUIRE(vriGetInterface(ctx.device, VRI_INTERFACE_INTEROP, sizeof(interop), &interop) == VriResult_Success);

    VriDeviceNativeHandles native{};
    REQUIRE(interop.GetDeviceNativeHandles(ctx.device, &native) == VriResult_Success);
    CHECK(native.api == VriGraphicsAPI_Vulkan);
    CHECK(native.u.vulkan.instance != nullptr);
    CHECK(native.u.vulkan.physicalDevice != nullptr);
    CHECK(native.u.vulkan.device != nullptr);
    CHECK(native.u.vulkan.graphicsQueue != nullptr);

    // create a texture, grab its native VkImage, wrap it back, and confirm the
    // wrapper points at the same native object (round-trip through the seam).
    VriTextureDesc td{};
    td.type = VriTextureType_2D;
    td.format = VriFormat_RGBA8_UNORM;
    td.width = 16; td.height = 16; td.depth = 1;
    td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1;
    td.usage = VriTextureUsage_ShaderResource;
    td.memoryLocation = VriMemoryLocation_Device;
    VriTexture* texture = nullptr;
    REQUIRE(c.CreateTexture(ctx.device, &td, &texture) == VriResult_Success);

    void* nativeImage = interop.GetTextureNativeHandle(texture);
    CHECK(nativeImage != nullptr);

    VriWrapTextureDesc wd{};
    wd.nativeTexture = nativeImage;
    wd.desc = td;
    VriTexture* wrapped = nullptr;
    REQUIRE(interop.WrapTexture(ctx.device, &wd, &wrapped) == VriResult_Success);
    CHECK(interop.GetTextureNativeHandle(wrapped) == nativeImage);

    c.DestroyTexture(wrapped); // borrowed -> must NOT free the underlying image
    c.DestroyTexture(texture); // owns + frees the image
}
