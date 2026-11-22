// Texture types the GL backend cannot create as described are refused as Unsupported, with no object and one error
// saying why, rather than created as another type: 1D and 1DArray textures (GLES and WebGL have none, and desktop
// GL's are not wired), multisample 2DArray textures (no multisample array target or per-layer attachment), and, on
// GLES, cube map arrays (ESSL 300 cannot sample one).
#include <doctest/doctest.h>

#include <vri/vri.h>

#include <string>
#include <vector>

namespace
{
    struct Messages
    {
        std::vector<std::string> errors;
    };

    void VRI_CALL OnMessage(void* userArg, VriMessageSeverity severity, const char* message)
    {
        if (severity == VriMessageSeverity_Error)
            static_cast<Messages*>(userArg)->errors.emplace_back(message ? message : "");
    }

    VriTextureDesc Texture(VriTextureType type, uint32_t height, uint32_t layers, uint32_t samples)
    {
        VriTextureDesc td {};
        td.type           = type;
        td.format         = VriFormat_RGBA8_UNORM;
        td.width          = 64;
        td.height         = height;
        td.depth          = 1;
        td.mipNum         = 1;
        td.layerNum       = layers;
        td.sampleNum      = samples;
        td.usage          = samples > 1 ? VriTextureUsage_ColorAttachment : VriTextureUsage_ShaderResource;
        td.memoryLocation = VriMemoryLocation_Device;
        return td;
    }

    void CheckUnsupported(const VriCoreInterface& c,
                          VriDevice*              dev,
                          Messages&               messages,
                          const VriTextureDesc&   desc,
                          const char*             error)
    {
        messages.errors.clear();
        VriTexture*     texture = reinterpret_cast<VriTexture*>(&messages); // must be reset
        const VriResult r       = c.CreateTexture(dev, &desc, &texture);
        CHECK(r == VriResult_Unsupported);
        CHECK(texture == nullptr);
        REQUIRE(messages.errors.size() == 1);
        CHECK(messages.errors[0] == error);
    }
} // namespace

TEST_CASE("GL refuses the texture types it cannot create as described")
{
    Messages             messages;
    VriCallbackInterface callbacks = {};
    callbacks.MessageCallback      = OnMessage;
    callbacks.userArg              = &messages;
    VriDeviceCreationDesc dc       = vriDefaultDeviceCreationDesc();
    dc.graphicsAPI                 = VriGraphicsAPI_OpenGL;
    dc.bestEffort                  = VRI_TRUE;
    dc.callbackInterface           = &callbacks;
    VriDevice* dev                 = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("[OpenGL] unavailable - skipped");
        return;
    }
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    CHECK(c.GetDeviceDesc(dev)->texture1DMaxDim == 0);

    const char* const no1D = "CreateTexture: GL has no 1D textures here; use a 2D texture 1 texel high";
    {
        INFO("1D");
        CheckUnsupported(c, dev, messages, Texture(VriTextureType_1D, 1, 1, 1), no1D);
    }
    {
        INFO("1DArray");
        CheckUnsupported(c, dev, messages, Texture(VriTextureType_1DArray, 1, 3, 1), no1D);
    }
    {
        INFO("multisample 2DArray");
        CheckUnsupported(c,
                         dev,
                         messages,
                         Texture(VriTextureType_2DArray, 32, 2, 4),
                         "CreateTexture: GL has no multisample 2DArray textures here");
    }
    // Android's GL device is an ES context: there cube map arrays are refused too.
    if (c.GetDeviceDesc(dev)->graphicsAPI == VriGraphicsAPI_OpenGLES)
    {
        INFO("CubeArray on GLES");
        VriTextureDesc cubes = Texture(VriTextureType_CubeArray, 64, 12, 1);
        CheckUnsupported(c,
                         dev,
                         messages,
                         cubes,
                         "CreateTexture: GL has cube map arrays on desktop GL 4.0+ only, not on GLES or WebGL");
    }
    vriDestroyDevice(dev);
}

TEST_CASE("GL reports the error a texture creation leaves, and the next creation is not blamed for it")
{
    Messages             messages;
    VriCallbackInterface callbacks = {};
    callbacks.MessageCallback      = OnMessage;
    callbacks.userArg              = &messages;
    VriDeviceCreationDesc dc       = vriDefaultDeviceCreationDesc();
    dc.graphicsAPI                 = VriGraphicsAPI_OpenGL;
    dc.bestEffort                  = VRI_TRUE;
    dc.callbackInterface           = &callbacks;
    VriDevice* dev                 = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("[OpenGL] unavailable - skipped");
        return;
    }
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);

    // Valid for the shared check (one mip), but wider than GL_MAX_TEXTURE_SIZE: glTexStorage2D gives
    // GL_INVALID_VALUE.
    const uint32_t maxDim = c.GetDeviceDesc(dev)->texture2DMaxDim;
    REQUIRE(maxDim > 0);
    VriTextureDesc tooWide = Texture(VriTextureType_2D, 4, 1, 1);
    tooWide.width          = maxDim * 2;
    messages.errors.clear();
    VriTexture*     texture = reinterpret_cast<VriTexture*>(&messages); // must be reset
    const VriResult r       = c.CreateTexture(dev, &tooWide, &texture);
    CHECK(r == VriResult_Failure);
    CHECK(texture == nullptr);
    REQUIRE(messages.errors.size() == 1);
    CHECK(messages.errors[0].rfind("CreateTexture: GL_", 0) == 0);
    MESSAGE(messages.errors[0]);

    // The error is consumed: a valid texture right after succeeds with no error.
    messages.errors.clear();
    const VriTextureDesc valid = Texture(VriTextureType_2D, 64, 1, 1);
    REQUIRE(c.CreateTexture(dev, &valid, &texture) == VriResult_Success);
    CHECK(messages.errors.empty());
    c.DestroyTexture(texture);
    vriDestroyDevice(dev);
}
