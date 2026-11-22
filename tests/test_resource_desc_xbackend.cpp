// Buffer and texture descriptions that no backend can create as described are refused the same way on every
// backend: VriResult_InvalidArgument, no object, and one error naming the field. Before, the same description was
// rejected by one backend, reported later by another and silently changed by a third (a Cube of 4 layers became 6
// faces, a width of 0 became 1). The valid edge cases next to each rule still succeed.
//
// The table runs twice: through the shared check itself (no device, so it runs everywhere) and through
// CreateTexture / CreateBuffer on each backend that comes up.
#include <doctest/doctest.h>

#include "core/resource_desc_checks.h"

#include <vri/vri.h>

#include <string>
#include <vector>

namespace
{
    VriTextureDesc Texture(VriTextureType type, uint32_t width, uint32_t height, uint32_t depth, uint32_t layers)
    {
        VriTextureDesc td {};
        td.type           = type;
        td.format         = VriFormat_RGBA8_UNORM;
        td.width          = width;
        td.height         = height;
        td.depth          = depth;
        td.mipNum         = 1;
        td.layerNum       = layers;
        td.sampleNum      = 1;
        td.usage          = VriTextureUsage_ShaderResource;
        td.memoryLocation = VriMemoryLocation_Device;
        return td;
    }

    VriTextureDesc Texture2D() { return Texture(VriTextureType_2D, 64, 32, 1, 1); }

    struct TextureCase
    {
        const char*    name;
        VriTextureDesc desc;
        const char*    error; // nullptr: valid
    };

    struct BufferCase
    {
        const char*   name;
        VriBufferDesc desc;
        const char*   error;
    };

    std::vector<TextureCase> TextureCases()
    {
        std::vector<TextureCase> cases;
        const auto               add = [&](const char* name, VriTextureDesc desc, const char* error) {
            cases.push_back({name, desc, error});
        };

        // 1. A 1D or 2D texture has one layer; more need the array type.
        add("2D with 4 layers",
            Texture(VriTextureType_2D, 64, 32, 1, 4),
            "CreateTexture: a 2D texture has 1 layer, not 4; use 2DArray for more (layerNum)");
        add("1D with 2 layers",
            Texture(VriTextureType_1D, 64, 1, 1, 2),
            "CreateTexture: a 1D texture has 1 layer, not 2; use 1DArray for more (layerNum)");
        add("2DArray with 4 layers", Texture(VriTextureType_2DArray, 64, 32, 1, 4), nullptr);

        // 2. A Cube has 6 layers, a CubeArray a non-zero multiple of 6.
        add("Cube with 4 layers",
            Texture(VriTextureType_Cube, 64, 64, 1, 4),
            "CreateTexture: a Cube has 6 layers, not 4 (layerNum)");
        add("CubeArray with 8 layers",
            Texture(VriTextureType_CubeArray, 64, 64, 1, 8),
            "CreateTexture: a CubeArray has a non-zero multiple of 6 layers, not 8 (layerNum)");
        add("Cube with 6 layers", Texture(VriTextureType_Cube, 64, 64, 1, 6), nullptr);
        add("CubeArray with 12 layers", Texture(VriTextureType_CubeArray, 64, 64, 1, 12), nullptr);

        // 3. A 3D texture has one layer (its slices are depth); only a 3D texture is deeper than 1, and a 1D texture
        //    is 1 texel high.
        add("3D with 2 layers",
            Texture(VriTextureType_3D, 16, 16, 4, 2),
            "CreateTexture: a 3D texture has 1 layer, not 2; its slices are depth (layerNum)");
        add("2D 4 deep",
            Texture(VriTextureType_2D, 64, 32, 4, 1),
            "CreateTexture: only a 3D texture is deeper than 1, not 4 for a 2D (depth)");
        add("1D 4 high",
            Texture(VriTextureType_1D, 64, 4, 1, 1),
            "CreateTexture: a 1D texture is 1 texel high, not 4 (height)");

        // 4. At most floor(log2(largest extent)) + 1 mips: 7 for 64 x 32, 5 for 16 x 16 x 8.
        VriTextureDesc mips = Texture2D();
        mips.mipNum         = 8;
        add("64x32 with 8 mips", mips, "CreateTexture: a texture 64 texels across has at most 7 mips, not 8 (mipNum)");
        mips.mipNum = 7;
        add("64x32 with 7 mips", mips, nullptr);
        VriTextureDesc volume = Texture(VriTextureType_3D, 16, 16, 8, 1);
        volume.mipNum         = 5;
        add("16x16x8 with 5 mips", volume, nullptr);

        // 5. Every extent is at least 1.
        add("width 0",
            Texture(VriTextureType_2D, 0, 32, 1, 1),
            "CreateTexture: a texture is at least 1 texel wide, not 0 (width)");
        add("height 0",
            Texture(VriTextureType_2D, 64, 0, 1, 1),
            "CreateTexture: a texture is at least 1 texel high, not 0 (height)");
        add("depth 0",
            Texture(VriTextureType_3D, 16, 16, 0, 1),
            "CreateTexture: a texture is at least 1 texel deep, not 0 (depth)");
        // ... and so are the layer, mip and sample counts.
        add("0 layers",
            Texture(VriTextureType_2D, 64, 32, 1, 0),
            "CreateTexture: a texture has at least 1 layer, not 0 (layerNum)");
        VriTextureDesc noMips = Texture2D();
        noMips.mipNum         = 0;
        add("0 mips", noMips, "CreateTexture: a texture has at least 1 mip, not 0 (mipNum)");
        VriTextureDesc noSamples = Texture2D();
        noSamples.sampleNum      = 0;
        add("0 samples", noSamples, "CreateTexture: a texture has at least 1 sample, not 0 (sampleNum)");

        // 6. Attachment and storage usages follow the format's kind.
        VriTextureDesc depthColor = Texture2D();
        depthColor.format         = VriFormat_D32_SFLOAT;
        depthColor.usage          = VriTextureUsage_ColorAttachment;
        add("D32 as a color attachment",
            depthColor,
            "CreateTexture: a depth/stencil format cannot be a color attachment (usage ColorAttachment)");
        VriTextureDesc depthStorage = Texture2D();
        depthStorage.format         = VriFormat_D32_SFLOAT;
        depthStorage.usage          = VriTextureUsage_ShaderResourceStorage;
        add("D32 as storage",
            depthStorage,
            "CreateTexture: a depth/stencil format cannot be a storage texture (usage ShaderResourceStorage)");
        VriTextureDesc colorDepth = Texture2D();
        colorDepth.usage          = VriTextureUsage_DepthStencilAttachment;
        add("RGBA8 as a depth attachment",
            colorDepth,
            "CreateTexture: a color format cannot be a depth/stencil attachment (usage DepthStencilAttachment)");
        VriTextureDesc depth = Texture2D();
        depth.format         = VriFormat_D32_SFLOAT;
        depth.usage          = VriTextureUsage_DepthStencilAttachment | VriTextureUsage_ShaderResource;
        add("D32 as a sampled depth attachment", depth, nullptr);
        return cases;
    }

    std::vector<BufferCase> BufferCases()
    {
        VriBufferDesc base {};
        base.size           = 96;
        base.usage          = VriBufferUsage_StorageBuffer;
        base.memoryLocation = VriMemoryLocation_Device;

        // 7. A buffer is at least 1 byte, and a structured one holds whole elements.
        std::vector<BufferCase> cases;
        VriBufferDesc           empty = base;
        empty.size                    = 0;
        cases.push_back({"size 0", empty, "CreateBuffer: a buffer is at least 1 byte, not 0 (size)"});
        VriBufferDesc partial   = base;
        partial.size            = 100;
        partial.structureStride = 16;
        cases.push_back(
            {"size 100, stride 16", partial, "CreateBuffer: size 100 is not a multiple of structureStride 16 (size)"});
        VriBufferDesc whole   = base;
        whole.structureStride = 16;
        cases.push_back({"size 96, stride 16", whole, nullptr});
        return cases;
    }

    struct Messages
    {
        std::vector<std::string> errors;
    };

    void VRI_CALL OnMessage(void* userArg, VriMessageSeverity severity, const char* message)
    {
        if (severity == VriMessageSeverity_Error)
            static_cast<Messages*>(userArg)->errors.emplace_back(message ? message : "");
    }

    void CheckBackend(VriGraphicsAPI api, const char* name)
    {
        Messages             messages;
        VriCallbackInterface callbacks = {};
        callbacks.MessageCallback      = OnMessage;
        callbacks.userArg              = &messages;
        VriDeviceCreationDesc dc       = vriDefaultDeviceCreationDesc();
        dc.graphicsAPI                 = api;
        dc.bestEffort                  = VRI_TRUE;
        dc.callbackInterface           = &callbacks;
        VriDevice* dev                 = nullptr;
        if (vriCreateDevice(&dc, &dev) != VriResult_Success)
        {
            MESSAGE("[" << std::string(name) << "] unavailable - skipped");
            return;
        }
        VriCoreInterface c {};
        REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);

        for (const TextureCase& tc : TextureCases())
        {
            INFO(std::string(name) << ": " << tc.name);
            messages.errors.clear();
            VriTexture*     texture = reinterpret_cast<VriTexture*>(&messages); // must be reset on failure
            const VriResult r       = c.CreateTexture(dev, &tc.desc, &texture);
            if (tc.error)
            {
                CHECK(r == VriResult_InvalidArgument);
                CHECK(texture == nullptr);
                REQUIRE(messages.errors.size() == 1);
                CHECK(messages.errors[0] == tc.error);
            }
            else if (api == VriGraphicsAPI_OpenGL && tc.desc.type == VriTextureType_CubeArray)
            {
                // GL creates one cube for any CubeArray, so Success here would not mean the array exists; the
                // follow-up GL cube-array fix makes this Success or Unsupported. Here only: not refused as invalid.
                CHECK(r != VriResult_InvalidArgument);
                if (r == VriResult_Success)
                    c.DestroyTexture(texture);
            }
            else
            {
                CHECK(r == VriResult_Success);
                CHECK(messages.errors.empty());
                if (r == VriResult_Success)
                    c.DestroyTexture(texture);
            }
        }
        for (const BufferCase& bc : BufferCases())
        {
            INFO(std::string(name) << ": " << bc.name);
            messages.errors.clear();
            VriBuffer*      buffer = reinterpret_cast<VriBuffer*>(&messages);
            const VriResult r      = c.CreateBuffer(dev, &bc.desc, &buffer);
            if (bc.error)
            {
                CHECK(r == VriResult_InvalidArgument);
                CHECK(buffer == nullptr);
                REQUIRE(messages.errors.size() == 1);
                CHECK(messages.errors[0] == bc.error);
            }
            else
            {
                CHECK(r == VriResult_Success);
                CHECK(messages.errors.empty());
                if (r == VriResult_Success)
                    c.DestroyBuffer(buffer);
            }
        }
        vriDestroyDevice(dev);
    }
} // namespace

TEST_CASE("Resource descriptions: the shared check names the field of each invalid description")
{
    for (const TextureCase& tc : TextureCases())
    {
        INFO(tc.name);
        CHECK(vri::core::CheckTextureDesc("CreateTexture", tc.desc) == std::string(tc.error ? tc.error : ""));
    }
    for (const BufferCase& bc : BufferCases())
    {
        INFO(bc.name);
        CHECK(vri::core::CheckBufferDesc("CreateBuffer", bc.desc) == std::string(bc.error ? bc.error : ""));
    }
}

TEST_CASE("Resource descriptions: every backend refuses the same invalid descriptions the same way")
{
    CheckBackend(VriGraphicsAPI_Vulkan, "Vulkan");
    CheckBackend(VriGraphicsAPI_D3D12, "D3D12");
    CheckBackend(VriGraphicsAPI_OpenGL, "OpenGL");
    CheckBackend(VriGraphicsAPI_WebGPU, "WebGPU");
    CheckBackend(VriGraphicsAPI_Metal, "Metal");
}
