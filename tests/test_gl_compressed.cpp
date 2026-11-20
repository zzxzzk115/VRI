// GL block-compressed formats: GetFormatSupport reports the families this context has; each one uploads,
// samples to its known decode, and refuses a readback. A format the context lacks fails to create.
#include "tests/gl_compressed_probe.h"

#include <doctest/doctest.h>

#include <vri/vri.h>

#include <string>

namespace
{
    void VRI_CALL CountErrors(void* userArg, VriMessageSeverity severity, const char*)
    {
        if (severity == VriMessageSeverity_Error)
            ++*static_cast<int*>(userArg);
    }
} // namespace

TEST_CASE("GL compressed formats: upload, sample against the reference decode, refuse readback")
{
    int                   errors    = 0;
    VriCallbackInterface  callbacks = {&errors, CountErrors};
    VriDeviceCreationDesc dc        = vriDefaultDeviceCreationDesc();
    dc.graphicsAPI                  = VriGraphicsAPI_OpenGL;
    dc.bestEffort                   = VRI_TRUE;
    dc.callbackInterface            = &callbacks;
    VriDevice* dev                  = nullptr;
    if (vriCreateDevice(&dc, &dev) != VriResult_Success)
    {
        MESSAGE("OpenGL unavailable - skipped");
        return;
    }
    VriCoreInterface c {};
    REQUIRE(vriGetInterface(dev, VRI_INTERFACE_CORE, sizeof(c), &c) == VriResult_Success);
    std::string supported;
    for (const vri::test::CompressedSample& sample : vri::test::CompressedSamples())
    {
        INFO(sample.name);
        const VriFormatSupportFlags flags = c.GetFormatSupport(dev, sample.format);
        if ((flags & VriFormatSupport_Texture) == 0)
        {
            CHECK(flags == VriFormatSupport_None);
            VriTextureDesc td {};
            td.type             = VriTextureType_2D;
            td.format           = sample.format;
            td.width            = 4;
            td.height           = 4;
            td.depth            = 1;
            td.mipNum           = 1;
            td.layerNum         = 1;
            td.sampleNum        = 1;
            td.usage            = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst;
            VriTexture* texture = nullptr;
            CHECK(c.CreateTexture(dev, &td, &texture) == VriResult_Unsupported);
            continue;
        }
        CHECK(flags == VriFormatSupport_Texture); // sampled only: no target, storage or readback
        supported += std::string(" ") + sample.name;
        uint8_t rgba[4] = {};
        CHECK(vri::test::SampleCompressed(dev, c, sample, rgba) == 0);
        INFO("got " << int(rgba[0]) << "," << int(rgba[1]) << "," << int(rgba[2]) << "," << int(rgba[3]));
        CHECK(vri::test::MatchesDecode(sample, rgba));
        CHECK(vri::test::CheckCompressedReadbackRefused(dev, c, sample, errors) == 0);
    }
    MESSAGE("OpenGL compressed formats:" << (supported.empty() ? " none" : supported));
    vriDestroyDevice(dev);
}
