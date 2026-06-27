// ktx.h - a minimal KTX1 reader for examples/cube: returns the base mip of an
// unconverted RGBA8 (.ktx) texture. NOT a general KTX loader (no compressed formats,
// cubemaps, arrays, or mip chains) - just enough to load Sascha Willems' *_rgba.ktx
// assets for the example. Approach borrowed (conceptually) from libvultra/vasset, which
// use the full libktx; here we parse by hand to avoid the dependency.
#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

struct KtxImage
{
    uint32_t             width  = 0;
    uint32_t             height = 0;
    std::vector<uint8_t> rgba; // width*height*4, base mip
};

// Loads the base mip of a KTX1 RGBA8 file. Returns false on any mismatch.
inline bool LoadKtxRgba(const char* path, KtxImage& out)
{
    std::FILE* f = std::fopen(path, "rb");
    if (!f)
        return false;
    struct Closer
    {
        std::FILE* f;
        ~Closer() { std::fclose(f); }
    } closer {f};

    uint8_t id[12] = {};
    if (std::fread(id, 1, 12, f) != 12)
        return false;
    static const uint8_t kKtx1[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
    for (int i = 0; i < 12; ++i)
        if (id[i] != kKtx1[i])
            return false;

    uint32_t h[13] = {}; // endianness .. bytesOfKeyValueData (13 u32 after the identifier)
    if (std::fread(h, 4, 13, f) != 13)
        return false;
    // h[0]=endianness h[1]=glType h[2]=glTypeSize h[3]=glFormat h[4]=glInternalFormat
    // h[5]=glBaseInternalFormat h[6]=pixelWidth h[7]=pixelHeight ... h[12]=bytesOfKeyValueData
    const uint32_t glFormat            = h[3]; // GL_RGBA == 0x1908
    const uint32_t glInternalFormat    = h[4]; // GL_RGBA8 == 0x8058
    const uint32_t pixelWidth          = h[6];
    const uint32_t pixelHeight         = h[7];
    const uint32_t bytesOfKeyValueData = h[12];
    if (glFormat != 0x1908 || glInternalFormat != 0x8058)
        return false; // RGBA8 only
    if (pixelWidth == 0 || pixelHeight == 0)
        return false;

    if (std::fseek(f, static_cast<long>(bytesOfKeyValueData), SEEK_CUR) != 0)
        return false;

    uint32_t imageSize = 0;
    if (std::fread(&imageSize, 4, 1, f) != 1)
        return false;
    const uint32_t expected = pixelWidth * pixelHeight * 4;
    if (imageSize < expected)
        return false;

    out.width  = pixelWidth;
    out.height = pixelHeight;
    out.rgba.resize(expected);
    if (std::fread(out.rgba.data(), 1, expected, f) != expected)
        return false;
    return true;
}
