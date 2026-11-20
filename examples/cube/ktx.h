// ktx.h - a minimal KTX1 reader for examples/cube: returns the base mip of an
// unconverted RGBA8 (.ktx) texture. NOT a general KTX loader (no compressed formats,
// cubemaps, arrays, or mip chains) - just enough to load Sascha Willems' *_rgba.ktx
// assets for the example. Approach borrowed (conceptually) from libvultra/vasset, which
// use the full libktx; here we parse by hand to avoid the dependency.
#pragma once

#include "../common/asset_file.h"

#include <cstdint>
#include <cstring>
#include <vector>

struct KtxImage
{
    uint32_t             width  = 0;
    uint32_t             height = 0;
    std::vector<uint8_t> rgba; // width*height*4, base mip
};

// Loads the base mip of a KTX1 RGBA8 file (an APK asset on Android, see asset_file.h). Returns
// false on any mismatch.
inline bool LoadKtxRgba(const char* path, KtxImage& out)
{
    std::vector<uint8_t> file;
    if (!vriex::ReadAssetFile(path, file))
        return false;
    // identifier (12 bytes), 13 u32 header fields, then key/value data and the base mip's imageSize
    if (file.size() < 12 + 13 * 4)
        return false;
    static const uint8_t kKtx1[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
    if (std::memcmp(file.data(), kKtx1, 12) != 0)
        return false;

    uint32_t h[13] = {}; // endianness .. bytesOfKeyValueData (13 u32 after the identifier)
    std::memcpy(h, file.data() + 12, sizeof(h));
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

    const size_t sizeOffset = 12 + sizeof(h) + size_t(bytesOfKeyValueData);
    if (file.size() < sizeOffset + 4)
        return false;
    uint32_t imageSize = 0;
    std::memcpy(&imageSize, file.data() + sizeOffset, 4);
    const uint32_t expected = pixelWidth * pixelHeight * 4;
    if (imageSize < expected || file.size() - (sizeOffset + 4) < expected)
        return false;

    out.width  = pixelWidth;
    out.height = pixelHeight;
    out.rgba.assign(file.begin() + std::ptrdiff_t(sizeOffset + 4),
                    file.begin() + std::ptrdiff_t(sizeOffset + 4 + expected));
    return true;
}
