// asset_file.h - where an example's run-time files live and how they are read.
//
// Desktop: next to the executable (each example's xmake.lua copies them there).
// Web: the MEMFS root (--preload-file).
// Android: the APK's assets (`xmake apk` packages each target's vri.android_assets), read through
// the AAssetManager because they are not files on disk.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#if defined(__ANDROID__)
#include <android/asset_manager.h>
#elif !defined(__EMSCRIPTEN__)
#include <SDL3/SDL.h>
#endif

namespace vriex
{
#if defined(__ANDROID__)
    // Set by android_main (examples/common/android_main.cpp) before the example starts.
    inline AAssetManager* g_androidAssets = nullptr;
#endif

    // The path an example passes to its loader for a file it ships.
    inline std::string AssetPath(const char* file)
    {
#if defined(__EMSCRIPTEN__)
        return std::string("/") + file;
#elif defined(__ANDROID__)
        return file; // relative to the APK assets root
#else
        const char* base = SDL_GetBasePath();
        return (base ? std::string(base) : std::string()) + file;
#endif
    }

    // Reads a whole file. On Android a relative path names an APK asset; anything else is a file.
    inline bool ReadAssetFile(const char* path, std::vector<uint8_t>& out)
    {
#if defined(__ANDROID__)
        if (path[0] != '/')
        {
            AAsset* asset = AAssetManager_open(g_androidAssets, path, AASSET_MODE_BUFFER);
            if (!asset)
                return false;
            out.resize(size_t(AAsset_getLength64(asset)));
            const int read = AAsset_read(asset, out.data(), out.size());
            AAsset_close(asset);
            return read >= 0 && size_t(read) == out.size();
        }
#endif
        std::FILE* f = std::fopen(path, "rb");
        if (!f)
            return false;
        std::fseek(f, 0, SEEK_END);
        const long size = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        bool ok = size >= 0;
        if (ok)
        {
            out.resize(size_t(size));
            ok = std::fread(out.data(), 1, out.size(), f) == out.size();
        }
        std::fclose(f);
        return ok;
    }
} // namespace vriex
