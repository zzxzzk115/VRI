// software_icd_vk.cpp - locate a software Vulkan ICD (SwiftShader) for CPU rendering.
//
// Kept in its own translation unit so <windows.h> (needed for GetModuleFileName) never leaks its
// macros (CreateSemaphore, min/max, ...) into the Vulkan TU device_vk.cpp - which is exactly why
// the backend otherwise avoids <windows.h> and uses string extension names.
#include "software_icd_vk.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__APPLE__)
#    include <mach-o/dyld.h>
#elif defined(__linux__)
#    include <unistd.h>
#endif

namespace vri::vk
{
    namespace
    {
        // Absolute path of the running executable, or "" if it can't be determined.
        std::string ExecutablePath()
        {
#if defined(_WIN32)
            char  buf[MAX_PATH];
            DWORD n = GetModuleFileNameA(nullptr, buf, static_cast<DWORD>(sizeof(buf)));
            if (n == 0 || n >= sizeof(buf))
                return {};
            return std::string(buf, n);
#elif defined(__APPLE__)
            char     buf[4096];
            uint32_t sz = sizeof(buf);
            if (_NSGetExecutablePath(buf, &sz) != 0)
                return {};
            return std::string(buf);
#elif defined(__linux__)
            char    buf[4096];
            ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
            if (n <= 0)
                return {};
            buf[n] = '\0';
            return std::string(buf, static_cast<size_t>(n));
#else
            return {};
#endif
        }

        bool FileExists(const std::string& path)
        {
            std::FILE* f = std::fopen(path.c_str(), "rb");
            if (!f)
                return false;
            std::fclose(f);
            return true;
        }

        void SetEnv(const char* name, const char* value)
        {
#if defined(_WIN32)
            _putenv_s(name, value);
#else
            setenv(name, value, 1);
#endif
        }
    } // namespace

    void TrySelectSoftwareICD()
    {
        // A caller-provided software ICD (e.g. Mesa lavapipe on Linux CI, pointed at via the
        // standard loader env vars) wins - only auto-select when nothing was chosen.
        const char* icd = std::getenv("VK_ICD_FILENAMES");
        const char* drv = std::getenv("VK_DRIVER_FILES");
        if ((icd && *icd) || (drv && *drv))
            return;

        const std::string exe = ExecutablePath();
        if (exe.empty())
            return;
        const size_t slash = exe.find_last_of("/\\");
        if (slash == std::string::npos)
            return;
        const std::string manifest = exe.substr(0, slash + 1) + "vk_swiftshader_icd.json";
        if (!FileExists(manifest))
            return; // no bundled software ICD next to the app; leave the loader untouched

        // Newer Vulkan loaders read VK_DRIVER_FILES; older ones VK_ICD_FILENAMES. Set both so the
        // manifest is the only enumerated ICD -> the software (CPU) device is the one VRI picks.
        SetEnv("VK_ICD_FILENAMES", manifest.c_str());
        SetEnv("VK_DRIVER_FILES", manifest.c_str());
    }
} // namespace vri::vk
