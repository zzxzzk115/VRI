// software_icd_vk.h - opt a software Vulkan ICD (SwiftShader) into the loader for CPU rendering.
#pragma once

namespace vri::vk
{
    // Software rendering: if the app hasn't already selected a Vulkan ICD (via the standard loader
    // env vars VK_ICD_FILENAMES / VK_DRIVER_FILES), and a SwiftShader manifest
    // (vk_swiftshader_icd.json) sits next to the executable, point the loader at it. Best-effort
    // and idempotent - call once before the first Vulkan entry point, so the loader reads the env
    // at ICD-scan time. A caller-provided ICD (e.g. Mesa lavapipe on Linux CI) always wins.
    void TrySelectSoftwareICD();
} // namespace vri::vk
