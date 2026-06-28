// imgui_vri.h - the built-in Dear ImGui renderer's interface getter.
#pragma once

#include <vri/vri.h>

namespace vri::core
{
    // The ImGui renderer function table. Registered by every backend (it is backend-agnostic --
    // it draws entirely through the device's public core interface).
    const VriImguiInterface* GetImguiInterface();
} // namespace vri::core
