// resource_desc_checks.h - the description checks every backend runs before creating a buffer or texture.
//
// Each backend's CreateBuffer / CreateTexture (and the exportable variants) calls these first, so a description no
// backend can create as described is refused the same way everywhere: VriResult_InvalidArgument and an error naming
// the field. Left to the native APIs, the same description was rejected by one backend, reported later by another and
// silently changed by a third (a Cube of 4 layers became 6 faces, a width of 0 became 1).
#pragma once

#include <vri/vri.h>

#include <string>

namespace vri::core
{
    // An empty string when `desc` is valid; otherwise "<function>: <reason> (<field>)".
    std::string CheckTextureDesc(const char* function, const VriTextureDesc& desc);
    std::string CheckBufferDesc(const char* function, const VriBufferDesc& desc);

    // Reports `error` as an error through the backend device and returns true when there is one.
    template<typename Device>
    bool RejectDesc(const Device& device, const std::string& error)
    {
        if (error.empty())
            return false;
        device.ReportError(error.c_str());
        return true;
    }
} // namespace vri::core
