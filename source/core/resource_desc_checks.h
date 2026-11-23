// resource_desc_checks.h - the description checks every backend runs before creating a buffer or texture, and
// before copying a texture into itself.
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

    // An empty string unless CmdCopyTexture copies within one texture and its source and destination share a
    // subresource: the same mip with intersecting layer ranges (one mip of a 3D texture, all its slices, is one
    // subresource). D3D12 and WebGPU forbid that, and a GLES or WebGL blit reports it as an error, so it is refused
    // everywhere; different mips or layers of one texture are copied by every backend.
    std::string CheckTextureCopy(const VriTexture* dst, const VriTexture* src, const VriTextureCopyDesc* region);

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
