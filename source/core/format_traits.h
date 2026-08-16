// format_traits.h - backend-neutral predicates over VriFormat.
//
// Whether a format carries depth or stencil is a property of the VRI enum, not
// of any one API, but every backend needs the answer: to pick a depth vs. color
// attachment slot, to choose an image aspect, and to report GetFormatSupport
// honestly. Keeping the predicates here means a new depth format is one edit
// rather than one per backend.
//
// Backend-neutral: depends on nothing but <vri/vri.h>, so it is safe to include
// from every backend translation unit (including the Objective-C++ Metal one).
#pragma once

#include <vri/vri.h>

namespace vri
{
    inline bool FormatHasDepth(VriFormat f)
    {
        return f == VriFormat_D16_UNORM || f == VriFormat_D32_SFLOAT || f == VriFormat_D24_UNORM_S8_UINT ||
               f == VriFormat_D32_SFLOAT_S8_UINT;
    }

    inline bool FormatHasStencil(VriFormat f)
    {
        return f == VriFormat_S8_UINT || f == VriFormat_D24_UNORM_S8_UINT || f == VriFormat_D32_SFLOAT_S8_UINT;
    }

    // A format that belongs on a depth/stencil attachment rather than a color one.
    inline bool FormatIsDepthStencil(VriFormat f) { return FormatHasDepth(f) || FormatHasStencil(f); }
} // namespace vri
