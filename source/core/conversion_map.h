// conversion_map.h - shared, backend-neutral skeleton for VRI enum -> native
// conversions.
//
// Every backend has to translate the same VRI enums (VriFormat, VriBlendFactor,
// buffer/texture usage flags, ...) into its own API's constants. That translation
// only ever takes two shapes:
//
//   * a 1:1 enum dispatch    - switch (x) { case A: return X; ... default: return F; }
//   * a flag-set accumulate  - if (x & A) r |= X; if (x & B) r |= Y; ...
//
// Historically each backend open-coded both skeletons inline, so the *dispatch*
// logic (and the "unmapped -> sentinel" policy) was duplicated ~5x on top of the
// mapping data that genuinely differs per backend. This header keeps the two
// skeletons once; each backend now supplies ONLY its mapping data as a constexpr
// table of rows. Adding a new format/enum value is a single new row, not a new
// `case` edited into every backend's switch.
//
// Backend-neutral: depends on nothing but <cstddef>, so it is safe to include
// from every backend translation unit (including the Objective-C++ Metal one).
#pragma once

#include <cstddef>

namespace vri
{
    // One "VriEnum -> native value" mapping row. `To` may be a scalar/enum or a
    // small aggregate (e.g. a {format, texelSize} struct) - anything copyable.
    template<class From, class To>
    struct ConvRow
    {
        From from;
        To   to;
    };

    // Enum dispatch. Returns the native value mapped from `value`, or `fallback`
    // when `value` is unmapped. Exactly equivalent to
    //   switch (value) { case row.from: return row.to; ... default: return fallback; }
    //
    // `fallback` is each backend's documented "unmapped" sentinel
    // (VK_FORMAT_UNDEFINED, MTLPixelFormatInvalid, WGPUTextureFormat_Undefined, ...)
    // which call sites already test to diagnose an unsupported format - so routing
    // the unmapped case through here preserves the existing "unmapped -> diagnostic"
    // contract without re-implementing the switch skeleton in every backend.
    template<class From, class To, std::size_t N>
    constexpr To MapOr(From value, const ConvRow<From, To> (&table)[N], To fallback)
    {
        for (const auto& row : table)
            if (row.from == value)
                return row.to;
        return fallback;
    }

    // Flag-set dispatch. ORs together the native bits whose VRI bit is set in
    // `value`. Equivalent to a chain of `if (value & row.from) r |= row.to;`.
    // Order-independent (OR is commutative) and duplicate-safe (OR is idempotent),
    // so a row whose `to` combines several native bits, or several rows that share
    // a native bit, behave exactly as the hand-written `if` chain did. `zero` is
    // the native "no flags" value the accumulator starts from.
    template<class From, class To, std::size_t N>
    constexpr To MapFlags(From value, const ConvRow<From, To> (&table)[N], To zero = To {})
    {
        To r = zero;
        for (const auto& row : table)
            if (value & row.from)
                r |= row.to;
        return r;
    }
} // namespace vri
