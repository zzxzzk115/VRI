/*
 * VRI - cross-API Render Hardware Interface
 *
 * vri_base.h - fixed-width types, result codes, and ABI macros.
 *
 * This header (and every header under include/vri/ except vri.hpp) is C-clean:
 * it must compile as C11 and C++. Keep it free of C++ constructs.
 */
#ifndef VRI_BASE_H
#define VRI_BASE_H

#include <stddef.h>
#include <stdint.h>

/* ---- versioning ------------------------------------------------------- */
/* VRI_VERSION_MAJOR/MINOR/PATCH are generated from set_version() in the root
 * xmake.lua (via add_configfiles) into vri/vri_version.h - the version lives in
 * exactly one place. Bump it there, not here. */
#include <vri/vri_version.h>

/* ---- ABI / calling-convention macros ---------------------------------- */
#if defined(_WIN32)
#    define VRI_CALL __cdecl
#    if defined(VRI_SHARED_LIBRARY)
#        if defined(VRI_BUILD)
#            define VRI_API __declspec(dllexport)
#        else
#            define VRI_API __declspec(dllimport)
#        endif
#    else
#        define VRI_API
#    endif
#else
#    define VRI_CALL
#    if defined(VRI_SHARED_LIBRARY) && defined(VRI_BUILD)
#        define VRI_API __attribute__((visibility("default")))
#    else
#        define VRI_API
#    endif
#endif

#if defined(__cplusplus)
#    define VRI_EXTERN_C_BEGIN extern "C" {
#    define VRI_EXTERN_C_END }
#else
#    define VRI_EXTERN_C_BEGIN
#    define VRI_EXTERN_C_END
#endif

VRI_EXTERN_C_BEGIN

/* ---- result codes ----------------------------------------------------- */
typedef enum VriResult
{
    VriResult_Success = 0,
    VriResult_Failure,
    VriResult_InvalidArgument,
    VriResult_OutOfMemory,
    VriResult_Unsupported,
    VriResult_DeviceLost,
    VriResult_OutOfDate,      /* swapchain out of date */
    VriResult_MaxEnum = 0x7fffffff
} VriResult;

/* 32-bit boolean for stable ABI across compilers. */
typedef uint32_t VriBool;
#define VRI_TRUE  1u
#define VRI_FALSE 0u

VRI_EXTERN_C_END

#endif /* VRI_BASE_H */
