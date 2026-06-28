/*
 * vri.h - umbrella include for the VRI C API.
 *
 * Include this for the full C interface. For the C++ RAII convenience layer,
 * include <vri/vri.hpp> instead (which includes this).
 *
 * Windowing integration helpers (GLFW/SDL/native) are intentionally NOT pulled
 * in here so VRI never forces those dependencies; include the one you need
 * from <vri/integration/...> explicitly.
 */
#ifndef VRI_H
#define VRI_H

#include "vri_base.h"
#include "vri_handles.h"
#include "vri_enums.h"
#include "vri_flags.h"
#include "vri_format.h"
#include "vri_structs.h"
#include "vri_device_desc.h"
#include "vri_resource.h"
#include "vri_descriptor.h"
#include "vri_pipeline.h"
#include "vri_command.h"
#include "vri_sync.h"
#include "vri_core.h"

#include "ext/vri_ext_swapchain.h"
#include "ext/vri_ext_interop.h"
#include "ext/vri_ext_raytracing.h"
#include "ext/vri_ext_omm.h"
#include "ext/vri_ext_meshshader.h"
#include "ext/vri_ext_vrs.h"
#include "ext/vri_ext_external.h"
#include "ext/vri_ext_query.h"

#endif /* VRI_H */
