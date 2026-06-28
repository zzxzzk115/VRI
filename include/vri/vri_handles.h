/*
 * vri_handles.h - opaque handle types.
 *
 * Each handle is a forward-declared incomplete struct used only by pointer.
 * The backend reinterpret_casts it to its concrete implementation; no type
 * information crosses the ABI boundary.
 */
#ifndef VRI_HANDLES_H
#define VRI_HANDLES_H

#include "vri_base.h"

VRI_EXTERN_C_BEGIN

typedef struct VriDevice          VriDevice;
typedef struct VriQueue           VriQueue;
typedef struct VriCommandAllocator VriCommandAllocator;
typedef struct VriCommandBuffer   VriCommandBuffer;
typedef struct VriBuffer        VriBuffer;
typedef struct VriTexture       VriTexture;
typedef struct VriDescriptor    VriDescriptor;     /* a single resource view */
typedef struct VriDescriptorSet VriDescriptorSet;
typedef struct VriDescriptorPool VriDescriptorPool;
typedef struct VriPipelineLayout VriPipelineLayout;
typedef struct VriPipeline      VriPipeline;
typedef struct VriPipelineCache VriPipelineCache; /* ext/vri_ext_pipeline_cache.h */
typedef struct VriMemory        VriMemory;
typedef struct VriFence         VriFence;
typedef struct VriSwapChain     VriSwapChain;

VRI_EXTERN_C_END

#endif /* VRI_HANDLES_H */
