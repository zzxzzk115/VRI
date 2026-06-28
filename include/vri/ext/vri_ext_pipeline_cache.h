/*
 * vri_ext_pipeline_cache.h - a serializable pipeline cache, queried via
 * vriGetInterface(device, VRI_INTERFACE_PIPELINE_CACHE, ...).
 *
 * A pipeline cache lets the driver reuse work (shader compilation, PSO assembly) across pipeline
 * creations and across application runs: serialize the cache to disk on exit, seed a new cache
 * with it on the next launch, and pipeline creation that hits the cache is dramatically faster.
 *
 * Usage:
 *   CreatePipelineCache(device, savedBlob, savedSize, &cache);   // savedBlob NULL/0 = empty
 *   ... set VriGraphicsPipelineDesc::pipelineCache / VriComputePipelineDesc::pipelineCache = cache
 *       and create pipelines as usual ...
 *   GetPipelineCacheData(cache, NULL, &size); buf = malloc(size);
 *   GetPipelineCacheData(cache, buf, &size);  // write buf to disk for next launch
 *   DestroyPipelineCache(cache);
 *
 * Stale or foreign blobs (different GPU / driver) are detected and ignored by the backend, so it
 * is always safe to seed a cache with whatever was saved last time. Backends that have no cache
 * concept (e.g. WebGPU) report VriResult_Unsupported from CreatePipelineCache.
 */
#ifndef VRI_EXT_PIPELINE_CACHE_H
#define VRI_EXT_PIPELINE_CACHE_H

#include "../vri_base.h"
#include "../vri_handles.h"

VRI_EXTERN_C_BEGIN

typedef struct VriPipelineCacheInterface
{
    /* Create a cache, optionally seeded with a previously serialized blob (initialData/initialSize;
     * NULL/0 for an empty cache). Incompatible seed data is ignored, not an error. */
    VriResult (VRI_CALL *CreatePipelineCache)(VriDevice* device,
                                              const void* initialData,
                                              size_t initialSize,
                                              VriPipelineCache** outCache);
    void (VRI_CALL *DestroyPipelineCache)(VriPipelineCache* cache);

    /* Serialize the cache. Call with data == NULL to read the required size into *size, then again
     * with a buffer of that size. *size is updated to the number of bytes written. */
    VriResult (VRI_CALL *GetPipelineCacheData)(VriPipelineCache* cache, void* data, size_t* size);
} VriPipelineCacheInterface;

VRI_EXTERN_C_END

#endif /* VRI_EXT_PIPELINE_CACHE_H */
