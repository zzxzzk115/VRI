// pipeline_cache_mtl.mm - VriPipelineCacheInterface for native Metal, backed by MTLBinaryArchive.
//
// MTLBinaryArchive only seeds from / serializes to a file URL (no in-memory blob API), so this
// bridges VRI's blob-based contract through a temporary file: a seed blob is written to a temp file
// the archive loads at creation, and GetPipelineCacheData serializes to a temp file it reads back.
// The archive is populated at pipeline creation (addRender/ComputePipelineFunctionsWithDescriptor,
// see core_mtl.mm). A stale/foreign/garbage seed makes newBinaryArchiveWithDescriptor fail; we fall
// back to an empty archive so seeding is always safe (matches the Vulkan/D3D12 backends).

#include "pipeline_cache_mtl.h"
#include "device_mtl.h"
#include "objects_mtl.h"

#include <cstring>
#include <string>

namespace vri::mtl
{
    namespace
    {
        inline DeviceMTL*        Dev(VriDevice* h) { return reinterpret_cast<DeviceMTL*>(h); }
        inline PipelineCacheMTL* PC(VriPipelineCache* h) { return reinterpret_cast<PipelineCacheMTL*>(h); }

        // A fresh temp-file URL under NSTemporaryDirectory(); autoreleased.
        NSURL* TempURL()
        {
            NSString* dir  = NSTemporaryDirectory();
            NSString* name = [@"vri_mtl_pcache_" stringByAppendingString:[[NSProcessInfo processInfo] globallyUniqueString]];
            return [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:name]];
        }

        VriResult VRI_CALL CreatePipelineCache(VriDevice*         device,
                                               const void*        initialData,
                                               size_t             initialSize,
                                               VriPipelineCache** out)
        {
            if (!out)
                return VriResult_InvalidArgument;
            DeviceMTL* d = Dev(device);

            @autoreleasepool
            {
                MTLBinaryArchiveDescriptor* ad = [[MTLBinaryArchiveDescriptor alloc] init];
                NSURL*                      seedURL = nil;
                if (initialData && initialSize)
                {
                    seedURL = TempURL();
                    NSData* blob = [NSData dataWithBytes:initialData length:initialSize];
                    if ([blob writeToURL:seedURL atomically:YES])
                        ad.url = seedURL; // archive loads the blob at creation
                }

                NSError*             err = nil;
                id<MTLBinaryArchive> ar  = [d->Device() newBinaryArchiveWithDescriptor:ad error:&err]; // +1
                if (!ar && ad.url)
                {
                    // Stale / foreign / garbage seed: discard it and start empty.
                    ad.url = nil;
                    err    = nil;
                    ar     = [d->Device() newBinaryArchiveWithDescriptor:ad error:&err];
                }
                [ad release];
                if (seedURL)
                    [[NSFileManager defaultManager] removeItemAtURL:seedURL error:nil];

                if (!ar)
                {
                    d->ReportError("newBinaryArchiveWithDescriptor failed");
                    return VriResult_Unsupported;
                }
                PipelineCacheMTL* p = new PipelineCacheMTL{};
                p->device  = d;
                p->archive = ar;
                p->dirty   = true; // serialize on first request
                *out = ToHandle(p);
            }
            return VriResult_Success;
        }

        void VRI_CALL DestroyPipelineCache(VriPipelineCache* h)
        {
            if (!h)
                return;
            PipelineCacheMTL* p = PC(h);
            if (p->archive) [p->archive release];
            delete p;
        }

        VriResult VRI_CALL GetPipelineCacheData(VriPipelineCache* h, void* data, size_t* size)
        {
            if (!h || !size)
                return VriResult_InvalidArgument;
            PipelineCacheMTL* p = PC(h);

            // Refresh the cached blob only when new pipelines have been added (serializeToURL: is
            // reliable only on an archive's first serialize, so we avoid re-serializing otherwise).
            if (p->dirty)
            {
                @autoreleasepool
                {
                    NSURL*   url = TempURL();
                    NSError* err = nil;
                    if ([p->archive serializeToURL:url error:&err])
                    {
                        if (NSData* blob = [NSData dataWithContentsOfURL:url])
                        {
                            p->blob.assign(static_cast<const uint8_t*>([blob bytes]),
                                           static_cast<const uint8_t*>([blob bytes]) + [blob length]);
                            p->dirty = false;
                        }
                    }
                    else if (p->blob.empty())
                    {
                        std::string m = "MTLBinaryArchive serializeToURL failed: ";
                        m += err ? [[err localizedDescription] UTF8String] : "unknown";
                        p->device->ReportError(m.c_str());
                        [[NSFileManager defaultManager] removeItemAtURL:url error:nil];
                        return VriResult_Failure;
                    }
                    [[NSFileManager defaultManager] removeItemAtURL:url error:nil];
                }
            }

            const size_t sz = p->blob.size();
            if (!data) // size query
            {
                *size = sz;
                return VriResult_Success;
            }
            if (*size < sz)
                return VriResult_InvalidArgument; // buffer too small
            std::memcpy(data, p->blob.data(), sz);
            *size = sz;
            return VriResult_Success;
        }

        const VriPipelineCacheInterface g_pipelineCacheMTL = {
            CreatePipelineCache,
            DestroyPipelineCache,
            GetPipelineCacheData,
        };
    } // namespace

    const VriPipelineCacheInterface* GetPipelineCacheInterfaceMTL() { return &g_pipelineCacheMTL; }
} // namespace vri::mtl
