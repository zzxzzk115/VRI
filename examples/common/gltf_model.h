// Minimal glTF 2.0 model loader for the VRI examples, built on tinygltf - the same library
// Sascha Willems' Vulkan-Examples use (base/VulkanglTFModel). It is intentionally scoped to what
// the examples need: one interleaved vertex buffer (position / normal / uv) + one 32-bit index
// buffer for the whole model, the per-primitive material's baseColor / occlusion texture indices,
// and one GPU texture per glTF texture. Node transforms are BAKED into the vertices at load
// (PreTransformVertices), so the geometry is world-space and needs no per-node transform plumbing.
//
// The vertex/index buffers are created with AccelerationBuildInput | StorageBuffer usage so the
// same model can feed a BLAS build AND be read back in a closesthit shader (see examples/raytracing)
// - while staying usable as ordinary vertex/index buffers for a raster pass.
//
// Define TINYGLTF_IMPLEMENTATION in exactly ONE translation unit before including this header.
#pragma once

#include "example_app.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

namespace vriex
{
    // Must match the Vertex struct in tests/shaders/rt_gltf.slang (32 bytes, tightly packed).
    struct GltfVertex
    {
        float pos[3];
        float normal[3];
        float uv[2];
    };

    // One glTF primitive == one BLAS geometry. firstIndex/indexCount address the model's global
    // index buffer; the texture indices select into GltfModel::textures (-1 == none).
    struct GltfPrimitive
    {
        uint32_t firstIndex = 0;
        uint32_t indexCount = 0;
        int32_t  baseColorTexture = -1;
        int32_t  occlusionTexture = -1;
    };

    struct GltfModel
    {
        VriBuffer*                  vertexBuffer = nullptr; // GltfVertex[], host-visible, AS build input + SSBO
        VriBuffer*                  indexBuffer = nullptr;  // uint32[], global (indices already offset per primitive)
        uint32_t                    vertexCount = 0;
        uint32_t                    indexCount = 0;
        std::vector<GltfPrimitive>  primitives;
        std::vector<VriTexture*>    textures;
        std::vector<VriDescriptor*> textureViews;
        // Optional 2D-array texture holding every glTF texture as one layer (all resampled to a
        // common size), built when Load(..., buildTextureArray=true). The bindless software ray-
        // tracing path (GL/WebGPU, no descriptor-array support) samples this by glTF texture index.
        VriTexture*                 textureArray = nullptr;
        VriDescriptor*              textureArrayView = nullptr;
        uint32_t                    textureArrayLayers = 0;
        std::vector<GltfVertex>     cpuVerts;
        std::vector<uint32_t>       cpuIndices;
        float                       boundsMin[3] = {0, 0, 0};
        float                       boundsMax[3] = {0, 0, 0};

        // ---- tiny 4x4 column-major matrix helpers (node transforms) ----
        struct Mat4 { float m[16]; };
        static Mat4 Identity() { Mat4 r{}; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f; return r; }
        static Mat4 Mul(const Mat4& a, const Mat4& b)
        {
            Mat4 r{};
            for (int c = 0; c < 4; ++c)
                for (int row = 0; row < 4; ++row)
                {
                    float s = 0;
                    for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[c * 4 + k];
                    r.m[c * 4 + row] = s;
                }
            return r;
        }
        static void XformPoint(const Mat4& a, const float* p, float* out)
        {
            for (int row = 0; row < 3; ++row)
                out[row] = a.m[0 * 4 + row] * p[0] + a.m[1 * 4 + row] * p[1] + a.m[2 * 4 + row] * p[2] + a.m[3 * 4 + row];
        }
        static void XformDir(const Mat4& a, const float* p, float* out)
        {
            for (int row = 0; row < 3; ++row)
                out[row] = a.m[0 * 4 + row] * p[0] + a.m[1 * 4 + row] * p[1] + a.m[2 * 4 + row] * p[2];
        }
        static Mat4 NodeLocal(const tinygltf::Node& n)
        {
            if (n.matrix.size() == 16)
            {
                Mat4 r{};
                for (int i = 0; i < 16; ++i) r.m[i] = static_cast<float>(n.matrix[i]);
                return r;
            }
            Mat4 t = Identity(), r = Identity(), s = Identity();
            if (n.translation.size() == 3) { t.m[12] = (float)n.translation[0]; t.m[13] = (float)n.translation[1]; t.m[14] = (float)n.translation[2]; }
            if (n.rotation.size() == 4)
            {
                const float x = (float)n.rotation[0], y = (float)n.rotation[1], z = (float)n.rotation[2], w = (float)n.rotation[3];
                r.m[0] = 1 - 2 * (y * y + z * z); r.m[1] = 2 * (x * y + z * w);     r.m[2] = 2 * (x * z - y * w);
                r.m[4] = 2 * (x * y - z * w);     r.m[5] = 1 - 2 * (x * x + z * z); r.m[6] = 2 * (y * z + x * w);
                r.m[8] = 2 * (x * z + y * w);     r.m[9] = 2 * (y * z - x * w);     r.m[10] = 1 - 2 * (x * x + y * y);
            }
            if (n.scale.size() == 3) { s.m[0] = (float)n.scale[0]; s.m[5] = (float)n.scale[1]; s.m[10] = (float)n.scale[2]; }
            return Mul(t, Mul(r, s));
        }

        // Read accessor element `i`, component `comp` (0..) as a float, regardless of storage type.
        static float ReadComponent(const tinygltf::Model& mdl, const tinygltf::Accessor& acc, size_t i, int comp, int numComp)
        {
            const tinygltf::BufferView& bv = mdl.bufferViews[acc.bufferView];
            const tinygltf::Buffer& buf = mdl.buffers[bv.buffer];
            const int compBytes = tinygltf::GetComponentSizeInBytes(acc.componentType);
            const size_t stride = acc.ByteStride(bv) ? (size_t)acc.ByteStride(bv) : (size_t)compBytes * numComp;
            const uint8_t* base = buf.data.data() + bv.byteOffset + acc.byteOffset + i * stride + (size_t)comp * compBytes;
            switch (acc.componentType)
            {
                case TINYGLTF_COMPONENT_TYPE_FLOAT: { float v; std::memcpy(&v, base, 4); return v; }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: { uint16_t v; std::memcpy(&v, base, 2); return acc.normalized ? v / 65535.0f : (float)v; }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: { uint8_t v = *base; return acc.normalized ? v / 255.0f : (float)v; }
                default: return 0.0f;
            }
        }
        static uint32_t ReadIndex(const tinygltf::Model& mdl, const tinygltf::Accessor& acc, size_t i)
        {
            const tinygltf::BufferView& bv = mdl.bufferViews[acc.bufferView];
            const uint8_t* base = mdl.buffers[bv.buffer].data.data() + bv.byteOffset + acc.byteOffset;
            switch (acc.componentType)
            {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: { uint32_t v; std::memcpy(&v, base + i * 4, 4); return v; }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: { uint16_t v; std::memcpy(&v, base + i * 2, 2); return v; }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: return base[i];
                default: return 0;
            }
        }

        // Load a .gltf (ASCII + external .bin/images) or .glb. Returns false on parse failure.
        // keepCpuGeometry retains cpuVerts/cpuIndices (for a CPU BVH build); buildTextureArray also
        // assembles textureArray (for the non-bindless software ray-tracing path).
        bool Load(ExampleApp& app, const char* path, bool keepCpuGeometry = false, bool buildTextureArray = false)
        {
            VriCoreInterface& c = app.c;
            tinygltf::Model mdl;
            tinygltf::TinyGLTF loader;
            std::string err, warn;
            const std::string p = path;
            const bool isBinary = p.size() > 4 && p.compare(p.size() - 4, 4, ".glb") == 0;
            const bool ok = isBinary ? loader.LoadBinaryFromFile(&mdl, &err, &warn, p)
                                     : loader.LoadASCIIFromFile(&mdl, &err, &warn, p);
            if (!warn.empty()) std::printf("[gltf] warning: %s\n", warn.c_str());
            if (!ok) { std::printf("[gltf] failed to load '%s': %s\n", path, err.c_str()); return false; }

            std::vector<GltfVertex> verts;
            std::vector<uint32_t>   indices;
            bool haveBounds = false;

            // Walk the default scene's node hierarchy, baking each node's world matrix into the
            // vertices it draws. One primitive -> one GltfPrimitive (BLAS geometry) entry.
            const tinygltf::Scene& scene = mdl.scenes[mdl.defaultScene >= 0 ? mdl.defaultScene : 0];
            struct StackItem { int node; Mat4 world; };
            std::vector<StackItem> stack;
            for (int n : scene.nodes) stack.push_back({n, Identity()});
            while (!stack.empty())
            {
                const StackItem item = stack.back();
                stack.pop_back();
                const tinygltf::Node& node = mdl.nodes[item.node];
                const Mat4 world = Mul(item.world, NodeLocal(node));
                for (int child : node.children) stack.push_back({child, world});
                if (node.mesh < 0) continue;

                for (const tinygltf::Primitive& prim : mdl.meshes[node.mesh].primitives)
                {
                    if (prim.indices < 0 || prim.attributes.count("POSITION") == 0) continue;
                    const uint32_t vertexStart = (uint32_t)verts.size();
                    const tinygltf::Accessor& posAcc = mdl.accessors[prim.attributes.at("POSITION")];
                    const bool hasNrm = prim.attributes.count("NORMAL") > 0;
                    const bool hasUv = prim.attributes.count("TEXCOORD_0") > 0;
                    const tinygltf::Accessor* nrmAcc = hasNrm ? &mdl.accessors[prim.attributes.at("NORMAL")] : nullptr;
                    const tinygltf::Accessor* uvAcc = hasUv ? &mdl.accessors[prim.attributes.at("TEXCOORD_0")] : nullptr;

                    for (size_t i = 0; i < posAcc.count; ++i)
                    {
                        GltfVertex v{};
                        float lp[3] = {ReadComponent(mdl, posAcc, i, 0, 3), ReadComponent(mdl, posAcc, i, 1, 3), ReadComponent(mdl, posAcc, i, 2, 3)};
                        XformPoint(world, lp, v.pos);
                        if (nrmAcc)
                        {
                            float ln[3] = {ReadComponent(mdl, *nrmAcc, i, 0, 3), ReadComponent(mdl, *nrmAcc, i, 1, 3), ReadComponent(mdl, *nrmAcc, i, 2, 3)};
                            float wn[3]; XformDir(world, ln, wn);
                            const float len = std::sqrt(wn[0] * wn[0] + wn[1] * wn[1] + wn[2] * wn[2]);
                            if (len > 0) { v.normal[0] = wn[0] / len; v.normal[1] = wn[1] / len; v.normal[2] = wn[2] / len; }
                        }
                        if (uvAcc) { v.uv[0] = ReadComponent(mdl, *uvAcc, i, 0, 2); v.uv[1] = ReadComponent(mdl, *uvAcc, i, 1, 2); }
                        for (int k = 0; k < 3; ++k)
                        {
                            if (!haveBounds) { boundsMin[k] = boundsMax[k] = v.pos[k]; }
                            else { boundsMin[k] = std::min(boundsMin[k], v.pos[k]); boundsMax[k] = std::max(boundsMax[k], v.pos[k]); }
                        }
                        haveBounds = true;
                        verts.push_back(v);
                    }                    GltfPrimitive gp{};
                    gp.firstIndex = (uint32_t)indices.size();
                    const tinygltf::Accessor& idxAcc = mdl.accessors[prim.indices];
                    gp.indexCount = (uint32_t)idxAcc.count;
                    for (size_t i = 0; i < idxAcc.count; ++i) indices.push_back(vertexStart + ReadIndex(mdl, idxAcc, i));
                    if (prim.material >= 0)
                    {
                        const tinygltf::Material& mat = mdl.materials[prim.material];
                        gp.baseColorTexture = mat.pbrMetallicRoughness.baseColorTexture.index;
                        gp.occlusionTexture = mat.occlusionTexture.index;
                    }
                    primitives.push_back(gp);
                }
            }

            vertexCount = (uint32_t)verts.size();
            indexCount = (uint32_t)indices.size();
            if (vertexCount == 0 || indexCount == 0) { std::printf("[gltf] '%s' has no triangles\n", path); return false; }

            // Geometry buffers: host-visible so the AS build reads them directly and the closesthit
            // shader reads them as SSBOs (no separate device copy needed for an example-sized model).
            const uint64_t vbSize = (uint64_t)vertexCount * sizeof(GltfVertex);
            const uint64_t ibSize = (uint64_t)indexCount * sizeof(uint32_t);
            VriBufferDesc vbd{}; vbd.size = vbSize; vbd.usage = VriBufferUsage_AccelerationBuildInput | VriBufferUsage_StorageBuffer; vbd.memoryLocation = VriMemoryLocation_HostUpload;
            c.CreateBuffer(app.dev, &vbd, &vertexBuffer);
            std::memcpy(c.MapBuffer(vertexBuffer, 0, vbSize), verts.data(), vbSize); c.UnmapBuffer(vertexBuffer);
            VriBufferDesc ibd{}; ibd.size = ibSize; ibd.usage = VriBufferUsage_AccelerationBuildInput | VriBufferUsage_StorageBuffer; ibd.memoryLocation = VriMemoryLocation_HostUpload;
            c.CreateBuffer(app.dev, &ibd, &indexBuffer);
            std::memcpy(c.MapBuffer(indexBuffer, 0, ibSize), indices.data(), ibSize); c.UnmapBuffer(indexBuffer);
            if (keepCpuGeometry) { cpuVerts = verts; cpuIndices = indices; }

            // One GPU texture per glTF texture (RGBA8). Decoded by tinygltf via stb_image.
            app.BeginUpload();
            std::vector<std::vector<uint8_t>> pixelHold; pixelHold.reserve(mdl.textures.size());
            std::vector<std::pair<uint32_t, uint32_t>> texDims; texDims.reserve(mdl.textures.size());
            for (const tinygltf::Texture& t : mdl.textures)
            {
                const tinygltf::Image& img = mdl.images[t.source >= 0 ? t.source : 0];
                const uint32_t w = (uint32_t)img.width, h = (uint32_t)img.height;
                texDims.push_back({w, h});
                std::vector<uint8_t> rgba((size_t)w * h * 4, 255);
                const int comp = img.component;
                for (size_t px = 0; px < (size_t)w * h; ++px)
                    for (int k = 0; k < 4; ++k)
                        rgba[px * 4 + k] = (k < comp) ? img.image[px * comp + k] : (k == 3 ? 255 : 0);
                pixelHold.push_back(std::move(rgba));

                VriTextureDesc td{}; td.type = VriTextureType_2D; td.format = VriFormat_RGBA8_UNORM; td.width = w; td.height = h; td.depth = 1;
                td.mipNum = 1; td.layerNum = 1; td.sampleNum = 1; td.usage = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst; td.memoryLocation = VriMemoryLocation_Device;
                VriTexture* tex = nullptr; c.CreateTexture(app.dev, &td, &tex);
                VriTextureViewDesc vd{}; vd.texture = tex; vd.viewType = VriTextureViewType_2D; vd.format = VriFormat_Unknown; vd.aspect = VriImageAspect_Color;
                VriDescriptor* view = nullptr; c.CreateTextureView(app.dev, &vd, &view);
                app.UploadTexture(tex, pixelHold.back().data(), w, h, 1, w * h * 4);
                textures.push_back(tex); textureViews.push_back(view);
            }

            // 2D-array texture (one layer per glTF texture, all nearest-resampled to a common size)
            // for the non-bindless software path. Capped at 1024 to bound memory; nearest-neighbor
            // keeps the loader dependency-free (no stb_image_resize) and is plenty for the examples.
            if (buildTextureArray && !pixelHold.empty())
            {
                constexpr uint32_t kDim = 1024;
                textureArrayLayers = (uint32_t)pixelHold.size();
                const size_t layerBytes = (size_t)kDim * kDim * 4;
                std::vector<uint8_t> packed(layerBytes * textureArrayLayers);
                for (uint32_t L = 0; L < textureArrayLayers; ++L)
                {
                    const uint32_t sw = texDims[L].first, sh = texDims[L].second;
                    const uint8_t* src = pixelHold[L].data();
                    uint8_t* dst = packed.data() + (size_t)L * layerBytes;
                    for (uint32_t y = 0; y < kDim; ++y)
                    {
                        const uint32_t sy = sh ? (y * sh) / kDim : 0;
                        for (uint32_t x = 0; x < kDim; ++x)
                        {
                            const uint32_t sx = sw ? (x * sw) / kDim : 0;
                            std::memcpy(dst + ((size_t)y * kDim + x) * 4, src + ((size_t)sy * sw + sx) * 4, 4);
                        }
                    }
                }
                VriTextureDesc ad{}; ad.type = VriTextureType_2D; ad.format = VriFormat_RGBA8_UNORM; ad.width = kDim; ad.height = kDim; ad.depth = 1;
                ad.mipNum = 1; ad.layerNum = textureArrayLayers; ad.sampleNum = 1; ad.usage = VriTextureUsage_ShaderResource | VriTextureUsage_TransferDst; ad.memoryLocation = VriMemoryLocation_Device;
                c.CreateTexture(app.dev, &ad, &textureArray);
                VriTextureViewDesc avd{}; avd.texture = textureArray; avd.viewType = VriTextureViewType_2DArray; avd.format = VriFormat_Unknown; avd.aspect = VriImageAspect_Color; avd.layerNum = textureArrayLayers;
                c.CreateTextureView(app.dev, &avd, &textureArrayView);
                app.UploadTexture(textureArray, packed.data(), kDim, kDim, textureArrayLayers, (uint32_t)layerBytes);
            }
            app.EndUpload();

            std::printf("[gltf] '%s': %u verts, %u indices, %zu primitives, %zu textures\n",
                        path, vertexCount, indexCount, primitives.size(), textures.size());
            return true;
        }

        void Destroy(ExampleApp& app)
        {
            VriCoreInterface& c = app.c;
            for (VriDescriptor* v : textureViews) c.DestroyDescriptor(v);
            for (VriTexture* t : textures) c.DestroyTexture(t);
            if (textureArrayView) c.DestroyDescriptor(textureArrayView);
            if (textureArray) c.DestroyTexture(textureArray);
            if (vertexBuffer) c.DestroyBuffer(vertexBuffer);
            if (indexBuffer) c.DestroyBuffer(indexBuffer);
            textureArrayView = nullptr; textureArray = nullptr; textureArrayLayers = 0;
            textureViews.clear(); textures.clear(); primitives.clear(); cpuVerts.clear(); cpuIndices.clear();
        }
    };
} // namespace vriex
