#pragma once

#include "gltf_model.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

namespace vriex
{
    struct BvhNode
    {
        float    aabbMin[3];
        uint32_t leftOrFirst;
        float    aabbMax[3];
        uint32_t countOrSecond;
    };

    struct BvhTriRef
    {
        uint32_t indexBase;
        uint32_t geomNodeIndex;
    };

    // Leaf nodes set this high bit in countOrSecond so the shader can tell them apart from interior
    // nodes (whose countOrSecond stores the right-child index, which can otherwise also be small).
    static constexpr uint32_t kBvhLeafFlag = 0x80000000u;

    struct Bvh
    {
        std::vector<BvhNode>   nodes;
        std::vector<BvhTriRef> triRefs;
    };

    namespace detail
    {
        struct BuildTri
        {
            float    bmin[3];
            float    bmax[3];
            float    centroid[3];
            uint32_t indexBase     = 0;
            uint32_t geomNodeIndex = 0;
        };

        inline void ResetBounds(float* bmin, float* bmax)
        {
            for (int k = 0; k < 3; ++k)
            {
                bmin[k] = FLT_MAX;
                bmax[k] = -FLT_MAX;
            }
        }

        inline void GrowBounds(float* bmin, float* bmax, const float* p)
        {
            for (int k = 0; k < 3; ++k)
            {
                bmin[k] = std::min(bmin[k], p[k]);
                bmax[k] = std::max(bmax[k], p[k]);
            }
        }

        inline uint32_t BuildNode(std::vector<BuildTri>& tris, uint32_t first, uint32_t count, Bvh& out)
        {
            const uint32_t nodeIndex = static_cast<uint32_t>(out.nodes.size());
            out.nodes.push_back({});

            float bmin[3], bmax[3], cmin[3], cmax[3];
            ResetBounds(bmin, bmax);
            ResetBounds(cmin, cmax);
            for (uint32_t i = first; i < first + count; ++i)
            {
                GrowBounds(bmin, bmax, tris[i].bmin);
                GrowBounds(bmin, bmax, tris[i].bmax);
                GrowBounds(cmin, cmax, tris[i].centroid);
            }

            BvhNode& node = out.nodes[nodeIndex];
            for (int k = 0; k < 3; ++k)
            {
                node.aabbMin[k] = bmin[k];
                node.aabbMax[k] = bmax[k];
            }

            if (count <= 4)
            {
                node.leftOrFirst   = static_cast<uint32_t>(out.triRefs.size());
                node.countOrSecond = count | kBvhLeafFlag;
                for (uint32_t i = first; i < first + count; ++i)
                    out.triRefs.push_back({tris[i].indexBase, tris[i].geomNodeIndex});
                return nodeIndex;
            }

            int         axis  = 0;
            const float ex[3] = {cmax[0] - cmin[0], cmax[1] - cmin[1], cmax[2] - cmin[2]};
            if (ex[1] > ex[axis])
                axis = 1;
            if (ex[2] > ex[axis])
                axis = 2;

            const uint32_t mid = first + count / 2;
            std::nth_element(
                tris.begin() + first,
                tris.begin() + mid,
                tris.begin() + first + count,
                [axis](const BuildTri& a, const BuildTri& b) { return a.centroid[axis] < b.centroid[axis]; });

            const uint32_t left                = BuildNode(tris, first, mid - first, out);
            const uint32_t right               = BuildNode(tris, mid, first + count - mid, out);
            out.nodes[nodeIndex].leftOrFirst   = left;
            out.nodes[nodeIndex].countOrSecond = right;
            return nodeIndex;
        }
    } // namespace detail

    inline Bvh BuildBvh(const GltfModel& model)
    {
        std::vector<detail::BuildTri> tris;
        tris.reserve(model.indexCount / 3);
        for (uint32_t geom = 0; geom < static_cast<uint32_t>(model.primitives.size()); ++geom)
        {
            const GltfPrimitive& prim = model.primitives[geom];
            for (uint32_t i = 0; i + 2 < prim.indexCount; i += 3)
            {
                const uint32_t   base = prim.firstIndex + i;
                detail::BuildTri tri {};
                detail::ResetBounds(tri.bmin, tri.bmax);
                for (uint32_t k = 0; k < 3; ++k)
                    detail::GrowBounds(tri.bmin, tri.bmax, model.cpuVerts[model.cpuIndices[base + k]].pos);
                for (int k = 0; k < 3; ++k)
                    tri.centroid[k] = (tri.bmin[k] + tri.bmax[k]) * 0.5f;
                tri.indexBase     = base;
                tri.geomNodeIndex = geom;
                tris.push_back(tri);
            }
        }

        Bvh out;
        if (!tris.empty())
        {
            out.nodes.reserve(tris.size() * 2);
            out.triRefs.reserve(tris.size());
            detail::BuildNode(tris, 0, static_cast<uint32_t>(tris.size()), out);
        }
        return out;
    }
} // namespace vriex
