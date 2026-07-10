// Tessellated terrain for examples/tessellation: 4-control-point QUAD patches, the hull
// stage sets the ImGui-driven tessellation factors, and the domain stage displaces the
// tessellated grid by the shared procedural height field (terrain_height.slangh). At
// factor 1 the terrain is the raw faceted patch grid; at 32 it is smooth - tessellation
// density made visible (add the wireframe toggle in the example to count triangles).
// Desktop-only stages: no WGSL output exists for this file (WebGPU has no hull/domain);
// the fallback path lives in terrain_basic.slang.
#include "terrain_height.slangh"

struct Camera
{
    float4x4 mvp;
    float4 params; // x = tess factor, y = height scale
};
[[vk::binding(0, 0)]] ConstantBuffer<Camera> uCam;

struct VSOut
{
    float3 pos : POSITION;
};
struct HSOut
{
    float3 pos : POSITION;
};
struct PatchConst
{
    float edges[4] : SV_TessFactor;
    float inside[2] : SV_InsideTessFactor;
};
struct DSOut
{
    float4 position : SV_Position;
    float height : TEXCOORD0;
    float3 normal : TEXCOORD1;
};

struct VSInput
{
    float3 pos : TEXCOORD0;
};

[shader("vertex")]
VSOut vertexMain(VSInput i)
{
    VSOut o;
    o.pos = i.pos;
    return o;
}

PatchConst constants(InputPatch<VSOut, 4> patch)
{
    PatchConst o;
    float f = uCam.params.x;
    o.edges[0] = f;
    o.edges[1] = f;
    o.edges[2] = f;
    o.edges[3] = f;
    o.inside[0] = f;
    o.inside[1] = f;
    return o;
}

[shader("hull")]
[domain("quad")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(4)]
[patchconstantfunc("constants")]
HSOut hullMain(InputPatch<VSOut, 4> patch, uint i : SV_OutputControlPointID)
{
    HSOut o;
    o.pos = patch[i].pos;
    return o;
}

[shader("domain")]
[domain("quad")]
DSOut domainMain(PatchConst c, float2 uv : SV_DomainLocation, const OutputPatch<HSOut, 4> patch)
{
    // control points arrive in (v0 v1 / v3 v2)-independent order: bilinear by uv
    float3 a = lerp(patch[0].pos, patch[1].pos, uv.x);
    float3 b = lerp(patch[3].pos, patch[2].pos, uv.x);
    float3 p = lerp(a, b, uv.y);
    float h = TerrainHeight(p.xz);
    p.y = h * uCam.params.y;

    DSOut o;
    o.position = mul(uCam.mvp, float4(p, 1.0));
    o.height = h;
    o.normal = TerrainNormal(p.xz);
    return o;
}

[shader("fragment")]
float4 fragmentMain(DSOut i) : SV_Target
{
    return float4(TerrainColor(i.height, normalize(i.normal)), 1.0);
}
