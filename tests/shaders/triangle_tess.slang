// Tessellation pass-through: a 3-control-point triangle patch is tessellated
// (hull sets tess factors, domain reconstructs positions from barycentric
// coordinates) and painted green. With cullMode=None the tessellated triangle
// covers the same area, so the center pixel is green - proving the legacy
// hull/domain (tessellation control/evaluation) path on Vulkan + desktop OpenGL.
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
    float edges[3] : SV_TessFactor;
    float inside   : SV_InsideTessFactor;
};

struct DSOut
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

static const float2 kPos[3] = { float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };

[shader("vertex")]
VSOut vertexMain(uint vid : SV_VertexID)
{
    VSOut o;
    o.pos = float3(kPos[vid], 0.0);
    return o;
}

PatchConst constants(InputPatch<VSOut, 3> patch)
{
    PatchConst o;
    o.edges[0] = 2.0;
    o.edges[1] = 2.0;
    o.edges[2] = 2.0;
    o.inside = 2.0;
    return o;
}

[shader("hull")]
[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("constants")]
HSOut hullMain(InputPatch<VSOut, 3> patch, uint i : SV_OutputControlPointID)
{
    HSOut o;
    o.pos = patch[i].pos;
    return o;
}

[shader("domain")]
[domain("tri")]
DSOut domainMain(PatchConst c, float3 bary : SV_DomainLocation, const OutputPatch<HSOut, 3> patch)
{
    DSOut o;
    float3 p = patch[0].pos * bary.x + patch[1].pos * bary.y + patch[2].pos * bary.z;
    o.position = float4(p, 1.0);
    o.color = float3(0.0, 1.0, 0.0);
    return o;
}

[shader("fragment")]
float4 fragmentMain(DSOut input) : SV_Target
{
    return float4(input.color, 1.0);
}
