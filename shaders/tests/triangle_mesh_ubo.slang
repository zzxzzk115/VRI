// Mesh shader that reads a constant buffer - a regression guard for descriptor
// binding to the MESH stage (the render encoder used to bind buffers only to
// vertex/fragment, so a mesh shader saw zeros). Sizes the triangle by a scale read
// from the CB: bound correctly -> scale 2 -> the triangle covers the center pixel;
// not bound -> scale reads 0 -> a degenerate triangle that misses it. Reuses
// triangle.slang's solid-red fragmentMain, so a covered center reads back red.
// MESH ENTRY ONLY (see triangle_mesh.slang on why the mesh stage stays single-entry).
struct Params
{
    float  scale;
    float3 pad;
};
[[vk::binding(0, 0)]] ConstantBuffer<Params> uParams;

struct VertexOut
{
    float4 position : SV_Position;
};

static const float2 kPositions[3] = {float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5)};

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshMain(out vertices VertexOut verts[3], out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);
    for (uint i = 0; i < 3; ++i)
        verts[i].position = float4(kPositions[i] * uParams.scale, 0.0, 1.0);
    tris[0] = uint3(0, 1, 2);
}
