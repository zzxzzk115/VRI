// Mesh-shader triangle in Slang. MESH ENTRY ONLY (single entry point): emits one
// triangle with the same positions as triangle.slang. Kept mesh-only on purpose -
// Slang's multi-entry SPIR-V link (getTargetCode) drops the OutputTrianglesEXT
// execution mode, producing SPIR-V that spirv-val rejects; a single-entry mesh
// module is emitted correctly. The fragment stage reuses triangle.slang's
// fragmentMain (g_triangleSpv). SPIR-V only - no WebGPU/D3D12 mesh stage here.
struct VertexOut
{
    float4 position : SV_Position;
};

static const float2 kPositions[3] = { float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshMain(
    out vertices VertexOut verts[3],
    out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);
    for (uint i = 0; i < 3; ++i)
        verts[i].position = float4(kPositions[i], 0.0, 1.0);
    tris[0] = uint3(0, 1, 2);
}
