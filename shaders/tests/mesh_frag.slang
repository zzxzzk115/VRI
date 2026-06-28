// Minimal fragment shader for mesh-shader pipelines: takes NO interpolated input
// (D3D12 requires PS inputs be a subset of the mesh shader's outputs; our mesh
// shader outputs only SV_Position), returns solid red. Used by the D3D12 mesh test
// paired with triangle_mesh's mesh entry point.
[shader("fragment")]
float4 fragmentMain() : SV_Target
{
    return float4(1.0, 0.0, 0.0, 1.0);
}
