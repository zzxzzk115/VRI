// Fragment-shader barycentrics: the fragment reads the rasterizer-generated
// barycentric coordinates (SV_Barycentrics) and outputs them as color. The three
// coords sum to 1 at every covered pixel, so R+G+B == 255 there - the test's signal
// that barycentrics are actually delivered. Maps to VK_KHR_fragment_shader_barycentric
// (SPIR-V) and SM6.1 SV_Barycentrics (DXIL).
struct VSOutput { float4 position : SV_Position; };

static const float2 kPositions[3] = { float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    VSOutput o;
    o.position = float4(kPositions[vid], 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 fragmentMain(float3 bary : SV_Barycentrics) : SV_Target
{
    return float4(bary, 1.0);
}
