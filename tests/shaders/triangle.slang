// Bufferless triangle in Slang. One module, two entry points (vertexMain +
// fragmentMain); the SPIR-V blob is referenced twice with the entry-point name.
struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

static const float2 kPositions[3] = { float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };
static const float3 kColors[3]    = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1) };

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    VSOutput o;
    o.position = float4(kPositions[vid], 0.0, 1.0);
    o.color = kColors[vid];
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    // Solid red so the center pixel is exactly (255, 0, 0, 255).
    return float4(1.0, 0.0, 0.0, 1.0);
}
