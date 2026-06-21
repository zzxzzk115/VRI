// A very thin (sub-pixel-tall) triangle for the conservative-rasterization test:
// normal rasterization covers few/no pixel centers, conservative rasterization covers
// every pixel the triangle touches - so the conservative draw lights up strictly more
// pixels. Solid red.
struct VSOutput { float4 position : SV_Position; };

static const float2 kPositions[3] = { float2(-0.9, 0.0), float2(0.9, 0.0), float2(0.9, 0.04) };

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    VSOutput o;
    o.position = float4(kPositions[vid], 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 fragmentMain() : SV_Target
{
    return float4(1.0, 0.0, 0.0, 1.0);
}
