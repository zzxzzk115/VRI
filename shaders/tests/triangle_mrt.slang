// Triangle that writes to TWO color render targets (MRT): target 0 red, target 1
// blue. Validates multiple color attachments across backends.
struct VSOutput
{
    float4 position : SV_Position;
};

struct FSOutput
{
    float4 color0 : SV_Target0;
    float4 color1 : SV_Target1;
};

static const float2 kPositions[3] = { float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    VSOutput o;
    o.position = float4(kPositions[vid], 0.0, 1.0);
    return o;
}

[shader("fragment")]
FSOutput fragmentMain()
{
    FSOutput o;
    o.color0 = float4(1.0, 0.0, 0.0, 1.0); // red   -> attachment 0
    o.color1 = float4(0.0, 0.0, 1.0, 1.0); // blue  -> attachment 1
    return o;
}
