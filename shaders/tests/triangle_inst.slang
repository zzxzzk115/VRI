// Instanced draw: two instances of a small triangle, offset left/right by the
// instance id, colored red (instance 0) and green (instance 1). Validates
// instanced drawing + SV_InstanceID across backends.
struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

static const float2 kBase[3] = { float2(0.0, -0.3), float2(0.3, 0.3), float2(-0.3, 0.3) };

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    float xoff = (iid == 0u) ? -0.4 : 0.4;
    VSOutput o;
    o.position = float4(kBase[vid].x + xoff, kBase[vid].y, 0.0, 1.0);
    o.color = (iid == 0u) ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
