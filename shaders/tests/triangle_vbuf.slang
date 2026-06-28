// Triangle whose vertices come from a VERTEX BUFFER (position + color attributes),
// drawn indexed. Validates vertex input + indexed draw across backends. Attribute
// order (POSITION=0, COLOR0=1) maps to VRI vertex-attribute indices 0 and 1.
struct VSInput
{
    float3 position : POSITION;
    float3 color : COLOR0;
};

struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput o;
    o.position = float4(input.position, 1.0);
    o.color = input.color;
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
