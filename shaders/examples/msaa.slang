// Scene shader for examples/msaa: a spinning pinwheel of thin, gap-separated triangle "blades"
// (lots of sharp high-contrast edges, where multisampling is most visible). The rotation +
// aspect correction come from a push constant; each vertex carries its blade color.
struct Push { float4 rot; }; // x = cos, y = sin, z = aspect (height/width), w unused
[[vk::push_constant]] ConstantBuffer<Push> pc;

struct VSInput
{
    float2 pos : TEXCOORD0;
    float3 color : TEXCOORD1;
};
struct VSOutput
{
    float4 position : SV_Position;
    float3 color : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertexMain(VSInput i)
{
    VSOutput o;
    float2 p = float2(i.pos.x * pc.rot.x - i.pos.y * pc.rot.y,
                      i.pos.x * pc.rot.y + i.pos.y * pc.rot.x);
    p.x *= pc.rot.z; // keep the pinwheel circular regardless of aspect ratio
    o.position = float4(p, 0.0, 1.0);
    o.color = i.color;
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput i) : SV_Target
{
    return float4(i.color, 1.0);
}
