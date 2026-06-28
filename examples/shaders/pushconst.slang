// Push-constant probe/example: a fullscreen triangle whose color comes entirely from a
// push constant. Used by the push-constant cross-backend test + examples/pushconstants.
// Slang lowers [[vk::push_constant]] per target (SPIR-V PushConstant block, an HLSL cbuffer,
// a WGSL uniform); the VRI backends route CmdSetConstants data to wherever it lands.
struct PushData
{
    float4 color;
};
[[vk::push_constant]] ConstantBuffer<PushData> pc;

struct VSOutput
{
    float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    // Oversized fullscreen triangle from the vertex id (no vertex buffer).
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VSOutput o;
    o.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return pc.color;
}
