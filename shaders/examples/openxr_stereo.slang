// OpenXR single-pass-stereo shader. One bufferless triangle rendered to both eyes of a 2-layer
// swapchain array in a single pass; SV_ViewID (-> SPIR-V gl_ViewIndex) selects the eye, so the
// left eye is tinted red and the right eye green -- an unmistakable proof the multiview viewMask
// drove per-eye rendering in the headset. The triangle is also nudged sideways per eye to give a
// crude stereo parallax.
struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

static const float2 kPositions[3] = { float2(0.0, -0.6), float2(0.6, 0.6), float2(-0.6, 0.6) };

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID, uint viewID : SV_ViewID)
{
    VSOutput o;
    float    parallax = (viewID == 0) ? 0.05 : -0.05; // shift the eyes apart a touch
    o.position = float4(kPositions[vid] + float2(parallax, 0.0), 0.0, 1.0);
    o.color    = (viewID == 0) ? float3(1, 0.2, 0.2)  // left eye  -> red
                               : float3(0.2, 1, 0.2); // right eye -> green
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
