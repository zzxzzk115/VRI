// Multiview test shader: one bufferless fullscreen triangle rendered to every view of a layered
// target in a single pass. Each view colors itself from SV_ViewID (-> SPIR-V gl_ViewIndex), so a
// 2-view stereo render fills layer 0 red and layer 1 green -- proving the viewMask drove per-layer
// rendering. SV_ViewID needs the device multiview feature (VkPhysicalDeviceVulkan11Features::multiview).
struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

// A big clip-space triangle that fully covers [-1,1]^2 (so every pixel of each layer is written).
static const float2 kPositions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID, uint viewID : SV_ViewID)
{
    VSOutput o;
    o.position = float4(kPositions[vid], 0.0, 1.0);
    o.color = (viewID == 0) ? float3(1, 0, 0)  // view 0 -> red
            : (viewID == 1) ? float3(0, 1, 0)  // view 1 -> green
            : (viewID == 2) ? float3(0, 0, 1)  // view 2 -> blue
                            : float3(1, 1, 1);
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
