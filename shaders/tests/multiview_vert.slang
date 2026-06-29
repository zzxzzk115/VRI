// Multiview test - vertex stage only. Single-entry-point module so the SV_ViewID (-> SPIR-V
// ViewIndex) builtin stays out of the fragment module: SPIRV-Cross (GL OVR_multiview2) only allows
// the view index in the vertex stage, and a shared two-entry-point module would leak it into the
// fragment's interface. One big clip-space triangle covers [-1,1]^2; each view colors itself.
struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

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
