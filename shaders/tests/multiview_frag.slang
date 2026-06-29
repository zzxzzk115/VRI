// Multiview test - fragment stage only (paired with multiview_vert.slang). Keeping the fragment in
// its own module guarantees no SV_ViewID / ViewIndex builtin here, which GL OVR_multiview2 forbids
// outside the vertex stage. Just passes the per-view vertex color through.
struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
