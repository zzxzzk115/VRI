// Read hit identity and barycentrics independently of the shading/material path.
[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]][format("rgba8")] RWTexture2D<float4> outImage;

[shader("compute")][numthreads(8, 8, 1)] void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint width, height;
    outImage.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
        return;
    RayDesc description;
    description.Origin = float3((float2(tid.xy) + 0.5) / float2(width, height) * 2 - 1, -1);
    description.Direction = float3(0, 0, 1);
    description.TMin = 0.001;
    description.TMax = 10;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(tlas, RAY_FLAG_NONE, 255, description);
    float4 candidate = 0;
    while (query.Proceed())
    {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            candidate = float4(float(query.CandidateInstanceID()) / 255.0, query.CandidateTriangleBarycentrics(), 1);
            query.CommitNonOpaqueTriangleHit();
        }
    }
    float4 result = 0;
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        result = float4(float(query.CommittedInstanceID()) / 255.0, query.CommittedTriangleBarycentrics(), 1);
    // Both alpha-test candidates and committed surfaces must identify the same instance/UV weights.
    if (any(abs(candidate - result) > 1e-5))
        result = 0;
    outImage[tid.xy] = result;
}
