// Ray-traced triangle in Slang -> SPIR-V. Three entry points: raygen casts an
// orthographic ray per pixel into the TLAS; closesthit writes red; miss writes
// black. SPIR-V only (no WebGPU/D3D mapping here). The storage image format is
// pinned to rgba8 so no shaderStorageImageWriteWithoutFormat feature is needed.
[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> outImage;

struct Payload { float3 color; };

[shader("raygeneration")]
void rayGenMain()
{
    uint2 idx = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;
    // pixel center -> [-1, 1]
    float2 uv = (float2(idx) + 0.5) / float2(dim) * 2.0 - 1.0;

    RayDesc ray;
    ray.Origin = float3(uv.x, uv.y, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.001;
    ray.TMax = 10.0;

    Payload payload;
    payload.color = float3(0.0, 0.0, 0.0);
    TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    outImage[idx] = float4(payload.color, 1.0);
}

[shader("miss")]
void missMain(inout Payload payload)
{
    payload.color = float3(0.0, 0.0, 0.0); // background
}

[shader("closesthit")]
void closestHitMain(inout Payload payload, BuiltInTriangleIntersectionAttributes attr)
{
    payload.color = float3(1.0, 0.0, 0.0); // solid red
}
