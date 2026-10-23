// Match VriCoordinateTransform. These helpers are explicit shader boundaries;
// do not apply them again to an already transformed projection matrix.
float4 vriTransformClipPosition(float4 position, float3 clipTransform)
{
    position.y *= clipTransform.x;
    position.z = position.z * clipTransform.y + position.w * clipTransform.z;
    return position;
}

float2 vriTransformUv(float2 uv, float2 uvTransform)
{
    uv.y = uv.y * uvTransform.x + uvTransform.y;
    return uv;
}
