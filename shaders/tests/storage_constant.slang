// Writes a push-constant value into every texel of the bound storage view.
struct Parameters
{
    float4 value;
};
[[vk::push_constant]] ConstantBuffer<Parameters> uParameters;
[[vk::binding(0, 0)]] RWTexture2D<float4> uTarget;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint w, h;
    uTarget.GetDimensions(w, h);
    if (tid.x < w && tid.y < h)
        uTarget[tid.xy] = uParameters.value;
}
