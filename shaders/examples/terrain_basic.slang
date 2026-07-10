// Fallback path for examples/tessellation, used where VriDeviceDesc::hasTessellation is
// false (Metal, WebGPU): a pre-triangulated fixed-density grid is displaced in the VERTEX
// stage by the same shared height field (terrain_height.slangh), so the scene shades
// identically - just without the ImGui-driven tessellation density.
#include "terrain_height.slangh"

struct Camera
{
    float4x4 mvp;
    float4 params; // x = tess factor (unused here), y = height scale
};
[[vk::binding(0, 0)]] ConstantBuffer<Camera> uCam;

struct VSInput
{
    float3 pos : TEXCOORD0;
};
struct VSOutput
{
    float4 position : SV_Position;
    float height : TEXCOORD0;
    float3 normal : TEXCOORD1;
};

[shader("vertex")]
VSOutput vertexMain(VSInput i)
{
    float3 p = i.pos;
    float h = TerrainHeight(p.xz);
    p.y = h * uCam.params.y;
    VSOutput o;
    o.position = mul(uCam.mvp, float4(p, 1.0));
    o.height = h;
    o.normal = TerrainNormal(p.xz);
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput i) : SV_Target
{
    return float4(TerrainColor(i.height, normalize(i.normal)), 1.0);
}
