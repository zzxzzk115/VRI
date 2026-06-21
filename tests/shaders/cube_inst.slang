// Instanced textured cube: per-vertex {position, uv} from stream 0, a per-instance model
// matrix (4 columns) from stream 1, and a shared view-projection in a constant buffer.
// The model columns are multiplied directly (M*v = c0*x + c1*y + c2*z + c3*w) so no
// matrix is reconstructed in the shader - column-major data flows straight through.
struct VSInput
{
    float3 position : POSITION;
    float2 uv : TEXCOORD0;
    float4 c0 : TEXCOORD1; // model column 0
    float4 c1 : TEXCOORD2; // model column 1
    float4 c2 : TEXCOORD3; // model column 2
    float4 c3 : TEXCOORD4; // model column 3
};
struct VSOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};
struct Camera { float4x4 viewProj; };

[[vk::binding(0, 0)]] ConstantBuffer<Camera> uCam;
[[vk::binding(1, 0)]] Texture2D<float4> uTex;
[[vk::binding(2, 0)]] SamplerState uSmp;

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    float4 p = float4(input.position, 1.0);
    float4 world = input.c0 * p.x + input.c1 * p.y + input.c2 * p.z + input.c3 * p.w;
    VSOutput o;
    o.position = mul(uCam.viewProj, world);
    o.uv = input.uv;
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return uTex.Sample(uSmp, input.uv);
}
