// Geometry-shader pass-through: the vertex stage emits a triangle, the geometry
// stage re-emits the three vertices and assigns a flat green color, the fragment
// stage outputs it. If the GS stage runs, the center pixel is green - proving the
// legacy geometry-shader path (Vulkan + desktop OpenGL; gated off WebGPU/WebGL2).
struct VSOut
{
    float4 position : SV_Position;
};

struct GSOut
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

static const float2 kPos[3] = { float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };

[shader("vertex")]
VSOut vertexMain(uint vid : SV_VertexID)
{
    VSOut o;
    o.position = float4(kPos[vid], 0.0, 1.0);
    return o;
}

[shader("geometry")]
[maxvertexcount(3)]
void geometryMain(triangle VSOut input[3], inout TriangleStream<GSOut> stream)
{
    for (int i = 0; i < 3; ++i)
    {
        GSOut o;
        o.position = input[i].position;
        o.color = float3(0.0, 1.0, 0.0); // GS-assigned: green marks the GS ran
        stream.Append(o);
    }
}

[shader("fragment")]
float4 fragmentMain(GSOut input) : SV_Target
{
    return float4(input.color, 1.0);
}
