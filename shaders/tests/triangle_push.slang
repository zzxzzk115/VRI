// Triangle transformed by a float4x4 in a PUSH CONSTANT (not a UBO). Exercises the
// matrix-in-push-constant path: Slang lowers a matrix inside a constant block to a
// _MatrixStorage wrapper struct, so a backend that emulates push constants by setting
// each member by name (OpenGL) must descend into that wrapper. If it doesn't, `mvp`
// stays zero, every vertex collapses to the origin, and nothing is drawn.
struct Push
{
    // column_major forces Slang to emit the _MatrixStorage_..._ColMajor wrapper (matches
    // vshadersystem's column-major default, i.e. the model-viewer). SPIRV-Cross then presents
    // the member as a struct rather than a native mat4, which is the case OpenGL's push-constant
    // emulation must flatten (row-major collapses to a native mat4 and works without it).
    column_major float4x4 mvp;
};

[[vk::push_constant]]
Push pc;

struct VSOutput
{
    float4 position : SV_Position;
};

static const float2 kPositions[3] = { float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5) };

[shader("vertex")]
VSOutput vertexMain(uint vid : SV_VertexID)
{
    VSOutput o;
    o.position = mul(pc.mvp, float4(kPositions[vid], 0.0, 1.0));
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return float4(0.0, 1.0, 0.0, 1.0); // green
}
