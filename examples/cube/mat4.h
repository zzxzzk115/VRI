// mat4.h - a tiny column-major 4x4 matrix helper for examples/cube (glm conventions:
// column-major storage, right-handed view, zero-to-one depth, matching VRI's Y-up clip).
// m[col*4 + row] == M[row][col]. Just enough for an MVP; not a general math library.
#pragma once

#include <cmath>

struct Mat4
{
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; // identity
};

// Transpose. Needed when uploading a matrix consumed by `mul(cbMatrix, vec)` in Slang:
// that lowers to OpVectorTimesMatrix on a column-major-packed matrix, i.e. it computes
// Mᵀ·v for the stored data — so uploading the transpose makes it compute M·v. (Verified
// identical on Vulkan/D3D12/OpenGL via test_cube_xbackend.)
inline Mat4 Transpose(const Mat4& a)
{
    Mat4 t {};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            t.m[c * 4 + r] = a.m[r * 4 + c];
    return t;
}

// C = A * B (both column-major)
inline Mat4 Mul(const Mat4& a, const Mat4& b)
{
    Mat4 c {};
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
        {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k)
                s += a.m[k * 4 + i] * b.m[j * 4 + k];
            c.m[j * 4 + i] = s;
        }
    return c;
}

// Right-handed perspective with depth in [0,1] (glm perspectiveRH_ZO).
inline Mat4 Perspective(float fovYRadians, float aspect, float zNear, float zFar)
{
    const float t = std::tan(fovYRadians * 0.5f);
    Mat4        r {};
    for (float& v : r.m)
        v = 0.0f;
    r.m[0]  = 1.0f / (aspect * t);
    r.m[5]  = 1.0f / t;
    r.m[10] = zFar / (zNear - zFar);
    r.m[11] = -1.0f;
    r.m[14] = -(zFar * zNear) / (zFar - zNear);
    return r;
}

// Right-handed orthographic with depth in [0,1] (glm orthoRH_ZO). For directional shadow maps.
inline Mat4 Ortho(float left, float right, float bottom, float top, float zNear, float zFar)
{
    Mat4 r {};
    for (float& v : r.m)
        v = 0.0f;
    r.m[0]  = 2.0f / (right - left);
    r.m[5]  = 2.0f / (top - bottom);
    r.m[10] = -1.0f / (zFar - zNear);
    r.m[12] = -(right + left) / (right - left);
    r.m[13] = -(top + bottom) / (top - bottom);
    r.m[14] = -zNear / (zFar - zNear);
    r.m[15] = 1.0f;
    return r;
}

// Right-handed look-at (glm lookAtRH).
inline Mat4 LookAt(const float eye[3], const float center[3], const float up[3])
{
    auto sub = [](const float a[3], const float b[3], float o[3]) {
        o[0] = a[0] - b[0];
        o[1] = a[1] - b[1];
        o[2] = a[2] - b[2];
    };
    auto norm = [](float v[3]) {
        float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        v[0] /= l;
        v[1] /= l;
        v[2] /= l;
    };
    auto cross = [](const float a[3], const float b[3], float o[3]) {
        o[0] = a[1] * b[2] - a[2] * b[1];
        o[1] = a[2] * b[0] - a[0] * b[2];
        o[2] = a[0] * b[1] - a[1] * b[0];
    };
    auto  dot = [](const float a[3], const float b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    float f[3];
    sub(center, eye, f);
    norm(f);
    float s[3];
    cross(f, up, s);
    norm(s);
    float u[3];
    cross(s, f, u);
    Mat4 r {};
    for (float& v : r.m)
        v = 0.0f;
    r.m[0]  = s[0];
    r.m[4]  = s[1];
    r.m[8]  = s[2];
    r.m[1]  = u[0];
    r.m[5]  = u[1];
    r.m[9]  = u[2];
    r.m[2]  = -f[0];
    r.m[6]  = -f[1];
    r.m[10] = -f[2];
    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] = dot(f, eye);
    r.m[15] = 1.0f;
    return r;
}

// Rotation about the Y axis.
inline Mat4 RotateY(float radians)
{
    const float c = std::cos(radians), s = std::sin(radians);
    Mat4        r {};
    r.m[0]  = c;
    r.m[2]  = -s;
    r.m[8]  = s;
    r.m[10] = c;
    return r;
}

// Rotation about the X axis.
inline Mat4 RotateX(float radians)
{
    const float c = std::cos(radians), s = std::sin(radians);
    Mat4        r {};
    r.m[5]  = c;
    r.m[6]  = s;
    r.m[9]  = -s;
    r.m[10] = c;
    return r;
}

// Translation.
inline Mat4 Translate(float x, float y, float z)
{
    Mat4 r {};
    r.m[12] = x;
    r.m[13] = y;
    r.m[14] = z;
    return r;
}

// Uniform scale.
inline Mat4 Scale(float s)
{
    Mat4 r {};
    r.m[0]  = s;
    r.m[5]  = s;
    r.m[10] = s;
    return r;
}
