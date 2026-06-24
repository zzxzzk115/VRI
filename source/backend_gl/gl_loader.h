// gl_loader.h - single include point for GL entry points across profiles.
//
// Three profiles share one backend source:
//   * Emscripten  -> WebGL2 == OpenGL ES 3.0; symbols/headers from the Emscripten sysroot.
//   * Native GLES -> OpenGL ES on an EGL context (embedded / Raspberry Pi); system
//                    <GLES3/gl3.h> headers, symbols from libGLESv2 (no loader needed).
//   * Desktop     -> glad (GL 4.6 core). The command path only touches the GLES3/WebGL2
//                    subset, so the same code compiles for all three.
//
// VRI_GL_ES_HEADERS is the umbrella switch for "GLES3/WebGL2 subset headers": the
// desktop-only DSA / 4.x entry points are undeclared, so the backend takes the same
// non-DSA path on Emscripten and native GLES alike. Genuinely web-only concerns
// (canvas sizing, the Emscripten GL runtime's getBufferSubData) stay keyed on
// __EMSCRIPTEN__; genuinely native-GLES concerns (EGL context/present) on
// VRI_GL_NATIVE_ES.
#pragma once

#if defined(__EMSCRIPTEN__) || defined(VRI_GL_NATIVE_ES)
#    define VRI_GL_ES_HEADERS 1
#endif

#if defined(__EMSCRIPTEN__)
#    include <GLES3/gl3.h>
#    include <GLES2/gl2ext.h>
// WebGL2's getBufferSubData is implemented by the Emscripten GL runtime but not
// declared in <GLES3/gl3.h>; declare it so the buffer-readback path can use it.
// (Native GLES has no glGetBufferSubData at all - see ReadBufferSubData in core_gl.cpp.)
extern "C" void glGetBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, void* data);
#elif defined(VRI_GL_NATIVE_ES)
// Native OpenGL ES: the system GLES headers + ext tokens. libGLESv2 exports the core
// entry points directly, so no runtime loader is needed. <GLES3/gl31.h> (ES 3.1) is a
// superset of ES 3.0 and additionally declares the indirect-draw / compute entry points
// the embedded path can use beyond the WebGL2 (ES 3.0) subset.
#    include <GLES3/gl31.h>
#    include <GLES2/gl2ext.h>
#else
#    include <glad/glad.h>
#endif

#if defined(VRI_GL_ES_HEADERS)
// GLES3/WebGL2 has no core BGRA client format. Map to the EXT token when present
// (EXT_texture_format_BGRA8888), else fall back to RGBA so code compiles; BGRA on
// GLES requires that extension at runtime and is otherwise best avoided.
#    if !defined(GL_BGRA)
#        if defined(GL_BGRA_EXT)
#            define GL_BGRA GL_BGRA_EXT
#        else
#            define GL_BGRA GL_RGBA
#        endif
#    endif
// SSBOs are ES 3.1 (absent in WebGL2 / ES 3.0); define the token so code compiles.
// Binding to it on WebGL would be a runtime no-op/error, but the triangle/UBO path never does.
#    if !defined(GL_SHADER_STORAGE_BUFFER)
#        define GL_SHADER_STORAGE_BUFFER 0x90D2
#    endif
#endif
