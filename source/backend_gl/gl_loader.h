// gl_loader.h - single include point for GL entry points across profiles.
//
// On Emscripten the GL backend IS the WebGL backend: WebGL2 == OpenGL ES 3.0, and
// the symbols/headers are provided directly by the Emscripten sysroot (no loader).
// On native platforms we use glad (desktop GL 4.6 core); the command path only
// touches the GLES3/WebGL2 subset, so the same code compiles for both.
#pragma once

#if defined(__EMSCRIPTEN__)
#    include <GLES3/gl3.h>
#    include <GLES2/gl2ext.h>
// GLES3/WebGL2 has no core BGRA client format. Map to the EXT token when present
// (EXT_texture_format_BGRA8888), else fall back to RGBA so code compiles; BGRA on
// the web requires that extension at runtime and is otherwise best avoided.
#    if !defined(GL_BGRA)
#        if defined(GL_BGRA_EXT)
#            define GL_BGRA GL_BGRA_EXT
#        else
#            define GL_BGRA GL_RGBA
#        endif
#    endif
// WebGL2's getBufferSubData is implemented by the Emscripten GL runtime but not
// declared in <GLES3/gl3.h>; declare it so the buffer-readback path can use it.
extern "C" void glGetBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, void* data);
#else
#    include <glad/glad.h>
#endif
