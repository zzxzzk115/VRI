// windows.h first so its APIENTRY definition wins before glad (pulled in by the
// headers below) defines its own - avoids a macro-redefinition warning.
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h> // WGL (wglGetCurrentContext/wglMakeCurrent) + GDI (GetDC/SwapBuffers/...)
#elif defined(__linux__) && !defined(__EMSCRIPTEN__)
// Declare the few GLX entry points we need ourselves (resolved from libGL via glfw's
// link) instead of including <GL/glx.h>, which pulls <GL/gl.h> and clashes with glad.
// GLXContext is an opaque pointer; Display* is opaque (void*); GLXDrawable is an XID
// (unsigned long); Bool is int.
extern "C" void*         glXGetCurrentContext(void);
extern "C" unsigned long glXGetCurrentDrawable(void);
extern "C" int           glXMakeCurrent(void* dpy, unsigned long drawable, void* ctx);
extern "C" void          glXSwapBuffers(void* dpy, unsigned long drawable);
#endif

#include "swapchain_gl.h"

#include "conversions_gl.h"
#include "device_gl.h"

namespace vri::gl
{
    namespace
    {
        DeviceGL*    Dev(VriDevice* d) { return reinterpret_cast<DeviceGL*>(d); }
        SwapChainGL* Swap(VriSwapChain* s) { return reinterpret_cast<SwapChainGL*>(s); }

        // A backbuffer is an ordinary renderable color texture; the app draws into it
        // via the normal FBO path, then Present blits it to the window.
        TextureGL* CreateBackbuffer(DeviceGL* d, uint32_t w, uint32_t h, VriFormat fmt)
        {
            const GLFormat gf = ToGLFormat(fmt);
            GLuint id = 0;
#if !defined(__EMSCRIPTEN__)
            if (d->Features().dsa)
            {
                glCreateTextures(GL_TEXTURE_2D, 1, &id);
                glTextureStorage2D(id, 1, gf.internalFormat, static_cast<GLsizei>(w), static_cast<GLsizei>(h));
            }
            else
#endif
            {
                glGenTextures(1, &id);
                glBindTexture(GL_TEXTURE_2D, id);
                glTexStorage2D(GL_TEXTURE_2D, 1, gf.internalFormat, static_cast<GLsizei>(w), static_cast<GLsizei>(h));
                glBindTexture(GL_TEXTURE_2D, 0);
            }
            TextureGL* t = new TextureGL{};
            t->device = d; t->id = id; t->target = GL_TEXTURE_2D;
            t->glFormat = gf.format; t->glType = gf.type;
            t->width = w; t->height = h; t->depth = 1; t->mipNum = 1; t->layerNum = 1; t->texelSize = gf.texelSize;
            return t;
        }

        void DestroyBackbuffers(SwapChainGL* s)
        {
            for (TextureGL* t : s->textures)
            {
                s->device->EvictFbosReferencing(t->id); // drop cached render-pass FBOs using it
                glDeleteTextures(1, &t->id);
                delete t;
            }
            s->textures.clear();
        }

        VriResult VRI_CALL CreateSwapChain(VriDevice* device, const VriSwapChainDesc* desc, VriSwapChain** out)
        {
            DeviceGL* d = Dev(device);
            const uint32_t n = desc->textureNum ? desc->textureNum : 2u;
            const uint32_t w = desc->width ? desc->width : 1u;
            const uint32_t h = desc->height ? desc->height : 1u;
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
            if (desc->window.type != VriWindowSystem_Win32)
            {
                d->ReportError("GL swapchain: only the Win32 window system is implemented on this platform");
                return VriResult_Unsupported;
            }
            HWND hwnd = static_cast<HWND>(desc->window.handle.win32.hwnd);
            if (!hwnd) return VriResult_InvalidArgument;
            HDC hdc = GetDC(hwnd);
            if (!hdc) { d->ReportError("GL swapchain: GetDC failed"); return VriResult_Failure; }
            // wglMakeCurrent requires the target window's pixel format to match the one
            // the device's context was created with. The device context is current here
            // (right after device creation), so copy its exact PFD onto the window. The
            // app's window must not already own a pixel format (create it without one).
            PIXELFORMATDESCRIPTOR pfd{};
            HDC devDc = wglGetCurrentDC();
            const int devPf = devDc ? GetPixelFormat(devDc) : 0;
            if (devPf && DescribePixelFormat(devDc, devPf, sizeof(pfd), &pfd) == 0) { /* fall through to default */ }
            if (!devPf)
            {
                pfd.nSize = sizeof(pfd); pfd.nVersion = 1;
                pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
                pfd.iPixelType = PFD_TYPE_RGBA; pfd.cColorBits = 32; pfd.cDepthBits = 24; pfd.cStencilBits = 8;
            }
            const int pf = ChoosePixelFormat(hdc, &pfd);
            if (!pf || !SetPixelFormat(hdc, pf, &pfd))
            {
                ReleaseDC(hwnd, hdc);
                d->ReportError("GL swapchain: ChoosePixelFormat/SetPixelFormat failed (window may already have a pixel format)");
                return VriResult_Failure;
            }

            SwapChainGL* s = new SwapChainGL{};
            s->device = d; s->hwnd = hwnd; s->hdc = hdc;
            s->width = w; s->height = h; s->format = desc->format; s->vsync = desc->vsync != VRI_FALSE;
            for (uint32_t i = 0; i < n; ++i) s->textures.push_back(CreateBackbuffer(d, w, h, desc->format));
            glGenFramebuffers(1, &s->blitFbo);
            *out = ToHandle(s);
            return VriResult_Success;
#elif defined(__linux__) && !defined(__EMSCRIPTEN__)
            // X11/GLX: present by retargeting the device's GLX context to the app window
            // (same idea as Win32). The window's visual must be compatible with the
            // device context's FBConfig; both default to a standard RGBA visual, but if
            // they differ glXMakeCurrent fails at Present (reported, not silent).
            if (desc->window.type != VriWindowSystem_Xlib)
            {
                d->ReportError("GL swapchain: only the Xlib window system is implemented on Linux (Wayland GLX TBD)");
                return VriResult_Unsupported;
            }
            SwapChainGL* s = new SwapChainGL{};
            s->device = d; s->xdisplay = desc->window.handle.xlib.display; s->xwindow = desc->window.handle.xlib.window;
            s->width = w; s->height = h; s->format = desc->format; s->vsync = desc->vsync != VRI_FALSE;
            for (uint32_t i = 0; i < n; ++i) s->textures.push_back(CreateBackbuffer(d, w, h, desc->format));
            glGenFramebuffers(1, &s->blitFbo);
            *out = ToHandle(s);
            return VriResult_Success;
#elif defined(__EMSCRIPTEN__)
            // Web: the device's WebGL2 context already owns the canvas (its default
            // framebuffer). Backbuffers are offscreen textures; Present blits the acquired
            // one onto FBO 0 (the canvas). No platform window/context glue is needed.
            if (desc->window.type != VriWindowSystem_Web && desc->window.type != VriWindowSystem_None)
                d->ReportError("GL swapchain (web): expected VriWindowSystem_Web (canvas); presenting to the default canvas");
            SwapChainGL* s = new SwapChainGL{};
            s->device = d; s->width = w; s->height = h; s->format = desc->format; s->vsync = desc->vsync != VRI_FALSE;
            for (uint32_t i = 0; i < n; ++i) s->textures.push_back(CreateBackbuffer(d, w, h, desc->format));
            glGenFramebuffers(1, &s->blitFbo);
            *out = ToHandle(s);
            return VriResult_Success;
#else
            (void)d; (void)desc; (void)out; (void)n; (void)w; (void)h;
            Dev(device)->ReportError("GL swapchain: windowed present is only implemented on Win32 / Web so far");
            return VriResult_Unsupported;
#endif
        }

        void VRI_CALL DestroySwapChain(VriSwapChain* swapChain)
        {
            if (!swapChain) return;
            SwapChainGL* s = Swap(swapChain);
            DestroyBackbuffers(s);
            if (s->blitFbo) glDeleteFramebuffers(1, &s->blitFbo);
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
            if (s->hdc && s->hwnd) ReleaseDC(static_cast<HWND>(s->hwnd), static_cast<HDC>(s->hdc));
#endif
            delete s;
        }

        VriResult VRI_CALL GetSwapChainTextures(VriSwapChain* swapChain, VriTexture** outTextures, uint32_t* ioCount)
        {
            SwapChainGL* s = Swap(swapChain);
            const uint32_t n = static_cast<uint32_t>(s->textures.size());
            if (!outTextures) { if (ioCount) *ioCount = n; return VriResult_Success; }
            const uint32_t cap = ioCount ? *ioCount : 0u;
            const uint32_t c = cap < n ? cap : n;
            for (uint32_t i = 0; i < c; ++i) outTextures[i] = ToHandle(s->textures[i]);
            if (ioCount) *ioCount = c;
            return VriResult_Success;
        }

        VriResult VRI_CALL AcquireNextTexture(VriSwapChain* swapChain, VriFence* acquireFence, uint64_t signalValue, uint32_t* outIndex)
        {
            SwapChainGL* s = Swap(swapChain);
            const uint32_t n = static_cast<uint32_t>(s->textures.size());
            if (!n) return VriResult_Failure;
            s->currentIndex = (s->currentIndex + 1u) % n;
            if (acquireFence) // GL is ordered on one context; the image is immediately ready
                reinterpret_cast<FenceGL*>(acquireFence)->value = signalValue;
            if (outIndex) *outIndex = s->currentIndex;
            return VriResult_Success;
        }

        VriResult VRI_CALL Present(VriSwapChain* swapChain, VriFence* /*waitFence*/, uint64_t /*waitValue*/)
        {
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
            SwapChainGL* s = Swap(swapChain);
            DeviceGL* d = s->device;
            // The device's GL context is current here (rendering just used it). Grab it +
            // its drawable so we can retarget it to the window for the blit, then restore.
            HGLRC rc = wglGetCurrentContext();
            HDC devDc = wglGetCurrentDC();
            HDC hdc = static_cast<HDC>(s->hdc);
            if (!rc)
            {
                d->ReportError("GL swapchain: no current GL context at Present (render before presenting)");
                return VriResult_Failure;
            }
            if (!wglMakeCurrent(hdc, rc))
            {
                d->ReportError("GL swapchain: wglMakeCurrent(window) failed (pixel-format mismatch?)");
                return VriResult_Failure;
            }
            // Blit the acquired backbuffer onto the window's default framebuffer. The
            // backbuffer is VRI top-left origin; FBO 0 is GL bottom-left, so flip Y.
            const TextureGL* bb = s->textures[s->currentIndex];
            glBindFramebuffer(GL_READ_FRAMEBUFFER, s->blitFbo);
            glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bb->id, 0);
            glReadBuffer(GL_COLOR_ATTACHMENT0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glDrawBuffer(GL_BACK);
            const GLint w = static_cast<GLint>(s->width), h = static_cast<GLint>(s->height);
            glBlitFramebuffer(0, 0, w, h, 0, h, w, 0, GL_COLOR_BUFFER_BIT, GL_LINEAR);
            SwapBuffers(hdc); // vsync follows the window's swap interval (WGL_EXT_swap_control not wired)
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            wglMakeCurrent(devDc, rc); // restore the device's drawable so later rendering is unaffected
            return VriResult_Success;
#elif defined(__linux__) && !defined(__EMSCRIPTEN__)
            SwapChainGL* s = Swap(swapChain);
            DeviceGL* d = s->device;
            void* rc = glXGetCurrentContext();
            const unsigned long devDrawable = glXGetCurrentDrawable();
            if (!rc) { d->ReportError("GL swapchain: no current GLX context at Present"); return VriResult_Failure; }
            if (!glXMakeCurrent(s->xdisplay, s->xwindow, rc))
            {
                d->ReportError("GL swapchain: glXMakeCurrent(window) failed (visual/FBConfig mismatch?)");
                return VriResult_Failure;
            }
            const TextureGL* bb = s->textures[s->currentIndex];
            glBindFramebuffer(GL_READ_FRAMEBUFFER, s->blitFbo);
            glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bb->id, 0);
            glReadBuffer(GL_COLOR_ATTACHMENT0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glDrawBuffer(GL_BACK);
            const GLint w = static_cast<GLint>(s->width), h = static_cast<GLint>(s->height);
            glBlitFramebuffer(0, 0, w, h, 0, h, w, 0, GL_COLOR_BUFFER_BIT, GL_LINEAR); // flip Y to upright
            glXSwapBuffers(s->xdisplay, s->xwindow);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            glXMakeCurrent(s->xdisplay, devDrawable, rc); // restore the device's drawable
            return VriResult_Success;
#elif defined(__EMSCRIPTEN__)
            // Web: blit the acquired backbuffer onto the canvas (default framebuffer 0).
            // The browser presents the canvas when control returns to the event loop.
            SwapChainGL* s = Swap(swapChain);
            const TextureGL* bb = s->textures[s->currentIndex];
            glBindFramebuffer(GL_READ_FRAMEBUFFER, s->blitFbo);
            glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, bb->id, 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // the canvas
            const GLint w = static_cast<GLint>(s->width), h = static_cast<GLint>(s->height);
            glBlitFramebuffer(0, 0, w, h, 0, h, w, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST); // flip Y to upright
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            return VriResult_Success;
#else
            (void)swapChain;
            return VriResult_Unsupported;
#endif
        }

        VriResult VRI_CALL Resize(VriSwapChain* swapChain, uint32_t width, uint32_t height)
        {
            SwapChainGL* s = Swap(swapChain);
            const uint32_t n = static_cast<uint32_t>(s->textures.size()); // preserve the backbuffer count
            DestroyBackbuffers(s);
            s->width = width ? width : 1u;
            s->height = height ? height : 1u;
            for (uint32_t i = 0; i < n; ++i) s->textures.push_back(CreateBackbuffer(s->device, s->width, s->height, s->format));
            s->currentIndex = 0;
            return VriResult_Success;
        }
    } // namespace

    const VriSwapChainInterface* GetSwapChainInterfaceGL()
    {
        static const VriSwapChainInterface table = {
            CreateSwapChain, DestroySwapChain, GetSwapChainTextures, AcquireNextTexture, Present, Resize,
        };
        return &table;
    }
} // namespace vri::gl
