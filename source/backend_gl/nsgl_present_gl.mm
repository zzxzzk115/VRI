// nsgl_present_gl.mm - macOS NSOpenGL presentation glue for the GL swapchain.
//
// GL has no swapchain object: the device renders into an offscreen FBO, and Present
// retargets the device's single GL context to the target window and blits. On Win32
// that's wglMakeCurrent(windowDC); on macOS the equivalent is pointing the device's
// NSOpenGLContext at the window's content view (-setView:) and -flushBuffer to swap.
//
// vsync: NSOpenGL's swap interval is NOT honored when the context is retargeted onto a
// foreign (SDL-owned) window's view - -flushBuffer free-runs even with the interval set.
// So we gate the present on a CVDisplayLink instead: it fires once per vblank on the
// window's display, and the present blocks on a semaphore it signals. This caps the GL
// backend to the real refresh rate when vsync is requested. Kept in Objective-C++ so
// swapchain_gl.cpp stays plain C++.

#import <Cocoa/Cocoa.h>
#import <CoreVideo/CoreVideo.h>

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#define GLFW_EXPOSE_NATIVE_NSGL
#include <GLFW/glfw3native.h>

// NSOpenGL is deprecated since macOS 10.14 but remains the only way to drive a GL
// context against an AppKit window; the GL backend is intentionally legacy-path here.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace
{
    NSView* ContentView(id obj)
    {
        if ([obj isKindOfClass:[NSWindow class]]) return [(NSWindow*)obj contentView];
        if ([obj isKindOfClass:[NSView class]])   return (NSView*)obj;
        return nil;
    }

    // One CVDisplayLink driving a counting semaphore: each vblank bumps it, and a vsync'd
    // present consumes one tick (blocking if it's ahead of the display). Lazily created.
    CVDisplayLinkRef     g_link = nullptr;
    dispatch_semaphore_t g_vblank = nullptr;

    CVReturn VblankCallback(CVDisplayLinkRef, const CVTimeStamp*, const CVTimeStamp*, CVOptionFlags, CVOptionFlags*, void*)
    {
        dispatch_semaphore_signal(g_vblank);
        return kCVReturnSuccess;
    }

    void EnsureDisplayLink()
    {
        if (g_link) return;
        g_vblank = dispatch_semaphore_create(0);
        if (CVDisplayLinkCreateWithActiveCGDisplays(&g_link) != kCVReturnSuccess || !g_link) return;
        CVDisplayLinkSetOutputCallback(g_link, &VblankCallback, nullptr);
        CVDisplayLinkStart(g_link);
    }
}

// Point the device's NSOpenGL context at the target window's view and make it current,
// so a subsequent blit to FBO 0 lands on the window. The view is left attached (the
// device renders into FBOs, so the default-framebuffer drawable is only used at present).
// Returns the view's backing-pixel size (Retina-aware) for scaling the present blit.
extern "C" void vriNSGLBeginPresent(GLFWwindow* devWindow, void* targetNSWindow, int* outW, int* outH)
{
    if (outW) *outW = 0;
    if (outH) *outH = 0;
    if (!devWindow || !targetNSWindow) return;

    NSOpenGLContext* ctx = glfwGetNSGLContext(devWindow);
    NSView* view = ContentView((__bridge id)targetNSWindow);
    if (!ctx || !view) return;

    view.wantsBestResolutionOpenGLSurface = YES; // full-res Retina drawable
    if (ctx.view != view) { [ctx setView:view]; [ctx update]; }
    [ctx makeCurrentContext];

    const NSSize px = [view convertSizeToBacking:view.bounds.size];
    if (outW) *outW = (int)px.width;
    if (outH) *outH = (int)px.height;
}

// Swap the target view's buffers, then (when vsync) block until the next vblank via the
// CVDisplayLink - NSOpenGL's own swap interval isn't honored on a retargeted foreign view.
extern "C" void vriNSGLEndPresent(GLFWwindow* devWindow, int vsync)
{
    if (!devWindow) return;
    NSOpenGLContext* ctx = glfwGetNSGLContext(devWindow);
    if (!ctx) return;
    [ctx flushBuffer];

    if (vsync)
    {
        EnsureDisplayLink();
        if (g_vblank)
        {
            // Drain any backlog so we land on the *next* vblank, not a stale one, then wait.
            while (dispatch_semaphore_wait(g_vblank, DISPATCH_TIME_NOW) == 0) {}
            dispatch_semaphore_wait(g_vblank, dispatch_time(DISPATCH_TIME_NOW, 100 * NSEC_PER_MSEC));
        }
    }
}
