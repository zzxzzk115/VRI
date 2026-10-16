# Showcase capture

Captured on 2026-10-16 in a newly created independent workspace. Source checkout: `5be370859a2f65a88fb6d97b7ac8d972e8f361e3`.

Build and run the actual native example from the repository root:

```sh
xmake f -y --vri_backend_vulkan=y --vri_backend_wgpu=n --vri_backend_gl=n --vri_build_tests=n
xmake build -y -j4 example-pathtracer
VRI_MAX_FRAMES=132 VRI_CAPTURE=/tmp/vri-pathtracer.bmp xmake run example-pathtracer
```

The built-in readback produces a 960×720 BMP. The capture shows the default 128 accumulated samples, four bounces, camera orbit 1.4, sky intensity 1.2 and light azimuth 0.7. Vulkan inline ray query was available on the capture machine. This image demonstrates the example, not a performance benchmark; the panel's FPS is not a measured cross-device result.

Model: the repository's vendored `assets/models/FlightHelmet/glTF/FlightHelmet.gltf` (FlightHelmet, CC0; see its included documentation). Only the screenshot is added here; model files are unchanged. The image is converted to WebP without added framing or retouching.
