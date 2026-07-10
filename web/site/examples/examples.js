/* Gallery metadata for the web-capable examples. Hand-written on purpose:
   descriptions need human wording, and keeping it a plain script (not fetched
   JSON) lets the gallery work from file:// during local preview.
   When a new web-capable example lands, append one entry here. */

var VRI_EXAMPLES = [
    { name: "triangle",           title: "Hello Triangle",       desc: "The minimal swapchain-to-present path: one pipeline, one 3-vertex draw.",                                    tags: ["core"] },
    { name: "cube",               title: "Textured Cube",        desc: "A textured, depth-tested, animated cube — vertex/index buffers, constant buffer, KTX texture.",             tags: ["core", "texture"] },
    { name: "instancing",         title: "Instancing",           desc: "A field of textured cubes drawn with a single indexed instanced draw call.",                                 tags: ["instancing"] },
    { name: "pbrbasic",           title: "PBR Basics",           desc: "Metallic-roughness sphere grid lit by four point lights with a Cook-Torrance BRDF.",                         tags: ["pbr", "lighting"] },
    { name: "pushconstants",      title: "Push Constants",       desc: "A fullscreen triangle colored entirely by a per-frame push constant.",                                       tags: ["constants"] },
    { name: "normalmapping",      title: "Normal Mapping",       desc: "Tangent-space normal mapping on a lit quad, color + normal maps.",                                           tags: ["texture", "lighting"] },
    { name: "computeshader",      title: "Compute Shader",       desc: "A compute pass writes an animated plasma into a storage image (fragment fallback on WebGL).",                tags: ["compute"] },
    { name: "offscreen",          title: "Offscreen & MRT",      desc: "Render-to-texture with multiple render targets, composited side by side.",                                   tags: ["rendertarget", "mrt"] },
    { name: "descriptorindexing", title: "Descriptor Indexing",  desc: "Bindless texturing through a 16-texture descriptor array, with a texture-array fallback.",                   tags: ["bindless"] },
    { name: "shadowmapping",      title: "Shadow Mapping",       desc: "Directional-light PCF shadow mapping: depth-only pass, comparison sampler.",                                 tags: ["shadows", "depth"] },
    { name: "msaa",               title: "MSAA",                 desc: "4x multisampling with resolve — toggle it off to watch the edges go jagged.",                                tags: ["msaa"] },
    { name: "cubemap",            title: "Cubemap",              desc: "A skybox and a reflective chrome sphere sharing one procedural cubemap.",                                    tags: ["cubemap"] },
    { name: "stenciloutline",     title: "Stencil Outline",      desc: "Two-pass stencil silhouette outline on a D24S8 depth-stencil buffer.",                                       tags: ["stencil"] },
    { name: "deferred",           title: "Deferred Shading",     desc: "A three-target G-buffer lit by six moving point lights.",                                                    tags: ["deferred", "mrt"] },
    { name: "rayquery",           title: "Ray Query Shadows",    desc: "Ray-traced hard shadows: hardware ray query with a software compute fallback.",                              tags: ["raytracing"] },
    { name: "raytracing",         title: "Ray Traced glTF",      desc: "The FlightHelmet glTF model traced by a raygen/miss/closest-hit pipeline (or inline ray query).",            tags: ["raytracing", "gltf"] },
    { name: "pathtracer",         title: "Path Tracer",          desc: "A progressive path tracer with diffuse global illumination, accumulating over frames.",                      tags: ["raytracing", "gi"] },
    { name: "waveops",            title: "Wave Operations",      desc: "Subgroup/wave intrinsics visualized as bands whose stripe period is the hardware wave size (compute/fragment fallbacks on the web).", tags: ["compute", "subgroups"] }
];
