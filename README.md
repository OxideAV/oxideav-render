# oxideav-render

[![CI](https://github.com/OxideAV/oxideav-render/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-render/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-render.svg)](https://crates.io/crates/oxideav-render) [![docs.rs](https://docs.rs/oxideav-render/badge.svg)](https://docs.rs/oxideav-render) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust 3D-scene → raster image renderer for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.
Consumes an [`oxideav_mesh3d::Scene3D`] and produces a packed RGBA8
[`RgbaImage`].

## Status

Three live CPU backends behind one trait, on a shared scene-preparation
layer:

| Backend     | Status                                                       |
| ----------- | ----------------------------------------------------------- |
| `Scanline`  | done — clipped, perspective-correct, tile-parallel rasteriser; glTF 2.0 metallic-roughness `Pbr` mode (textures, normal / occlusion / emissive maps, vertex colours, unlit, punctual lights, OPAQUE / MASK / BLEND, back-face culling, shadow maps with PCF) plus the legacy Flat / Gouraud / Phong / Wireframe / NormalDebug / DepthDebug modes |
| `Raycast`   | done — Whitted recursive ray tracer on the prep + trace layers: the scanline glTF `Pbr` model per hit (pixel-identical to scanline without secondary rays) plus ray-traced hard shadows for every light type, Fresnel-weighted mirror reflection, refraction (`KHR_materials_transmission` / `_volume` / `_ior`), MASK any-hit and BLEND see-through rays; legacy modes unchanged; HDR output; tile-parallel |
| `PathTrace` | done — unbiased Monte Carlo path tracer: NEE to punctual + emissive-triangle lights with MIS, glTF metallic-roughness BSDF with GGX VNDF sampling plus transmission / volume / ior / specular / clearcoat / sheen, MASK / stochastic BLEND, uniform sky or HDR environment map, Owen-scrambled Sobol' sampling, progressive `PathTracer` |

### Shared layers (public, backend-agnostic)

Every backend — including the out-of-tree GPU backend — consumes the
same building blocks, so they frame, light and shade a scene
identically:

* **`prepare`** — `PreparedScene::build(&scene, &PrepareOptions,
  &mut TextureCache)` turns a `Scene3D` (+ animation time) into a flat
  world-space render list: per `DrawItem` de-indexed positions /
  normals / tangents / UV sets / `COLOR_0` (`Vec<[f32; N]>`, ready to
  upload), triangles / lines / points, a material index; animation
  sampled at `RenderOptions::time`, morph weights (animation > node >
  mesh) and CPU skinning applied, CCW-front winding normalised,
  missing normals / tangents synthesised. Plus the resolved material
  table (`PreparedMaterial`), texture table, world-space
  `KHR_lights_punctual` lights (`PreparedLight::sample` implements the
  range / cone attenuation; the options' `LightSpec` is the fallback)
  and scene camera instances.
* **`camera`** — `Camera::resolve(&prepared, &opts, w, h)`: a scene
  camera (`RenderOptions::scene_camera`) or the historical auto-frame
  / orbit camera; view / projection matrices in either depth
  convention (`DepthRange::{NegOneToOne, ZeroToOne}`), eye / basis,
  near / far, primary-ray generation.
* **`texture`** — `TextureResolver` (bytes + MIME → texels), the
  memoising `TextureCache`, sRGB / linear box-filtered mip chains and
  a glTF-sampler-exact CPU sampler (wrap modes, nearest / bilinear,
  mip nearest / trilinear, LOD from UV derivatives). Under the
  `registry` feature `RegistryTextureResolver` decodes through an
  `oxideav_core::RuntimeContext` (whatever image codecs the caller
  registered) + `oxideav-pixfmt`; the crate itself links no codec.
* **`brdf`** — glTF Appendix B: GGX, height-correlated Smith,
  Schlick, Lambert.
* **`hdr`** — `HdrImage` (scene-linear `f32` RGBA, via
  `Renderer::render_hdr`, for EXR-style output), `ToneMap::{Clamp,
  Reinhard, AcesFitted}` + `exposure`, sRGB transfer.
* **`trace`** — the ray-tracing layer shared by `Raycast` and
  `PathTrace`: `TraceScene` (one world-space SAH / object-median BVH
  over the prepared triangles, watertight intersection, hits mapped
  back to draw items, filtered closest / any-hit queries), hit →
  surface and glTF material evaluation (`material_oriented` for
  double-sided back faces), texture LOD from ray differentials
  (`TexLod::Grad`) or ray cones (`TexLod::Cone`), shadow
  transmittance (MASK / BLEND / transmission), BRDF direct lighting,
  robust origin offsetting, Snell / Schlick helpers.
* **`testscenes`** — procedural reference scenes (Cornell-style box,
  metallic × roughness sphere grid, checker floor, textured quad,
  alpha MASK / BLEND planes, shadow box, skinned + morph-animated
  beam, normal-mapped quad) and image metrics (MAE, PSNR, region
  mean / stddev) for cross-backend tests.

### Path tracer

`RenderBackend::PathTrace` (registry name `"pathtrace"`) renders
`RenderOptions::path_trace` samples per pixel:

```rust
use oxideav_render::{make_renderer, PathTraceOptions, RenderBackend, RenderOptions};

let opts = RenderOptions {
    scene_camera: Some(0),
    path_trace: PathTraceOptions {
        samples_per_pixel: 256, // default 64
        max_bounces: 8,         // 1 = direct lighting only
        clamp: 0.0,             // > 0 = biased firefly clamp
        ..PathTraceOptions::default()
    },
    ..RenderOptions::default()
};
let img = make_renderer(RenderBackend::PathTrace)?.render(&scene, &opts)?;
```

Interactive callers drive the progressive accumulator instead:
`PathTracer::sync(&scene, &opts)` (re-prepares / resets only when the
scene was invalidated or a radiance-relevant option changed — exposure,
tone map, background and the sample target never reset),
`refine(n)` (adds `n` samples per pixel; `refine(a); refine(b)` is
bit-identical to `refine(a + b)`), `image()` / `hdr()`,
`invalidate_scene()`, and `set_environment(Some(Arc<EnvironmentMap>))`
for an importance-sampled equirectangular `HdrImage` sky (otherwise
`RenderOptions::ambient` is a constant sky radiance). Renders are
deterministic per `PathTraceOptions::seed`. Uncovered pixels keep the
background bytes exactly; `aa` and `shading` are ignored.

The `pathtrace` module docs are the normative estimator specification
(sequence dimensions, lobe-selection probabilities, light-selection
pdf, MIS weights, roulette rule) for ports such as the GPU backend.

## Usage

```rust,no_run
use oxideav_render::{make_renderer, RenderBackend, RenderOptions, Result, ShadingMode, ToneMap};
use oxideav_mesh3d::Scene3D;

fn render_one(scene: &Scene3D) -> Result<()> {
    let mut renderer = make_renderer(RenderBackend::Scanline)?;
    let opts = RenderOptions {
        width: 1024,
        height: 768,
        shading: ShadingMode::Pbr,
        tone_map: ToneMap::AcesFitted,
        shadows: true,
        scene_camera: Some(0), // first camera in the scene, if any
        time: Some(0.5),       // sample animation 0 at t = 0.5 s
        ..Default::default()
    };
    let _image = renderer.render(scene, &opts)?;
    let _linear = renderer.render_hdr(scene, &opts)?; // f32 radiance
    Ok(())
}
```

`RenderOptions` carries the framebuffer size, `ShadingMode`,
`Projection`, `BackgroundColor`, a fallback directional `LightSpec`, an
orbit `CameraSpec`, SSAA (`aa ∈ 1..=8`), and the newer `tone_map`,
`exposure`, `time` / `animation`, `scene_camera`, `use_scene_lights`,
`ambient`, `shadows` / `shadow_map_size`, `material_variant`,
`max_ray_depth` and `reflection_roughness_cutoff` fields. The default is 512×512, Phong shading, perspective
projection, clamp tone map — i.e. the historical output. Always build
it with `..RenderOptions::default()` so new options default.

### Raycast specifics

`ShadingMode::Pbr` on `Raycast` evaluates exactly the scanline formulas
at each hit, so with `max_ray_depth: 0` and `shadows: false` the two
backends agree to rounding on every shared test scene. On top:

* `shadows: true` traces a hard shadow ray per light (directional,
  point and spot) instead of shadow maps; MASK cut-outs pass light,
  BLEND surfaces pass `1 − α`, transmissive surfaces tint.
* Reflection: one Fresnel-weighted (`F(n·v)` of the material `F0`)
  mirror ray; it fades in below `reflection_roughness_cutoff`
  (default 0.5) as `(1 − roughness / cutoff)²`, the uniform
  `ambient · ao · F0` environment term covering the rest.
* Refraction: `KHR_materials_transmission` replaces the transmitted
  share of the diffuse lobe by a refracted ray — straight through for
  thin walls, Snell + total internal reflection + Beer–Lambert
  absorption for `KHR_materials_volume` bodies.
* BLEND surfaces continue the ray and composite *over* in linear space.
* Escaping secondary rays see a uniform environment of radiance
  `ambient`; camera rays see the background.
* `max_ray_depth` (default 4) bounds the bounces.
* Texture LOD: ray differentials on camera rays (identical filtering to
  scanline), ray cones after a bounce.

Textures decode through the renderer's `TextureCache`; construct
`ScanlineRenderer::with_texture_resolver(Arc::new(
RegistryTextureResolver::new(ctx)))` (or `RaycastRenderer::…`, or
`Renderer::set_texture_resolver` on any backend) to decode PNG / JPEG
/ … through the framework registry.

`RenderOptions::validate() -> Result<()>` runs a typed pre-flight
check of every field and surfaces the first offending one via
`Error::InvalidOptions(String)`. `Renderer::render` does not call it
automatically — backends clamp silently — so a caller wanting strict
failure on a malformed job opts in before calling `render`.

## Output

A renderer emits an `RgbaImage` (RGBA8, packed). Typed accessors keep
downstream consumers free of stride-aware byte arithmetic:
`pixel(x, y)` / `set_pixel(x, y, rgba)`, `pixel_count()`,
`is_empty()`, `pixels_rgba()` (row-major `[u8; 4]` iterator), and
`rows()` (per-row `&[u8]` slice iterator). Downstream encoders
(`oxideav-png`, `oxideav-mjpeg`, `oxideav-openexr`) consume the
surface directly; `oxideav-cli-convert` handles encoder dispatch, so
this crate pulls in no image-encoder deps.

## Standalone build

Drop the `registry` feature to build without `oxideav-core`:

```toml
oxideav-render = { version = "0.0", default-features = false }
```

The standalone build exposes `Renderer` / `RenderOptions` /
`RgbaImage` / `make_renderer` without the framework dependency tree.
The 3D input type stays `oxideav_mesh3d::Scene3D`.

## Benchmarks

`benches/render.rs` (criterion) tracks the backends on procedural
scenes (including the PBR / shadow-map paths and the path tracer);
baseline numbers + analysis live in [`BENCHMARKS.md`](BENCHMARKS.md).
The CPU backends are parallel over std scoped threads (bands / tiles)
with deterministic output. The path tracer renders the 256² Cornell
box at 64 spp in ~0.28 s on a 64-thread machine.

`cargo run -p oxideav-render --release --example dump_testscenes --
out/ [scanline|raycast|pathtrace] [spp]` renders every reference scene
to PNG.

## Clean-room policy

Render math is sourced from published papers and specifications —
glTF 2.0 + Appendix B, `KHR_lights_punctual`, Pineda 1988,
Sutherland–Hodgman 1974, Liang–Barsky 1984, Heckbert–Moreton 1991,
Williams 1978 / 1983, Reeves et al. 1987, Whitted 1980, Igehy 1999,
Akenine-Möller et al. 2019, Woop et al. 2013, Wald 2007, Wächter &
Binder 2019, Walter et al. 2007, Heitz
2014, Schlick 1994, Burley 2012, Reinhard et al. 2002, Narkowicz's
ACES fit, Porter–Duff 1984, IEC 61966-2-1; for the path tracer Kajiya
1986, Veach 1997, Heitz 2018 (VNDF), Arvo–Kirk 1990, Sobol' 1967 /
Joe–Kuo 2008, Burley 2020, O'Neill 2014, Jarzynski–Olano 2020,
Shirley–Chiu 1997, Duff et al. 2017, Turk 1990, Estevez–Kulla 2017,
Woop et al. 2013, Wächter–Binder 2019, Akenine-Möller et al. 2019, the
KHR material extension specs and the *Physically Based Rendering*
book text. Reference renderer source code is not consulted. glTF KHR extensions provide the
material vocabulary.

## License

MIT — see `LICENSE`.

[`oxideav_mesh3d::Scene3D`]: https://docs.rs/oxideav-mesh3d
[`oxideav_mesh3d::Bvh`]: https://docs.rs/oxideav-mesh3d
[`RgbaImage`]: https://docs.rs/oxideav-render
