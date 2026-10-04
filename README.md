# oxideav-render

[![CI](https://github.com/OxideAV/oxideav-render/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-render/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-render.svg)](https://crates.io/crates/oxideav-render) [![docs.rs](https://docs.rs/oxideav-render/badge.svg)](https://docs.rs/oxideav-render) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust 3D-scene → raster image renderer for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.
Consumes an [`oxideav_mesh3d::Scene3D`] and produces a packed RGBA8
[`RgbaImage`].

## Status

Two live CPU backends behind one trait, on a shared scene-preparation
layer:

| Backend     | Status                                                       |
| ----------- | ----------------------------------------------------------- |
| `Scanline`  | done — clipped, perspective-correct, tile-parallel rasteriser; glTF 2.0 metallic-roughness `Pbr` mode (textures, normal / occlusion / emissive maps, vertex colours, unlit, punctual lights, OPAQUE / MASK / BLEND, back-face culling, shadow maps with PCF) plus the legacy Flat / Gouraud / Phong / Wireframe / NormalDebug / DepthDebug modes |
| `Raycast`   | done — Whitted recursive ray tracer: six legacy shading modes + raytraced hard shadows, recursive reflection / refraction in `Phong` mode (renders `Pbr` as `Phong` for now); BVH-accelerated |
| `PathTrace` | not yet — path tracing + physically-based BRDF              |

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
* **`testscenes`** — procedural reference scenes (Cornell-style box,
  metallic × roughness sphere grid, checker floor, textured quad,
  alpha MASK / BLEND planes, shadow box, skinned + morph-animated
  beam, normal-mapped quad) and image metrics (MAE, PSNR, region
  mean / stddev) for cross-backend tests.

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
`ambient`, `shadows` / `shadow_map_size` and `material_variant`
fields. The default is 512×512, Phong shading, perspective
projection, clamp tone map — i.e. the historical output. Always build
it with `..RenderOptions::default()` so new options default.

Textures decode through the renderer's `TextureCache`; construct
`ScanlineRenderer::with_texture_resolver(Arc::new(
RegistryTextureResolver::new(ctx)))` to decode PNG / JPEG / … through
the framework registry.

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

`benches/render.rs` (criterion) tracks both backends on procedural
scenes (including the PBR / shadow-map paths); baseline numbers +
analysis live in [`BENCHMARKS.md`](BENCHMARKS.md). Both backends are
band-parallel over std scoped threads with deterministic output.

`cargo run -p oxideav-render --release --example dump_testscenes --
out/` renders every reference scene to PNG.

## Clean-room policy

Render math is sourced from published papers and specifications —
glTF 2.0 + Appendix B, `KHR_lights_punctual`, Pineda 1988,
Sutherland–Hodgman 1974, Liang–Barsky 1984, Heckbert–Moreton 1991,
Williams 1978 / 1983, Reeves et al. 1987, Walter et al. 2007, Heitz
2014, Schlick 1994, Burley 2012, Reinhard et al. 2002, Narkowicz's
ACES fit, Porter–Duff 1984, IEC 61966-2-1. Reference renderer source
code is not consulted. glTF KHR extensions provide the
material vocabulary.

## License

MIT — see `LICENSE`.

[`oxideav_mesh3d::Scene3D`]: https://docs.rs/oxideav-mesh3d
[`oxideav_mesh3d::Bvh`]: https://docs.rs/oxideav-mesh3d
[`RgbaImage`]: https://docs.rs/oxideav-render
