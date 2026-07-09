# oxideav-render

[![CI](https://github.com/OxideAV/oxideav-render/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-render/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-render.svg)](https://crates.io/crates/oxideav-render) [![docs.rs](https://docs.rs/oxideav-render/badge.svg)](https://docs.rs/oxideav-render) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust 3D-scene → raster image renderer for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.
Consumes an [`oxideav_mesh3d::Scene3D`] and produces a packed RGBA8
[`RgbaImage`].

## Status

Two live backends behind one trait:

| Backend     | Status                                                       |
| ----------- | ----------------------------------------------------------- |
| `Scanline`  | done — Gouraud / Phong / Flat / Wireframe / NormalDebug / DepthDebug shading, perspective + orthographic projection, directional light |
| `Raycast`   | done — Whitted recursive ray tracer: same six shading modes + raytraced hard shadows, recursive reflection (metallic / roughness, Schlick Fresnel) and refraction (transmission / IOR, TIR fallback) in `Phong` mode; BVH-accelerated |
| `PathTrace` | not yet — path tracing + physically-based BRDF              |

Both backends share camera framing, light resolution, and sRGB
output encoding, so a scene renders with matching coverage and
colours from either; `Flat` mode is pixel-exact across the two. The
raycast backend bakes the scene graph into a world-space triangle
soup once per render and traverses an [`oxideav_mesh3d::Bvh`]
allocation-free per ray. Line / point topologies have no surface
area and are invisible to rays — the scanline backend remains the
renderer for wire/point content. The scanline backend has no global
illumination and no raytraced shadows.

## Usage

```rust,no_run
use oxideav_render::{make_renderer, RenderBackend, RenderOptions, Result};
use oxideav_mesh3d::Scene3D;

fn render_one(scene: &Scene3D) -> Result<()> {
    let mut renderer = make_renderer(RenderBackend::Scanline)?;
    let opts = RenderOptions {
        width: 1024,
        height: 768,
        ..Default::default()
    };
    let _image = renderer.render(scene, &opts)?;
    Ok(())
}
```

`RenderOptions` carries the framebuffer size, `ShadingMode`,
`Projection`, `BackgroundColor`, a directional `LightSpec`, a
`CameraSpec`, and an anti-aliasing factor (`aa ∈ 1..=8`). The default
is 512×512, Phong shading, perspective projection.

`RenderOptions::validate() -> Result<()>` runs a typed pre-flight
check (width/height ≥ 1, `fov_deg ∈ (0, 180)`, `aa ∈ 1..=8`, finite +
non-negative `light.intensity`, finite light/camera angles, positive
finite `camera.distance`) and surfaces the first offending field via
`Error::InvalidOptions(String)`. `Renderer::render` does not call it
automatically — backends still clamp silently — so a caller wanting
strict failure on a malformed job opts in before calling `render`.

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
scenes; baseline numbers + analysis live in
[`BENCHMARKS.md`](BENCHMARKS.md). Headline: on a 960-triangle sphere
at 256×256 Phong, the rasteriser takes ~2.4 ms and the Whitted ray
tracer ~20 ms (single-threaded, shadow rays included); the BVH keeps
raycast triangle-scaling logarithmic (4× triangles → 1.33× time).

## Clean-room policy

Render math is sourced from published academic papers. Reference
renderer source code is not consulted. glTF KHR extensions provide the
material vocabulary.

## License

MIT — see `LICENSE`.

[`oxideav_mesh3d::Scene3D`]: https://docs.rs/oxideav-mesh3d
[`oxideav_mesh3d::Bvh`]: https://docs.rs/oxideav-mesh3d
[`RgbaImage`]: https://docs.rs/oxideav-render
