# oxideav-render

Pure-Rust 3D-scene → raster image renderer for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.
Consumes an [`oxideav_mesh3d::Scene3D`] and produces a packed RGBA8
[`RgbaImage`].

## Status

The **scanline backend** is implemented and working:
`make_renderer(RenderBackend::Scanline)` returns a live renderer
backed by a half-space edge-function rasteriser with a per-pixel
z-buffer. Raytraced and path-traced backends are not yet implemented.

| Backend     | Status                                                       |
| ----------- | ----------------------------------------------------------- |
| `Scanline`  | done — Gouraud / Phong / Flat / Wireframe / NormalDebug / DepthDebug shading, perspective + orthographic projection, directional light |
| `Raycast`   | not yet — Whitted primary + shadow + reflection / refraction |
| `PathTrace` | not yet — path tracing + physically-based BRDF              |

The scanline backend has no global illumination and no raytraced
shadows. Pipeline integration (an `oxideav-pipeline` frame source) is
a follow-up.

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

## Clean-room policy

Render math is sourced from published academic papers. Reference
renderer source code is not consulted. glTF KHR extensions provide the
material vocabulary.

## License

MIT — see `LICENSE`.

[`oxideav_mesh3d::Scene3D`]: https://docs.rs/oxideav-mesh3d
[`RgbaImage`]: https://docs.rs/oxideav-render
