# oxideav-render

Pure-Rust 3D-scene → raster image/video renderer for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

**Status:** Phase A scaffold (2026-06-07) — `Renderer` trait +
`RenderBackend` selector + `make_renderer` stub. Backends land in
follow-up phases.

## Roadmap

| Phase | Surface added                                                                       |
|-------|------------------------------------------------------------------------------------|
| **A** *(now)* | `Renderer` trait + `RenderBackend::Scanline` variant + `make_renderer` stub. |
| **B**         | Scanline backend (Gouraud / Phong / Wireframe / Flat / NormalDebug / Depth).  |
| **C**         | `oxideav-pipeline` `DagNode::Render3D` source — emits `Frame::Video`.         |
| **D**         | `RenderBackend::Raycast` — Whitted primary + shadow + reflection / refraction.|
| **E**         | `RenderBackend::PathTrace` — Kajiya path tracing + Disney/Burley BRDF.        |

## Usage (Phase A surface)

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

Phase A `make_renderer` returns `Err(Error::NotImplemented)` for every
backend. Phase B (migration of the existing scanline rasteriser from
`oxideav-cli-convert`) flips `Scanline` to a working implementation.

## Output

A renderer emits an `RgbaImage` (RGBA8, packed). Downstream encoders
in the workspace (`oxideav-png`, `oxideav-mjpeg`, `oxideav-openexr`)
consume that surface directly. `oxideav-cli-convert` handles the
encoder dispatch — the render crate does not pull in image-encoder
deps.

## Standalone build

Drop the `registry` feature to build without `oxideav-core`:

```toml
oxideav-render = { version = "0.0", default-features = false }
```

The standalone build exposes `Renderer` / `RenderOptions` /
`RgbaImage` / `make_renderer` without the framework dependency tree.
The 3D input type stays `oxideav_mesh3d::Scene3D`.

## Clean-room policy

Render math is sourced from published academic papers (Möller–
Trumbore 1997, Burley 2012 SIGGRAPH course, Kajiya 1986). Reference
renderer source code (PBRT, Cycles, EEVEE, Embree, OptiX, Mitsuba,
`.blend` file format) is **not** consulted at any phase. glTF KHR
extensions provide the material vocabulary; downstream encoder
crosswalk uses the existing oxideav PNG / OpenEXR / video crates.

## License

MIT — see `LICENSE`.
