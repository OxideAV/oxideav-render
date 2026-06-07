# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.3](https://github.com/OxideAV/oxideav-render/compare/v0.0.2...v0.0.3) - 2026-06-07

### Added

- RenderSource impl FrameSource — Phase C-3d pipeline source bridge

### Added

- **`RenderSource`** — `oxideav_core::FrameSource` impl wrapping a
  `Scene3D` + `Box<dyn Renderer>` + `RenderOptions`. Phase C-3d of
  the pipeline integration. Emits one `Frame::Video` for the
  still-scene case then `Error::Eof`. Animation-aware variant
  deferred to a future phase. Used by the cli-convert-installed
  `render_source_factory` callback on oxideav-pipeline's RunContext
  to bridge the renderer into the pipeline DAG. Gated on the
  `registry` cargo feature alongside the `oxideav-core` dep — the
  standalone build does not expose this type because `FrameSource`
  itself lives in `oxideav-core`.

## [0.0.2](https://github.com/OxideAV/oxideav-render/compare/v0.0.1...v0.0.2) - 2026-06-07

### Added

- RenderRegistry — Phase C-1 named backend lookup
- Phase B — scanline backend lands behind make_renderer(Scanline)

### Other

- release v0.0.1 ([#1](https://github.com/OxideAV/oxideav-render/pull/1))

## [0.0.1](https://github.com/OxideAV/oxideav-render/releases/tag/v0.0.1) - 2026-06-07

### Added

- **Phase A scaffold** — 3D-scene → raster renderer Phase 1.
  - `Renderer` trait: `render(&Scene3D, &RenderOptions) -> Result<RgbaImage>`.
    Object-safe so `make_renderer` can return a boxed renderer.
  - `RenderBackend::Scanline` variant (only variant for now). Phase D
    adds `Raycast`, Phase E adds `PathTrace`.
  - `make_renderer(RenderBackend) -> Result<Box<dyn Renderer>>` —
    Phase A returns `Err(Error::NotImplemented)` for every variant.
    Phase B fills in the scanline implementation.
  - `RenderOptions { width, height, background, shading, projection,
    fov_deg, aa }` — defaults to `512 × 512`, Phong-shaded, perspective,
    60° FOV, no SSAA.
  - `ShadingMode` enum: Flat / Gouraud / Phong / Wireframe /
    NormalDebug / DepthDebug.
  - `Projection` enum: Perspective / Orthographic.
  - `RgbaImage { width, height, pixels, stride }` — packed RGBA8
    output buffer, the renderer's contract with downstream encoders.
  - `Error::NotImplemented` + `Error::Mesh3D(oxideav_mesh3d::Error)` +
    `Error::Core(oxideav_core::Error)` (feature-gated).
  - `register(ctx)` hook (feature-gated) for `oxideav-core`
    `RuntimeContext`. Phase A is a no-op so `oxideav-meta`'s
    `build.rs` can auto-discover the crate.
  - `oxideav_core::register!("render", register)` invocation.

[Unreleased]: https://github.com/OxideAV/oxideav-render/compare/v0.0.1...HEAD
