# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Phase B — scanline backend lands.** `make_renderer(Scanline)` now
  returns a working renderer instead of `Err(NotImplemented)`. The
  rasteriser migrates verbatim from `oxideav-cli-convert/src/mesh3d_render.rs`
  (rounds 44 + 45 in that crate) and lives under a new `scanline`
  module. Half-space edge-function pipeline with per-pixel z-buffer;
  supports Flat / Gouraud / Phong / Wireframe shading + NormalDebug /
  DepthDebug visualisers, SSAA (1..=8) via box-filter downsample,
  perspective + orthographic projection with auto-frame or orbit
  camera, and a single directional light with constant ambient term.
- **`ScanlineRenderer` public type** — direct constructor parallel to
  the [`make_renderer`] dispatch path, matching the dual-API
  convention used across the workspace.
- **Extended `RenderOptions`**: `light: LightSpec`,
  `camera: Option<CameraSpec>` — promoted from `Mesh3DOptions` in
  cli-convert so the framework-wide options struct carries every knob
  the rasteriser honours. `LightSpec::default_light()` matches the
  rasteriser baseline (azimuth 45°, elevation 45°, intensity 1.0).
- Algorithmic provenance documented in `scanline.rs` module header:
  Pineda 1988 (half-space rasterisation), Bresenham 1965 (line walker),
  IEC 61966-2-1 (sRGB encoding), OpenGL right-handed conventions for
  look-at / perspective / orthographic matrices.

### Changed

- Module documentation for the lib root reflects Phase B status; the
  `make_renderer` doc-comment promises `Scanline` now succeeds.

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
