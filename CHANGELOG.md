# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1](https://github.com/OxideAV/oxideav-render/releases/tag/v0.0.1) - 2026-06-07

### Added

- oxideav-render Phase A scaffold (Renderer trait + RenderBackend stub)

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
