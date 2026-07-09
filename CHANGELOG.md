# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`RenderBackend::Raycast` — Phase D Whitted ray tracer.**
  `make_renderer(RenderBackend::Raycast)` returns a live
  `RaycastRenderer`; the `RenderRegistry` built-ins now include
  `"raycast"` alongside `"scanline"`. The backend flattens the scene
  graph once per render into a world-space triangle soup (strips /
  fans pre-unrolled with correct winding, per-vertex or face normals,
  per-triangle material slots), builds an `oxideav_mesh3d::Bvh` over
  it, and walks the BVH allocation-free per ray. All six
  `ShadingMode`s are honoured: `Flat` (unlit base colour, pixel-exact
  with the scanline backend), `Gouraud` (per-vertex lighting
  interpolated barycentrically), `Phong` (full Whitted: per-pixel
  lighting + raytraced hard shadows + recursive reflection driven by
  material metallic/roughness with a Schlick Fresnel weight +
  refraction driven by `KHR_materials_transmission` /
  `KHR_materials_ior` with total-internal-reflection fallback, depth
  cap 4), `Wireframe` (barycentric edge-band detection),
  `NormalDebug`, and `DepthDebug` (hit depth mapped onto the
  projection's NDC scale, matching the rasteriser's colour key).
  Camera framing, light, background, SSAA (`aa` 1..=8), and both
  projections behave identically to the scanline backend — a
  cross-backend test pins per-pixel coverage agreement. Line / point
  topologies have no surface area and are invisible to rays.
- **`RgbaImage` typed accessors** — `set_pixel(x, y, rgba) -> bool`
  mirror of the existing `pixel(x, y)` getter, `pixel_count() -> u64`,
  `is_empty() -> bool`, `pixels_rgba()` iterator yielding `[u8; 4]`
  per pixel in row-major order, and `rows()` iterator yielding
  per-row `&[u8]` slices (stride-aware so a future padded layout
  doesn't break the walker). Lets downstream consumers stitch into
  or stream out of the renderer output without hand-rolling
  `stride`-aware byte arithmetic.
- **`RenderOptions::validate() -> Result<()>`** — typed pre-flight
  check returning the new `Error::InvalidOptions(String)` variant on
  zero `width`/`height`, out-of-range `fov_deg` / `aa`, non-finite
  or negative `light.intensity`, non-finite light angles, or a bogus
  `camera` override (non-finite angles / `distance <= 0`). Not
  called automatically by `Renderer::render` (backends still
  silently clamp) — opt-in for `oxideav-pipeline`'s `Render3D` DAG
  node which wants strict failure on a malformed job graph.

### Fixed

- **Scene-graph walks are now cycle-safe and depth-unbounded.** The
  scene graph is an arena of parent → child index references, so a
  corrupt or hostile scene can contain a self-referential node, a
  multi-node cycle, or a diamond-shared child. Every walk in this
  crate (camera auto-frame bbox, scanline draw, raycast bake) now
  runs through one shared iterative pre-order traversal that claims
  each node once at first arrival — the same contract as
  `oxideav_mesh3d::Scene3D`'s own ray / bounds walks. Previously a
  cyclic graph recursed forever and a ~10k-deep parent chain
  overflowed the call stack; both now render fine (regression tests
  cover self-cycle, two-node cycle, diamond sharing — pixel-identical
  to the equivalent plain scene — out-of-range node/mesh ids,
  NaN/Inf-poisoned vertices, garbage index buffers, 1×1 outputs, and
  a 10 000-deep hierarchy on both backends).

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
