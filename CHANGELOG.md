# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`RenderBackend::PathTrace`** (`pathtrace`, registry name
  `"pathtrace"`): unbiased unidirectional Monte Carlo path tracer
  (Kajiya 1986). Next-event estimation to every `KHR_lights_punctual`
  light and to one power-sampled emissive triangle (spherical-triangle
  sampling, Arvo 1995, with a uniform-area fallback for tiny solid
  angles) or environment texel per vertex, power-heuristic MIS against BSDF
  sampling (Veach 1997), Russian roulette, optional firefly clamp.
  BSDF: glTF metallic-roughness with GGX visible-normal sampling
  (Heitz 2018), cosine diffuse and Fresnel-weighted lobe selection
  (mixture pdfs), `KHR_materials_transmission` (thin-walled and
  Walter 2007 rough refraction), `_volume` (Beer-Lambert), `_ior`,
  `_specular`, `_clearcoat`, `_sheen` (Charlie), emissive. MASK any-hit,
  BLEND stochastic transparency, normal maps, double-sided surfaces.
  Owen-scrambled Sobol' sequences (Burley 2020) with PCG-hash seeding
  and a documented dimension layout; deterministic output.
  `RenderOptions::path_trace: PathTraceOptions` (samples per pixel,
  max bounces, roulette start, clamp, seed, `LightStrategy`).
- **Progressive `PathTracer`**: `sync` / `refine` / `image` / `hdr`
  with radiance-aware reset; `EnvironmentMap` (equirectangular
  `HdrImage`, 2-D luminance CDF importance sampling);
  `PathTraceRenderer`.
- **`trace`**: shared ray-tracing layer (`TraceScene` world-space BVH
  with filtered closest / any-hit queries, surface interpolation,
  material evaluation at a hit, ray-cone texture LOD, Wächter-Binder
  ray offsets).
- `dump_testscenes` example takes an optional backend + spp.

- **Public scene-preparation layer** (`prepare`): `PreparedScene` —
  world-space de-indexed `DrawItem`s (triangles / lines / points) with
  animation sampled at a time, morph weights, CPU skinning, generated
  flat normals / tangents, CCW-front winding; resolved material and
  texture tables; world-space `KHR_lights_punctual` lights
  (`PreparedLight::sample` = spec attenuation) with the options light as
  fallback; scene camera instances. Shared by every backend and the GPU
  backend.
- **Public camera** (`camera::Camera`): `resolve` (scene camera or
  auto-frame / orbit), `from_scene_camera`, `projection_matrix` /
  `view_projection` with `DepthRange::{NegOneToOne, ZeroToOne}`.
- **Textures** (`texture`): `TextureResolver` trait, `TextureCache`,
  `RegistryTextureResolver` (registry feature; decodes via an
  `oxideav_core::RuntimeContext` + `oxideav-pixfmt`), built-in raw RGBA8
  container, sRGB / linear mip chains, glTF-sampler CPU sampling
  (wrap modes, nearest / bilinear / trilinear, derivative LOD).
- **HDR** (`hdr`): `HdrImage`, `Renderer::render_hdr`, `ToneMap::{Clamp,
  Reinhard, AcesFitted}`, `exposure`.
- **`ShadingMode::Pbr`** in the scanline backend: glTF 2.0 Appendix B
  metallic-roughness BRDF (public `brdf` module), base colour /
  metallic-roughness / normal / occlusion / emissive textures, vertex
  colours, `KHR_materials_unlit` / `_emissive_strength` / `_ior`,
  multiple punctual lights + ambient, OPAQUE / MASK / BLEND (sorted,
  over-composited), double-sided vs back-face culling, optional shadow
  maps for directional / spot lights with PCF.
- `RenderOptions`: `tone_map`, `exposure`, `time`, `animation`,
  `scene_camera`, `use_scene_lights`, `ambient`, `shadows`,
  `shadow_map_size`, `material_variant`.
- `RenderOptions::camera_target_offset` pans the auto-frame / orbit
  camera; an orbit `CameraSpec::distance` now zooms orthographic
  views; `Renderer::set_texture_resolver` (default no-op, implemented
  by the scanline backend) gives any backend a texture decoder.
- **Raycast `ShadingMode::Pbr`**: the Whitted ray tracer now renders the
  shared `PreparedScene` (animation, morphs, skinning) through
  `Camera::resolve` and the new `trace` layer, evaluating the scanline
  glTF formulas at every hit (all texture slots, vertex colours, normal
  maps, `KHR_texture_transform`, unlit, emissive, double-sided /
  culled back faces) — pixel-identical to scanline without secondary
  rays — plus ray-traced hard shadows for every light type (MASK /
  BLEND / transmission aware), MASK any-hit filtering, BLEND by
  continued rays composited in linear space, Fresnel-weighted mirror
  reflection with a roughness cut-off, and refraction through
  `KHR_materials_transmission` (thin-walled) / `KHR_materials_volume`
  (Snell, total internal reflection, Beer–Lambert absorption).
  Texture LOD from ray differentials on camera rays, ray cones after
  bounces.
- `RaycastRenderer`: native `render_hdr`, `set_texture_resolver`,
  `with_texture_resolver`, `texture_cache_mut`.
- `RenderOptions::max_ray_depth` (default 4) and
  `RenderOptions::reflection_roughness_cutoff` (default 0.5).
- `trace` additions: `TexLod::Grad` + `TraceScene::barycentric_differentials`
  (ray differentials), `material_oriented`, `shadow_transmittance`,
  `direct_light`, `with_build_options`, `reflect` / `refract` /
  `schlick` / `barycentric_of` / `pixel_spread`.
- `tests/raycast_pbr.rs`: scanline parity on eight test scenes,
  shadow, reflection, refraction, MASK / BLEND, HDR and resolver tests.
- `testscenes`: procedural reference scenes + image metrics for
  cross-backend tests; `tests/scanline_pbr.rs` property suite with raw
  goldens; `examples/dump_testscenes`.

### Changed

- Raycast backend rebuilt on the prep + trace layers: framing, posing
  and lighting match scanline / GPU; every mode resolves through the
  scanline frame contract (linear SSAA, tone map, verbatim
  background); 16×16 tiles on `std::thread::scope` workers with an
  atomic queue (deterministic); object-median BVH when few rays per
  triangle are traced, binned SAH otherwise. Legacy modes keep their
  output (`Phong` keeps its Whitted rays). Lines / points stay
  invisible to rays.
- Scanline backend rebuilt on the prep layer: homogeneous frustum
  clipping (triangles crossing the near plane are clipped instead of
  dropped), perspective-correct interpolation, top-left fill rule
  (crack-free shared edges), visibility buffer with one shade per
  pixel in linear `f32`, premultiplied SSAA resolve, band-parallel
  `std::thread::scope` workers with deterministic output.
  `ScanlineRenderer` now owns a `TextureCache`
  (`with_texture_resolver`). Legacy shading modes keep their output.

## [0.0.4](https://github.com/OxideAV/oxideav-render/compare/v0.0.3...v0.0.4) - 2026-08-16

### Added

- raycast material fidelity — KHR_materials_unlit + emissive
- RenderBackend::Raycast — Phase D Whitted recursive ray tracer
- typed RgbaImage accessors + RenderOptions::validate

### Fixed

- cycle-safe, depth-unbounded scene walks via shared iterative pre-order traversal

### Other

- state the source-independence contract without an enumerated list
- RenderSource drives the raycast backend; Error::NotImplemented doc reflects Phase D
- raycast traces rows in parallel bands — 20.1 ms -> 2.57 ms (-88%) on the Phong head-to-head
- cross-backend parity suite — interior pixels agree between backends
- criterion suite for both backends + BENCHMARKS.md baseline
- extract shared camera / math / shade modules + camera ray generation
- add CI / crates.io / docs.rs / MIT-license badges
- refresh to current status, drop per-round changelog cruft
- drop release-plz.toml — use release-plz defaults across the workspace

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

- **Raycast material fidelity: `KHR_materials_unlit` + emissive.**
  In the Whitted (`Phong`) path, an unlit material constant-shades
  its base colour — no lighting, no shadow ray, no secondary rays —
  and every material adds `emissive_factor ×
  KHR_materials_emissive_strength` after the diffuse term
  (self-illumination unaffected by shadowing; the sRGB encode
  saturates >1 sums toward white).
- **Cross-backend parity test suite.** On scenes where the two
  backends' models coincide (camera-facing triangle, uniform
  normals, no occlusion), interior pixels — full 3×3 painted
  neighbourhood in both images — must agree within ±1/255 for Phong
  / Gouraud / NormalDebug (±2 for DepthDebug, exact for Flat), in
  both perspective and orthographic projection. Pins camera framing,
  shading maths, and sRGB encoding to a single behaviour across the
  rasteriser and the ray tracer.
- **Criterion benchmark suite + `BENCHMARKS.md`.** `benches/render.rs`
  covers scanline-vs-raycast head-to-heads (Phong + Flat at 256×256 on
  a 960-triangle procedural UV sphere), triangle-count scaling on the
  BVH walk (~4× triangles → ~1.33× time), Whitted mirror-floor
  recursion, SSAA 4× scaling on both backends, and an isolated
  bake+BVH-build row (~0.21 ms at 4k triangles). Baseline numbers and
  the untapped optimisation headroom are documented in
  `BENCHMARKS.md`.

### Changed

- **Raycast renders rows in parallel bands** across
  `available_parallelism()` std scoped threads (no new dependency).
  Each band owns a disjoint slice of the output buffer, so the image
  stays bit-identical across runs and thread counts (pinned by a
  determinism test). Measured on the criterion suite: the 960-triangle
  Phong head-to-head drops 20.1 ms → 2.57 ms (−88%), landing the ray
  tracer within ~7% of the single-threaded rasteriser on the same
  scene; all raycast rows improve 73–88%. A near-child-first ordered
  BVH traversal was also tried and measured 8–13% *slower* at these
  scene sizes — kept out; the negative result is recorded in
  `BENCHMARKS.md`.

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
