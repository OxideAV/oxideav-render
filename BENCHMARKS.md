# oxideav-render benchmarks

Criterion suite in `benches/render.rs`. All scenes are synthesised
procedurally in the bench source (UV sphere, optional mirror floor) —
no fixture files. Run with:

```sh
cargo bench --bench render
```

## Baseline — round 400 (2026-07-09)

Apple Silicon (aarch64-apple-darwin), rustc 1.8x release bench
profile. The scanline backend is single-threaded; the raycast
backend traces rows in parallel bands across
`available_parallelism()` std scoped threads (see below — the
"1-thread" column is the pre-parallelism measurement kept for the
per-ray cost model).

| Scenario | Backend | Time | 1-thread |
| --- | --- | --- | --- |
| `scanline_phong_960tri_256` | Scanline | 2.40 ms | — |
| `raycast_phong_960tri_256` | Raycast | 2.57 ms | 20.1 ms |
| `scanline_flat_960tri_256` | Scanline | 0.75 ms | — |
| `raycast_flat_960tri_256` | Raycast | 1.97 ms | 12.1 ms |
| `raycast_phong_3968tri_256` | Raycast | 3.92 ms | 26.8 ms |
| `raycast_mirror_floor_256` | Raycast | 1.72 ms | 5.58 ms |
| `scanline_phong_960tri_aa4_128` | Scanline | 9.32 ms | — |
| `raycast_phong_960tri_aa4_128` | Raycast | 9.96 ms | 77.5 ms |
| `raycast_bake_only_3968tri_1` | Raycast | 0.22 ms | 0.21 ms |

## Prep-layer scanline + PBR (2026-10-04)

x86_64 Linux, 64 hardware threads (bands capped at 8 workers), rustc
1.98 release bench profile — a different machine from the baseline
above, so compare rows within this table only.

| Scenario | Time |
| --- | --- |
| `scanline_flat_960tri_256` | 2.47 ms |
| `scanline_phong_960tri_256` | 2.64 ms |
| `scanline_pbr_960tri_256` | 2.72 ms |
| `scanline_pbr_shadows_960tri_256` | 6.28 ms |
| `scanline_pbr_cornell_shadows_aa2_256` | 6.82 ms |
| `scanline_phong_960tri_aa4_128` | 5.02 ms |

The scanline backend now prepares the scene (morph / skin / bake /
de-index), clips, writes a visibility buffer and shades each visible
pixel once, all band-parallel. At 256² the per-pass worker spawns and
the serial prepare + triangle setup dominate, which is why Flat, Phong
and PBR cost about the same. Shadow maps add one 1024² depth pass per
directional or spot light.

## Raycast on the prep + trace layers (2026-10-04)

Same x86_64 Linux host as the table above (64 hardware threads, shared
with other jobs — expect ±10 % noise), rustc 1.98 release bench
profile. The raycast backend traces 16×16 tiles on every hardware
thread (≥ 4 tiles per worker).

| Scenario | Scanline | Raycast |
| --- | --- | --- |
| `*_flat_960tri_256` | 1.83 ms | 3.37 ms |
| `*_phong_960tri_256` (raycast: + shadow / mirror rays) | 1.97 ms | 3.47 ms |
| `*_pbr_960tri_256` | 2.54 ms | 4.48 ms |
| `*_pbr_shadows_960tri_256` (maps vs shadow rays) | 7.15 ms | 3.93 ms |
| `*_pbr_cornell_shadows_aa2_256` (point light: maps skip it, rays don't) | 5.94 ms | 7.32 ms |
| `*_pbr_sphere_grid_shadows_256` (20.7k tri; raycast adds reflections) | 14.17 ms | 6.52 ms |
| `*_phong_960tri_aa4_128` | 4.41 ms | 7.08 ms |
| `raycast_phong_3968tri_256` | — | 5.39 ms |
| `raycast_mirror_floor_256` | — | 3.12 ms |
| `raycast_bake_only_3968tri_1` (prepare + BVH) | — | 0.25 ms |

- **Shadows are cheaper by ray** at these sizes: one occlusion ray per
  lit pixel beats rasterising a 1024² shadow map per light, and point
  lights get shadows at all.
- **BVH builder choice dominates mid-size scenes.** The binned-SAH
  build costs ~0.5 µs per triangle (10.5 ms for the 20.7k-triangle
  sphere grid) — more than tracing a 256² frame. Raycast now builds an
  object-median BVH (1.7 ms, SAH cost +17 %) unless the frame traces
  ≥ 16 camera samples per triangle: sphere grid 15.7 → 6.5 ms,
  bake-only 1.9 → 0.25 ms.
- **Worker count:** capping tile workers at 8 / 16 / 32 measured
  5.6 / 4.7 / 3.5 ms on `raycast_pbr_960tri_256` — tracing is
  compute-bound, so the backend uses every hardware thread.
- Flat / Phong raycast rows are slower than the old band-parallel
  numbers on this host because every mode now prepares the scene
  (morph / skin / de-index + material table) and resolves through the
  float frame shared with scanline (linear SSAA + tone map) instead of
  writing sRGB bytes directly.

## Reading the numbers

- **Banded row parallelism** (std scoped threads, zero new
  dependencies, bit-identical output — each band owns a disjoint
  slice of the framebuffer) took the raycast Phong head-to-head from
  20.1 ms to 2.57 ms (−88%, ~8.7× on this 8-performance-core
  machine), landing the ray tracer within ~7% of the single-threaded
  rasteriser on the same scene. Bake-only is unchanged — the bake +
  BVH build stays sequential.
- **Head-to-head (960-triangle sphere, 256×256, Phong):** per ray
  (single-threaded numbers), the tracer pays a BVH walk per primary
  ray plus a shadow ray per lit hit — ~8.5× the rasteriser's
  per-pixel cost; parallelism buys that back.
- **Flat vs Phong on the raycast backend** (12.1 → 20.1 ms
  single-threaded) isolates the per-hit lighting + shadow-ray cost
  at ~8 ms for this scene; the rest is primary-ray traversal.
- **Triangle scaling:** 960 → 3968 triangles (~4.1×) moves the
  Phong trace from 2.57 to 3.92 ms (~1.5×) — the logarithmic BVH
  depth curve, not the linear soup walk.
- **SSAA:** `aa = 4` renders 16× the samples; both backends scale
  close to linearly in sample count.
- **`raycast_mirror_floor_256`** is not comparable to the sphere-only
  rows: the auto-framed camera zooms out to include the 8×8 floor,
  so the sphere covers fewer pixels; the row exists to keep the
  Whitted reflection recursion (one extra bounce per floor pixel) on
  a tracked curve.
- **`raycast_bake_only_3968tri_1`** (1×1 output) isolates scene
  flatten + BVH build: ~0.21 ms at 4k triangles — re-baking per
  `render` call is the right simplicity trade-off at these scene
  sizes.

## Negative result — ordered BVH traversal (tried, measured, dropped)

Near-child-first traversal with pre-push slab tests and
entry-distance culling (`(node, t_enter)` stack; drop subtrees whose
entry distance exceeds the current best hit) measured **8–13%
slower** than the plain test-on-pop walk on these scenes: at ~960–4k
triangles the BVH is shallow enough that two pre-push slab tests per
interior node cost more than the far-subtree culling saves. Re-try
only with much larger scenes (100k+ triangles) where deep-tree
culling has room to pay off.

## Optimisation headroom (untapped, tracked for future rounds)

- The scanline backend is single-threaded; band parallelism there
  needs per-band z-buffer ownership or triangle binning (its 2.4 ms
  is now the head-to-head floor).
- The raycast primary loop re-derives the camera basis per pixel
  through `Camera::primary_ray`; a per-row delta form would shave
  constant work.
- The scanline inner loop evaluates three edge functions per pixel
  from scratch; incremental edge stepping is the classic next step.

## Path tracer (2026-10-04)

Same machine as the prep-layer table (AMD Ryzen Threadripper 9970X,
64 hardware threads; the path tracer uses every thread, one
4-row band per work item), rustc 1.98 release bench profile.
`cargo bench --bench render -- pathtrace`.

| Scenario | Time |
| --- | --- |
| `pathtrace_cornell_64spp_256` | 281 ms |
| `pathtrace_cornell_refine1_256` | 8.5 ms |
| `pathtrace_sphere_960tri_16spp_256` | 154 ms |

- **Cornell box, 256², 64 spp, 8 bounces** (`testscenes::cornell_box`,
  point light + 0.2 sky through the open side): 4.2 M camera paths in
  0.28 s, ~15 M paths/s including NEE shadow rays and roulette.
- **`refine1`** is one progressive pass (1 spp over 256²) plus the
  display resolve — the per-frame cost an interactive viewer pays, so
  ~100 fps of refinement at this size.
- The sphere row is slower per sample than the Cornell box despite
  shorter paths: its 960-triangle BVH is ~20× the Cornell box's
  triangle count, and every bounce off the convex sphere traverses it
  for both the continuation ray and the shadow ray.
