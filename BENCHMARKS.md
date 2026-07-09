# oxideav-render benchmarks

Criterion suite in `benches/render.rs`. All scenes are synthesised
procedurally in the bench source (UV sphere, optional mirror floor) —
no fixture files. Run with:

```sh
cargo bench --bench render
```

## Baseline — round 400 (2026-07-09)

Apple Silicon (aarch64-apple-darwin), rustc 1.8x release bench
profile, single-threaded renderers.

| Scenario | Backend | Time |
| --- | --- | --- |
| `scanline_phong_960tri_256` | Scanline | 2.37 ms |
| `raycast_phong_960tri_256` | Raycast | 20.1 ms |
| `scanline_flat_960tri_256` | Scanline | 0.73 ms |
| `raycast_flat_960tri_256` | Raycast | 12.1 ms |
| `raycast_phong_3968tri_256` | Raycast | 26.8 ms |
| `raycast_mirror_floor_256` | Raycast | 5.58 ms |
| `scanline_phong_960tri_aa4_128` | Scanline | 9.01 ms |
| `raycast_phong_960tri_aa4_128` | Raycast | 77.5 ms |
| `raycast_bake_only_3968tri_1` | Raycast | 0.21 ms |

## Reading the numbers

- **Head-to-head (960-triangle sphere, 256×256, Phong):** the ray
  tracer costs ~8.5× the rasteriser. That is the expected shape: the
  raycast row pays per-render scene baking + BVH build (~0.2 ms at 4k
  triangles, see the bake-only row), a BVH walk per primary ray, and
  a shadow ray per lit hit — the rasteriser touches each triangle
  once and each covered pixel a constant number of times.
- **Flat vs Phong on the raycast backend** (12.1 → 20.1 ms) isolates
  the per-hit lighting + shadow-ray cost at ~8 ms for this scene;
  the remaining ~12 ms is primary-ray traversal.
- **Triangle scaling:** 960 → 3968 triangles (~4.1×) moves the
  Phong trace from 20.1 to 26.8 ms (~1.33×) — the logarithmic BVH
  depth curve, not the linear soup walk.
- **SSAA:** `aa = 4` renders 16× the samples; both backends scale
  close to linearly in sample count (scanline 2.37 → 9.01 ms with a
  quarter-size output; raycast 20.1 → 77.5 ms).
- **`raycast_mirror_floor_256`** is not comparable to the sphere-only
  rows: the auto-framed camera zooms out to include the 8×8 floor,
  so the sphere covers fewer pixels; the row exists to keep the
  Whitted reflection recursion (one extra bounce per floor pixel) on
  a tracked curve.
- **`raycast_bake_only_3968tri_1`** (1×1 output) isolates scene
  flatten + BVH build: ~0.21 ms at 4k triangles, i.e. ~1% of the
  256×256 trace — re-baking per `render` call is the right
  simplicity trade-off at these scene sizes.

## Optimisation headroom (untapped, tracked for future rounds)

- Both backends are single-threaded; rows and tiles are
  embarrassingly parallel.
- The raycast primary loop re-derives the camera basis per pixel
  through `Camera::primary_ray`; a per-row delta form would shave
  constant work.
- The scanline inner loop evaluates three edge functions per pixel
  from scratch; incremental edge stepping is the classic next step.
