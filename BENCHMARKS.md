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
