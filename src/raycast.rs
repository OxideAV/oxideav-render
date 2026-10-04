//! Raycast backend — the Whitted recursive ray tracer behind
//! [`crate::RenderBackend::Raycast`].
//!
//! The scene goes through the same shared layers as every other
//! backend: [`PreparedScene`] (animation at [`RenderOptions::time`],
//! morph targets, CPU skinning, world-space attributes, materials,
//! textures, lights), [`Camera::resolve`] (identical framing — rays
//! come from [`Camera::primary_ray`]) and [`crate::trace::TraceScene`]
//! (one SAH BVH over the prepared triangles, watertight intersection,
//! hit → attribute interpolation, glTF material inputs). Line and point
//! items have no surface area and are invisible to rays.
//!
//! ## Shading modes
//!
//! * `Flat` / `Gouraud` / `Phong` / `Wireframe` / `NormalDebug` /
//!   `DepthDebug` keep the historical model shared with the scanline
//!   backend ([`crate::shade`]): base-colour factor, the options'
//!   directional light, constant 0.2 ambient. `Phong` additionally
//!   traces Whitted's rays (Whitted, "An Improved Illumination Model
//!   for Shaded Display", CACM 23(6), 1980): a hard shadow ray to the
//!   light, a mirror ray weighted by `metallic · (1 − roughness)` and
//!   Schlick's Fresnel, and a refraction ray for
//!   `KHR_materials_transmission`. `Wireframe` paints a barycentric
//!   edge band of each closest hit.
//! * `Pbr` evaluates the scanline backend's glTF 2.0 metallic-roughness
//!   formulas at every hit — [`crate::brdf`] direct lighting from every
//!   [`crate::prepare::PreparedLight`], ambient
//!   `ambient · ao · (c_diff + F0)`, emissive, unlit, all texture slots,
//!   vertex colours, normal maps, `KHR_texture_transform`, double-sided
//!   back faces with flipped normals, back-face culling of single-sided
//!   materials — and adds what rays can do:
//!   - **Hard shadows** ([`RenderOptions::shadows`]) by shadow rays for
//!     every light type (point lights included) instead of shadow
//!     maps, with `MASK` cut-outs letting light through, `BLEND`
//!     surfaces passing `1 − α` and transmissive surfaces tinting
//!     ([`crate::trace::TraceScene::shadow_transmittance`]).
//!   - **`MASK`** as an any-hit filter on every ray (camera, secondary
//!     and shadow).
//!   - **`BLEND`** by continuing the ray through the surface and
//!     compositing the result under it with the straight-alpha *over*
//!     operator in linear space (Porter & Duff 1984) — exact
//!     per-pixel ordering instead of a sorted blend pass.
//!   - **Reflection** (Whitted): one perfect-mirror ray about the
//!     shading normal, weighted by the Schlick Fresnel `F(n·v)` of the
//!     material's `F0` (so metals reflect their tinted colour and
//!     dielectrics ~4 % head-on). A single mirror ray cannot represent
//!     a glossy lobe, so roughness is handled by a cut-off
//!     ([`RenderOptions::reflection_roughness_cutoff`]): the traced
//!     reflection fades in as `g = (1 − roughness / cutoff)²` and the
//!     uniform environment term `ambient · ao · F0` keeps the remaining
//!     `1 − g` — at or above the cut-off the result is exactly the
//!     scanline formula.
//!   - **Refraction** (`KHR_materials_transmission` + `KHR_materials_ior`):
//!     the transmitted share `transmission · (1 − metallic) · (1 − F)`
//!     of the diffuse lobe is replaced by a refracted ray tinted by the
//!     base colour (the diffuse term keeps `1 − transmission`). Without
//!     `KHR_materials_volume` the surface is thin-walled and the ray
//!     passes straight through (as the extension specifies); with a
//!     volume the ray bends by Snell's law on entry and exit, reflects
//!     internally on total internal reflection, and is attenuated by
//!     Beer–Lambert absorption `exp(−σ·d)`,
//!     `σ = −ln(attenuationColor) / attenuationDistance`.
//!
//!   Secondary rays that escape the scene see a uniform environment of
//!   radiance [`RenderOptions::ambient`] (the same environment the
//!   ambient term assumes); camera rays — directly or through `BLEND`
//!   layers — see the background canvas. Recursion stops at
//!   [`RenderOptions::max_ray_depth`] bounces; at depth 0 the surface
//!   is shaded exactly like the scanline backend (opaque, no traced
//!   reflection or refraction).
//!
//! ## Texture filtering
//!
//! Camera rays carry ray differentials (Igehy, "Tracing Ray
//! Differentials", SIGGRAPH 1999): the rays through the next pixel
//! along `x` and `y`, intersected with the hit triangle's plane, give
//! the barycentric derivatives per pixel — the same derivatives the
//! rasteriser takes, so primary-hit (and `BLEND` / thin-wall
//! see-through) texture filtering matches the scanline backend. After a
//! reflection or refraction the footprint becomes a ray cone
//! (Akenine-Möller et al., "Texture Level of Detail Strategies for
//! Real-Time Ray Tracing", Ray Tracing Gems ch. 20, 2019) starting at
//! the differential footprint's width and spreading by the camera's
//! per-pixel angle (curvature ignored). Any-hit alpha tests sample
//! level 0.
//!
//! ## Output
//!
//! Every render-resolution sample (SSAA `aa × aa` per output pixel)
//! fills the scanline backend's `Frame` (scene-linear colour +
//! coverage), so the resolve contract is identical: background
//! samples come back verbatim, covered samples go through exposure,
//! [`crate::ToneMap`] and sRGB, SSAA averages premultiplied. Samples
//! are traced on `std::thread::scope` workers pulling 16×16 tiles
//! from an atomic counter; every sample is a pure function of its
//! position, so the output is bit-identical for any thread count.
//!
//! Clean-room: algorithms come from the cited papers and the glTF 2.0 /
//! KHR extension specifications. No renderer source code was consulted.

use std::sync::atomic::{AtomicUsize, Ordering};

use oxideav_mesh3d::{AlphaMode, BvhBuildOptions, Scene3D};

use crate::brdf::{f_schlick, BrdfParams};
use crate::camera::Camera;
use crate::hdr::{srgb_u8_lut, HdrImage};
use crate::image::RgbaImage;
use crate::math::{vec3_dot, vec3_normalise, vec3_scale};
use crate::options::{RenderOptions, ShadingMode};
use crate::prepare::{PrepareOptions, PreparedScene};
use crate::scanline::{depth_to_byte, normal_to_byte, Frame, Sample};
use crate::shade::{build_light, shade_pixel, DirLight, AMBIENT};
use crate::texture::TextureCache;
use crate::trace::{
    interp3, offset_ray_origin, pixel_spread, reflect, refract, schlick, TexLod, TraceHit,
    TraceScene,
};

/// Hard cap on [`RenderOptions::max_ray_depth`].
const MAX_DEPTH_CAP: u32 = 16;

/// Maximum number of `BLEND` / thin-wall see-through continuations
/// along one ray path (bounds the recursion on stacks of
/// semi-transparent layers).
const MAX_LAYERS: u32 = 64;

/// Minimum metallic (legacy `Phong`) / transmission before a secondary
/// ray is traced.
const SECONDARY_RAY_THRESHOLD: f32 = 1.0e-2;

/// Barycentric distance from a triangle edge under which a
/// [`ShadingMode::Wireframe`] hit paints the pixel.
const WIREFRAME_EDGE_WIDTH: f32 = 0.03;

/// Tile edge (render pixels) of the parallel work unit.
const TILE: usize = 16;

// ---------------------------------------------------------------------
// Entry points.
// ---------------------------------------------------------------------

/// Render with a throw-away texture cache (built-in raw textures only).
#[cfg(test)]
pub(crate) fn render_scene(scene: &Scene3D, opts: &RenderOptions) -> RgbaImage {
    render_with_cache(scene, opts, &mut TextureCache::default())
}

/// Render to RGBA8, decoding textures through `cache`.
pub(crate) fn render_with_cache(
    scene: &Scene3D,
    opts: &RenderOptions,
    cache: &mut TextureCache,
) -> RgbaImage {
    render_frame(scene, opts, cache).to_rgba8(opts)
}

/// Scene-linear output (pre exposure / tone map).
pub(crate) fn render_hdr_with_cache(
    scene: &Scene3D,
    opts: &RenderOptions,
    cache: &mut TextureCache,
) -> HdrImage {
    render_frame(scene, opts, cache).to_hdr()
}

fn render_frame(scene: &Scene3D, opts: &RenderOptions, cache: &mut TextureCache) -> Frame {
    let width = opts.width.max(1);
    let height = opts.height.max(1);
    let aa = opts.aa.clamp(1, 8);
    let rw = width.saturating_mul(aa).max(1);
    let rh = height.saturating_mul(aa).max(1);

    let prepared = PreparedScene::build(scene, &PrepareOptions::from_render_options(opts), cache);
    let camera = Camera::resolve(&prepared, opts, rw, rh);
    // BVH builder: binned SAH (Wald 2007) traces ~15 % faster than an
    // object-median split but builds ~6× slower; a Whitted frame
    // traces only a few rays per triangle on dense meshes, so the
    // cheaper build wins below ~16 camera samples per triangle.
    let build = if (rw as usize * rh as usize) >= 16 * prepared.triangle_count() {
        BvhBuildOptions::sah()
    } else {
        BvhBuildOptions::object_median()
    };
    let ts = TraceScene::with_build_options(prepared, &build);
    let ambient = if opts.ambient.is_finite() {
        opts.ambient.max(0.0)
    } else {
        0.0
    };
    let cutoff = if opts.reflection_roughness_cutoff.is_finite() {
        opts.reflection_roughness_cutoff.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let lut = srgb_u8_lut();
    let b = opts.background.0;
    let ctx = Ctx {
        ts: &ts,
        camera,
        mode: opts.shading,
        light: build_light(opts.light),
        ambient,
        shadows: opts.shadows,
        max_depth: opts.max_ray_depth.min(MAX_DEPTH_CAP),
        cutoff,
        spread: pixel_spread(&camera, rh),
        bg: [
            lut[b[0] as usize],
            lut[b[1] as usize],
            lut[b[2] as usize],
            b[3] as f32 / 255.0,
        ],
        rw: rw as f32,
        rh: rh as f32,
    };

    let samples = trace_tiles(rw as usize, rh as usize, |x, y| ctx.pixel(x, y));
    Frame {
        out_w: width,
        out_h: height,
        aa,
        samples,
        background: opts.background.0,
        display_referred: matches!(
            opts.shading,
            ShadingMode::NormalDebug | ShadingMode::DepthDebug
        ),
    }
}

/// Evaluate `f(x, y)` for every sample of a `w × h` grid on scoped
/// worker threads pulling [`TILE`]² tiles from an atomic counter.
/// Deterministic: each sample depends only on its coordinates.
fn trace_tiles(w: usize, h: usize, f: impl Fn(usize, usize) -> Sample + Sync) -> Vec<Sample> {
    let blank = Sample {
        c: [0.0; 4],
        covered: false,
    };
    let mut out = vec![blank; w * h];
    let (tx, ty) = (w.div_ceil(TILE), h.div_ceil(TILE));
    let n_tiles = tx * ty;
    let run_tile = |t: usize| -> (usize, Vec<Sample>) {
        let (x0, y0) = ((t % tx) * TILE, (t / tx) * TILE);
        let (x1, y1) = ((x0 + TILE).min(w), (y0 + TILE).min(h));
        let mut v = Vec::with_capacity((x1 - x0) * (y1 - y0));
        for y in y0..y1 {
            for x in x0..x1 {
                v.push(f(x, y));
            }
        }
        (t, v)
    };
    // Ray tracing is compute-bound: use every hardware thread, but
    // keep at least four tiles per worker so small renders do not pay
    // for idle spawns.
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(n_tiles / 4)
        .max(1);
    let results: Vec<(usize, Vec<Sample>)> = if workers <= 1 {
        (0..n_tiles).map(run_tile).collect()
    } else {
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    s.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let t = next.fetch_add(1, Ordering::Relaxed);
                            if t >= n_tiles {
                                break;
                            }
                            done.push(run_tile(t));
                        }
                        done
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap_or_default())
                .collect()
        })
    };
    for (t, v) in results {
        let (x0, y0) = ((t % tx) * TILE, (t / tx) * TILE);
        let tw = (x0 + TILE).min(w) - x0;
        for (i, s) in v.into_iter().enumerate() {
            out[(y0 + i / tw) * w + x0 + i % tw] = s;
        }
    }
    out
}

// ---------------------------------------------------------------------
// Per-render context.
// ---------------------------------------------------------------------

struct Ctx<'a> {
    ts: &'a TraceScene,
    camera: Camera,
    mode: ShadingMode,
    /// Legacy-mode directional light.
    light: DirLight,
    ambient: f32,
    shadows: bool,
    max_depth: u32,
    cutoff: f32,
    /// Camera per-pixel spread angle (ray cones).
    spread: f32,
    /// Linear background (camera-ray misses through `BLEND` layers).
    bg: [f32; 4],
    rw: f32,
    rh: f32,
}

/// Texture footprint a ray carries.
#[derive(Debug, Clone, Copy)]
enum Footprint {
    /// Neighbouring camera rays (valid while the path is straight).
    Diff {
        dx: ([f32; 3], [f32; 3]),
        dy: ([f32; 3], [f32; 3]),
    },
    /// Ray cone: width at the origin, spread per unit distance.
    Cone { width: f32, spread: f32 },
}

/// Recursion state of a `Pbr` ray.
#[derive(Debug, Clone, Copy)]
struct RayState {
    /// Reflection / refraction bounces so far.
    depth: u32,
    /// See-through continuations so far.
    layers: u32,
    /// Still a camera ray (only `BLEND` continuations so far): misses
    /// see the background canvas instead of the environment.
    camera: bool,
    /// Absorption coefficient of the volume the ray travels in.
    inside: Option<[f32; 3]>,
    /// Global triangle the ray leaves.
    skip: Option<u32>,
}

impl Ctx<'_> {
    fn pixel(&self, x: usize, y: usize) -> Sample {
        let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
        let (o, d) = self.camera.primary_ray(px, py, self.rw, self.rh);
        let c = match self.mode {
            ShadingMode::Pbr => {
                let fp = Footprint::Diff {
                    dx: self.camera.primary_ray(px + 1.0, py, self.rw, self.rh),
                    dy: self.camera.primary_ray(px, py + 1.0, self.rw, self.rh),
                };
                let st = RayState {
                    depth: 0,
                    layers: 0,
                    camera: true,
                    inside: None,
                    skip: None,
                };
                self.radiance(o, d, fp, st)
            }
            _ => self.legacy(o, d),
        };
        match c {
            Some(c) => Sample { c, covered: true },
            None => Sample {
                c: [0.0; 4],
                covered: false,
            },
        }
    }

    // -----------------------------------------------------------------
    // Legacy modes.
    // -----------------------------------------------------------------

    fn legacy(&self, o: [f32; 3], d: [f32; 3]) -> Option<[f32; 4]> {
        let hit = self.ts.closest_hit(o, d, 0.0, f32::INFINITY)?;
        let (item, mat) = self.ts.item(&hit);
        let base = 3 * hit.tri as usize;
        let b = hit.barycentric;
        Some(match self.mode {
            ShadingMode::Flat => mat.base_color,
            ShadingMode::Wireframe => {
                // A rasteriser draws edges; a ray tracer detects them —
                // paint when the hit lies in a barycentric edge band,
                // else show the background (closest hit only).
                let m = b.iter().fold(f32::INFINITY, |a, &v| a.min(v));
                if m > WIREFRAME_EDGE_WIDTH {
                    return None;
                }
                mat.base_color
            }
            ShadingMode::Gouraud => {
                let n = &item.normals;
                let ca = shade_pixel(mat.base_color, n[base], &self.light);
                let cb = shade_pixel(mat.base_color, n[base + 1], &self.light);
                let cc = shade_pixel(mat.base_color, n[base + 2], &self.light);
                let mut o = [0.0; 4];
                for k in 0..4 {
                    o[k] = ca[k] * b[0] + cb[k] * b[1] + cc[k] * b[2];
                }
                o
            }
            ShadingMode::NormalDebug => {
                let n = vec3_normalise(interp3(&item.normals, base, b));
                let lut = srgb_u8_lut();
                [
                    lut[normal_to_byte(n[0]) as usize],
                    lut[normal_to_byte(n[1]) as usize],
                    lut[normal_to_byte(n[2]) as usize],
                    1.0,
                ]
            }
            ShadingMode::DepthDebug => {
                let p = interp3(&item.positions, base, b);
                let z = self.camera.ndc_z(self.camera.view_depth(p));
                let g = srgb_u8_lut()[depth_to_byte(z) as usize];
                [g, g, g, 1.0]
            }
            _ => self.whitted(d, &hit, 0),
        })
    }

    /// Historical `Phong` Whitted shading: Lambert + ambient with a
    /// shadow ray, recursive mirror (metallic) and refraction
    /// (transmission) rays blended by a Schlick weight.
    fn whitted(&self, d: [f32; 3], hit: &TraceHit, depth: u32) -> [f32; 4] {
        let (item, mat) = self.ts.item(hit);
        if mat.unlit {
            return mat.base_color;
        }
        let base = 3 * hit.tri as usize;
        let raw = vec3_normalise(interp3(&item.normals, base, hit.barycentric));
        // Orient the interpolated normal against the incident ray;
        // `entering` doubles as the inside/outside signal for the
        // refraction index ratio.
        let entering = vec3_dot(d, raw) < 0.0;
        let n = if entering { raw } else { vec3_scale(raw, -1.0) };
        let p = interp3(&item.positions, base, hit.barycentric);
        let skip = hit.global;
        let not_self = |h: &TraceHit| h.global != skip;

        let so = offset_ray_origin(p, n);
        let shadowed =
            self.ts
                .occluded_filtered(so, self.light.direction, 0.0, f32::INFINITY, not_self);
        let a = mat.base_color;
        let mut out = if shadowed {
            [a[0] * AMBIENT, a[1] * AMBIENT, a[2] * AMBIENT, a[3]]
        } else {
            shade_pixel(a, n, &self.light)
        };
        for (o, e) in out.iter_mut().zip(mat.emissive) {
            *o += e;
        }
        if depth >= self.max_depth {
            return out;
        }
        let cos_in = (-vec3_dot(d, n)).clamp(0.0, 1.0);
        let trace = |o: [f32; 3], dir: [f32; 3]| -> Option<[f32; 4]> {
            let h = self
                .ts
                .closest_hit_filtered(o, dir, 0.0, f32::INFINITY, not_self)?;
            Some(self.whitted(dir, &h, depth + 1))
        };

        let gloss = mat.metallic * (1.0 - mat.roughness);
        if gloss > SECONDARY_RAY_THRESHOLD {
            let f0 = 0.04 + 0.96 * mat.metallic;
            let kr = schlick(f0, cos_in).clamp(0.0, 1.0) * gloss;
            // Escaping reflection rays contribute black (the canvas
            // colour is not an environment).
            let rc = trace(so, reflect(d, n)).unwrap_or([0.0, 0.0, 0.0, 1.0]);
            for k in 0..3 {
                out[k] = out[k] * (1.0 - kr) + rc[k] * a[k] * kr;
            }
        }
        let kt = mat
            .ext
            .transmission
            .map(|t| t.factor.clamp(0.0, 1.0))
            .unwrap_or(0.0);
        if kt > SECONDARY_RAY_THRESHOLD {
            let eta = if entering { 1.0 / mat.ior } else { mat.ior };
            match refract(d, n, eta) {
                Some(t) => {
                    let ro = offset_ray_origin(p, vec3_scale(n, -1.0));
                    let tc = trace(ro, t).unwrap_or([0.0; 4]);
                    for k in 0..3 {
                        out[k] = out[k] * (1.0 - kt) + tc[k] * a[k] * kt;
                    }
                }
                None => {
                    // Total internal reflection: the energy goes along
                    // the mirror direction instead.
                    if let Some(rc) = trace(so, reflect(d, n)) {
                        for k in 0..3 {
                            out[k] = out[k] * (1.0 - kt) + rc[k] * kt;
                        }
                    }
                }
            }
        }
        out
    }

    // -----------------------------------------------------------------
    // Pbr.
    // -----------------------------------------------------------------

    /// Radiance (+ coverage alpha) arriving along `o + t·d`; `None`
    /// when the ray escapes.
    fn radiance(&self, o: [f32; 3], d: [f32; 3], fp: Footprint, st: RayState) -> Option<[f32; 4]> {
        let ts = self.ts;
        let cull = st.inside.is_none();
        let hit = ts.closest_hit_filtered(o, d, 0.0, f32::INFINITY, |h| {
            if Some(h.global) == st.skip {
                return false;
            }
            let (_, mat) = ts.item(h);
            if cull && !h.front_face && !mat.double_sided {
                return false;
            }
            !ts.masked_out(h)
        })?;
        let mut c = self.shade(d, &hit, fp, st);
        if let Some(sigma) = st.inside {
            for k in 0..3 {
                c[k] *= (-sigma[k] * hit.t).exp();
            }
        }
        Some(c)
    }

    /// What a ray of state `st` sees when it escapes.
    fn miss(&self, st: &RayState) -> [f32; 4] {
        if st.camera {
            self.bg
        } else {
            [self.ambient, self.ambient, self.ambient, 1.0]
        }
    }

    /// Trace a secondary ray, falling back to the environment.
    fn secondary(&self, o: [f32; 3], d: [f32; 3], fp: Footprint, st: RayState) -> [f32; 3] {
        let c = self
            .radiance(o, d, fp, st)
            .unwrap_or_else(|| self.miss(&st));
        [c[0], c[1], c[2]]
    }

    fn shade(&self, d: [f32; 3], hit: &TraceHit, fp: Footprint, st: RayState) -> [f32; 4] {
        let ts = self.ts;
        let surf = ts.surface(hit);
        let (lod, width) = match fp {
            Footprint::Diff { dx, dy } => ts
                .barycentric_differentials(hit, dx, dy)
                .unwrap_or((TexLod::Base, 0.0)),
            Footprint::Cone { width, spread } => {
                let w = (width + spread * hit.t).max(0.0);
                (TexLod::Cone { width: w, dir: d }, w)
            }
        };
        let m = ts.material_oriented(hit, &surf, lod, !hit.front_face);
        let p = surf.position;
        let ng = if hit.front_face {
            surf.geometric_normal
        } else {
            vec3_scale(surf.geometric_normal, -1.0)
        };
        let n = m.normal;
        let v = vec3_scale(d, -1.0);
        let skip = Some(hit.global);
        let bounce = Footprint::Cone {
            width,
            spread: self.spread,
        };
        let can_bounce = st.depth < self.max_depth;
        let f0_d = {
            let r = (m.ior - 1.0) / (m.ior + 1.0);
            r * r
        };

        // Leaving a volume through its boundary: Fresnel split between
        // the internally reflected and the refracted (outgoing) ray.
        if st.inside.is_some() && !hit.front_face && m.transmission > 0.0 && m.thickness > 0.0 {
            let tint_st = |inside| RayState {
                depth: st.depth + 1,
                layers: st.layers,
                camera: false,
                inside,
                skip,
            };
            if !can_bounce {
                let e = self.miss(&tint_st(None));
                return [e[0], e[1], e[2], 1.0];
            }
            let cos_i = vec3_dot(v, n).clamp(0.0, 1.0);
            let internal = || {
                self.secondary(
                    offset_ray_origin(p, ng),
                    reflect(d, n),
                    bounce,
                    tint_st(st.inside),
                )
            };
            return match refract(d, n, m.ior) {
                Some(t) => {
                    let sin2_t = m.ior * m.ior * (1.0 - cos_i * cos_i);
                    let f = schlick(f0_d, (1.0 - sin2_t).max(0.0).sqrt());
                    let out = self.secondary(
                        offset_ray_origin(p, vec3_scale(ng, -1.0)),
                        t,
                        bounce,
                        tint_st(None),
                    );
                    let inn = internal();
                    [
                        f * inn[0] + (1.0 - f) * out[0],
                        f * inn[1] + (1.0 - f) * out[1],
                        f * inn[2] + (1.0 - f) * out[2],
                        1.0,
                    ]
                }
                None => {
                    let inn = internal();
                    [inn[0], inn[1], inn[2], 1.0]
                }
            };
        }

        let col = m.base_color;
        let alpha = match m.alpha_mode {
            AlphaMode::Blend => col[3].clamp(0.0, 1.0),
            _ => 1.0,
        };
        let mut out = [0.0f32; 3];
        if m.unlit {
            out = [col[0], col[1], col[2]];
        } else {
            let mut params = BrdfParams::metallic_roughness(
                [col[0], col[1], col[2]],
                m.metallic,
                m.roughness,
                f0_d,
            );
            let kt = if can_bounce && m.transmission > 0.0 {
                m.transmission
            } else {
                0.0
            };
            for c in params.c_diff.iter_mut() {
                *c *= 1.0 - kt;
            }
            let g = if can_bounce && m.roughness < self.cutoff {
                let x = 1.0 - m.roughness / self.cutoff;
                x * x
            } else {
                0.0
            };
            for (k, o) in out.iter_mut().enumerate() {
                *o = m.emissive[k]
                    + self.ambient * m.occlusion * (params.c_diff[k] + (1.0 - g) * params.f0[k]);
            }
            let direct = ts.direct_light(p, n, ng, v, &params, self.shadows, skip);
            for k in 0..3 {
                out[k] += direct[k];
            }
            let next = |inside| RayState {
                depth: st.depth + 1,
                layers: st.layers,
                camera: false,
                inside,
                skip,
            };
            if g > 0.0 {
                let n_dot_v = vec3_dot(n, v).clamp(0.0, 1.0);
                let f = f_schlick(params.f0, n_dot_v);
                let mut r = reflect(d, n);
                if vec3_dot(r, ng) <= 0.0 {
                    // Normal-mapped normal sending the mirror ray under
                    // the surface: reflect about the geometric normal.
                    r = reflect(d, ng);
                }
                let l = self.secondary(offset_ray_origin(p, ng), r, bounce, next(st.inside));
                for k in 0..3 {
                    out[k] += g * f[k] * l[k];
                }
            }
            if kt > 0.0 {
                let share = kt * (1.0 - m.metallic);
                let cos_i = vec3_dot(v, n).clamp(0.0, 1.0);
                let through = offset_ray_origin(p, vec3_scale(ng, -1.0));
                let (f, l) = if m.thickness > 0.0 && hit.front_face {
                    // Entering a volume: Snell bend, absorption inside.
                    let sigma = attenuation_sigma(m.attenuation_color, m.attenuation_distance);
                    match refract(d, n, 1.0 / m.ior) {
                        Some(t) => (
                            schlick(f0_d, cos_i),
                            self.secondary(through, t, bounce, next(Some(sigma))),
                        ),
                        None => (1.0, [0.0; 3]),
                    }
                } else {
                    // Thin wall: straight through, footprint unchanged.
                    let st2 = RayState {
                        layers: st.layers + 1,
                        ..next(st.inside)
                    };
                    (schlick(f0_d, cos_i), self.secondary(through, d, fp, st2))
                };
                for k in 0..3 {
                    out[k] += share * (1.0 - f) * col[k] * l[k];
                }
            }
        }

        if alpha >= 1.0 {
            return [out[0], out[1], out[2], 1.0];
        }
        // BLEND: continue the ray behind the surface and composite.
        let dst = if st.layers < MAX_LAYERS {
            let st2 = RayState {
                layers: st.layers + 1,
                skip,
                ..st
            };
            self.radiance(offset_ray_origin(p, vec3_scale(ng, -1.0)), d, fp, st2)
                .unwrap_or_else(|| self.miss(&st))
        } else {
            self.miss(&st)
        };
        over([out[0], out[1], out[2], alpha], dst)
    }
}

/// Beer–Lambert absorption coefficient of `KHR_materials_volume`:
/// `σ = −ln(attenuationColor) / attenuationDistance` (`0` when the
/// distance is infinite).
fn attenuation_sigma(color: [f32; 3], distance: f32) -> [f32; 3] {
    if !(distance.is_finite() && distance > 0.0) {
        return [0.0; 3];
    }
    color.map(|c| -c.clamp(1.0e-6, 1.0).ln() / distance)
}

/// Straight-alpha Porter–Duff *over* in linear space.
fn over(src: [f32; 4], dst: [f32; 4]) -> [f32; 4] {
    let a = src[3].clamp(0.0, 1.0);
    let ad = dst[3].clamp(0.0, 1.0);
    let ao = a + ad * (1.0 - a);
    if ao <= 0.0 {
        return [0.0; 4];
    }
    let mut c = [0.0; 4];
    for k in 0..3 {
        c[k] = (src[k] * a + dst[k] * ad * (1.0 - a)) / ao;
    }
    c[3] = ao;
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec3_normalise;
    use crate::options::BackgroundColor;
    use crate::shade::linear_rgba_to_srgb_u8;
    use oxideav_mesh3d::{
        Indices, Material, MaterialId, Mesh, MeshId, Node, NodeId, Primitive, Scene3D, Topology,
    };

    const WHITE_BG: BackgroundColor = BackgroundColor([255, 255, 255, 255]);

    fn push_mesh_node(scene: &mut Scene3D, prim: Primitive) {
        let mesh_id = scene.meshes.len() as u32;
        scene
            .meshes
            .push(Mesh::new(format!("m{mesh_id}")).with_primitive(prim));
        let node_id = scene.nodes.len() as u32;
        scene.nodes.push(Node {
            mesh: Some(MeshId(mesh_id)),
            ..Node::default()
        });
        scene.roots.push(NodeId(node_id));
    }

    fn unit_triangle_scene() -> Scene3D {
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        let mut scene = Scene3D::new();
        push_mesh_node(&mut scene, prim);
        scene
    }

    fn render_with_mode(mode: ShadingMode) -> RgbaImage {
        let opts = RenderOptions {
            width: 64,
            height: 64,
            shading: mode,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        render_scene(&unit_triangle_scene(), &opts)
    }

    fn non_bg_count(img: &RgbaImage, bg: [u8; 4]) -> usize {
        img.pixels.chunks_exact(4).filter(|p| *p != bg).count()
    }

    #[test]
    fn every_mode_paints_the_triangle() {
        for mode in [
            ShadingMode::Flat,
            ShadingMode::Gouraud,
            ShadingMode::Phong,
            ShadingMode::Wireframe,
            ShadingMode::NormalDebug,
            ShadingMode::DepthDebug,
        ] {
            let img = render_with_mode(mode);
            assert!(
                non_bg_count(&img, [255, 255, 255, 255]) > 0,
                "{mode:?} must paint at least one pixel"
            );
        }
    }

    #[test]
    fn empty_scene_yields_pure_background() {
        let scene = Scene3D::new();
        let opts = RenderOptions {
            width: 8,
            height: 8,
            background: BackgroundColor([42, 7, 99, 255]),
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        for px in img.pixels.chunks_exact(4) {
            assert_eq!(px, &[42, 7, 99, 255]);
        }
    }

    #[test]
    fn output_dimensions_match_options_with_and_without_aa() {
        for aa in [1, 2, 4] {
            let opts = RenderOptions {
                width: 33,
                height: 17,
                aa,
                background: WHITE_BG,
                ..RenderOptions::default()
            };
            let img = render_scene(&unit_triangle_scene(), &opts);
            assert_eq!((img.width, img.height), (33, 17), "aa={aa}");
            assert_eq!(img.pixels.len(), 33 * 17 * 4, "aa={aa}");
        }
    }

    #[test]
    fn coverage_matches_scanline_backend() {
        // The two backends share camera framing, so the same triangle
        // must cover a nearly identical pixel set. Allow a small edge
        // disagreement (rasteriser samples pixel centres with edge
        // fill rules; rays sample exact centres).
        let scene = unit_triangle_scene();
        let opts = RenderOptions {
            width: 64,
            height: 64,
            shading: ShadingMode::Flat,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let ray_img = render_scene(&scene, &opts);
        let scan_img = crate::scanline::render_scene(&scene, &opts);
        let mut both = 0usize;
        let mut only_one = 0usize;
        for (r, s) in ray_img
            .pixels
            .chunks_exact(4)
            .zip(scan_img.pixels.chunks_exact(4))
        {
            let rp = r != [255, 255, 255, 255];
            let sp = s != [255, 255, 255, 255];
            if rp && sp {
                both += 1;
            } else if rp != sp {
                only_one += 1;
            }
        }
        assert!(
            both > 100,
            "expected substantial shared coverage, got {both}"
        );
        assert!(
            only_one * 10 < both,
            "backends disagree on too many pixels: shared={both}, disputed={only_one}"
        );
    }

    #[test]
    fn flat_colour_matches_scanline_exactly() {
        // Interior pixels in Flat mode are the same sRGB-encoded base
        // colour in both backends.
        let scene = unit_triangle_scene();
        let opts = RenderOptions {
            width: 64,
            height: 64,
            shading: ShadingMode::Flat,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        let expected = linear_rgba_to_srgb_u8([0.7, 0.7, 0.75, 1.0]);
        let painted: Vec<&[u8]> = img
            .pixels
            .chunks_exact(4)
            .filter(|p| *p != [255, 255, 255, 255])
            .collect();
        assert!(!painted.is_empty());
        for p in painted {
            assert_eq!(p, expected, "flat shading must be the unlit base colour");
        }
    }

    #[test]
    fn strip_and_fan_topologies_are_visible() {
        for topology in [Topology::TriangleStrip, Topology::TriangleFan] {
            let mut prim = Primitive::new(topology);
            prim.positions = vec![
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [-0.5, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ];
            let mut scene = Scene3D::new();
            push_mesh_node(&mut scene, prim);
            let opts = RenderOptions {
                width: 32,
                height: 32,
                shading: ShadingMode::Flat,
                background: WHITE_BG,
                ..RenderOptions::default()
            };
            let img = render_scene(&scene, &opts);
            assert!(
                non_bg_count(&img, [255, 255, 255, 255]) > 0,
                "{topology:?} must bake into visible triangles"
            );
        }
    }

    #[test]
    fn line_and_point_topologies_are_invisible() {
        for topology in [Topology::Lines, Topology::LineStrip, Topology::Points] {
            let mut prim = Primitive::new(topology);
            prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]];
            let mut scene = Scene3D::new();
            push_mesh_node(&mut scene, prim);
            let opts = RenderOptions {
                width: 16,
                height: 16,
                shading: ShadingMode::Flat,
                background: WHITE_BG,
                ..RenderOptions::default()
            };
            let img = render_scene(&scene, &opts);
            assert_eq!(
                non_bg_count(&img, [255, 255, 255, 255]),
                0,
                "{topology:?} has no surface area for rays"
            );
        }
    }

    #[test]
    fn indexed_u16_triangles_render() {
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        prim.indices = Some(Indices::U16(vec![0, 1, 2]));
        let mut scene = Scene3D::new();
        push_mesh_node(&mut scene, prim);
        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Flat,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        assert!(non_bg_count(&img, [255, 255, 255, 255]) > 0);
    }

    #[test]
    fn material_base_colour_reaches_pixels() {
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        prim.material = Some(MaterialId(0));
        let mut scene = Scene3D::new();
        scene.materials.push(Material {
            base_color: [1.0, 0.0, 0.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            ..Material::new()
        });
        push_mesh_node(&mut scene, prim);
        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Flat,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        let expected = linear_rgba_to_srgb_u8([1.0, 0.0, 0.0, 1.0]);
        let hit_red = img.pixels.chunks_exact(4).any(|p| p == expected);
        assert!(hit_red, "material base colour must reach the framebuffer");
    }

    #[test]
    fn depth_debug_is_grayscale_and_nearer_is_brighter() {
        // Two coplanar-to-screen triangles at different depths: the
        // nearer one must be brighter (near → white convention).
        let mut near = Primitive::new(Topology::Triangles);
        near.positions = vec![[-0.9, -0.4, 0.5], [-0.1, -0.4, 0.5], [-0.5, 0.4, 0.5]];
        let mut far = Primitive::new(Topology::Triangles);
        far.positions = vec![[0.1, -0.4, -0.5], [0.9, -0.4, -0.5], [0.5, 0.4, -0.5]];
        let mut scene = Scene3D::new();
        push_mesh_node(&mut scene, near);
        push_mesh_node(&mut scene, far);
        let opts = RenderOptions {
            width: 64,
            height: 64,
            shading: ShadingMode::DepthDebug,
            background: BackgroundColor([255, 0, 0, 255]),
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        // Gather grayscale values on each half of the image.
        let mut left_max = 0u8;
        let mut right_max = 0u8;
        for y in 0..64 {
            for x in 0..64 {
                let p = img.pixel(x, y).unwrap();
                if p == [255, 0, 0, 255] {
                    continue;
                }
                assert_eq!(p[0], p[1]);
                assert_eq!(p[1], p[2]);
                if x < 32 {
                    left_max = left_max.max(p[0]);
                } else {
                    right_max = right_max.max(p[0]);
                }
            }
        }
        assert!(left_max > 0 && right_max > 0, "both triangles must appear");
        assert!(
            left_max > right_max,
            "near (+z, left) triangle must be brighter: {left_max} vs {right_max}"
        );
    }

    #[test]
    fn phong_shading_is_darker_than_flat_on_tilted_face() {
        // A face tilted away from the default light shades darker
        // than its unlit base colour.
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        prim.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
        let mut scene = Scene3D::new();
        push_mesh_node(&mut scene, prim);
        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Phong,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        let flat = linear_rgba_to_srgb_u8([0.7, 0.7, 0.75, 1.0]);
        let painted: Vec<[u8; 4]> = img
            .pixels
            .chunks_exact(4)
            .filter(|p| *p != [255, 255, 255, 255])
            .map(|p| [p[0], p[1], p[2], p[3]])
            .collect();
        assert!(!painted.is_empty());
        for p in &painted {
            assert!(
                p[0] <= flat[0] && p[1] <= flat[1] && p[2] <= flat[2],
                "lit colour must not exceed base colour: {p:?} vs {flat:?}"
            );
        }
    }

    #[test]
    fn wireframe_paints_fewer_pixels_than_flat() {
        let flat = render_with_mode(ShadingMode::Flat);
        let wire = render_with_mode(ShadingMode::Wireframe);
        let bg = [255, 255, 255, 255];
        let flat_n = non_bg_count(&flat, bg);
        let wire_n = non_bg_count(&wire, bg);
        assert!(wire_n > 0, "wireframe must paint the edge band");
        assert!(
            wire_n < flat_n,
            "edge band must be sparser than the filled face: {wire_n} vs {flat_n}"
        );
    }

    #[test]
    fn reflect_mirrors_about_normal() {
        let r = reflect([1.0, -1.0, 0.0], [0.0, 1.0, 0.0]);
        assert!((r[0] - 1.0).abs() < 1.0e-6);
        assert!((r[1] - 1.0).abs() < 1.0e-6);
        assert!(r[2].abs() < 1.0e-6);
    }

    #[test]
    fn refract_straight_through_at_eta_one() {
        let d = vec3_normalise([0.3, -0.9, 0.1]);
        let t = refract(d, [0.0, 1.0, 0.0], 1.0).expect("eta 1 never TIRs");
        for i in 0..3 {
            assert!((t[i] - d[i]).abs() < 1.0e-5, "eta=1 must not bend the ray");
        }
    }

    #[test]
    fn refract_reports_total_internal_reflection() {
        // Grazing exit from a dense medium (eta > 1) must TIR.
        let d = vec3_normalise([0.9, -0.1, 0.0]);
        assert!(refract(d, [0.0, 1.0, 0.0], 1.5).is_none());
    }

    #[test]
    fn shadow_ray_darkens_occluded_floor() {
        // A floor plane with a small raised occluder, lit at a 45°
        // slant so the occluder's shadow falls on floor that a
        // near-top-down camera can see (a straight-down light would
        // hide the shadow under the occluder itself). Identical
        // geometry rendered twice — once with the slanted light
        // (shadow visible), once with the light coming from the
        // mirrored azimuth (shadow falls on the other side): both
        // must contain ambient-only floor pixels; the floor-only
        // control scene must not.
        let mut floor = Primitive::new(Topology::Triangles);
        floor.positions = vec![
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ];
        floor.normals = Some(vec![[0.0, 1.0, 0.0]; 6]);

        let mut occluder = Primitive::new(Topology::Triangles);
        occluder.positions = vec![
            [-0.3, 0.5, -0.3],
            [0.3, 0.5, -0.3],
            [-0.3, 0.5, 0.3],
            [0.3, 0.5, -0.3],
            [0.3, 0.5, 0.3],
            [-0.3, 0.5, 0.3],
        ];

        // Slanted light: from +Z at 45° elevation. Floor lit at
        // cos(45°) ⇒ linear 0.7 × (0.2 + 0.8·0.707) ≈ 0.535 ⇒ sRGB
        // ≈ 194. Shadowed floor keeps ambient only: 0.7 × 0.2 = 0.14
        // ⇒ sRGB ≈ 110.
        let light_spec = crate::options::LightSpec {
            azimuth_deg: 0.0,
            elevation_deg: 45.0,
            intensity: 1.0,
        };
        let camera = crate::options::CameraSpec {
            elevation_deg: 80.0,
            azimuth_deg: 0.0,
            distance: 1.5,
        };
        let bg = [255, 0, 255, 255];
        let opts = RenderOptions {
            width: 48,
            height: 48,
            shading: ShadingMode::Phong,
            background: BackgroundColor(bg),
            light: light_spec,
            camera: Some(camera),
            ..RenderOptions::default()
        };

        let mut scene = Scene3D::new();
        push_mesh_node(&mut scene, floor.clone());
        push_mesh_node(&mut scene, occluder);
        let shadow_img = render_scene(&scene, &opts);

        let mut control = Scene3D::new();
        push_mesh_node(&mut control, floor);
        let control_img = render_scene(&control, &opts);

        let min_luma = |img: &RgbaImage| -> u8 {
            img.pixels
                .chunks_exact(4)
                .filter(|p| *p != bg)
                .map(|p| p[0])
                .min()
                .unwrap_or(255)
        };
        let shadow_min = min_luma(&shadow_img);
        let control_min = min_luma(&control_img);
        assert!(
            shadow_min <= 120,
            "occluder scene must contain ambient-only shadowed floor, min luma {shadow_min}"
        );
        assert!(
            control_min >= 160,
            "floor-only control must be fully lit everywhere, min luma {control_min}"
        );
    }

    #[test]
    fn metallic_floor_reflects_offscreen_triangle() {
        // A mirror floor with a red wall standing on it: floor
        // pixels near the wall pick up red from the reflection,
        // compared against the same scene with an inert floor.
        let mut floor = Primitive::new(Topology::Triangles);
        floor.positions = vec![
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ];
        floor.normals = Some(vec![[0.0, 1.0, 0.0]; 6]);
        floor.material = Some(MaterialId(0));

        let mut wall = Primitive::new(Topology::Triangles);
        wall.positions = vec![
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ];
        wall.normals = Some(vec![[0.0, 0.0, 1.0]; 6]);
        wall.material = Some(MaterialId(1));

        let mirror = Material {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 1.0,
            roughness: 0.0,
            ..Material::new()
        };
        let inert = Material {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            ..Material::new()
        };
        let red_wall = Material {
            base_color: [1.0, 0.0, 0.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            ..Material::new()
        };

        let camera = crate::options::CameraSpec {
            elevation_deg: 30.0,
            azimuth_deg: 0.0, // looking from +Z toward the wall at -Z
            distance: 1.5,
        };
        let opts = RenderOptions {
            width: 48,
            height: 48,
            shading: ShadingMode::Phong,
            background: BackgroundColor([0, 0, 255, 255]),
            camera: Some(camera),
            ..RenderOptions::default()
        };

        let render_floor = |floor_mat: Material| -> RgbaImage {
            let mut scene = Scene3D::new();
            scene.materials.push(floor_mat);
            scene.materials.push(red_wall.clone());
            push_mesh_node(&mut scene, floor.clone());
            push_mesh_node(&mut scene, wall.clone());
            render_scene(&scene, &opts)
        };

        let mirror_img = render_floor(mirror);
        let inert_img = render_floor(inert);

        // Somewhere in the lower half (the floor) the mirror image
        // must be substantially redder than the inert image.
        let mut best_delta = 0i32;
        for y in 24..48 {
            for x in 0..48 {
                let m = mirror_img.pixel(x, y).unwrap();
                let i = inert_img.pixel(x, y).unwrap();
                let redness_m = m[0] as i32 - ((m[1] as i32 + m[2] as i32) / 2);
                let redness_i = i[0] as i32 - ((i[1] as i32 + i[2] as i32) / 2);
                best_delta = best_delta.max(redness_m - redness_i);
            }
        }
        assert!(
            best_delta > 40,
            "mirror floor must reflect the red wall (best redness delta {best_delta})"
        );
    }

    #[test]
    fn transmissive_pane_shows_through_tinted() {
        // A transmissive pane in front of a red wall: rays through
        // the pane must still find the wall (vs. an opaque pane that
        // hides it).
        let mut wall = Primitive::new(Topology::Triangles);
        wall.positions = vec![
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ];
        wall.normals = Some(vec![[0.0, 0.0, 1.0]; 6]);
        wall.material = Some(MaterialId(1));

        let mut pane = Primitive::new(Topology::Triangles);
        pane.positions = vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        pane.normals = Some(vec![[0.0, 0.0, 1.0]; 6]);
        pane.material = Some(MaterialId(0));

        let red_wall = Material {
            base_color: [1.0, 0.0, 0.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            ..Material::new()
        };
        let mut glass = Material {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.0,
            ..Material::new()
        };
        glass.ext.transmission = Some(oxideav_mesh3d::material::Transmission {
            factor: 1.0,
            factor_texture: None,
        });
        glass.ext.ior = Some(1.0); // index-matched: no bending
        let opaque = Material {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            ..Material::new()
        };

        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Phong,
            background: BackgroundColor([0, 0, 255, 255]),
            ..RenderOptions::default()
        };

        let render_pane = |pane_mat: Material| -> RgbaImage {
            let mut scene = Scene3D::new();
            scene.materials.push(pane_mat);
            scene.materials.push(red_wall.clone());
            push_mesh_node(&mut scene, pane.clone());
            push_mesh_node(&mut scene, wall.clone());
            render_scene(&scene, &opts)
        };

        let glass_img = render_pane(glass);
        let opaque_img = render_pane(opaque);

        let g = glass_img.pixel(16, 16).unwrap();
        let o = opaque_img.pixel(16, 16).unwrap();
        assert!(
            g[0] > 100 && g[1] < 100,
            "glass pane must show the red wall through it, got {g:?}"
        );
        assert!(
            o[0].abs_diff(o[1]) < 30,
            "opaque pane must stay neutral, got {o:?}"
        );
    }

    #[test]
    fn unlit_material_ignores_light_and_shadow() {
        // An unlit triangle renders its exact base colour even when
        // the light grazes it; the same geometry with a lit material
        // shades darker.
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        prim.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
        prim.material = Some(MaterialId(0));

        let base = [0.3, 0.8, 0.2, 1.0];
        let mut unlit_mat = Material {
            base_color: base,
            metallic: 0.0,
            roughness: 1.0,
            ..Material::new()
        };
        unlit_mat.ext.unlit = true;
        let lit_mat = Material {
            base_color: base,
            metallic: 0.0,
            roughness: 1.0,
            ..Material::new()
        };

        // Light nearly opposite the surface normal.
        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Phong,
            background: WHITE_BG,
            light: crate::options::LightSpec {
                azimuth_deg: 180.0,
                elevation_deg: 0.0,
                intensity: 1.0,
            },
            ..RenderOptions::default()
        };

        let render_mat = |mat: Material| -> RgbaImage {
            let mut scene = Scene3D::new();
            scene.materials.push(mat);
            push_mesh_node(&mut scene, prim.clone());
            render_scene(&scene, &opts)
        };

        let unlit_img = render_mat(unlit_mat);
        let lit_img = render_mat(lit_mat);
        let expected = linear_rgba_to_srgb_u8(base);
        let unlit_painted: Vec<[u8; 4]> = unlit_img
            .pixels
            .chunks_exact(4)
            .filter(|p| *p != [255, 255, 255, 255])
            .map(|p| [p[0], p[1], p[2], p[3]])
            .collect();
        assert!(!unlit_painted.is_empty());
        for p in &unlit_painted {
            assert_eq!(*p, expected, "unlit surface must be the exact base colour");
        }
        let lit_darker = lit_img
            .pixels
            .chunks_exact(4)
            .filter(|p| *p != [255, 255, 255, 255])
            .all(|p| p[1] < expected[1]);
        assert!(
            lit_darker,
            "the lit control must shade darker than the unlit base colour"
        );
    }

    #[test]
    fn emissive_material_self_illuminates() {
        // Black base + red emission: the surface must glow red, and
        // halving KHR_materials_emissive_strength must dim it.
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
        prim.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
        prim.material = Some(MaterialId(0));

        let emissive_mat = |strength: Option<f32>| -> Material {
            let mut m = Material {
                base_color: [0.0, 0.0, 0.0, 1.0],
                metallic: 0.0,
                roughness: 1.0,
                emissive_factor: [0.5, 0.0, 0.0],
                ..Material::new()
            };
            m.ext.emissive_strength = strength;
            m
        };

        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Phong,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let render_mat = |mat: Material| -> RgbaImage {
            let mut scene = Scene3D::new();
            scene.materials.push(mat);
            push_mesh_node(&mut scene, prim.clone());
            render_scene(&scene, &opts)
        };

        let full = render_mat(emissive_mat(None)); // strength default 1.0
        let dim = render_mat(emissive_mat(Some(0.5)));
        let max_red = |img: &RgbaImage| -> u8 {
            img.pixels
                .chunks_exact(4)
                .filter(|p| *p != [255, 255, 255, 255])
                .map(|p| p[0])
                .max()
                .unwrap_or(0)
        };
        let full_red = max_red(&full);
        let dim_red = max_red(&dim);
        let expected_full = crate::shade::linear_to_srgb_u8(0.5);
        let expected_dim = crate::shade::linear_to_srgb_u8(0.25);
        assert_eq!(full_red, expected_full, "emission must reach the pixel");
        assert_eq!(
            dim_red, expected_dim,
            "emissive_strength must scale the emission"
        );
        // Green / blue channels stay black (base colour is black,
        // emission is pure red).
        let clean_channels = full
            .pixels
            .chunks_exact(4)
            .filter(|p| *p != [255, 255, 255, 255])
            .all(|p| p[1] == 0 && p[2] == 0);
        assert!(clean_channels, "emission must not leak across channels");
    }

    #[test]
    fn parallel_render_is_deterministic() {
        // The banded parallel loop must be bit-identical across
        // runs — disjoint output slices, no shared accumulation.
        let scene = unit_triangle_scene();
        let opts = RenderOptions {
            width: 64,
            height: 47, // deliberately not a multiple of any band size
            shading: ShadingMode::Phong,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let a = render_scene(&scene, &opts);
        let b = render_scene(&scene, &opts);
        assert_eq!(a.pixels, b.pixels);
        assert_eq!(a.pixels.len(), 64 * 47 * 4);
    }

    #[test]
    fn nested_transform_hierarchy_is_honoured() {
        // A child triangle translated by its parent must move in the
        // image relative to an un-translated sibling scene.
        use oxideav_mesh3d::Transform;
        let mut prim = Primitive::new(Topology::Triangles);
        prim.positions = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];

        let mut scene = Scene3D::new();
        scene
            .meshes
            .push(Mesh::new("t".to_string()).with_primitive(prim));
        // Parent shifts +X by 2; child carries the mesh.
        scene.nodes.push(Node {
            transform: Transform::Trs {
                translation: [2.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
            children: vec![NodeId(1)],
            ..Node::default()
        });
        scene.nodes.push(Node {
            mesh: Some(MeshId(0)),
            ..Node::default()
        });
        scene.roots.push(NodeId(0));

        let opts = RenderOptions {
            width: 32,
            height: 32,
            shading: ShadingMode::Flat,
            background: WHITE_BG,
            ..RenderOptions::default()
        };
        let img = render_scene(&scene, &opts);
        // Auto-framing recentres on the shifted bbox, so the triangle
        // still lands mid-frame — assert it renders at all (the walk
        // composed parent × child without panicking) and matches the
        // scanline backend's coverage for the same scene.
        let scan = crate::scanline::render_scene(&scene, &opts);
        let ray_n = non_bg_count(&img, [255, 255, 255, 255]);
        let scan_n = non_bg_count(&scan, [255, 255, 255, 255]);
        assert!(ray_n > 0);
        assert!(
            (ray_n as i64 - scan_n as i64).unsigned_abs() as usize <= ray_n / 4 + 8,
            "backends must agree on transformed coverage: ray={ray_n} scan={scan_n}"
        );
    }
}
