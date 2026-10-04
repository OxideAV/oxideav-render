//! Unbiased Monte Carlo path tracer — the backend behind
//! [`crate::RenderBackend::PathTrace`], plus the progressive
//! [`PathTracer`] accumulator interactive viewers drive frame by frame.
//!
//! The renderer estimates the rendering equation (Kajiya, "The
//! Rendering Equation", SIGGRAPH 1986) with unidirectional path
//! tracing: next-event estimation (NEE) to every punctual light and to
//! one sampled emissive triangle / environment texel per vertex,
//! combined with BSDF sampling by multiple importance sampling (Veach,
//! "Robust Monte Carlo Methods for Light Transport Simulation", PhD
//! thesis, Stanford 1997, ch. 9), and Russian roulette (Arvo & Kirk,
//! "Particle Transport and Image Synthesis", SIGGRAPH 1990). Geometry,
//! hits, material inputs and ray offsets come from [`crate::trace`].
//!
//! This documentation is the **normative estimator specification**: a
//! GPU port that follows it (same sequences, same dimension layout,
//! same lobe / light probabilities) produces the same estimator — the
//! same expected value and the same per-sample noise structure. Only
//! floating-point rounding and BVH traversal order (which affects
//! nothing but the order filters are invoked in) differ.
//!
//! # 1. Pixel loop and frame contract
//!
//! Sample `s` (0-based, global across [`PathTracer::refine`] calls) of
//! pixel `(x, y)` traces the camera ray
//! [`Camera::primary_ray`]`(x + ξ₀, y + ξ₁, W, H)` (box pixel filter,
//! `ξ` from pattern 0 below). The camera is [`Camera::resolve`] at the
//! output resolution, identical to the other backends.
//!
//! Per pixel the accumulator keeps `Σ L` over *covered* samples
//! (primary ray hit geometry) and the covered / total sample counts.
//! The resolve follows the scanline backend's contract: uncovered
//! samples carry the background colour, covered ones carry their
//! radiance, and the two are combined premultiplied in display-linear
//! space — `c = (f·T(m)·1 + (1−f)·b·b_a) / (f + (1−f)·b_a)`,
//! `alpha = f + (1−f)·b_a`, with `f` the covered fraction, `m` the mean
//! radiance of covered samples, `T` exposure + tone map, `b` the
//! decoded background. A fully uncovered pixel therefore returns the
//! background bytes exactly. The one deliberate difference from the
//! scanline SSAA resolve: radiance is averaged *before* tone mapping
//! (a Monte Carlo estimate must be averaged in linear space).
//! [`PathTracer::hdr`] is the same mix without `T` (scene-linear).
//! [`RenderOptions::aa`] and [`RenderOptions::shading`] are ignored.
//!
//! # 2. Random numbers and dimension allocation
//!
//! * **Hash** — `pcg_hash(x)`: one PCG step + RXS-M-XS output
//!   permutation (O'Neill, "PCG: A Family of Simple Fast
//!   Space-Efficient Statistically Good Algorithms for Random Number
//!   Generation", 2014; the single-round hash form recommended by
//!   Jarzynski & Olano, "Hash Functions for GPU Rendering", JCGT 9(3),
//!   2020): `state = x·747796405 + 2891336453;
//!   word = ((state >> ((state >> 28) + 4)) ^ state)·277803737;
//!   return (word >> 22) ^ word` (wrapping `u32`).
//!   `mix(a, b) = pcg_hash(a ^ pcg_hash(b))`.
//! * **Seeds** — `pixel_seed = mix(mix(seed, y), x)` with
//!   `seed = PathTraceOptions::seed`;
//!   `pattern_seed(p) = mix(pixel_seed, p)`.
//! * **Low-discrepancy points** — 4-D Sobol' points (Sobol' 1967;
//!   direction numbers from Joe & Kuo, "Constructing Sobol sequences
//!   with better two-dimensional projections", SIAM J. Sci. Comput.
//!   30(5), 2008: dimension 0 is the van der Corput radical inverse,
//!   dimensions 1–3 use the primitive polynomials `(s, a, m) = (1, 0,
//!   [1])`, `(2, 1, [1, 3])`, `(3, 1, [1, 3, 1])`), Owen-scrambled by
//!   hashing (Burley, "Practical Hash-based Owen Scrambling", JCGT
//!   9(4), 2020). Point `s` of pattern `p`, dimension `d ∈ 0..4`:
//!   `i = nested_uniform_scramble(s, pattern_seed(p))`,
//!   `v = nested_uniform_scramble(sobol(i, d), mix(pattern_seed(p), d))`,
//!   `ξ = (v >> 8)·2⁻²⁴`, where `nested_uniform_scramble(x, k) =
//!   reverse_bits(lk(reverse_bits(x), k))` and `lk` is Burley's
//!   Laine-Karras-style hash `x += k; x ^= x·0x6c50b47c; x ^=
//!   x·0xb82f1e52; x ^= x·0xc7afe638; x ^= x·0x8d22f6e6`. Each pattern
//!   shuffles its own point order ("padding" of independent 4-D
//!   sets, Burley §4).
//! * **Patterns** — pattern `0`: `[pixel ξ₀, pixel ξ₁, –, –]`. At path
//!   vertex `k` (0 = first hit): pattern `1 + 2k` =
//!   `[bsdf u₀, bsdf u₁, lobe select, roulette]`, pattern `2 + 2k` =
//!   `[light u₀, light u₁, light select, –]`.
//! * **Stochastic transparency** — the any-hit coin for candidate
//!   triangle `g` on ray `r` is `(mix(mix(path_seed, r), g) >> 8)·2⁻²⁴`
//!   with `path_seed = mix(pixel_seed, s)`; ray ids are `r = k << 16`
//!   for the ray that finds vertex `k`, `(k << 16) | 0x8000 | j` for
//!   the shadow ray to punctual light `j` from vertex `k`, and
//!   `(k << 16) | 0xffff` for the area / environment shadow ray. The
//!   coin is a pure function of `(path, ray, triangle)`, independent
//!   of traversal order.
//!
//! # 3. Path construction
//!
//! Vertex `k` is the `k`-th surface hit (`k = 0` is the camera hit),
//! `β` the path throughput (starts at 1).
//!
//! 1. **Intersect** — closest hit accepting a candidate unless: the
//!    material is `MASK` and alpha < cutoff; the material is `BLEND`
//!    and the coin ≥ alpha (pass-through probability `1 − alpha`); or
//!    `k = 0`, the hit is a back face and the material is
//!    single-sided (rasteriser-compatible culling — secondary rays see
//!    both sides of every surface). Alpha = base-colour factor ×
//!    texture (level 0) × `COLOR_0`.
//! 2. **Medium** — inside a `KHR_materials_volume` medium the segment
//!    multiplies `β` by `exp(−σ·t)`, `σ = −ln(attenuationColor) /
//!    attenuationDistance` (Beer-Lambert). A single medium is tracked
//!    (entered on refraction through a front face, left through a back
//!    face).
//! 3. **Miss** — at `k = 0`: the sample is *uncovered* (background).
//!    Otherwise add `β·L_env(d)·w`, where `L_env` is the environment
//!    map (when installed, [`PathTracer::set_environment`]) or the
//!    constant sky radiance [`RenderOptions::ambient`]. The constant
//!    sky is reached by BSDF sampling only (`w = 1`); the environment
//!    map uses `w` from (6). Terminate.
//! 4. **Emission** — on a front-face hit (or any hit of a double-sided
//!    material) add `β·Le·w`: `Le = emissiveFactor × emissiveTexture
//!    × strength`; `w = 1` at `k = 0`, otherwise the MIS weight of
//!    (6) against the light-sampling pdf of that triangle.
//!    `KHR_materials_unlit` surfaces add `β·baseColor` and terminate.
//! 5. **Depth** — terminate when `k = max_bounces`.
//! 6. **NEE** (shading frame of §4) —
//!    * every punctual light `j` (`KHR_lights_punctual`, delta):
//!      `β·f(v, l)·|n·l|·E_j(p)·V(p, l)` with `E_j` from
//!      [`PreparedLight::sample`] — weight 1 (cannot be hit);
//!    * one sample of the *area set*: with an environment map **and**
//!      emissive triangles the environment is chosen with probability
//!      `P_env = ½` (light-select `ξ < ½`, rescaled `ξ' = 2ξ` / `2ξ−1`
//!      for the next choice); with only one of them it gets
//!      probability 1. An emissive triangle `t` is picked from the
//!      discrete CDF of its power `Φ_t = A_t · lum(Le_t) · (2 if
//!      double-sided else 1)` (`lum` = Rec. 709 luminance; textured
//!      emission uses the texture's mean, i.e. its 1×1 mip). With
//!      `Ω` = the triangle's solid angle from the vertex position `p`
//!      ([`triangle_solid_angle`], Van Oosterom & Strackee 1983):
//!      if `Ω ≥` [`MIN_SPHERICAL_SOLID_ANGLE`] the direction is drawn
//!      uniformly over the spherical triangle ([`sample_spherical_triangle`],
//!      Arvo, "Stratified Sampling of Spherical Triangles", SIGGRAPH
//!      1995, with light `u₀` → sub-triangle area, `u₁` → arc
//!      position) and the light point is that ray's hit on the
//!      triangle's plane, pdf `p_L = P_area · P_t / Ω`; otherwise a
//!      point is drawn uniformly by area — barycentrics
//!      `(1 − √u₀, √u₀(1 − u₁), √u₀·u₁)` (Turk, "Generating Random
//!      Points in Triangles", Graphics Gems 1990) — with
//!      `p_L = P_area · P_t · d² / (A_t · |cos θ_L|)`. The same
//!      `Ω`-threshold decision, evaluated from the previous vertex
//!      position, gives `p_L` for emitters found by BSDF rays. The
//!      environment
//!      map is sampled from its luminance × `sin θ` 2-D CDF (§5).
//!      Contribution `β·f·|n·l|·Le·w_L / p_L`, with the power heuristic
//!      `w_L = p_L² / (p_L² + p_B²)` (`p_B` = the BSDF mixture pdf of
//!      §4 for `l`). [`LightStrategy::LightOnly`] uses `w_L = 1` and
//!      drops BSDF-found emission; [`LightStrategy::BsdfOnly`] skips
//!      this step and uses weight 1 on hits.
//! 7. **BSDF sample** — choose a lobe (§4), sample `l`, set
//!    `β ← β·f(v, l)·|n·l| / p_B(l)`, remember `p_B` for the MIS
//!    weight `w_B = p_B² / (p_B² + p_L²)` at the next emitter.
//! 8. **Roulette** — for the vertex index `k + 1 ≥ rr_start`: survive
//!    with `q = clamp(max(β), 0.05, 1)` (roulette dimension `ξ < q`),
//!    then `β ← β / q`.
//! 9. Offset the origin with [`crate::trace::offset_ray_origin`] to
//!    the side of the geometric normal `l` leaves on; continue at 1.
//!
//! Every contribution except camera-visible emission (`k = 0`, step 4)
//! is clamped so `max(rgb) ≤ clamp` when [`PathTraceOptions::clamp`]
//! is non-zero (biased).
//!
//! # 4. BSDF
//!
//! The glTF 2.0 metallic-roughness model (Appendix B) with the
//! KHR extension layering formulas, all lobes GGX / Trowbridge-Reitz
//! (Walter, Marschner, Li, Torrance, "Microfacet Models for Refraction
//! through Rough Surfaces", EGSR 2007) with `α = roughness²` (min
//! `1e-3`) and height-correlated Smith masking
//! `G₂ = 1 / (1 + Λ(v) + Λ(l))`, `Λ(ω) = (√(α² + (1−α²)cos²θ)/cosθ − 1)/2`
//! (Heitz, "Understanding the Masking-Shadowing Function", JCGT 3(2),
//! 2014). Frame: `n` is the (normal-mapped) shading normal flipped to
//! the incoming side (for double-sided back faces the interpolated
//! normal is reversed *before* the normal map applies, glTF §3.9.3); `n_g` the geometric normal likewise; if
//! `n·v ≤ 0` the shading normal is replaced by `n_g`. A direction is a
//! *reflection* when `n_g·l > 0`, a *transmission* otherwise.
//!
//! Terms (`m` metallic, `s` = `KHR_materials_specular` strength, `F₀ᵈ =
//! min(((η−1)/(η+1))² · specularColor, 1)`, `F_d(h) = Schlick(F₀ᵈ,
//! v·h)`; when leaving a volume (`η_i > η_t`) Schlick is evaluated at
//! the transmitted cosine and is 1 under total internal reflection;
//! `F_m(h) = Schlick(baseColor, v·h)`, `t` transmission):
//!
//! * specular reflection `D·G₂/(4|n·v||n·l|) · ((1−m)·s·F_d + m·F_m)`;
//! * diffuse `(1−m)(1−t)(1 − s·max(F_d)) · baseColor/π`;
//! * transmission `(1−m)·t·baseColor·(1 − s·max(F_d)) · BTDF` — thin
//!   walls (volume thickness 0): the reflection lobe mirrored through
//!   the shading plane (`l' = l − 2(l·n)n`, value `D·G₂/(4|n·v||n·l'|)`
//!   at `h = norm(v + l')`); volumes: Walter 2007 eq. 21 with
//!   `h = ±norm(η_i v + η_t l)`, `|v·h||l·h| η_t² D G₂ / (|n·v||n·l|
//!   (η_i v·h + η_t l·h)²)` — the `η²` radiance compression is *not*
//!   applied (exact when camera and lights share a medium);
//! * sheen (`KHR_materials_sheen`): Charlie `D` (Estevez & Kulla,
//!   "Production Friendly Microfacet Sheen BRDF", 2017) with the
//!   Neubelt-Pettineo visibility `1/(4(n·l + n·v − n·l·n·v))`, and the
//!   base layers scaled by `min(1 − max(sheenColor)·E(n·v), 1 −
//!   max(sheenColor)·E(n·l))`, `E` the sheen directional albedo
//!   (precomputed by quadrature);
//! * clearcoat (`KHR_materials_clearcoat`): `F_c = cc·Schlick(0.04,
//!   n_c·v)`; `f = (1 − F_c)·f_base + F_c·D_c G₂_c/(4|n_c·v||n_c·l|)`
//!   with its own normal and roughness.
//!
//! Inside a volume (back face of a material with thickness > 0) the
//! sheen and clearcoat layers are disabled.
//! The occlusion texture is ignored: the path tracer computes
//! occlusion itself (glTF §3.9.4 defines it as a hint for indirect
//! lighting approximations).
//!
//! **Lobe selection** (from `v` only, `c_v = n·v`):
//! `F̄_d = s·max(F_d at c_v)`, `F̄_m = mean(Schlick(baseColor, c_v))`,
//! `w_cc = F_c`, `S = (1 − w_cc)(1 − max(sheenColor)·E(c_v))`,
//! `p ∝ [diffuse: S(1−m)(1−t)(1−F̄_d)·mean(baseColor),
//! specular: S((1−m)F̄_d + m F̄_m), transmission: S(1−m)t(1−F̄_d)·mean(baseColor),
//! clearcoat: w_cc, sheen: (1 − w_cc)·max(sheenColor)·E(c_v)]`,
//! normalised; the CDF is walked in that order with the lobe-select
//! dimension. Diffuse and sheen sample the cosine-weighted hemisphere
//! about `n` (Shirley-Chiu concentric map, then Malley's projection);
//! specular, transmission and clearcoat sample GGX visible normals
//! (Heitz, "Sampling the GGX Distribution of Visible Normals", JCGT
//! 7(4), 2018) — reflect for specular / clearcoat, mirror (thin) or
//! refract (volume) for transmission. The returned pdf is always the
//! full **mixture** `p_B(l) = Σ_lobes p_i·pdf_i(l)` and the weight
//! `f(v,l)|n·l| / p_B(l)` uses the full BSDF (one-sample MIS over the
//! lobes, balance heuristic). VNDF pdfs: reflection
//! `G₁(v)D(h)/(4 n·v)`; refraction `G₁(v)(v·h)D(h)/(n·v) ·
//! η_t²|l·h|/(η_i v·h + η_t l·h)²`. Tangent frames follow Duff et al.,
//! "Building an Orthonormal Basis, Revisited", JCGT 6(1), 2017.
//!
//! # 5. Environment map
//!
//! [`EnvironmentMap`]: an equirectangular [`HdrImage`] (`u = ½ +
//! atan2(d_x, −d_z)/2π`, `v = acos(d_y)/π`, so the image centre looks
//! down `−Z` and the top row is `+Y`), nearest-texel lookup × an
//! intensity. Importance sampling uses the piecewise-constant 2-D
//! distribution `w_ij = lum(texel)·sin θ_j` (marginal over rows,
//! conditional within a row; Pharr, Jakob, Humphreys, *Physically
//! Based Rendering* 3rd ed., §13.6.7 / §14.2.4), pdf
//! `p(ω) = p(u, v) / (2π² sin θ)`.

use std::f32::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

use oxideav_mesh3d::{AlphaMode, Scene3D};

use crate::camera::Camera;
use crate::hdr::{linear_to_srgb_byte, srgb_u8_lut, HdrImage};
use crate::image::RgbaImage;
use crate::math::{vec3_cross, vec3_dot, vec3_normalise};
use crate::options::{LightStrategy, PathTraceOptions, RenderOptions};
use crate::prepare::{PrepareOptions, PreparedLight, PreparedScene};
use crate::texture::{ColorSpace, TextureCache, TextureResolver};
use crate::trace::{interp2, offset_ray_origin, MaterialSample, TexLod, TraceHit, TraceScene};

// =====================================================================
// Sampling primitives.
// =====================================================================

/// PCG-based 32-bit hash (O'Neill 2014; Jarzynski & Olano 2020).
#[inline]
pub fn pcg_hash(x: u32) -> u32 {
    let state = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// `pcg_hash(a ^ pcg_hash(b))` — order-sensitive seed combination.
#[inline]
pub fn hash_mix(a: u32, b: u32) -> u32 {
    pcg_hash(a ^ pcg_hash(b))
}

/// Sobol' generator matrices (direction numbers) for dimensions 0–3.
fn sobol_matrices() -> &'static [[u32; 32]; 4] {
    static M: OnceLock<[[u32; 32]; 4]> = OnceLock::new();
    M.get_or_init(|| {
        let mut out = [[0u32; 32]; 4];
        for (k, v) in out[0].iter_mut().enumerate() {
            *v = 1u32 << (31 - k);
        }
        // (degree s, coefficients a, initial m) — Joe & Kuo 2008.
        let polys: [(usize, u32, &[u32]); 3] = [(1, 0, &[1]), (2, 1, &[1, 3]), (3, 1, &[1, 3, 1])];
        for (d, (s, a, init)) in polys.iter().enumerate() {
            let mut m = [0u32; 32];
            m[..*s].copy_from_slice(init);
            for k in *s..32 {
                let mut v = m[k - s] ^ (m[k - s] << s);
                for j in 1..*s {
                    if (a >> (s - 1 - j)) & 1 == 1 {
                        v ^= m[k - j] << j;
                    }
                }
                m[k] = v;
            }
            for k in 0..32 {
                out[d + 1][k] = m[k] << (31 - k);
            }
        }
        out
    })
}

#[inline]
fn sobol(index: u32, dim: usize) -> u32 {
    let m = &sobol_matrices()[dim];
    let mut r = 0u32;
    let mut i = index;
    let mut k = 0;
    while i != 0 {
        if i & 1 == 1 {
            r ^= m[k];
        }
        i >>= 1;
        k += 1;
    }
    r
}

#[inline]
fn laine_karras(mut x: u32, seed: u32) -> u32 {
    x = x.wrapping_add(seed);
    x ^= x.wrapping_mul(0x6c50_b47c);
    x ^= x.wrapping_mul(0xb82f_1e52);
    x ^= x.wrapping_mul(0xc7af_e638);
    x ^= x.wrapping_mul(0x8d22_f6e6);
    x
}

/// Hash-based nested uniform (Owen) scramble (Burley 2020).
#[inline]
fn nested_uniform_scramble(x: u32, seed: u32) -> u32 {
    laine_karras(x.reverse_bits(), seed).reverse_bits()
}

#[inline]
fn to_unit(v: u32) -> f32 {
    (v >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Point `index` of the Owen-scrambled 4-D Sobol' pattern seeded by
/// `pattern_seed` (§2 of the module docs).
#[inline]
pub fn sobol_owen_4d(index: u32, pattern_seed: u32) -> [f32; 4] {
    let i = nested_uniform_scramble(index, pattern_seed);
    let mut out = [0.0; 4];
    for (d, o) in out.iter_mut().enumerate() {
        *o = to_unit(nested_uniform_scramble(
            sobol(i, d),
            hash_mix(pattern_seed, d as u32),
        ));
    }
    out
}

/// Per-sample sequence handle.
#[derive(Clone, Copy)]
struct Sampler {
    pixel_seed: u32,
    index: u32,
    path_seed: u32,
}

impl Sampler {
    fn new(seed: u32, x: u32, y: u32, index: u32) -> Self {
        let pixel_seed = hash_mix(hash_mix(seed, y), x);
        Self {
            pixel_seed,
            index,
            path_seed: hash_mix(pixel_seed, index),
        }
    }

    fn pattern(&self, p: u32) -> [f32; 4] {
        sobol_owen_4d(self.index, hash_mix(self.pixel_seed, p))
    }

    fn coin(&self, ray: u32, tri: u32) -> f32 {
        to_unit(hash_mix(hash_mix(self.path_seed, ray), tri))
    }
}

// =====================================================================
// Vector helpers.
// =====================================================================

#[inline]
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn mul(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}
#[inline]
fn neg(a: [f32; 3]) -> [f32; 3] {
    [-a[0], -a[1], -a[2]]
}
#[inline]
fn max3(a: [f32; 3]) -> f32 {
    a[0].max(a[1]).max(a[2])
}
#[inline]
fn mean3(a: [f32; 3]) -> f32 {
    (a[0] + a[1] + a[2]) * (1.0 / 3.0)
}
#[inline]
fn luminance(a: [f32; 3]) -> f32 {
    0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2]
}
#[inline]
fn is_black(a: [f32; 3]) -> bool {
    !(a[0] > 0.0 || a[1] > 0.0 || a[2] > 0.0)
}
#[inline]
fn finite3(a: [f32; 3]) -> bool {
    a[0].is_finite() && a[1].is_finite() && a[2].is_finite()
}

/// Orthonormal basis `(t, b)` around unit `n` (Duff et al. 2017).
#[inline]
fn basis(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let sign = 1.0f32.copysign(n[2]);
    let a = -1.0 / (sign + n[2]);
    let b = n[0] * n[1] * a;
    (
        [1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]],
        [b, sign + n[1] * n[1] * a, -n[1]],
    )
}

#[inline]
fn to_world(t: [f32; 3], b: [f32; 3], n: [f32; 3], v: [f32; 3]) -> [f32; 3] {
    [
        t[0] * v[0] + b[0] * v[1] + n[0] * v[2],
        t[1] * v[0] + b[1] * v[1] + n[1] * v[2],
        t[2] * v[0] + b[2] * v[1] + n[2] * v[2],
    ]
}

/// Cosine-weighted hemisphere direction about `+Z` (Shirley & Chiu
/// 1997 concentric disk, Malley projection).
fn cosine_hemisphere(u0: f32, u1: f32) -> [f32; 3] {
    let a = 2.0 * u0 - 1.0;
    let b = 2.0 * u1 - 1.0;
    let (r, phi) = if a == 0.0 && b == 0.0 {
        (0.0, 0.0)
    } else if a.abs() > b.abs() {
        (a, std::f32::consts::FRAC_PI_4 * (b / a))
    } else {
        (
            b,
            std::f32::consts::FRAC_PI_2 - std::f32::consts::FRAC_PI_4 * (a / b),
        )
    };
    let (x, y) = (r * phi.cos(), r * phi.sin());
    [x, y, (1.0 - x * x - y * y).max(0.0).sqrt()]
}

// =====================================================================
// Microfacet terms.
// =====================================================================

/// Smallest GGX `α` (mirror-like surfaces stay non-delta).
const MIN_ALPHA: f32 = 1.0e-3;

#[inline]
fn d_ggx(n_dot_h: f32, a: f32) -> f32 {
    if n_dot_h <= 0.0 {
        return 0.0;
    }
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d)
}

#[inline]
fn lambda(cos: f32, a: f32) -> f32 {
    let c = cos.abs().max(1.0e-7);
    let a2 = a * a;
    ((a2 + (1.0 - a2) * c * c).sqrt() / c - 1.0) * 0.5
}

#[inline]
fn g1(cos: f32, a: f32) -> f32 {
    1.0 / (1.0 + lambda(cos, a))
}

#[inline]
fn g2(cv: f32, cl: f32, a: f32) -> f32 {
    1.0 / (1.0 + lambda(cv, a) + lambda(cl, a))
}

#[inline]
fn schlick(f0: [f32; 3], cos: f32) -> [f32; 3] {
    let k = (1.0 - cos.clamp(0.0, 1.0)).powi(5);
    [
        f0[0] + (1.0 - f0[0]) * k,
        f0[1] + (1.0 - f0[1]) * k,
        f0[2] + (1.0 - f0[2]) * k,
    ]
}

/// GGX visible-normal sample in the local frame (`+Z` = normal),
/// Heitz 2018 (JCGT 7(4)), isotropic `α`.
fn sample_vndf(v: [f32; 3], a: f32, u0: f32, u1: f32) -> [f32; 3] {
    let vh = vec3_normalise([a * v[0], a * v[1], v[2]]);
    let lensq = vh[0] * vh[0] + vh[1] * vh[1];
    let t1 = if lensq > 0.0 {
        let il = 1.0 / lensq.sqrt();
        [-vh[1] * il, vh[0] * il, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let t2 = vec3_cross(vh, t1);
    let r = u0.sqrt();
    let phi = 2.0 * PI * u1;
    let p1 = r * phi.cos();
    let mut p2 = r * phi.sin();
    let s = 0.5 * (1.0 + vh[2]);
    p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2;
    let pz = (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
    let nh = [
        p1 * t1[0] + p2 * t2[0] + pz * vh[0],
        p1 * t1[1] + p2 * t2[1] + pz * vh[1],
        p1 * t1[2] + p2 * t2[2] + pz * vh[2],
    ];
    vec3_normalise([a * nh[0], a * nh[1], nh[2].max(1.0e-6)])
}

/// Reflection pdf (solid angle of `l`) of VNDF sampling about `n`.
#[inline]
fn vndf_reflect_pdf(n: [f32; 3], v: [f32; 3], h: [f32; 3], a: f32) -> f32 {
    let nv = vec3_dot(n, v);
    let nh = vec3_dot(n, h);
    if nv <= 0.0 || nh <= 0.0 || vec3_dot(v, h) <= 0.0 {
        return 0.0;
    }
    g1(nv, a) * d_ggx(nh, a) / (4.0 * nv)
}

#[inline]
fn reflect(v: [f32; 3], h: [f32; 3]) -> [f32; 3] {
    let d = 2.0 * vec3_dot(v, h);
    [d * h[0] - v[0], d * h[1] - v[1], d * h[2] - v[2]]
}

/// Refract `v` (pointing away from the surface, `v·h > 0`) through the
/// microfacet `h` with relative index `eta = η_i / η_t`.
fn refract(v: [f32; 3], h: [f32; 3], eta: f32) -> Option<[f32; 3]> {
    let ci = vec3_dot(v, h);
    let s2 = eta * eta * (1.0 - ci * ci).max(0.0);
    if s2 >= 1.0 {
        return None;
    }
    let ct = (1.0 - s2).sqrt();
    let k = eta * ci - ct;
    Some(vec3_normalise([
        -eta * v[0] + k * h[0],
        -eta * v[1] + k * h[1],
        -eta * v[2] + k * h[2],
    ]))
}

// ---------------------------------------------------------------------
// Sheen (Charlie distribution + Neubelt-Pettineo visibility).
// ---------------------------------------------------------------------

#[inline]
fn d_charlie(n_dot_h: f32, a: f32) -> f32 {
    let inv = 1.0 / a.max(1.0e-3);
    let sin2 = (1.0 - n_dot_h * n_dot_h).max(0.0);
    (2.0 + inv) * sin2.powf(inv * 0.5) / (2.0 * PI)
}

#[inline]
fn v_neubelt(nl: f32, nv: f32) -> f32 {
    1.0 / (4.0 * (nl + nv - nl * nv)).max(1.0e-6)
}

const SHEEN_LUT_COS: usize = 32;
const SHEEN_LUT_ROUGH: usize = 16;

/// Directional albedo `E(cos θ_v, roughness)` of the white sheen lobe,
/// tabulated by midpoint quadrature on first use.
fn sheen_lut() -> &'static [f32] {
    static LUT: OnceLock<Vec<f32>> = OnceLock::new();
    LUT.get_or_init(|| {
        let (nt, np) = (48usize, 24usize);
        let mut out = vec![0.0f32; SHEEN_LUT_COS * SHEEN_LUT_ROUGH];
        for ri in 0..SHEEN_LUT_ROUGH {
            let r = ri as f32 / (SHEEN_LUT_ROUGH - 1) as f32;
            let a = (r * r).max(1.0e-3);
            for ci in 0..SHEEN_LUT_COS {
                let cv = ((ci as f32 + 0.5) / SHEEN_LUT_COS as f32).max(1.0e-3);
                let v = [(1.0 - cv * cv).sqrt(), 0.0, cv];
                let mut sum = 0.0f32;
                for i in 0..nt {
                    let th = (i as f32 + 0.5) / nt as f32 * std::f32::consts::FRAC_PI_2;
                    let (st, ct) = th.sin_cos();
                    for j in 0..np {
                        let ph = (j as f32 + 0.5) / np as f32 * PI; // symmetric half
                        let l = [st * ph.cos(), st * ph.sin(), ct];
                        let h = vec3_normalise(add(v, l));
                        let f = d_charlie(h[2], a) * v_neubelt(ct, cv);
                        sum += f * ct * st;
                    }
                }
                let dw = (std::f32::consts::FRAC_PI_2 / nt as f32) * (PI / np as f32) * 2.0;
                out[ri * SHEEN_LUT_COS + ci] = (sum * dw).clamp(0.0, 1.0);
            }
        }
        out
    })
}

fn sheen_albedo(cos: f32, rough: f32) -> f32 {
    let lut = sheen_lut();
    let x =
        (cos.clamp(0.0, 1.0) * SHEEN_LUT_COS as f32 - 0.5).clamp(0.0, (SHEEN_LUT_COS - 1) as f32);
    let y = (rough.clamp(0.0, 1.0) * (SHEEN_LUT_ROUGH - 1) as f32)
        .clamp(0.0, (SHEEN_LUT_ROUGH - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = (
        (x0 + 1).min(SHEEN_LUT_COS - 1),
        (y0 + 1).min(SHEEN_LUT_ROUGH - 1),
    );
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let at = |xx: usize, yy: usize| lut[yy * SHEEN_LUT_COS + xx];
    let a = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
    let b = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
    a + (b - a) * fy
}

// =====================================================================
// BSDF.
// =====================================================================

const LOBE_DIFFUSE: usize = 0;
const LOBE_SPECULAR: usize = 1;
const LOBE_TRANSMISSION: usize = 2;
const LOBE_CLEARCOAT: usize = 3;
const LOBE_SHEEN: usize = 4;

/// The layered glTF BSDF at one shading point (§4 of the module
/// docs). `v` is fixed at construction.
#[derive(Debug, Clone)]
pub struct Bsdf {
    v: [f32; 3],
    n: [f32; 3],
    ng: [f32; 3],
    ncc: [f32; 3],
    base: [f32; 3],
    metallic: f32,
    alpha: f32,
    specular: f32,
    f0d: [f32; 3],
    transmission: f32,
    thin: bool,
    eta_i: f32,
    eta_t: f32,
    clearcoat: f32,
    cc_alpha: f32,
    sheen: [f32; 3],
    sheen_rough: f32,
    sheen_alpha: f32,
    /// `F_c` at `v` (clearcoat weight).
    w_cc: f32,
    /// Lobe-selection probabilities (diffuse, specular, transmission,
    /// clearcoat, sheen).
    probs: [f32; 5],
}

impl Bsdf {
    /// Build from material inputs. `v` points away from the surface
    /// toward the previous vertex; `ng` is the CCW-front geometric
    /// normal; `front` whether the ray hit the front side.
    pub fn new(m: &MaterialSample, v: [f32; 3], ng: [f32; 3], front: bool) -> Self {
        let flip = |a: [f32; 3]| if front { a } else { neg(a) };
        let ng = flip(ng);
        let mut n = flip(m.normal);
        if vec3_dot(n, v) <= 1.0e-4 {
            n = ng;
        }
        let mut ncc = flip(m.clearcoat_normal);
        if vec3_dot(ncc, v) <= 1.0e-4 {
            ncc = n;
        }
        let thin = m.thickness <= 0.0;
        let ior = if m.ior.is_finite() && m.ior >= 1.0 {
            m.ior
        } else {
            1.5
        };
        let (eta_i, eta_t) = if thin || front {
            (1.0, ior)
        } else {
            (ior, 1.0)
        };
        let inside = !thin && !front;
        let r0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
        let base = [m.base_color[0], m.base_color[1], m.base_color[2]];
        let mut b = Self {
            v,
            n,
            ng,
            ncc,
            base,
            metallic: m.metallic,
            alpha: (m.roughness * m.roughness).max(MIN_ALPHA),
            specular: m.specular.clamp(0.0, 1.0),
            f0d: [
                (r0 * m.specular_color[0]).min(1.0),
                (r0 * m.specular_color[1]).min(1.0),
                (r0 * m.specular_color[2]).min(1.0),
            ],
            transmission: m.transmission,
            thin,
            eta_i,
            eta_t,
            clearcoat: if inside { 0.0 } else { m.clearcoat },
            cc_alpha: (m.clearcoat_roughness * m.clearcoat_roughness).max(MIN_ALPHA),
            sheen: if inside { [0.0; 3] } else { m.sheen_color },
            sheen_rough: m.sheen_roughness,
            sheen_alpha: (m.sheen_roughness * m.sheen_roughness).max(1.0e-3),
            w_cc: 0.0,
            probs: [0.0; 5],
        };
        let cv = vec3_dot(n, v).max(1.0e-4);
        b.w_cc = if b.clearcoat > 0.0 {
            b.clearcoat * schlick([0.04; 3], vec3_dot(ncc, v))[0]
        } else {
            0.0
        };
        let sheen_max = max3(b.sheen);
        let e_v = if sheen_max > 0.0 {
            sheen_albedo(cv, b.sheen_rough)
        } else {
            0.0
        };
        let s_scale = (1.0 - b.w_cc) * (1.0 - sheen_max * e_v);
        let fd = b.specular * max3(b.fresnel_d(cv));
        let fm = mean3(schlick(base, cv));
        let m_ = b.metallic;
        let base_mean = mean3(base).max(0.0);
        let mut p = [
            s_scale * (1.0 - m_) * (1.0 - b.transmission) * (1.0 - fd) * base_mean,
            s_scale * ((1.0 - m_) * fd + m_ * fm),
            s_scale * (1.0 - m_) * b.transmission * (1.0 - fd) * base_mean,
            b.w_cc,
            (1.0 - b.w_cc) * sheen_max * e_v,
        ];
        let sum: f32 = p.iter().sum();
        if sum > 0.0 && sum.is_finite() {
            for x in &mut p {
                *x /= sum;
            }
        } else {
            p = [0.0; 5];
        }
        b.probs = p;
        b
    }

    /// `true` when no lobe can scatter (black surface).
    pub fn is_black(&self) -> bool {
        self.probs.iter().all(|&p| p <= 0.0)
    }

    /// Lobe-selection probabilities (diffuse, specular, transmission,
    /// clearcoat, sheen).
    pub fn lobe_probabilities(&self) -> [f32; 5] {
        self.probs
    }

    /// Unit geometric normal on the incoming side.
    pub fn geometric_normal(&self) -> [f32; 3] {
        self.ng
    }

    /// Dielectric Fresnel at microfacet cosine `c` (per channel,
    /// before the specular strength). Under total internal reflection
    /// it is 1.
    fn fresnel_d(&self, c: f32) -> [f32; 3] {
        if self.eta_i > self.eta_t {
            let eta = self.eta_i / self.eta_t;
            let s2 = eta * eta * (1.0 - c * c).max(0.0);
            if s2 >= 1.0 {
                return [1.0; 3];
            }
            return schlick(self.f0d, (1.0 - s2).sqrt());
        }
        schlick(self.f0d, c)
    }

    /// `f(v, l)·|n·l|` and the mixture sampling pdf of `l`.
    pub fn eval(&self, l: [f32; 3]) -> ([f32; 3], f32) {
        let v = self.v;
        let n = self.n;
        let nv = vec3_dot(n, v).max(1.0e-4);
        let p = &self.probs;
        if vec3_dot(self.ng, l) > 0.0 {
            // ---- Reflection.
            let nl = vec3_dot(n, l);
            let h = vec3_normalise(add(v, l));
            let vh = vec3_dot(v, h).max(0.0);
            let nh = vec3_dot(n, h);
            let mut f = [0.0; 3];
            let mut pdf = 0.0;
            if nl > 0.0 {
                let fd = self.fresnel_d(vh);
                let fm = schlick(self.base, vh);
                let spec_c = d_ggx(nh, self.alpha) * g2(nv, nl, self.alpha) / (4.0 * nv * nl);
                let m = self.metallic;
                let diff_k =
                    (1.0 - m) * (1.0 - self.transmission) * (1.0 - self.specular * max3(fd)) / PI;
                let mut layer = [0.0; 3];
                for k in 0..3 {
                    layer[k] = spec_c * ((1.0 - m) * self.specular * fd[k] + m * fm[k])
                        + diff_k * self.base[k];
                }
                let smax = max3(self.sheen);
                if smax > 0.0 {
                    let scale = (1.0 - smax * sheen_albedo(nv, self.sheen_rough))
                        .min(1.0 - smax * sheen_albedo(nl, self.sheen_rough));
                    let sh = d_charlie(nh, self.sheen_alpha) * v_neubelt(nl, nv);
                    for (lk, sk) in layer.iter_mut().zip(self.sheen) {
                        *lk = *lk * scale + sk * sh;
                    }
                }
                for k in 0..3 {
                    f[k] = layer[k] * (1.0 - self.w_cc);
                }
                pdf += (p[LOBE_DIFFUSE] + p[LOBE_SHEEN]) * nl / PI;
                pdf += p[LOBE_SPECULAR] * vndf_reflect_pdf(n, v, h, self.alpha);
            }
            if self.clearcoat > 0.0 {
                let cnv = vec3_dot(self.ncc, v).max(1.0e-4);
                let cnl = vec3_dot(self.ncc, l);
                if cnl > 0.0 {
                    let cnh = vec3_dot(self.ncc, h);
                    let c = self.w_cc * d_ggx(cnh, self.cc_alpha) * g2(cnv, cnl, self.cc_alpha)
                        / (4.0 * cnv * cnl);
                    for fk in &mut f {
                        *fk += c;
                    }
                    pdf += p[LOBE_CLEARCOAT] * vndf_reflect_pdf(self.ncc, v, h, self.cc_alpha);
                }
            }
            let cos = nl.max(0.0);
            (scale(f, cos), pdf)
        } else {
            // ---- Transmission.
            if self.transmission <= 0.0 || self.metallic >= 1.0 {
                return ([0.0; 3], 0.0);
            }
            let nl = -vec3_dot(n, l);
            if nl <= 0.0 {
                return ([0.0; 3], 0.0);
            }
            let smax = max3(self.sheen);
            let layer_scale = (1.0 - self.w_cc)
                * if smax > 0.0 {
                    1.0 - smax * sheen_albedo(nv, self.sheen_rough)
                } else {
                    1.0
                };
            let w = (1.0 - self.metallic) * self.transmission * layer_scale;
            let (val, pdf_t) = if self.thin {
                let lm = add(l, scale(n, 2.0 * nl)); // mirrored above
                let h = vec3_normalise(add(v, lm));
                let vh = vec3_dot(v, h);
                let nh = vec3_dot(n, h);
                if vh <= 0.0 || nh <= 0.0 {
                    return ([0.0; 3], 0.0);
                }
                let fd = self.specular * max3(self.fresnel_d(vh));
                let val =
                    d_ggx(nh, self.alpha) * g2(nv, nl, self.alpha) / (4.0 * nv * nl) * (1.0 - fd);
                (val, vndf_reflect_pdf(n, v, h, self.alpha))
            } else {
                let (ei, et) = (self.eta_i, self.eta_t);
                let mut h = vec3_normalise([
                    -(ei * v[0] + et * l[0]),
                    -(ei * v[1] + et * l[1]),
                    -(ei * v[2] + et * l[2]),
                ]);
                if vec3_dot(h, n) < 0.0 {
                    h = neg(h);
                }
                let vh = vec3_dot(v, h);
                let lh = vec3_dot(l, h);
                let nh = vec3_dot(n, h);
                if vh <= 0.0 || lh >= 0.0 || nh <= 0.0 {
                    return ([0.0; 3], 0.0);
                }
                let denom = ei * vh + et * lh;
                if denom.abs() < 1.0e-6 {
                    return ([0.0; 3], 0.0);
                }
                let d = d_ggx(nh, self.alpha);
                let fd = self.specular * max3(self.fresnel_d(vh));
                let jac = et * et * lh.abs() / (denom * denom);
                let val = vh * lh.abs() * d * g2(nv, nl, self.alpha) * et * et * (1.0 - fd)
                    / (nv * nl * denom * denom);
                (val, g1(nv, self.alpha) * vh * d / nv * jac)
            };
            let f = scale(self.base, w * val * nl);
            (f, p[LOBE_TRANSMISSION] * pdf_t)
        }
    }

    /// Sample a direction with `u = [u₀, u₁, lobe]`. Returns
    /// `(l, f·|n·l|, pdf)` or `None` when the sample carries no
    /// energy.
    pub fn sample(&self, u: [f32; 3]) -> Option<([f32; 3], [f32; 3], f32)> {
        let mut acc = 0.0;
        let mut lobe = usize::MAX;
        for (i, &p) in self.probs.iter().enumerate() {
            acc += p;
            if p > 0.0 && u[2] < acc {
                lobe = i;
                break;
            }
        }
        if lobe == usize::MAX {
            lobe = self.probs.iter().rposition(|&p| p > 0.0)?;
        }
        let v = self.v;
        let l = match lobe {
            LOBE_DIFFUSE | LOBE_SHEEN => {
                let (t, b) = basis(self.n);
                to_world(t, b, self.n, cosine_hemisphere(u[0], u[1]))
            }
            LOBE_SPECULAR | LOBE_CLEARCOAT | LOBE_TRANSMISSION => {
                let (nn, a) = if lobe == LOBE_CLEARCOAT {
                    (self.ncc, self.cc_alpha)
                } else {
                    (self.n, self.alpha)
                };
                let (t, b) = basis(nn);
                let vl = [vec3_dot(v, t), vec3_dot(v, b), vec3_dot(v, nn)];
                if vl[2] <= 0.0 {
                    return None;
                }
                let h = to_world(t, b, nn, sample_vndf(vl, a, u[0], u[1]));
                if lobe != LOBE_TRANSMISSION {
                    reflect(v, h)
                } else if self.thin {
                    let r = reflect(v, h);
                    let d = vec3_dot(r, self.n);
                    add(r, scale(self.n, -2.0 * d))
                } else {
                    refract(v, h, self.eta_i / self.eta_t)?
                }
            }
            _ => return None,
        };
        if !finite3(l) || vec3_dot(l, l) < 0.5 {
            return None;
        }
        // A lobe sample landing on the other side of the surface is
        // discarded (zero contribution): the mixture pdf `eval`
        // reports for a side only counts the lobes that live there.
        let reflection = vec3_dot(self.ng, l) > 0.0;
        if reflection == (lobe == LOBE_TRANSMISSION) {
            return None;
        }
        let (f, pdf) = self.eval(l);
        if pdf <= 0.0 || !pdf.is_finite() || is_black(f) || !finite3(f) {
            return None;
        }
        Some((l, f, pdf))
    }
}

// =====================================================================
// Lights.
// =====================================================================

/// Power-proportional table of emissive triangles.
#[derive(Debug, Clone, Default)]
struct EmissiveLights {
    /// Global triangle id per light.
    tris: Vec<u32>,
    /// Selection probability per light.
    pmf: Vec<f32>,
    /// Inclusive CDF.
    cdf: Vec<f32>,
    /// Global triangle → light index (`u32::MAX` = not a light).
    lookup: Vec<u32>,
}

impl EmissiveLights {
    fn build(ts: &TraceScene) -> Self {
        let p = &ts.prepared;
        let mut tris = Vec::new();
        let mut power = Vec::new();
        let mut lookup = vec![u32::MAX; ts.triangle_count()];
        // Mean emission per material (factor × texture mean).
        let mean_e: Vec<[f32; 3]> = p
            .materials
            .iter()
            .map(|m| {
                if m.unlit {
                    return [0.0; 3];
                }
                let mut e = m.emissive;
                if let Some(t) = m.emissive_texture.as_ref().and_then(|b| p.texture(b)) {
                    if let Some(top) = t.data.mips(ColorSpace::Srgb).last() {
                        let c = top.texels[0];
                        e = mul(e, [c[0], c[1], c[2]]);
                    }
                }
                e
            })
            .collect();
        for (g, r) in ts.tri_refs().iter().enumerate() {
            let item = &p.items[r.item as usize];
            let e = mean_e[item.material];
            let lum = luminance(e);
            if lum.is_nan() || lum <= 0.0 {
                continue;
            }
            let [a, b, c] = ts.triangle_positions(g as u32);
            let cr = vec3_cross(crate::math::vec3_sub(b, a), crate::math::vec3_sub(c, a));
            let area = 0.5 * vec3_dot(cr, cr).sqrt();
            if area.is_nan() || area <= 0.0 || !area.is_finite() {
                continue;
            }
            let sides = if p.materials[item.material].double_sided {
                2.0
            } else {
                1.0
            };
            lookup[g] = tris.len() as u32;
            tris.push(g as u32);
            power.push(area * lum * sides);
        }
        let total: f64 = power.iter().map(|&x| x as f64).sum();
        let mut pmf = Vec::with_capacity(power.len());
        let mut cdf = Vec::with_capacity(power.len());
        let mut acc = 0.0f64;
        for &pw in &power {
            let pr = (pw as f64 / total) as f32;
            pmf.push(pr);
            acc += pw as f64 / total;
            cdf.push(acc as f32);
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        Self {
            tris,
            pmf,
            cdf,
            lookup,
        }
    }

    fn is_empty(&self) -> bool {
        self.tris.is_empty()
    }

    fn pick(&self, u: f32) -> usize {
        self.cdf
            .partition_point(|&c| c <= u)
            .min(self.tris.len() - 1)
    }
}

/// Equirectangular HDR environment with a luminance-importance 2-D
/// sampling distribution (§5 of the module docs).
#[derive(Debug, Clone)]
pub struct EnvironmentMap {
    image: Arc<HdrImage>,
    intensity: f32,
    /// Marginal CDF over rows (`height + 1` entries, last = 1).
    marginal: Vec<f32>,
    /// Per-row conditional CDFs (`height × (width + 1)`).
    conditional: Vec<f32>,
    /// Per-texel probability mass (normalised).
    mass: Vec<f32>,
    uniform: bool,
}

impl EnvironmentMap {
    /// Wrap `image` (scene-linear RGB, row 0 = `+Y`) scaled by
    /// `intensity`.
    pub fn new(image: Arc<HdrImage>, intensity: f32) -> Self {
        let (w, h) = (image.width.max(1) as usize, image.height.max(1) as usize);
        let valid = image.pixels.len() >= w * h * 4 && image.width > 0 && image.height > 0;
        let mut weights = vec![0.0f64; w * h];
        if valid {
            for y in 0..h {
                let st = (PI * (y as f32 + 0.5) / h as f32).sin() as f64;
                for x in 0..w {
                    let i = (y * w + x) * 4;
                    let px = &image.pixels[i..i + 3];
                    let l = luminance([px[0], px[1], px[2]]);
                    weights[y * w + x] = if l.is_finite() && l > 0.0 {
                        l as f64 * st
                    } else {
                        0.0
                    };
                }
            }
        }
        let total: f64 = weights.iter().sum();
        let uniform = !(total > 0.0 && total.is_finite());
        if uniform {
            for (i, wv) in weights.iter_mut().enumerate() {
                let y = i / w;
                *wv = (PI * (y as f32 + 0.5) / h as f32).sin() as f64;
            }
        }
        let total: f64 = weights.iter().sum();
        let mut marginal = Vec::with_capacity(h + 1);
        let mut conditional = Vec::with_capacity(h * (w + 1));
        let mut acc = 0.0f64;
        marginal.push(0.0);
        for y in 0..h {
            let row: f64 = weights[y * w..(y + 1) * w].iter().sum();
            let mut racc = 0.0f64;
            conditional.push(0.0);
            for x in 0..w {
                racc += weights[y * w + x];
                conditional.push(if row > 0.0 {
                    (racc / row) as f32
                } else {
                    (x + 1) as f32 / w as f32
                });
            }
            acc += row;
            marginal.push((acc / total) as f32);
        }
        if let Some(l) = marginal.last_mut() {
            *l = 1.0;
        }
        let mass = weights.iter().map(|&x| (x / total) as f32).collect();
        Self {
            image,
            intensity: if intensity.is_finite() {
                intensity.max(0.0)
            } else {
                0.0
            },
            marginal,
            conditional,
            mass,
            uniform,
        }
    }

    fn dims(&self) -> (usize, usize) {
        (
            self.image.width.max(1) as usize,
            self.image.height.max(1) as usize,
        )
    }

    fn texel_of(&self, d: [f32; 3]) -> (usize, usize, f32) {
        let (w, h) = self.dims();
        let u = 0.5 + d[0].atan2(-d[2]) / (2.0 * PI);
        let th = d[1].clamp(-1.0, 1.0).acos();
        let v = th / PI;
        let x = ((u * w as f32) as usize).min(w - 1);
        let y = ((v * h as f32) as usize).min(h - 1);
        (x, y, th.sin())
    }

    /// Radiance arriving from direction `d` (unit, pointing away from
    /// the scene).
    pub fn radiance(&self, d: [f32; 3]) -> [f32; 3] {
        let (w, _) = self.dims();
        let (x, y, _) = self.texel_of(d);
        let i = (y * w + x) * 4;
        match self.image.pixels.get(i..i + 3) {
            Some(p) if !self.uniform => {
                let c = [p[0], p[1], p[2]];
                if finite3(c) {
                    scale(
                        [c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)],
                        self.intensity,
                    )
                } else {
                    [0.0; 3]
                }
            }
            _ => [0.0; 3],
        }
    }

    /// Solid-angle pdf of [`Self::sample`] producing `d`.
    pub fn pdf(&self, d: [f32; 3]) -> f32 {
        let (w, h) = self.dims();
        let (x, y, st) = self.texel_of(d);
        if st <= 1.0e-6 {
            return 0.0;
        }
        self.mass[y * w + x] * (w * h) as f32 / (2.0 * PI * PI * st)
    }

    /// Sample a direction from `u`: `(direction, radiance, pdf)`.
    pub fn sample(&self, u0: f32, u1: f32) -> Option<([f32; 3], [f32; 3], f32)> {
        let (w, h) = self.dims();
        let y = (self.marginal.partition_point(|&c| c <= u0)).clamp(1, h) - 1;
        let row = &self.conditional[y * (w + 1)..(y + 1) * (w + 1)];
        let x = (row.partition_point(|&c| c <= u1)).clamp(1, w) - 1;
        // Continuous offset within the texel.
        let my = (u0 - self.marginal[y]) / (self.marginal[y + 1] - self.marginal[y]).max(1e-12);
        let mx = (u1 - row[x]) / (row[x + 1] - row[x]).max(1e-12);
        let u = (x as f32 + mx.clamp(0.0, 0.999_99)) / w as f32;
        let v = (y as f32 + my.clamp(0.0, 0.999_99)) / h as f32;
        let phi = 2.0 * PI * (u - 0.5);
        let th = PI * v;
        let (st, ct) = th.sin_cos();
        let d = [st * phi.sin(), ct, -st * phi.cos()];
        let pdf = self.pdf(d);
        if pdf <= 0.0 || !pdf.is_finite() {
            return None;
        }
        Some((d, self.radiance(d), pdf))
    }
}

// =====================================================================
// Spherical triangles (Arvo 1995).
// =====================================================================

/// Solid angles below this use uniform-area sampling instead of
/// spherical-triangle sampling (too small to sample robustly).
pub const MIN_SPHERICAL_SOLID_ANGLE: f32 = 1.0e-4;

/// Solid angle of triangle `abc` seen from `p` (Van Oosterom &
/// Strackee, "The Solid Angle of a Plane Triangle", IEEE Trans.
/// Biomed. Eng. 30(2), 1983): `tan(Ω/2) = |A·(B×C)| / (1 + A·B + B·C
/// + C·A)` with `A, B, C` the unit directions to the vertices.
pub fn triangle_solid_angle(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let u = |q: [f32; 3]| -> [f64; 3] {
        let d = [
            (q[0] - p[0]) as f64,
            (q[1] - p[1]) as f64,
            (q[2] - p[2]) as f64,
        ];
        let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if l > 0.0 {
            [d[0] / l, d[1] / l, d[2] / l]
        } else {
            [0.0; 3]
        }
    };
    let (a, b, c) = (u(a), u(b), u(c));
    let dot = |x: [f64; 3], y: [f64; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let bc = [
        b[1] * c[2] - b[2] * c[1],
        b[2] * c[0] - b[0] * c[2],
        b[0] * c[1] - b[1] * c[0],
    ];
    let num = dot(a, bc).abs();
    let den = 1.0 + dot(a, b) + dot(b, c) + dot(c, a);
    let o = 2.0 * num.atan2(den);
    if o.is_finite() {
        o.max(0.0) as f32
    } else {
        0.0
    }
}

/// Arvo, "Stratified Sampling of Spherical Triangles", SIGGRAPH 1995:
/// a direction from `p` uniformly distributed (pdf `1/Ω`) over the
/// spherical projection of triangle `abc`. `None` when degenerate.
pub fn sample_spherical_triangle(
    p: [f32; 3],
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
    u0: f32,
    u1: f32,
) -> Option<[f32; 3]> {
    type V = [f64; 3];
    let sub = |x: [f32; 3]| -> V {
        [
            (x[0] - p[0]) as f64,
            (x[1] - p[1]) as f64,
            (x[2] - p[2]) as f64,
        ]
    };
    let dot = |x: V, y: V| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let cross = |x: V, y: V| -> V {
        [
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ]
    };
    let norm = |x: V| -> Option<V> {
        let l = dot(x, x).sqrt();
        (l > 1.0e-300 && l.is_finite()).then(|| [x[0] / l, x[1] / l, x[2] / l])
    };
    let (a, b, c) = (norm(sub(a))?, norm(sub(b))?, norm(sub(c))?);
    // Interior angles from the great-circle plane normals.
    let nab = norm(cross(a, b))?;
    let nbc = norm(cross(b, c))?;
    let nca = norm(cross(c, a))?;
    let ang = |x: V, y: V| (-dot(x, y)).clamp(-1.0, 1.0).acos();
    let alpha = ang(nab, nca);
    let beta = ang(nbc, nab);
    let gamma = ang(nca, nbc);
    let area = alpha + beta + gamma - std::f64::consts::PI;
    if area.is_nan() || area <= 0.0 {
        return None;
    }
    // Sub-triangle area → new vertex C' on arc AC.
    let ap = u0 as f64 * area;
    let (s, t) = (ap - alpha).sin_cos();
    let (sa, ca) = alpha.sin_cos();
    let cos_c = dot(a, b);
    let uu = t - ca;
    let vv = s + sa * cos_c;
    let den = (vv * s + uu * t) * sa;
    if den.abs() < 1.0e-300 {
        return None;
    }
    let q = (((vv * t - uu * s) * ca - vv) / den).clamp(-1.0, 1.0);
    let ca_perp = norm([
        c[0] - dot(c, a) * a[0],
        c[1] - dot(c, a) * a[1],
        c[2] - dot(c, a) * a[2],
    ])?;
    let r = (1.0 - q * q).max(0.0).sqrt();
    let cp = [
        q * a[0] + r * ca_perp[0],
        q * a[1] + r * ca_perp[1],
        q * a[2] + r * ca_perp[2],
    ];
    // Uniform point on arc B–C'.
    let z = 1.0 - u1 as f64 * (1.0 - dot(cp, b));
    let cb_perp = norm([
        cp[0] - dot(cp, b) * b[0],
        cp[1] - dot(cp, b) * b[1],
        cp[2] - dot(cp, b) * b[2],
    ]);
    let w = (1.0 - z * z).max(0.0).sqrt();
    let d = match cb_perp {
        Some(perp) => [
            z * b[0] + w * perp[0],
            z * b[1] + w * perp[1],
            z * b[2] + w * perp[2],
        ],
        None => b,
    };
    let d = norm(d)?;
    Some([d[0] as f32, d[1] as f32, d[2] as f32])
}

// =====================================================================
// Integrator.
// =====================================================================

/// Everything a path needs, fixed for one accumulation.
struct Integrator<'a> {
    ts: &'a TraceScene,
    lights: &'a EmissiveLights,
    env: Option<&'a EnvironmentMap>,
    camera: &'a Camera,
    width: u32,
    height: u32,
    ambient: f32,
    pt: PathTraceOptions,
    /// World cone spread per unit distance of a primary ray.
    pixel_spread: f32,
    orthographic: bool,
}

#[inline]
fn power_heuristic(a: f32, b: f32) -> f32 {
    let (a2, b2) = (a * a, b * b);
    if a2 + b2 > 0.0 && (a2 + b2).is_finite() {
        a2 / (a2 + b2)
    } else if a.is_infinite() {
        1.0
    } else {
        0.0
    }
}

impl Integrator<'_> {
    fn clamp(&self, c: [f32; 3]) -> [f32; 3] {
        let m = max3(c);
        if self.pt.clamp > 0.0 && m > self.pt.clamp {
            scale(c, self.pt.clamp / m)
        } else {
            c
        }
    }

    /// Probability that the area-light sampler picks the environment.
    fn p_env(&self) -> f32 {
        match (self.env.is_some(), self.lights.is_empty()) {
            (true, false) => 0.5,
            (true, true) => 1.0,
            _ => 0.0,
        }
    }

    /// Any-hit acceptance (MASK, stochastic BLEND, primary culling).
    fn accept(&self, c: &TraceHit, s: &Sampler, ray: u32, primary: bool) -> bool {
        let (_, mat) = self.ts.item(c);
        if primary && !c.front_face && !mat.double_sided {
            return false;
        }
        match mat.alpha_mode {
            AlphaMode::Opaque => true,
            AlphaMode::Mask { cutoff } => self.ts.alpha(c) >= cutoff,
            AlphaMode::Blend => s.coin(ray, c.global) < self.ts.alpha(c),
        }
    }

    fn visible(
        &self,
        o: [f32; 3],
        d: [f32; 3],
        t_max: f32,
        s: &Sampler,
        ray: u32,
        skip: u32,
    ) -> bool {
        !self.ts.occluded_filtered(o, d, 0.0, t_max, |c| {
            c.global != skip && self.accept(c, s, ray, false)
        })
    }

    /// Emitted radiance of global triangle `hit` toward `-dir`.
    fn emission(&self, hit: &TraceHit) -> [f32; 3] {
        let (item, mat) = self.ts.item(hit);
        if mat.unlit || is_black(mat.emissive) {
            return [0.0; 3];
        }
        let mut e = mat.emissive;
        if let Some(b) = &mat.emissive_texture {
            if let (Some(tex), Some(uvs)) = (self.ts.prepared.texture(b), item.uv_set(b.uv_set)) {
                if uvs.len() == item.positions.len() {
                    let uv = b.transform_uv(interp2(uvs, 3 * hit.tri as usize, hit.barycentric));
                    let t = tex.sample_lod(uv, 0.0, ColorSpace::Srgb);
                    e = mul(e, [t[0], t[1], t[2]]);
                }
            }
        }
        e
    }

    /// Light-sampling solid-angle pdf of choosing emissive triangle
    /// `g` (light index `li`) and the direction toward `q` on it from
    /// vertex `p`: spherical (`1/Ω`) when `Ω ≥`
    /// [`MIN_SPHERICAL_SOLID_ANGLE`], uniform-area otherwise.
    fn tri_pdf(&self, li: usize, g: u32, p: [f32; 3], q: [f32; 3]) -> f32 {
        let [a, b, c] = self.ts.triangle_positions(g);
        let sel = (1.0 - self.p_env()) * self.lights.pmf[li];
        let omega = triangle_solid_angle(p, a, b, c);
        if omega >= MIN_SPHERICAL_SOLID_ANGLE {
            return sel / omega;
        }
        let cr = vec3_cross(crate::math::vec3_sub(b, a), crate::math::vec3_sub(c, a));
        let len = vec3_dot(cr, cr).sqrt();
        let to = crate::math::vec3_sub(q, p);
        let d2 = vec3_dot(to, to);
        if len <= 0.0 || d2 <= 0.0 {
            return 0.0;
        }
        let cos = (vec3_dot(cr, to) / (len * d2.sqrt())).abs();
        if cos <= 0.0 {
            return 0.0;
        }
        sel * d2 / (0.5 * len * cos)
    }

    /// Light-sampling pdf of reaching emissive `hit` (at `q`) from the
    /// previous vertex `p` — 0 when the triangle is not in the table.
    fn light_pdf(&self, hit: &TraceHit, p: [f32; 3], q: [f32; 3]) -> f32 {
        let li = self.lights.lookup.get(hit.global as usize).copied();
        let Some(li) = li.filter(|&l| l != u32::MAX) else {
            return 0.0;
        };
        self.tri_pdf(li as usize, hit.global, p, q)
    }

    /// Trace sample `index` of pixel `(x, y)`. `None` = uncovered.
    fn sample(&self, x: u32, y: u32, index: u32) -> Option<[f32; 3]> {
        let s = Sampler::new(self.pt.seed, x, y, index);
        let p0 = s.pattern(0);
        let (mut o, mut d) = self.camera.primary_ray(
            x as f32 + p0[0],
            y as f32 + p0[1],
            self.width as f32,
            self.height as f32,
        );
        let mut beta = [1.0f32; 3];
        let mut radiance = [0.0f32; 3];
        let mut prev_pdf = 0.0f32;
        let mut prev_pos = [0.0f32; 3];
        let mut medium: Option<[f32; 3]> = None;
        let strategy = self.pt.strategy;
        let mut k: u32 = 0;
        loop {
            let ray_id = k << 16;
            let hit = self.ts.closest_hit_filtered(o, d, 0.0, f32::INFINITY, |c| {
                self.accept(c, &s, ray_id, k == 0)
            });
            // Beer-Lambert along the segment.
            if let Some(sigma) = medium {
                let Some(h) = &hit else { break };
                for c in 0..3 {
                    beta[c] *= (-sigma[c] * h.t).exp();
                }
            }
            let Some(hit) = hit else {
                if k == 0 {
                    return None;
                }
                let (le, w) = match self.env {
                    Some(env) => {
                        let le = env.radiance(d);
                        let w = match strategy {
                            LightStrategy::Mis => {
                                power_heuristic(prev_pdf, self.p_env() * env.pdf(d))
                            }
                            LightStrategy::LightOnly => 0.0,
                            _ => 1.0,
                        };
                        (le, w)
                    }
                    None => ([self.ambient; 3], 1.0),
                };
                if w > 0.0 {
                    radiance = add(radiance, self.clamp(scale(mul(beta, le), w)));
                }
                break;
            };
            let surf = self.ts.surface(&hit);
            let (_, pmat) = self.ts.item(&hit);

            // Emission.
            if (hit.front_face || pmat.double_sided) && !pmat.unlit && !is_black(pmat.emissive) {
                let le = self.emission(&hit);
                if !is_black(le) {
                    if k == 0 {
                        radiance = add(radiance, mul(beta, le));
                    } else {
                        let pl = self.light_pdf(&hit, prev_pos, surf.position);
                        let w = if pl <= 0.0 {
                            1.0
                        } else {
                            match strategy {
                                LightStrategy::Mis => power_heuristic(prev_pdf, pl),
                                LightStrategy::LightOnly => 0.0,
                                _ => 1.0,
                            }
                        };
                        if w > 0.0 {
                            radiance = add(radiance, self.clamp(scale(mul(beta, le), w)));
                        }
                    }
                }
            }
            let lod = if k == 0 {
                let width = if self.orthographic {
                    self.pixel_spread
                } else {
                    self.pixel_spread * hit.t
                };
                TexLod::Cone { width, dir: d }
            } else {
                TexLod::Base
            };
            // Double-sided back faces: glTF reverses the normal before
            // normal mapping (as the scanline backend does). `Bsdf::new`
            // expects front-side normals and flips them itself, so the
            // oriented result is negated back here.
            let flip = !hit.front_face && pmat.double_sided;
            let mut mat: MaterialSample = self.ts.material_oriented(&hit, &surf, lod, flip);
            if flip {
                mat.normal = neg(mat.normal);
                mat.clearcoat_normal = neg(mat.clearcoat_normal);
            }
            if mat.unlit {
                let c = [mat.base_color[0], mat.base_color[1], mat.base_color[2]];
                let contrib = mul(beta, c);
                radiance = add(radiance, if k == 0 { contrib } else { self.clamp(contrib) });
                break;
            }
            if k >= self.pt.max_bounces {
                break;
            }
            let v = neg(d);
            let bsdf = Bsdf::new(&mat, v, surf.geometric_normal, hit.front_face);
            if bsdf.is_black() {
                break;
            }
            let p = surf.position;
            let ng = bsdf.geometric_normal();
            let spawn = |dir: [f32; 3]| {
                if vec3_dot(dir, ng) >= 0.0 {
                    offset_ray_origin(p, ng)
                } else {
                    offset_ray_origin(p, neg(ng))
                }
            };

            // ---- NEE: punctual lights.
            for (j, light) in self.ts.prepared.lights.iter().enumerate() {
                let Some(ls) = PreparedLight::sample(light, p) else {
                    continue;
                };
                let (f, _) = bsdf.eval(ls.l);
                if is_black(f) {
                    continue;
                }
                let o2 = spawn(ls.l);
                let tmax = if ls.distance.is_finite() {
                    ls.distance * (1.0 - 1.0e-4)
                } else {
                    f32::INFINITY
                };
                if self.visible(
                    o2,
                    ls.l,
                    tmax,
                    &s,
                    ray_id | 0x8000 | (j as u32 & 0x7fff),
                    u32::MAX,
                ) {
                    radiance = add(radiance, self.clamp(mul(mul(beta, f), ls.radiance)));
                }
            }

            // ---- NEE: area set (emissive triangles / environment).
            let pl_u = s.pattern(2 + 2 * k);
            if strategy != LightStrategy::BsdfOnly
                && (self.env.is_some() || !self.lights.is_empty())
            {
                let pe = self.p_env();
                let shadow_id = ray_id | 0xffff;
                if pl_u[2] < pe {
                    if let Some(env) = self.env {
                        if let Some((l, le, pdf)) = env.sample(pl_u[0], pl_u[1]) {
                            let pl = pe * pdf;
                            let (f, pb) = bsdf.eval(l);
                            if !is_black(f) && !is_black(le) && pl > 0.0 {
                                let w = if strategy == LightStrategy::Mis {
                                    power_heuristic(pl, pb)
                                } else {
                                    1.0
                                };
                                if self.visible(spawn(l), l, f32::INFINITY, &s, shadow_id, u32::MAX)
                                {
                                    let c = scale(mul(mul(beta, f), le), w / pl);
                                    radiance = add(radiance, self.clamp(c));
                                }
                            }
                        }
                    }
                } else if !self.lights.is_empty() {
                    let us = if pe > 0.0 {
                        ((pl_u[2] - pe) / (1.0 - pe)).clamp(0.0, 0.999_999)
                    } else {
                        pl_u[2]
                    };
                    let li = self.lights.pick(us);
                    let g = self.lights.tris[li];
                    let [a, b, c] = self.ts.triangle_positions(g);
                    // Direction + point on the light: spherical when the
                    // solid angle allows, uniform area otherwise.
                    let target = if triangle_solid_angle(p, a, b, c) >= MIN_SPHERICAL_SOLID_ANGLE {
                        sample_spherical_triangle(p, a, b, c, pl_u[0], pl_u[1]).and_then(|l| {
                            let n = vec3_cross(
                                crate::math::vec3_sub(b, a),
                                crate::math::vec3_sub(c, a),
                            );
                            let dn = vec3_dot(l, n);
                            if dn == 0.0 {
                                return None;
                            }
                            let t = vec3_dot(crate::math::vec3_sub(a, p), n) / dn;
                            (t > 0.0 && t.is_finite()).then(|| add(p, scale(l, t)))
                        })
                    } else {
                        let su = pl_u[0].sqrt();
                        let w = [1.0 - su, su * (1.0 - pl_u[1]), su * pl_u[1]];
                        Some([
                            a[0] * w[0] + b[0] * w[1] + c[0] * w[2],
                            a[1] * w[0] + b[1] * w[1] + c[1] * w[2],
                            a[2] * w[0] + b[2] * w[1] + c[2] * w[2],
                        ])
                    };
                    let bary = target.and_then(|q| crate::trace::barycentric_of(q, a, b, c));
                    if let (Some(q), Some(bary)) = (target, bary) {
                        let bary = [
                            bary[0].clamp(0.0, 1.0),
                            bary[1].clamp(0.0, 1.0),
                            bary[2].clamp(0.0, 1.0),
                        ];
                        let to = crate::math::vec3_sub(q, p);
                        let dist2 = vec3_dot(to, to);
                        let r = self.ts.tri_refs()[g as usize];
                        let lh = TraceHit {
                            t: dist2.sqrt(),
                            global: g,
                            item: r.item,
                            tri: r.tri,
                            barycentric: bary,
                            front_face: true,
                        };
                        let lsurf = self.ts.surface(&lh);
                        let (_, lmat) = self.ts.item(&lh);
                        if dist2 > 1.0e-12 {
                            let dist = dist2.sqrt();
                            let l = scale(to, 1.0 / dist);
                            let cos_l = -vec3_dot(lsurf.geometric_normal, l);
                            let pl = self.tri_pdf(li, g, p, q);
                            if (cos_l > 0.0 || (lmat.double_sided && cos_l < 0.0))
                                && pl > 0.0
                                && pl.is_finite()
                            {
                                let (f, pb) = bsdf.eval(l);
                                let le = self.emission(&lh);
                                if !is_black(f) && !is_black(le) {
                                    let w = if strategy == LightStrategy::Mis {
                                        power_heuristic(pl, pb)
                                    } else {
                                        1.0
                                    };
                                    let o2 = spawn(l);
                                    let tmax = (dist * (1.0 - 1.0e-4)).max(0.0);
                                    if self.visible(o2, l, tmax, &s, shadow_id, g) {
                                        let c = scale(mul(mul(beta, f), le), w / pl);
                                        radiance = add(radiance, self.clamp(c));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ---- BSDF sampling.
            let pb_u = s.pattern(1 + 2 * k);
            let Some((l, f, pdf)) = bsdf.sample([pb_u[0], pb_u[1], pb_u[2]]) else {
                break;
            };
            beta = mul(beta, scale(f, 1.0 / pdf));
            if !finite3(beta) {
                break;
            }
            prev_pdf = pdf;
            prev_pos = p;
            // Medium transitions through volume boundaries.
            if vec3_dot(l, ng) < 0.0 && mat.thickness > 0.0 && mat.transmission > 0.0 {
                if hit.front_face {
                    let dist = mat.attenuation_distance;
                    medium = if dist.is_finite() {
                        let c = mat.attenuation_color;
                        Some([
                            -c[0].max(1.0e-6).ln() / dist,
                            -c[1].max(1.0e-6).ln() / dist,
                            -c[2].max(1.0e-6).ln() / dist,
                        ])
                    } else {
                        None
                    };
                } else {
                    medium = None;
                }
            }
            // Russian roulette.
            if k + 1 >= self.pt.rr_start {
                let q = max3(beta).clamp(0.05, 1.0);
                if q < 1.0 {
                    if pb_u[3] >= q {
                        break;
                    }
                    beta = scale(beta, 1.0 / q);
                }
            }
            o = spawn(l);
            d = l;
            k += 1;
        }
        if finite3(radiance) {
            Some(radiance)
        } else {
            Some([0.0; 3])
        }
    }
}

// =====================================================================
// Progressive accumulator.
// =====================================================================

struct State {
    ts: TraceScene,
    lights: EmissiveLights,
    camera: Camera,
    width: u32,
    height: u32,
}

/// Rows per parallel work unit.
const BAND_ROWS: usize = 4;

/// Progressive path tracer: keeps an accumulation buffer for a fixed
/// scene + options and refines it on demand.
///
/// ```no_run
/// use oxideav_render::{pathtrace::PathTracer, RenderOptions};
/// # fn demo(scene: &oxideav_mesh3d::Scene3D) {
/// let opts = RenderOptions::default();
/// let mut pt = PathTracer::new();
/// pt.sync(scene, &opts); // (re)prepares only when needed
/// while pt.samples() < 64 {
///     pt.refine(4); // cheap; call once per UI frame
///     let _frame = pt.image();
/// }
/// # }
/// ```
///
/// The accumulation resets whenever the radiance-relevant inputs
/// change: [`Self::sync`] with different camera / resolution /
/// lighting / path-trace options, [`Self::invalidate_scene`],
/// [`Self::set_environment`] or a new texture resolver. Display-only
/// options (background, tone map, exposure, sample target) never
/// reset it.
pub struct PathTracer {
    cache: TextureCache,
    state: Option<State>,
    opts: RenderOptions,
    prep: Option<PrepareOptions>,
    env: Option<Arc<EnvironmentMap>>,
    dirty: bool,
    /// Per pixel: `Σr, Σg, Σb` over covered samples, covered count.
    accum: Vec<[f64; 4]>,
    samples: u32,
}

impl std::fmt::Debug for PathTracer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PathTracer")
            .field("width", &self.opts.width)
            .field("height", &self.opts.height)
            .field("samples", &self.samples)
            .field("prepared", &self.state.is_some())
            .finish()
    }
}

impl Default for PathTracer {
    fn default() -> Self {
        Self::new()
    }
}

/// Options with every field that cannot change the radiance estimate
/// normalised away.
fn radiance_key(o: &RenderOptions) -> RenderOptions {
    let d = RenderOptions::default();
    RenderOptions {
        background: d.background,
        shading: d.shading,
        aa: d.aa,
        tone_map: d.tone_map,
        exposure: d.exposure,
        shadows: d.shadows,
        shadow_map_size: d.shadow_map_size,
        path_trace: PathTraceOptions {
            samples_per_pixel: d.path_trace.samples_per_pixel,
            ..o.path_trace
        },
        ..o.clone()
    }
}

impl PathTracer {
    /// An empty tracer resolving only built-in raw textures.
    pub fn new() -> Self {
        Self {
            cache: TextureCache::default(),
            state: None,
            opts: RenderOptions::default(),
            prep: None,
            env: None,
            dirty: true,
            accum: Vec::new(),
            samples: 0,
        }
    }

    /// An empty tracer decoding textures through `resolver`.
    pub fn with_texture_resolver(resolver: Arc<dyn TextureResolver>) -> Self {
        Self {
            cache: TextureCache::new(resolver),
            ..Self::new()
        }
    }

    /// Replace the texture decoder (re-prepares on the next sync).
    pub fn set_texture_resolver(&mut self, resolver: Arc<dyn TextureResolver>) {
        self.cache.set_resolver(resolver);
        self.dirty = true;
    }

    /// Install (or remove) an HDR environment map; replaces the
    /// constant [`RenderOptions::ambient`] sky. Resets accumulation.
    pub fn set_environment(&mut self, env: Option<Arc<EnvironmentMap>>) {
        self.env = env;
        self.reset();
    }

    /// Mark the scene content as changed: the next [`Self::sync`]
    /// re-prepares it.
    pub fn invalidate_scene(&mut self) {
        self.dirty = true;
    }

    /// Discard accumulated samples.
    pub fn reset(&mut self) {
        self.samples = 0;
        self.accum.fill([0.0; 4]);
    }

    /// Bring the tracer in line with `scene` + `opts`. Re-prepares the
    /// scene when it was invalidated (or never prepared) or when a
    /// preparation input changed (animation time, variant, lights);
    /// resets the accumulation when anything affecting radiance
    /// changed. Returns `true` when the accumulation was reset. Cheap
    /// when nothing changed (an options comparison).
    pub fn sync(&mut self, scene: &Scene3D, opts: &RenderOptions) -> bool {
        let prep = PrepareOptions::from_render_options(opts);
        let need_prepare = self.dirty || self.state.is_none() || self.prep.as_ref() != Some(&prep);
        let changed = need_prepare || radiance_key(opts) != radiance_key(&self.opts);
        self.opts = opts.clone();
        if !changed {
            return false;
        }
        let width = opts.width.max(1);
        let height = opts.height.max(1);
        if need_prepare {
            let prepared = PreparedScene::build(scene, &prep, &mut self.cache);
            let ts = TraceScene::new(prepared);
            let lights = EmissiveLights::build(&ts);
            let camera = Camera::resolve(&ts.prepared, opts, width, height);
            self.state = Some(State {
                ts,
                lights,
                camera,
                width,
                height,
            });
            self.prep = Some(prep);
            self.dirty = false;
        } else if let Some(st) = &mut self.state {
            st.camera = Camera::resolve(&st.ts.prepared, opts, width, height);
            st.width = width;
            st.height = height;
        }
        let n = width as usize * height as usize;
        self.accum.clear();
        self.accum.resize(n, [0.0; 4]);
        self.samples = 0;
        true
    }

    /// Samples accumulated per pixel so far.
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// `true` once [`PathTraceOptions::samples_per_pixel`] samples are
    /// in.
    pub fn is_converged(&self) -> bool {
        self.samples >= self.opts.path_trace.samples_per_pixel
    }

    /// The prepared scene being traced (after a [`Self::sync`]).
    pub fn prepared(&self) -> Option<&PreparedScene> {
        self.state.as_ref().map(|s| &s.ts.prepared)
    }

    /// Add `samples` samples to every pixel (multi-threaded over row
    /// bands with `std::thread::scope`). Sample indices continue from
    /// [`Self::samples`], so `refine(a); refine(b)` equals
    /// `refine(a + b)` bit for bit. No-op before the first
    /// [`Self::sync`].
    pub fn refine(&mut self, samples: u32) {
        let Some(st) = &self.state else { return };
        if samples == 0 {
            return;
        }
        let samples = samples.min(u32::MAX - self.samples);
        let (cam_h, cam_w) = (st.camera.half_h, st.camera.half_w);
        let orthographic = st.camera.projection == crate::options::Projection::Orthographic;
        let spread = 2.0 * cam_h / st.height as f32;
        let _ = cam_w;
        let integ = Integrator {
            ts: &st.ts,
            lights: &st.lights,
            env: self.env.as_deref(),
            camera: &st.camera,
            width: st.width,
            height: st.height,
            ambient: if self.opts.ambient.is_finite() {
                self.opts.ambient.max(0.0)
            } else {
                0.0
            },
            pt: self.opts.path_trace,
            pixel_spread: spread,
            orthographic,
        };
        let w = st.width as usize;
        let first = self.samples;
        let bands = Mutex::new(self.accum.chunks_mut(w * BAND_ROWS).enumerate());
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(st.height as usize / BAND_ROWS + 1)
            .max(1);
        let work = |integ: &Integrator| loop {
            let next = bands.lock().map(|mut it| it.next()).unwrap_or(None);
            let Some((band, rows)) = next else { break };
            let y0 = band * BAND_ROWS;
            for (i, px) in rows.iter_mut().enumerate() {
                let (x, y) = ((i % w) as u32, (y0 + i / w) as u32);
                for si in 0..samples {
                    if let Some(c) = integ.sample(x, y, first + si) {
                        px[0] += c[0] as f64;
                        px[1] += c[1] as f64;
                        px[2] += c[2] as f64;
                        px[3] += 1.0;
                    }
                }
            }
        };
        if threads <= 1 {
            work(&integ);
        } else {
            std::thread::scope(|sc| {
                for _ in 0..threads {
                    sc.spawn(|| work(&integ));
                }
            });
        }
        self.samples += samples;
    }

    fn bg_linear(&self) -> [f32; 4] {
        let lut = srgb_u8_lut();
        let b = self.opts.background.0;
        [
            lut[b[0] as usize],
            lut[b[1] as usize],
            lut[b[2] as usize],
            b[3] as f32 / 255.0,
        ]
    }

    /// Mix covered mean `m` (already mapped) with the background per
    /// the module-level resolve contract.
    fn mix(m: [f32; 3], f: f32, bg: [f32; 4]) -> [f32; 4] {
        let a_sum = f + (1.0 - f) * bg[3];
        if a_sum > 0.0 {
            let mut o = [0.0; 4];
            for k in 0..3 {
                o[k] = (f * m[k] + (1.0 - f) * bg[k] * bg[3]) / a_sum;
            }
            o[3] = a_sum;
            o
        } else {
            [
                f * m[0] + (1.0 - f) * bg[0],
                f * m[1] + (1.0 - f) * bg[1],
                f * m[2] + (1.0 - f) * bg[2],
                0.0,
            ]
        }
    }

    fn dims(&self) -> (u32, u32) {
        self.state
            .as_ref()
            .map(|s| (s.width, s.height))
            .unwrap_or((self.opts.width.max(1), self.opts.height.max(1)))
    }

    /// Resolve pixel `a` (accumulator entry) — `None` when no sample
    /// covered it (background).
    fn resolve_pixel(
        &self,
        a: &[f64; 4],
        bg: [f32; 4],
        map: &impl Fn([f32; 3]) -> [f32; 3],
    ) -> Option<[f32; 4]> {
        if a[3] <= 0.0 || self.samples == 0 {
            return None;
        }
        let m = [
            (a[0] / a[3]) as f32,
            (a[1] / a[3]) as f32,
            (a[2] / a[3]) as f32,
        ];
        let f = (a[3] / self.samples as f64).clamp(0.0, 1.0) as f32;
        let c = map(m);
        Some(if f >= 1.0 {
            [c[0], c[1], c[2], 1.0]
        } else {
            Self::mix(c, f, bg)
        })
    }

    /// Fill `out` (one entry per pixel) in parallel: `f(pixel, slot)`.
    fn par_pixels<T: Send>(out: &mut [T], f: impl Fn(usize, &mut T) + Sync) {
        let n = out.len();
        let threads = std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(1)
            .min(n / 16_384 + 1)
            .max(1);
        if threads <= 1 {
            for (i, o) in out.iter_mut().enumerate() {
                f(i, o);
            }
            return;
        }
        let chunk = n.div_ceil(threads);
        std::thread::scope(|sc| {
            for (ci, part) in out.chunks_mut(chunk).enumerate() {
                let f = &f;
                sc.spawn(move || {
                    for (i, o) in part.iter_mut().enumerate() {
                        f(ci * chunk + i, o);
                    }
                });
            }
        });
    }

    /// Current estimate as display bytes (exposure, tone map, sRGB;
    /// uncovered pixels keep the background bytes exactly). Resolved
    /// in parallel; cheap enough to call every UI frame.
    pub fn image(&self) -> RgbaImage {
        let (w, h) = self.dims();
        let (tm, ex) = (self.opts.tone_map, self.opts.exposure);
        let bg_bytes = self.opts.background.0;
        let bg = self.bg_linear();
        let map = |c: [f32; 3]| tm.apply(scale(c, ex));
        let n = w as usize * h as usize;
        let mut px = vec![bg_bytes; n];
        if self.accum.len() == n {
            Self::par_pixels(&mut px, |i, o| {
                if let Some(c) = self.resolve_pixel(&self.accum[i], bg, &map) {
                    *o = [
                        linear_to_srgb_byte(c[0]),
                        linear_to_srgb_byte(c[1]),
                        linear_to_srgb_byte(c[2]),
                        (c[3].clamp(0.0, 1.0) * 255.0).round() as u8,
                    ];
                }
            });
        }
        RgbaImage {
            width: w,
            height: h,
            stride: w as usize * 4,
            pixels: px.into_iter().flatten().collect(),
        }
    }

    /// Current estimate in scene-linear floats (no exposure / tone
    /// map; uncovered pixels hold the decoded background).
    pub fn hdr(&self) -> HdrImage {
        let (w, h) = self.dims();
        let bg = self.bg_linear();
        let n = w as usize * h as usize;
        let mut px = vec![bg; n];
        if self.accum.len() == n {
            Self::par_pixels(&mut px, |i, o| {
                if let Some(c) = self.resolve_pixel(&self.accum[i], bg, &|c| c) {
                    *o = c;
                }
            });
        }
        HdrImage {
            width: w,
            height: h,
            pixels: px.into_iter().flatten().collect(),
        }
    }
}

// =====================================================================
// Renderer.
// =====================================================================

/// [`crate::Renderer`] wrapper over [`PathTracer`]: every `render`
/// re-prepares the scene and takes
/// [`PathTraceOptions::samples_per_pixel`] samples.
#[derive(Debug, Default)]
pub struct PathTraceRenderer {
    tracer: PathTracer,
}

impl PathTraceRenderer {
    /// A renderer resolving only built-in raw textures.
    pub fn new() -> Self {
        Self::default()
    }

    /// The underlying progressive tracer (e.g. to install an
    /// environment map).
    pub fn tracer_mut(&mut self) -> &mut PathTracer {
        &mut self.tracer
    }

    fn run(&mut self, scene: &Scene3D, opts: &RenderOptions) {
        self.tracer.invalidate_scene();
        self.tracer.sync(scene, opts);
        self.tracer.refine(opts.path_trace.samples_per_pixel.max(1));
    }
}

impl crate::Renderer for PathTraceRenderer {
    fn render(&mut self, scene: &Scene3D, opts: &RenderOptions) -> crate::Result<RgbaImage> {
        self.run(scene, opts);
        Ok(self.tracer.image())
    }

    fn render_hdr(&mut self, scene: &Scene3D, opts: &RenderOptions) -> crate::Result<HdrImage> {
        self.run(scene, opts);
        Ok(self.tracer.hdr())
    }

    fn set_texture_resolver(&mut self, resolver: Arc<dyn TextureResolver>) {
        self.tracer.set_texture_resolver(resolver);
    }
}

// =====================================================================
// Tests.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::{BackgroundColor, Projection, ShadingMode};
    use crate::testscenes::{add_camera, cuboid, diffuse_material, quad};
    use crate::Renderer;
    use oxideav_mesh3d::{
        Light, Material, Mesh, Node, Primitive, Specular, Transform, Transmission, Volume,
    };

    // ---------------- helpers ----------------

    fn add_prim(scene: &mut Scene3D, prim: Primitive) {
        let m = scene.add_mesh(Mesh::new(None).with_primitive(prim));
        let n = scene.add_node(
            Node::new()
                .with_mesh(m)
                .with_transform(Transform::identity()),
        );
        scene.add_root(n);
    }

    fn lambert(rgb: [f32; 3]) -> Material {
        let mut m = diffuse_material(rgb, 1.0);
        m.ext.specular = Some(Specular {
            factor: 0.0,
            ..Specular::default()
        });
        m
    }

    fn opts(w: u32, spp: u32, bounces: u32) -> RenderOptions {
        RenderOptions {
            width: w,
            height: w,
            scene_camera: Some(0),
            ambient: 0.0,
            // No fallback directional light for light-less scenes.
            light: crate::LightSpec {
                intensity: 0.0,
                ..crate::LightSpec::default()
            },
            background: BackgroundColor([0, 0, 0, 255]),
            path_trace: PathTraceOptions {
                samples_per_pixel: spp,
                max_bounces: bounces,
                ..PathTraceOptions::default()
            },
            ..RenderOptions::default()
        }
    }

    fn hdr(scene: &Scene3D, o: &RenderOptions) -> HdrImage {
        PathTraceRenderer::new().render_hdr(scene, o).unwrap()
    }

    fn mean_rgb(img: &HdrImage) -> [f64; 3] {
        let mut s = [0.0f64; 3];
        let n = (img.width * img.height) as f64;
        for p in img.pixels.chunks_exact(4) {
            for k in 0..3 {
                s[k] += p[k] as f64 / n;
            }
        }
        s
    }

    fn sample_ms(base: [f32; 3], metallic: f32, roughness: f32) -> MaterialSample {
        MaterialSample {
            base_color: [base[0], base[1], base[2], 1.0],
            metallic,
            roughness,
            normal: [0.0, 0.0, 1.0],
            occlusion: 1.0,
            emissive: [0.0; 3],
            unlit: false,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            ior: 1.5,
            specular: 1.0,
            specular_color: [1.0; 3],
            transmission: 0.0,
            thickness: 0.0,
            attenuation_color: [1.0; 3],
            attenuation_distance: f32::INFINITY,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            clearcoat_normal: [0.0, 0.0, 1.0],
            sheen_color: [0.0; 3],
            sheen_roughness: 0.0,
        }
    }

    // ---------------- sampling primitives ----------------

    #[test]
    fn sobol_unscrambled_dims_match_published_points() {
        // Dimension 0: van der Corput; dimension 1: (1/2, 3/4, 1/4, ...).
        // Joe & Kuo's published first eight points (Gray-code order,
        // index g = i ^ (i >> 1)) for their dimensions 1–4.
        let table: [[f32; 4]; 8] = [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.75, 0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75, 0.75],
            [0.375, 0.375, 0.625, 0.875],
            [0.875, 0.875, 0.125, 0.375],
            [0.625, 0.125, 0.875, 0.625],
            [0.125, 0.625, 0.375, 0.125],
        ];
        for (i, row) in table.iter().enumerate() {
            let g = (i ^ (i >> 1)) as u32;
            let got: Vec<f32> = (0..4).map(|d| to_unit(sobol(g, d))).collect();
            assert_eq!(&got[..], &row[..], "point {i}");
        }
        // Dimensions 0 and 1 form a (0,2)-sequence: one point per
        // elementary 4×4 cell at 16 points.
        let mut cells = [0u32; 16];
        for i in 0..16 {
            let x = (to_unit(sobol(i, 0)) * 4.0) as usize;
            let y = (to_unit(sobol(i, 1)) * 4.0) as usize;
            cells[y * 4 + x] += 1;
        }
        assert!(cells.iter().all(|&c| c == 1), "{cells:?}");
    }

    #[test]
    fn owen_scrambling_preserves_stratification() {
        for seed in [1u32, 77, 0xdead_beef] {
            let mut cells = [0u32; 64];
            for i in 0..64 {
                let p = sobol_owen_4d(i, seed);
                cells[(p[0] * 8.0) as usize * 8 + (p[1] * 8.0) as usize] += 1;
                assert!(p.iter().all(|v| (0.0..1.0).contains(v)));
            }
            assert!(cells.iter().all(|&c| c == 1), "seed {seed}");
        }
    }

    // ---------------- BSDF ----------------

    /// `∫ f |cos| dω` by quadrature over the sphere, and `∫ pdf dω`.
    fn quadrature(b: &Bsdf) -> ([f64; 3], f64) {
        let (nt, np) = (360usize, 180usize);
        let mut e = [0.0f64; 3];
        let mut pdf_sum = 0.0f64;
        for i in 0..nt {
            let th = (i as f64 + 0.5) / nt as f64 * std::f64::consts::PI;
            for j in 0..np {
                // v lies in the xz-plane: integrate φ ∈ [0, π], ×2.
                let ph = (j as f64 + 0.5) / np as f64 * std::f64::consts::PI;
                let l = [
                    (th.sin() * ph.cos()) as f32,
                    (th.sin() * ph.sin()) as f32,
                    th.cos() as f32,
                ];
                let dw = th.sin()
                    * (std::f64::consts::PI / nt as f64)
                    * (std::f64::consts::PI / np as f64)
                    * 2.0;
                let (f, pdf) = b.eval(l);
                for k in 0..3 {
                    e[k] += f[k] as f64 * dw;
                }
                pdf_sum += pdf as f64 * dw;
            }
        }
        (e, pdf_sum)
    }

    fn monte_carlo(b: &Bsdf, n: u32) -> [f64; 3] {
        let mut e = [0.0f64; 3];
        for i in 0..n {
            let u = sobol_owen_4d(i, 12345);
            if let Some((_, f, pdf)) = b.sample([u[0], u[1], u[2]]) {
                for k in 0..3 {
                    e[k] += (f[k] / pdf) as f64 / n as f64;
                }
            }
        }
        e
    }

    #[test]
    fn bsdf_sampling_matches_quadrature_and_conserves_energy() {
        let v = vec3_normalise([0.6, 0.0, 0.8]);
        let ng = [0.0, 0.0, 1.0];
        let mut cases: Vec<(&str, MaterialSample, bool)> = vec![
            (
                "rough dielectric",
                sample_ms([0.8, 0.5, 0.2], 0.0, 0.8),
                true,
            ),
            (
                "glossy dielectric",
                sample_ms([0.8, 0.8, 0.8], 0.0, 0.45),
                true,
            ),
            ("rough metal", sample_ms([0.95, 0.64, 0.54], 1.0, 0.6), true),
            ("half metal", sample_ms([0.9, 0.9, 0.9], 0.5, 0.5), true),
        ];
        let mut thin = sample_ms([1.0; 3], 0.0, 0.5);
        thin.transmission = 1.0;
        cases.push(("thin transmission", thin, true));
        let mut vol = sample_ms([1.0; 3], 0.0, 0.5);
        vol.transmission = 1.0;
        vol.thickness = 1.0;
        cases.push(("volume entering", vol, true));
        // Leaving: the ray arrives from the back side, so the
        // CCW-front normals point away from `v`.
        let mut leave = vol;
        leave.normal = [0.0, 0.0, -1.0];
        leave.clearcoat_normal = [0.0, 0.0, -1.0];
        cases.push(("volume leaving", leave, false));
        let mut cc = sample_ms([0.2, 0.3, 0.8], 0.0, 0.7);
        cc.clearcoat = 1.0;
        cc.clearcoat_roughness = 0.4;
        cases.push(("clearcoat", cc, true));
        let mut sh = sample_ms([0.3, 0.1, 0.1], 0.0, 0.8);
        sh.sheen_color = [0.9, 0.7, 0.7];
        sh.sheen_roughness = 0.6;
        cases.push(("sheen", sh, true));
        for (name, m, front) in cases {
            let ng = if front { ng } else { neg(ng) };
            let b = Bsdf::new(&m, v, ng, front);
            let (eq, pdf_int) = quadrature(&b);
            let em = monte_carlo(&b, 1 << 16);
            for k in 0..3 {
                let tol = 0.02 * eq[k].max(0.05);
                assert!(
                    (eq[k] - em[k]).abs() < tol,
                    "{name} ch{k}: quadrature {} vs sampled {}",
                    eq[k],
                    em[k]
                );
                // White-ish lobes never reflect more than they receive.
                assert!(eq[k] <= 1.01, "{name} ch{k}: albedo {}", eq[k]);
            }
            // The mixture pdf is a (sub-)density over the sphere.
            assert!(pdf_int <= 1.01 && pdf_int > 0.5, "{name}: ∫pdf = {pdf_int}");
        }
    }

    #[test]
    fn white_lambert_albedo_is_exact() {
        let b = Bsdf::new(
            &MaterialSample {
                specular: 0.0,
                ..sample_ms([1.0; 3], 0.0, 1.0)
            },
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            true,
        );
        let e = monte_carlo(&b, 4096);
        for c in e {
            assert!((c - 1.0).abs() < 1e-3, "{c}");
        }
    }

    // ---------------- spherical triangles ----------------

    #[test]
    fn octant_triangle_subtends_half_pi() {
        let o = triangle_solid_angle([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert!((o - std::f32::consts::FRAC_PI_2).abs() < 1e-5, "{o}");
    }

    #[test]
    fn spherical_triangle_samples_are_uniform_in_solid_angle() {
        let p = [0.1, -0.2, 0.0];
        let (a, b, c) = ([-0.6, 0.3, 1.0], [1.2, -0.4, 0.8], [0.2, 1.1, 1.4]);
        let omega = triangle_solid_angle(p, a, b, c) as f64;
        // E[g(ω)] under pdf 1/Ω equals (1/Ω)∫ g dω; integrate the right
        // side over the triangle's area with dω = cos θ_l dA / r².
        let g = |d: [f32; 3]| (d[0] as f64 + 0.3).powi(2) + d[1] as f64;
        let n = vec3_cross(crate::math::vec3_sub(b, a), crate::math::vec3_sub(c, a));
        let area2 = vec3_dot(n, n).sqrt();
        let nn = scale(n, 1.0 / area2);
        let steps = 600;
        let mut quad = 0.0f64;
        let mut omega_q = 0.0f64;
        for i in 0..steps {
            for j in 0..steps - i {
                // Midpoint of a sub-triangle cell (upright cells only;
                // the inverted ones are covered by symmetry at this
                // resolution).
                let (u, v) = (
                    (i as f32 + 1.0 / 3.0) / steps as f32,
                    (j as f32 + 1.0 / 3.0) / steps as f32,
                );
                let q = [
                    a[0] + (b[0] - a[0]) * u + (c[0] - a[0]) * v,
                    a[1] + (b[1] - a[1]) * u + (c[1] - a[1]) * v,
                    a[2] + (b[2] - a[2]) * u + (c[2] - a[2]) * v,
                ];
                let to = crate::math::vec3_sub(q, p);
                let r2 = vec3_dot(to, to);
                let d = scale(to, 1.0 / r2.sqrt());
                let dw = (vec3_dot(nn, d).abs() / r2) as f64;
                quad += g(d) * dw;
                omega_q += dw;
            }
        }
        let expect = quad / omega_q;
        assert!(
            (omega_q * 0.5 * area2 as f64 / (steps * steps) as f64 * 2.0 / omega - 1.0).abs()
                < 0.02
        );
        let m = 1 << 15;
        let mut sum = 0.0f64;
        for i in 0..m {
            let u = sobol_owen_4d(i, 5);
            let d = sample_spherical_triangle(p, a, b, c, u[0], u[1]).unwrap();
            // Every direction hits the triangle.
            let t = vec3_dot(crate::math::vec3_sub(a, p), n) / vec3_dot(d, n);
            let q = add(p, scale(d, t));
            let w = crate::trace::barycentric_of(q, a, b, c).unwrap();
            assert!(w.iter().all(|&x| x > -1e-3), "{w:?}");
            sum += g(d);
        }
        let got = sum / m as f64;
        assert!(
            (got - expect).abs() < 0.01 * expect.abs().max(0.1),
            "{got} vs {expect}"
        );
    }

    // ---------------- environment map ----------------

    #[test]
    fn environment_pdf_is_normalised_and_matches_samples() {
        let (w, h) = (32u32, 16u32);
        let mut img = HdrImage::filled(w, h, [0.1, 0.1, 0.1, 1.0]);
        // A bright "sun" block.
        for y in 3..5 {
            for x in 20..23 {
                let i = ((y * w + x) * 4) as usize;
                img.pixels[i..i + 3].copy_from_slice(&[50.0, 40.0, 30.0]);
            }
        }
        let env = EnvironmentMap::new(Arc::new(img), 1.0);
        let n = 400;
        let mut sum = 0.0f64;
        for i in 0..n {
            let th = (i as f64 + 0.5) / n as f64 * std::f64::consts::PI;
            for j in 0..2 * n {
                let ph = (j as f64 + 0.5) / (2 * n) as f64 * 2.0 * std::f64::consts::PI;
                let d = [
                    (th.sin() * ph.cos()) as f32,
                    th.cos() as f32,
                    (th.sin() * ph.sin()) as f32,
                ];
                sum += env.pdf(d) as f64
                    * th.sin()
                    * (std::f64::consts::PI / n as f64)
                    * (std::f64::consts::PI / n as f64);
            }
        }
        assert!((sum - 1.0).abs() < 0.01, "∫pdf = {sum}");
        // Sampling: the pdf returned equals pdf(d) and most samples land
        // on the sun.
        let mut sun = 0;
        for i in 0..1000 {
            let u = sobol_owen_4d(i, 9);
            let (d, le, pdf) = env.sample(u[0], u[1]).unwrap();
            assert!((pdf - env.pdf(d)).abs() <= 1e-3 * pdf.max(1.0));
            if le[0] > 10.0 {
                sun += 1;
            }
        }
        assert!(sun > 800, "{sun}");
    }

    // ---------------- integrator ----------------

    /// Closed box of double-sided emissive Lambertian walls (albedo ρ,
    /// emission Le) viewed from inside: L = Le / (1 − ρ) everywhere.
    fn furnace_scene(rho: f32, le: f32) -> Scene3D {
        let mut scene = Scene3D::new();
        let mut m = lambert([rho; 3]);
        m.emissive_factor = [le; 3];
        m.double_sided = true;
        let mat = scene.add_material(m);
        let mut b = cuboid([-1.0; 3], [1.0; 3]);
        b.material = Some(mat);
        add_prim(&mut scene, b);
        add_camera(&mut scene, [0.2, 0.1, 0.4], [0.0, -0.3, -1.0], 1.2);
        scene
    }

    #[test]
    fn white_furnace_converges_to_analytic_radiance() {
        let scene = furnace_scene(0.5, 1.0);
        for strategy in [
            LightStrategy::Mis,
            LightStrategy::LightOnly,
            LightStrategy::BsdfOnly,
        ] {
            let mut o = opts(8, 64, 64);
            o.path_trace.strategy = strategy;
            let m = mean_rgb(&hdr(&scene, &o));
            for c in m {
                assert!((c - 2.0).abs() < 0.04, "{strategy:?}: {m:?} (expected 2.0)");
            }
        }
    }

    /// White floor (diffuse + glossy) under a down-facing emissive quad.
    fn area_light_scene(roughness: f32, metallic: f32) -> Scene3D {
        let mut scene = Scene3D::new();
        let mut fm = diffuse_material([0.8, 0.8, 0.8], roughness);
        fm.metallic = metallic;
        let floor_mat = scene.add_material(fm);
        let mut lm = lambert([0.0; 3]);
        lm.emissive_factor = [4.0, 4.0, 4.0];
        let light_mat = scene.add_material(lm);
        let mut floor = quad(
            [
                [-2.0, 0.0, 2.0],
                [2.0, 0.0, 2.0],
                [2.0, 0.0, -2.0],
                [-2.0, 0.0, -2.0],
            ],
            [1.0, 1.0],
        );
        floor.material = Some(floor_mat);
        add_prim(&mut scene, floor);
        // Faces -Y (toward the floor).
        let mut light = quad(
            [
                [-0.5, 1.0, -0.5],
                [0.5, 1.0, -0.5],
                [0.5, 1.0, 0.5],
                [-0.5, 1.0, 0.5],
            ],
            [1.0, 1.0],
        );
        light.material = Some(light_mat);
        add_prim(&mut scene, light);
        add_camera(&mut scene, [0.0, 2.5, 3.0], [0.0, 0.0, 0.0], 0.9);
        scene
    }

    #[test]
    fn mis_light_only_and_bsdf_only_agree() {
        for (rough, metal) in [(1.0, 0.0), (0.35, 0.5)] {
            let scene = area_light_scene(rough, metal);
            let mut means = Vec::new();
            for strategy in [
                LightStrategy::Mis,
                LightStrategy::LightOnly,
                LightStrategy::BsdfOnly,
            ] {
                let mut o = opts(12, 256, 1);
                o.path_trace.strategy = strategy;
                means.push(mean_rgb(&hdr(&scene, &o))[1]);
            }
            let reference = means[0];
            assert!(reference > 0.05, "{means:?}");
            for m in &means {
                assert!(
                    (m - reference).abs() < 0.03 * reference,
                    "roughness {rough}: {means:?}"
                );
            }
        }
    }

    fn rmse(a: &HdrImage, b: &HdrImage) -> f64 {
        let mut s = 0.0f64;
        for (x, y) in a.pixels.iter().zip(&b.pixels) {
            s += ((x - y) as f64).powi(2);
        }
        (s / a.pixels.len() as f64).sqrt()
    }

    #[test]
    fn error_decreases_with_sample_count() {
        let scene = area_light_scene(0.5, 0.0);
        let reference = hdr(&scene, &opts(10, 1024, 3));
        let errs: Vec<f64> = [4u32, 16, 64]
            .iter()
            .map(|&spp| {
                let mut o = opts(10, spp, 3);
                o.path_trace.seed = 99; // independent of the reference
                rmse(&hdr(&scene, &o), &reference)
            })
            .collect();
        assert!(
            errs[1] < errs[0] * 0.8 && errs[2] < errs[1] * 0.8,
            "{errs:?}"
        );
    }

    #[test]
    fn cornell_box_red_wall_bleeds_onto_floor() {
        let scene = crate::testscenes::cornell_box();
        let red_ratio = |bounces: u32| {
            let img = hdr(&scene, &opts(32, 64, bounces));
            let px = |x: u32, y: u32| img.pixel(x, y).unwrap();
            let avg = |x0: u32| {
                let mut s = [0.0f32; 3];
                for y in 28..31 {
                    for x in x0..x0 + 3 {
                        let p = px(x, y);
                        for k in 0..3 {
                            s[k] += p[k];
                        }
                    }
                }
                s
            };
            let (l, r) = (avg(4), avg(25));
            (l[0] / l[1]) / (r[0] / r[1])
        };
        let direct = red_ratio(1);
        let gi = red_ratio(6);
        assert!((direct - 1.0).abs() < 0.05, "direct-only ratio {direct}");
        assert!(gi > 1.15, "GI must tint the floor near the red wall: {gi}");
    }

    #[test]
    fn direct_lighting_matches_scanline_pbr() {
        // Double-sided normal-mapped quad seen from behind, lit from
        // behind: exercises the back-face normal-map orientation.
        let mut back = crate::testscenes::normal_mapped_quad(0.8);
        for m in &mut back.materials {
            m.double_sided = true;
        }
        // Only the back light: the front light must not reach the back
        // face (the scanline shadow map's bias lets it leak through a
        // single quad when the normal map tilts toward it).
        for l in &mut back.lights {
            if let Light::Directional { intensity, .. } = l {
                *intensity = 0.0;
            }
        }
        crate::testscenes::add_light(
            &mut back,
            Light::Directional {
                color: [1.0; 3],
                intensity: 2.0,
            },
            [0.0; 3],
            [0.4, -0.3, 1.0],
        );
        add_camera(&mut back, [0.3, 0.2, -2.5], [0.0; 3], 0.9);
        for (name, scene, cam) in [
            ("shadow_box", crate::testscenes::shadow_box(), 0),
            ("sphere_grid", crate::testscenes::sphere_grid(3, 2), 0),
            ("normal_map_back", back, 1),
        ] {
            let base = RenderOptions {
                width: 48,
                height: 48,
                scene_camera: Some(cam),
                ambient: 0.0,
                shading: ShadingMode::Pbr,
                shadows: true,
                background: BackgroundColor([10, 20, 30, 255]),
                ..RenderOptions::default()
            };
            // 4×4 SSAA so both backends box-filter edge coverage.
            let scan = crate::make_renderer(crate::RenderBackend::Scanline)
                .unwrap()
                .render(
                    &scene,
                    &RenderOptions {
                        aa: 4,
                        ..base.clone()
                    },
                )
                .unwrap();
            let mut o = base.clone();
            o.path_trace.max_bounces = 1;
            o.path_trace.samples_per_pixel = 16;
            let pt = PathTraceRenderer::new().render(&scene, &o).unwrap();
            let mae = crate::testscenes::mean_abs_error(&scan, &pt);
            if let Ok(dir) = std::env::var("PT_DUMP") {
                for (tag, img) in [("scan", &scan), ("pt", &pt)] {
                    let png = oxideav_png::encode_rgba8(
                        img.width,
                        img.height,
                        &img.pixels,
                        &Default::default(),
                    )
                    .unwrap();
                    std::fs::write(format!("{dir}/{name}_{tag}.png"), png).unwrap();
                }
            }
            eprintln!("{name}: direct-lighting MAE vs scanline = {mae:.3}");
            assert!(mae < 2.5, "{name}: mean abs error {mae}");
        }
    }

    #[test]
    fn renders_are_deterministic_and_refinement_is_additive() {
        let scene = area_light_scene(0.5, 0.0);
        let o = opts(9, 8, 4);
        let a = hdr(&scene, &o);
        let b = hdr(&scene, &o);
        assert_eq!(a, b);
        let mut pt = PathTracer::new();
        pt.sync(&scene, &o);
        pt.refine(3);
        pt.refine(5);
        assert_eq!(pt.samples(), 8);
        assert_eq!(pt.hdr(), a, "refine(3)+refine(5) must equal refine(8)");
        let mut seeded = o.clone();
        seeded.path_trace.seed = 1;
        assert_ne!(hdr(&scene, &seeded), a, "seed must change the noise");
    }

    #[test]
    fn sync_resets_only_on_radiance_changes() {
        let scene = area_light_scene(0.5, 0.0);
        let o = opts(6, 8, 2);
        let mut pt = PathTracer::new();
        assert!(pt.sync(&scene, &o));
        pt.refine(2);
        let mut display = o.clone();
        display.exposure = 2.0;
        display.tone_map = crate::ToneMap::AcesFitted;
        display.path_trace.samples_per_pixel = 1000;
        assert!(!pt.sync(&scene, &display));
        assert_eq!(pt.samples(), 2);
        assert!(!pt.is_converged());
        let mut moved = display.clone();
        moved.scene_camera = None;
        moved.projection = Projection::Orthographic;
        assert!(pt.sync(&scene, &moved));
        assert_eq!(pt.samples(), 0);
        pt.invalidate_scene();
        pt.refine(1);
        assert!(pt.sync(&scene, &moved));
    }

    #[test]
    fn primary_misses_keep_background_bytes() {
        let scene = area_light_scene(0.5, 0.0);
        let mut o = opts(16, 4, 1);
        o.background = BackgroundColor([200, 10, 77, 128]);
        o.scene_camera = None;
        o.camera = Some(crate::CameraSpec {
            elevation_deg: 0.0,
            azimuth_deg: 0.0,
            distance: 3.0,
        });
        let img = PathTraceRenderer::new().render(&scene, &o).unwrap();
        // Top rows look over the floor's horizon into empty space.
        assert_eq!(img.pixel(0, 0).unwrap(), [200, 10, 77, 128]);
        let empty = PathTraceRenderer::new()
            .render(&Scene3D::new(), &o)
            .unwrap();
        assert!(empty.pixels_rgba().all(|p| p == [200, 10, 77, 128]));
    }

    #[test]
    fn uniform_environment_lights_a_lambert_plane() {
        // White ρ = 0.5 Lambert plane under a uniform sky of radiance 1:
        // L = ρ · 1 = 0.5, by every estimator; the constant `ambient`
        // sky gives the same answer.
        let mut scene = Scene3D::new();
        let mat = scene.add_material(lambert([0.5; 3]));
        let mut floor = quad(
            [
                [-1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, -1.0],
                [-1.0, 0.0, -1.0],
            ],
            [1.0, 1.0],
        );
        floor.material = Some(mat);
        add_prim(&mut scene, floor);
        add_camera(&mut scene, [0.0, 2.0, 0.01], [0.0, 0.0, 0.0], 0.4);
        let env = Arc::new(EnvironmentMap::new(
            Arc::new(HdrImage::filled(16, 8, [1.0, 1.0, 1.0, 1.0])),
            1.0,
        ));
        for strategy in [
            LightStrategy::Mis,
            LightStrategy::LightOnly,
            LightStrategy::BsdfOnly,
        ] {
            let mut o = opts(6, 64, 2);
            o.path_trace.strategy = strategy;
            let mut r = PathTraceRenderer::new();
            r.tracer_mut().set_environment(Some(env.clone()));
            let m = mean_rgb(&r.render_hdr(&scene, &o).unwrap());
            assert!((m[0] - 0.5).abs() < 0.02, "{strategy:?}: {m:?}");
        }
        let mut o = opts(6, 64, 2);
        o.ambient = 1.0;
        let m = mean_rgb(&hdr(&scene, &o));
        assert!((m[0] - 0.5).abs() < 0.02, "ambient sky: {m:?}");
    }

    /// Emissive backdrop (Le = 1) seen through a glass cube, straight on.
    fn glass_scene(volume: bool) -> Scene3D {
        let mut scene = Scene3D::new();
        let mut glass = diffuse_material([1.0; 3], 0.0);
        glass.ext.transmission = Some(Transmission {
            factor: 1.0,
            factor_texture: None,
        });
        if volume {
            glass.ext.volume = Some(Volume {
                thickness: 1.0,
                thickness_texture: None,
                attenuation_distance: Some(1.0),
                attenuation_color: [0.5, 0.5, 0.5],
            });
        }
        let gm = scene.add_material(glass);
        let mut lm = lambert([0.0; 3]);
        lm.emissive_factor = [1.0; 3];
        let em = scene.add_material(lm);
        let mut cube = cuboid([-0.5; 3], [0.5; 3]);
        cube.material = Some(gm);
        add_prim(&mut scene, cube);
        let mut back = quad(
            [
                [-3.0, -3.0, -2.0],
                [3.0, -3.0, -2.0],
                [3.0, 3.0, -2.0],
                [-3.0, 3.0, -2.0],
            ],
            [1.0, 1.0],
        );
        back.material = Some(em);
        add_prim(&mut scene, back);
        add_camera(&mut scene, [0.0, 0.0, 4.0], [0.0, 0.0, 0.0], 0.15);
        scene
    }

    #[test]
    fn smooth_glass_transmits_with_fresnel_and_beer_lambert() {
        // Two interfaces at normal incidence: (1 − 0.04)² ≈ 0.92; the
        // 1-unit volume with attenuation 0.5 / 1 halves that.
        let thin = mean_rgb(&hdr(&glass_scene(false), &opts(4, 16, 8)))[0];
        let vol = mean_rgb(&hdr(&glass_scene(true), &opts(4, 16, 8)))[0];
        assert!((thin - 0.92).abs() < 0.03, "thin-walled: {thin}");
        assert!((vol - 0.46).abs() < 0.03, "volume: {vol}");
    }

    #[test]
    fn point_lights_and_clamp() {
        let mut scene = Scene3D::new();
        let mat = scene.add_material(lambert([1.0; 3]));
        let mut floor = quad(
            [
                [-1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, -1.0],
                [-1.0, 0.0, -1.0],
            ],
            [1.0, 1.0],
        );
        floor.material = Some(mat);
        add_prim(&mut scene, floor);
        crate::testscenes::add_light(
            &mut scene,
            Light::Point {
                color: [1.0; 3],
                intensity: 4.0,
                range: None,
            },
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        );
        add_camera(&mut scene, [0.0, 3.0, 0.001], [0.0, 0.0, 0.0], 0.05);
        // Straight below a 4 cd point light at 1 m: E = 4, L = E/π.
        let l = mean_rgb(&hdr(&scene, &opts(2, 4, 1)))[0];
        assert!((l - 4.0 / PI as f64).abs() < 0.01, "{l}");
        let mut o = opts(2, 4, 1);
        o.path_trace.clamp = 0.5;
        let c = mean_rgb(&hdr(&scene, &o))[0];
        assert!((c - 0.5).abs() < 1e-3, "clamped: {c}");
    }
}
