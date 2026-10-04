//! glTF 2.0 metallic-roughness BRDF (glTF 2.0 specification,
//! Appendix B "BRDF Implementation"), shared by every backend.
//!
//! * Microfacet distribution — Trowbridge-Reitz / GGX
//!   (Walter, Marschner, Li, Torrance, "Microfacet Models for
//!   Refraction through Rough Surfaces", EGSR 2007):
//!   `D = α² / (π ((N·H)² (α² − 1) + 1)²)` with `α = roughness²`
//!   (Burley, "Physically-Based Shading at Disney", SIGGRAPH 2012
//!   course notes).
//! * Masking-shadowing — Smith height-correlated form for GGX
//!   (Heitz, "Understanding the Masking-Shadowing Function in
//!   Microfacet-Based BRDFs", JCGT 3(2), 2014), folded with the
//!   `1 / (4 |N·L| |N·V|)` denominator into the visibility term
//!   `V = 0.5 / (N·L √((N·V)²(1 − α²) + α²) + N·V √((N·L)²(1 − α²) + α²))`.
//! * Fresnel — Schlick, "An Inexpensive BRDF Model for
//!   Physically-based Rendering", Eurographics 1994:
//!   `F = F0 + (1 − F0)(1 − |V·H|)⁵`.
//! * Diffuse — Lambert, `c_diff / π`.
//!
//! Material mixing per Appendix B: `c_diff = lerp(baseColor, 0,
//! metallic)`, `F0 = lerp(f0_dielectric, baseColor, metallic)`,
//! `f = (1 − F) · c_diff / π + F · D · V`.

use std::f32::consts::PI;

/// Smallest GGX `α` used, keeping point-light highlights on perfectly
/// smooth surfaces finite.
pub const MIN_ALPHA: f32 = 1.0e-3;

/// GGX / Trowbridge-Reitz normal distribution `D(h)`.
pub fn d_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d).max(1.0e-12)
}

/// Height-correlated Smith visibility `V = G / (4 N·L N·V)` for GGX.
pub fn v_smith_ggx_correlated(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let gv = n_dot_l * (n_dot_v * n_dot_v * (1.0 - a2) + a2).sqrt();
    let gl = n_dot_v * (n_dot_l * n_dot_l * (1.0 - a2) + a2).sqrt();
    let s = gv + gl;
    if s > 0.0 {
        0.5 / s
    } else {
        0.0
    }
}

/// Schlick Fresnel.
pub fn f_schlick(f0: [f32; 3], v_dot_h: f32) -> [f32; 3] {
    let k = (1.0 - v_dot_h.clamp(0.0, 1.0)).powi(5);
    [
        f0[0] + (1.0 - f0[0]) * k,
        f0[1] + (1.0 - f0[1]) * k,
        f0[2] + (1.0 - f0[2]) * k,
    ]
}

/// Inputs of [`eval`] derived from the material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrdfParams {
    /// Diffuse colour `c_diff`.
    pub c_diff: [f32; 3],
    /// Normal-incidence specular reflectance `F0`.
    pub f0: [f32; 3],
    /// GGX `α` (`roughness²`, clamped to [`MIN_ALPHA`]).
    pub alpha: f32,
}

impl BrdfParams {
    /// Appendix B parameterisation from base colour (linear RGB),
    /// metallic, perceptual roughness and the dielectric `F0` (0.04
    /// for IOR 1.5).
    pub fn metallic_roughness(
        base: [f32; 3],
        metallic: f32,
        roughness: f32,
        f0_dielectric: f32,
    ) -> Self {
        let m = metallic.clamp(0.0, 1.0);
        let r = roughness.clamp(0.0, 1.0);
        Self {
            c_diff: [
                base[0] * (1.0 - m),
                base[1] * (1.0 - m),
                base[2] * (1.0 - m),
            ],
            f0: [
                f0_dielectric + (base[0] - f0_dielectric) * m,
                f0_dielectric + (base[1] - f0_dielectric) * m,
                f0_dielectric + (base[2] - f0_dielectric) * m,
            ],
            alpha: (r * r).max(MIN_ALPHA),
        }
    }
}

/// Evaluate `f(l, v) · (N·L)` for unit vectors `n`, `v` (toward the
/// viewer) and `l` (toward the light). Zero when either direction is
/// below the surface.
pub fn eval(p: &BrdfParams, n: [f32; 3], v: [f32; 3], l: [f32; 3]) -> [f32; 3] {
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let n_dot_l = dot(n, l);
    if n_dot_l <= 0.0 {
        return [0.0; 3];
    }
    // Viewer slightly below the shading normal (normal maps): clamp.
    let n_dot_v = dot(n, v).max(1.0e-4);
    let h = [l[0] + v[0], l[1] + v[1], l[2] + v[2]];
    let hl = dot(h, h).sqrt();
    let h = if hl > 0.0 {
        [h[0] / hl, h[1] / hl, h[2] / hl]
    } else {
        n
    };
    let n_dot_h = dot(n, h).max(0.0);
    let v_dot_h = dot(v, h).max(0.0);
    let f = f_schlick(p.f0, v_dot_h);
    let spec = d_ggx(n_dot_h, p.alpha) * v_smith_ggx_correlated(n_dot_l, n_dot_v, p.alpha);
    let mut out = [0.0; 3];
    for k in 0..3 {
        out[k] = ((1.0 - f[k]) * p.c_diff[k] / PI + f[k] * spec) * n_dot_l;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ggx_integrates_to_one_over_projected_hemisphere() {
        // ∫ D(h) (n·h) dω = 1 — midpoint quadrature in θ.
        for alpha in [0.1f32, 0.5, 1.0] {
            let n = 4000;
            let mut sum = 0.0f64;
            for i in 0..n {
                let th = (i as f64 + 0.5) / n as f64 * std::f64::consts::FRAC_PI_2;
                let c = th.cos() as f32;
                sum += d_ggx(c, alpha) as f64
                    * c as f64
                    * th.sin()
                    * 2.0
                    * std::f64::consts::PI
                    * (std::f64::consts::FRAC_PI_2 / n as f64);
            }
            assert!((sum - 1.0).abs() < 0.02, "alpha {alpha}: {sum}");
        }
    }

    #[test]
    fn white_furnace_lambert_bound() {
        // A rough dielectric white surface must not reflect more energy
        // than it receives (rough hemispherical estimate).
        let p = BrdfParams::metallic_roughness([1.0; 3], 0.0, 1.0, 0.04);
        let n = [0.0, 0.0, 1.0];
        let v = n;
        let mut sum = 0.0;
        let steps = 200;
        for i in 0..steps {
            for j in 0..steps {
                let th = (i as f32 + 0.5) / steps as f32 * std::f32::consts::FRAC_PI_2;
                let ph = (j as f32 + 0.5) / steps as f32 * 2.0 * PI;
                let l = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
                let dw = th.sin()
                    * (std::f32::consts::FRAC_PI_2 / steps as f32)
                    * (2.0 * PI / steps as f32);
                sum += eval(&p, n, v, l)[0] * dw;
            }
        }
        assert!(sum <= 1.02 && sum > 0.8, "{sum}");
    }

    #[test]
    fn metals_have_no_diffuse_and_tinted_f0() {
        let p = BrdfParams::metallic_roughness([1.0, 0.5, 0.0], 1.0, 0.5, 0.04);
        assert_eq!(p.c_diff, [0.0; 3]);
        assert_eq!(p.f0, [1.0, 0.5, 0.0]);
        assert_eq!(f_schlick([0.04; 3], 0.0), [1.0; 3]);
    }

    #[test]
    fn below_horizon_is_black() {
        let p = BrdfParams::metallic_roughness([1.0; 3], 0.0, 0.5, 0.04);
        assert_eq!(
            eval(&p, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]),
            [0.0; 3]
        );
    }
}
