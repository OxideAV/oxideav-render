//! Shading helpers shared by the scanline and raycast backends —
//! directional light construction, the Lambert-with-ambient pixel
//! shade, and linear → sRGB output encoding (IEC 61966-2-1).
//!
//! Keeping these in one module guarantees the two backends produce
//! matching colours for a surface with the same normal under the same
//! light, which the cross-backend consistency tests rely on.

use crate::options::LightSpec;

/// Constant ambient term added to every shaded pixel so back-faces
/// stay visible at the price of contrast.
pub(crate) const AMBIENT: f32 = 0.2;

/// Resolved directional light.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DirLight {
    /// Unit vector pointing TOWARD the light source from the surface.
    pub(crate) direction: [f32; 3],
    /// Diffuse multiplier.
    pub(crate) intensity: f32,
}

/// Resolve a [`LightSpec`] (azimuth / elevation in degrees) into a
/// unit direction + intensity.
pub(crate) fn build_light(spec: LightSpec) -> DirLight {
    let az = spec.azimuth_deg.to_radians();
    let el = spec.elevation_deg.to_radians();
    let cos_el = el.cos();
    let dir = [cos_el * az.sin(), el.sin(), cos_el * az.cos()];
    DirLight {
        direction: crate::math::vec3_normalise(dir),
        intensity: spec.intensity,
    }
}

/// Lambertian diffuse + ambient. `normal` is in world space, `light`
/// carries a unit direction TOWARD the light. Result is in linear
/// space, alpha pulled straight from the base colour.
pub(crate) fn shade_pixel(base: [f32; 4], normal: [f32; 3], light: &DirLight) -> [f32; 4] {
    let cos_theta = crate::math::vec3_dot(normal, light.direction).max(0.0);
    let diffuse = AMBIENT + (1.0 - AMBIENT) * cos_theta * light.intensity;
    let factor = diffuse.clamp(0.0, 1.0);
    [
        base[0] * factor,
        base[1] * factor,
        base[2] * factor,
        base[3],
    ]
}

/// Encode a linear-space RGBA colour (`0..=1` floats) into packed
/// sRGB bytes; alpha stays linear.
pub(crate) fn linear_rgba_to_srgb_u8(c: [f32; 4]) -> [u8; 4] {
    [
        linear_to_srgb_u8(c[0]),
        linear_to_srgb_u8(c[1]),
        linear_to_srgb_u8(c[2]),
        (c[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

/// Encode one linear-space channel into an sRGB byte per the
/// IEC 61966-2-1 piecewise curve.
pub(crate) fn linear_to_srgb_u8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let v = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_to_srgb_handles_endpoints() {
        assert_eq!(linear_to_srgb_u8(0.0), 0);
        assert_eq!(linear_to_srgb_u8(1.0), 255);
    }

    #[test]
    fn linear_to_srgb_clamps_out_of_range_and_nan() {
        assert_eq!(linear_to_srgb_u8(-1.0), 0);
        assert_eq!(linear_to_srgb_u8(2.0), 255);
        // NaN propagates through `clamp` and the curve; the final
        // float → integer cast saturates NaN to 0 — pin that.
        assert_eq!(linear_to_srgb_u8(f32::NAN), 0);
    }

    #[test]
    fn shade_full_normal_alignment_is_brightest() {
        let light = build_light(LightSpec {
            azimuth_deg: 0.0,
            elevation_deg: 90.0,
            intensity: 1.0,
        });
        let aligned = shade_pixel([1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0], &light);
        let opposed = shade_pixel([1.0, 1.0, 1.0, 1.0], [0.0, -1.0, 0.0], &light);
        assert!((aligned[0] - 1.0).abs() < 1.0e-5);
        assert!(
            (opposed[0] - AMBIENT).abs() < 1.0e-5,
            "back-face gets ambient only"
        );
    }

    #[test]
    fn shade_preserves_alpha() {
        let light = build_light(LightSpec::default_light());
        let c = shade_pixel([0.5, 0.5, 0.5, 0.25], [0.0, 1.0, 0.0], &light);
        assert!((c[3] - 0.25).abs() < 1.0e-6);
    }

    #[test]
    fn zero_intensity_light_leaves_ambient() {
        let light = build_light(LightSpec {
            intensity: 0.0,
            ..LightSpec::default_light()
        });
        let c = shade_pixel([1.0, 1.0, 1.0, 1.0], light.direction, &light);
        assert!((c[0] - AMBIENT).abs() < 1.0e-5);
    }

    #[test]
    fn build_light_direction_is_unit() {
        let l = build_light(LightSpec::default_light());
        let len = (l.direction[0] * l.direction[0]
            + l.direction[1] * l.direction[1]
            + l.direction[2] * l.direction[2])
            .sqrt();
        assert!((len - 1.0).abs() < 1.0e-5);
    }
}
