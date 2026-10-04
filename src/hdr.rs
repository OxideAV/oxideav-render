//! Linear-light HDR framebuffer output, tone mapping, and the sRGB
//! transfer function.
//!
//! Backends shade in **scene-linear** RGB (`f32`, unbounded above).
//! [`HdrImage`] is that radiance buffer, exposed so callers can write
//! floating-point formats (OpenEXR, PFM) without the 8-bit quantisation
//! of [`crate::RgbaImage`]. [`HdrImage::to_rgba8`] runs the display
//! transform: `exposure` scale → [`ToneMap`] operator → sRGB encode.
//!
//! Sources:
//!
//! * sRGB transfer function — IEC 61966-2-1:1999 piecewise curve.
//! * [`ToneMap::Reinhard`] — Reinhard, Stark, Shirley, Ferwerda,
//!   "Photographic Tone Reproduction for Digital Images", SIGGRAPH
//!   2002, eq. 3 (`L_d = L / (1 + L)`), applied to luminance
//!   (Rec. 709 weights) with the chromaticity preserved.
//! * [`ToneMap::AcesFitted`] — Krzysztof Narkowicz, "ACES Filmic Tone
//!   Mapping Curve" (2016 blog post): the rational fit
//!   `x(2.51x + 0.03) / (x(2.43x + 0.59) + 0.14)` with the 0.6
//!   pre-scale the post applies so that the fit matches the ACES
//!   reference rendering transform's mid-grey.

use crate::image::RgbaImage;

/// Tone-mapping operator applied when converting scene-linear
/// radiance to display values.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToneMap {
    /// Hard clamp to `[0, 1]` — the historical behaviour of every
    /// backend and the default. Values above 1 clip.
    #[default]
    Clamp,
    /// Reinhard et al. 2002 global operator `L / (1 + L)` on
    /// luminance, chromaticity preserved.
    Reinhard,
    /// Narkowicz's fitted ACES filmic curve (per channel).
    AcesFitted,
}

impl ToneMap {
    /// Map one scene-linear colour (already exposure-scaled) to
    /// display-linear `[0, 1]`.
    pub fn apply(self, rgb: [f32; 3]) -> [f32; 3] {
        let rgb = [
            sanitize(rgb[0]).max(0.0),
            sanitize(rgb[1]).max(0.0),
            sanitize(rgb[2]).max(0.0),
        ];
        match self {
            ToneMap::Clamp => [rgb[0].min(1.0), rgb[1].min(1.0), rgb[2].min(1.0)],
            ToneMap::Reinhard => {
                let l = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
                if l <= 0.0 {
                    return [0.0; 3];
                }
                let s = 1.0 / (1.0 + l);
                [
                    (rgb[0] * s).min(1.0),
                    (rgb[1] * s).min(1.0),
                    (rgb[2] * s).min(1.0),
                ]
            }
            ToneMap::AcesFitted => [aces(rgb[0]), aces(rgb[1]), aces(rgb[2])],
        }
    }
}

fn sanitize(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else {
        v
    }
}

fn aces(x: f32) -> f32 {
    let x = x * 0.6;
    let v = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    v.clamp(0.0, 1.0)
}

/// sRGB-encoded `[0, 1]` value → linear (IEC 61966-2-1).
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear `[0, 1]` value → sRGB-encoded `[0, 1]` (IEC 61966-2-1).
/// Input is clamped; NaN maps to 0.
pub fn linear_to_srgb(c: f32) -> f32 {
    let c = sanitize(c).clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// 256-entry sRGB byte → linear lookup table.
pub(crate) fn srgb_u8_lut() -> &'static [f32; 256] {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (i, v) in t.iter_mut().enumerate() {
            *v = srgb_to_linear(i as f32 / 255.0);
        }
        t
    })
}

/// Linear `[0, 1]` → sRGB byte.
///
/// Equivalent to `round(linear_to_srgb(c) * 255)`, evaluated as a
/// binary search over the 255 linear-light decision thresholds
/// (`srgb_to_linear((k + 0.5) / 255)`) instead of a `powf` per call.
pub(crate) fn linear_to_srgb_byte(c: f32) -> u8 {
    static T: std::sync::OnceLock<[f32; 255]> = std::sync::OnceLock::new();
    let t = T.get_or_init(|| {
        let mut t = [0.0f32; 255];
        for (k, v) in t.iter_mut().enumerate() {
            *v = srgb_to_linear((k as f32 + 0.5) / 255.0);
        }
        t
    });
    if c.is_nan() {
        return 0;
    }
    t.partition_point(|&th| th <= c) as u8
}

/// Scene-linear RGBA `f32` image — the pre-tone-map output of
/// [`crate::Renderer::render_hdr`].
///
/// `pixels` is row-major, tightly packed, 4 floats per pixel
/// (`R, G, B, A`), no stride padding. Colour is linear-light
/// (Rec. 709 / sRGB primaries), unbounded above; alpha is coverage in
/// `[0, 1]` (straight, not premultiplied).
#[derive(Debug, Clone, PartialEq)]
pub struct HdrImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height * 4` floats.
    pub pixels: Vec<f32>,
}

impl HdrImage {
    /// A `width × height` image filled with `rgba`.
    pub fn filled(width: u32, height: u32, rgba: [f32; 4]) -> Self {
        let n = width as usize * height as usize;
        let mut pixels = Vec::with_capacity(n * 4);
        for _ in 0..n {
            pixels.extend_from_slice(&rgba);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Pixel at `(x, y)`, `None` when out of bounds.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[f32; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = (y as usize * self.width as usize + x as usize) * 4;
        Some([
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ])
    }

    /// Display transform: multiply colour by `exposure`, apply
    /// `tone_map`, sRGB-encode to bytes. Alpha is quantised linearly.
    pub fn to_rgba8(&self, tone_map: ToneMap, exposure: f32) -> RgbaImage {
        let mut out = Vec::with_capacity(self.pixels.len());
        for p in self.pixels.chunks_exact(4) {
            let m = tone_map.apply([p[0] * exposure, p[1] * exposure, p[2] * exposure]);
            out.push(linear_to_srgb_byte(m[0]));
            out.push(linear_to_srgb_byte(m[1]));
            out.push(linear_to_srgb_byte(m[2]));
            out.push((sanitize(p[3]).clamp(0.0, 1.0) * 255.0).round() as u8);
        }
        RgbaImage {
            width: self.width,
            height: self.height,
            stride: self.width as usize * 4,
            pixels: out,
        }
    }

    /// Inverse of an identity display transform: decode an 8-bit sRGB
    /// image to linear floats. Used as the default
    /// [`crate::Renderer::render_hdr`] for backends without a native
    /// float path.
    pub fn from_rgba8(img: &RgbaImage) -> Self {
        let lut = srgb_u8_lut();
        let mut pixels = Vec::with_capacity(img.width as usize * img.height as usize * 4);
        for p in img.pixels_rgba() {
            pixels.push(lut[p[0] as usize]);
            pixels.push(lut[p[1] as usize]);
            pixels.push(lut[p[2] as usize]);
            pixels.push(p[3] as f32 / 255.0);
        }
        Self {
            width: img.width,
            height: img.height,
            pixels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_round_trips_every_byte() {
        let lut = srgb_u8_lut();
        for b in 0..=255u8 {
            assert_eq!(linear_to_srgb_byte(lut[b as usize]), b);
        }
    }

    #[test]
    fn threshold_encoder_matches_powf_curve() {
        for i in 0..=20_000 {
            let c = i as f32 / 20_000.0;
            let want = (linear_to_srgb(c) * 255.0).round() as i32;
            let got = linear_to_srgb_byte(c) as i32;
            assert!((want - got).abs() <= 1, "{c}: {want} vs {got}");
        }
        assert_eq!(linear_to_srgb_byte(-1.0), 0);
        assert_eq!(linear_to_srgb_byte(7.0), 255);
    }

    #[test]
    fn clamp_is_identity_in_range() {
        assert_eq!(ToneMap::Clamp.apply([0.25, 0.5, 1.0]), [0.25, 0.5, 1.0]);
        assert_eq!(ToneMap::Clamp.apply([4.0, -1.0, f32::NAN]), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn reinhard_compresses_and_is_monotone() {
        let a = ToneMap::Reinhard.apply([1.0, 1.0, 1.0]);
        assert!((a[0] - 0.5).abs() < 1e-5, "L=1 maps to 0.5, got {a:?}");
        let b = ToneMap::Reinhard.apply([100.0, 100.0, 100.0]);
        assert!(b[0] > a[0] && b[0] < 1.0);
    }

    #[test]
    fn aces_fit_matches_published_points() {
        // f(0) = 0, saturates to 1 for large input, monotone.
        assert_eq!(ToneMap::AcesFitted.apply([0.0; 3])[0], 0.0);
        assert!((ToneMap::AcesFitted.apply([1000.0; 3])[0] - 1.0).abs() < 1e-3);
        let mut last = 0.0;
        for i in 1..100 {
            let v = ToneMap::AcesFitted.apply([i as f32 * 0.1; 3])[0];
            assert!(v >= last);
            last = v;
        }
        // Mid-grey 0.18 lands near 0.18*0.6 fitted ≈ 0.14..0.2.
        let g = ToneMap::AcesFitted.apply([0.18; 3])[0];
        assert!((0.1..0.25).contains(&g), "{g}");
    }

    #[test]
    fn hdr_to_rgba8_and_back() {
        let img = HdrImage::filled(2, 1, [0.5, 0.0, 2.0, 1.0]);
        let ldr = img.to_rgba8(ToneMap::Clamp, 1.0);
        assert_eq!(ldr.pixel(0, 0), Some([188, 0, 255, 255]));
        let back = HdrImage::from_rgba8(&ldr);
        assert!((back.pixel(1, 0).unwrap()[0] - 0.5).abs() < 0.01);
        let dark = img.to_rgba8(ToneMap::Clamp, 0.0);
        assert_eq!(dark.pixel(0, 0), Some([0, 0, 0, 255]));
    }
}
